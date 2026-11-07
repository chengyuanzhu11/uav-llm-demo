"""Gazebo RGB-D transport with synchronized images and measured camera transforms."""
import base64
from collections import deque
from copy import deepcopy
import io
import math
import threading
import time

from .runtime import MAX_FEEDBACK_GAP_SECONDS, RuntimeUnavailableError, StaleFeedbackError


def stamp(message):
    t=message.header.stamp
    return t.sec+t.nsec*1e-9


def rotation(q):
    x,y,z,w=q.x,q.y,q.z,q.w
    import numpy as np
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


class GazeboCamera:
    # 250 ms of simulator motion is 1 m at the 4 m/s transit limit.
    MAX_POSE_AGE = .25

    def __init__(self, world, vehicle, partition="uav_llm"):
        from gz.transport13 import Node, NodeOptions, SubscribeOptions
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.camera_info_pb2 import CameraInfo
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.msgs10.clock_pb2 import Clock
        from gz.msgs10.double_pb2 import Double
        options=NodeOptions();options.partition=partition
        self.node=Node(options);self.world=world;self.vehicle=vehicle
        self.time=None;self._clock_published=None;self._closed=False;self._calibration=None
        self.vehicle_sample=None;self.vehicle_sample_received=0.
        self._vehicle_path=deque(maxlen=512)  # ~5 s at the 100 Hz pose publication rate.
        self._condition=threading.Condition()
        self._capture_lock=threading.Lock()
        self._images={};self._depths={};self._poses=deque(maxlen=400)
        self._subscriptions=[]
        root=f"/model/{vehicle}"
        self._yaw=self.node.advertise(root+'/gimbal_yaw/cmd_pos',Double)
        self._pitch=self.node.advertise(root+'/gimbal_pitch/cmd_pos',Double)
        self._image_topics=[(Image,root+'/rgbd/image',self._rgb),
                            (Image,root+'/rgbd/depth_image',self._depth)]
        for typ,topic,cb in [(CameraInfo,root+'/rgbd/camera_info',self._camera_info),
                             (Pose_V,root+'/pose',self._pose),
                             (Clock,f'/world/{world}/clock',self._clock)]:
            options=SubscribeOptions()
            if typ is Clock:
                options.msgs_per_sec=50  # Control runs at 20 Hz. Avoid queuing 250 clocks/s behind RGB-D.
            if not self.node.subscribe(typ,topic,cb,options):raise RuntimeUnavailableError('Cannot subscribe to Gazebo camera transport')
            self._subscriptions.append(topic)

    def _clock(self, msg):
        with self._condition:
            self.time=msg.sim.sec+msg.sim.nsec*1e-9
            published=msg.system.sec+msg.system.nsec*1e-9
            self._clock_published=time.monotonic()-max(0.,time.time()-published)

    def _camera_info(self, msg):
        with self._condition:
            self._calibration=msg
            self._condition.notify_all()

    def _store(self, mapping, msg):
        with self._condition:
            mapping[round(stamp(msg),9)]=msg
            while len(mapping)>8:mapping.pop(next(iter(mapping)))
            self._condition.notify_all()

    def _rgb(self,msg):self._store(self._images,msg)
    def _depth(self,msg):self._store(self._depths,msg)

    def _pose(self,msg):
        import numpy as np
        # PosePublisher supplies a transform tree. Compose measured link poses,
        # rather than reconstructing a level camera from requested angles.
        poses={}
        for p in msg.pose:
            headers={d.key:d.value[0] for d in p.header.data if d.value}
            poses[headers.get('child_frame_id',p.name)]=(headers.get('frame_id',self.world),p)
        resolved={}
        def resolve(name, seen):
            if name in resolved:return resolved[name]
            if name in seen:raise ValueError('Cyclic camera frame tree')
            if name not in poses:
                if name in (self.world,'world',''):return np.zeros(3),np.eye(3)
                raise ValueError('Missing camera parent transform')
            parent,pose=poses[name]
            pp,rr=resolve(parent,seen|{name})
            p=pose.position
            resolved[name]=(pp+rr@np.array([p.x,p.y,p.z]),rr@rotation(pose.orientation))
            return resolved[name]
        try:
            vehicle_time=stamp(poses[self.vehicle][1])
            vehicle_p,vehicle_r=resolve(self.vehicle,set())
            with self._condition:
                previous=self.vehicle_sample
                if previous is None or vehicle_time>previous[0]:
                    velocity=(np.zeros(3) if previous is None else
                              (vehicle_p-previous[1])/(vehicle_time-previous[0]))
                    self.vehicle_sample=(vehicle_time,vehicle_p,velocity,vehicle_r)
                    self.vehicle_sample_received=time.monotonic()
                    self._vehicle_path.append((vehicle_time,vehicle_p))
            if self.time is not None and self.time-vehicle_time>self.MAX_POSE_AGE:
                return  # Keep late measurements for path verification, never for new motion.
            name=next(n for n in poses if n.endswith('camera_link'))
            p,r=resolve(name,set())
            body=next((n for n in poses if n.endswith('base_link')),None)
            _,br=resolve(body,set()) if body else (None,r)
            with self._condition:
                # PosePublisher stamps each Pose, not the surrounding Pose_V.
                self._poses.append((stamp(poses[name][1]),p,r,br))
                self._condition.notify_all()
        except (KeyError,StopIteration,ValueError):pass

    def reference_fresh(self, sampled):
        with self._condition:
            return (sampled is not None and self._clock_published is not None
                    and time.monotonic()-self._clock_published <= self.MAX_POSE_AGE
                    and abs(self.time-sampled) <= self.MAX_POSE_AGE)

    def vehicle_state(self):
        with self._condition:
            now=time.monotonic()
            if self.vehicle_sample is not None and now-self.vehicle_sample_received > MAX_FEEDBACK_GAP_SECONDS:
                raise RuntimeUnavailableError('超过 1 秒没有收到新的 Gazebo 飞行位置，已停止任务。')
            if (self.vehicle_sample is None or now-self.vehicle_sample_received>self.MAX_POSE_AGE
                    or not self.reference_fresh(self.vehicle_sample[0])):
                raise StaleFeedbackError('Gazebo vehicle pose is missing or stale')
            sampled,position,velocity,orientation=self.vehicle_sample
            return position.copy(),velocity.copy(),orientation.copy(),sampled

    def vehicle_path(self, start_time, end_time):
        with self._condition:
            return [(t,p) for t,p in self._vehicle_path if start_time < t < end_time]

    def capture(self, request, clock_origin):
        # Subscribe to the large pixel streams only for an observation. Route
        # tracking still receives pose and world time throughout the flight.
        with self._capture_lock:
            if self._closed:raise RuntimeUnavailableError('Camera closed')
            topics=[]
            try:
                for typ,topic,cb in self._image_topics:
                    if not self.node.subscribe(typ,topic,cb):
                        raise RuntimeUnavailableError('Cannot subscribe to Gazebo RGB-D')
                    topics.append(topic)
                return self._capture(request, clock_origin)
            finally:
                for topic in topics:self.node.unsubscribe(topic)
                with self._condition:
                    self._images.clear()
                    self._depths.clear()

    def _capture(self, request, clock_origin):
        import numpy as np
        from PIL import Image
        from gz.msgs10.double_pb2 import Double
        deadline=time.monotonic()+15
        with self._condition:
            while not self._poses or self._calibration is None:
                if self._closed:raise RuntimeUnavailableError('Camera closed')
                if time.monotonic()>deadline:raise RuntimeUnavailableError('Measured camera pose unavailable')
                self._condition.wait(.05)
            body=self._poses[-1][3]
        desired=np.array([math.cos(math.radians(request['pitch_deg']))*math.cos(math.radians(request['yaw_deg'])),
                          -math.cos(math.radians(request['pitch_deg']))*math.sin(math.radians(request['yaw_deg'])),
                          math.sin(math.radians(request['pitch_deg']))])
        # Resolve the requested world direction in the measured body frame,
        # including roll/pitch; yaw-only compensation mis-aims a tilted UAV.
        local=body.T@desired
        target_yaw=math.atan2(local[1],local[0])
        target_pitch=math.atan2(-local[2],math.hypot(local[0],local[1]))
        self._yaw.publish(Double(data=target_yaw));self._pitch.publish(Double(data=target_pitch))
        after=(self.time or 0)+.5
        last_ts=aligned_since=None
        while not self._closed:
            with self._condition:
                shared=sorted(k for k in self._images.keys() & self._depths.keys() if k>=after)
                if shared and self._poses:
                    ts=shared[-1];pose=min(self._poses,key=lambda p:abs(p[0]-ts))
                    if abs(pose[0]-ts)<=.015:
                        rgb,depth=self._images[ts],self._depths[ts]
                        p,r=pose[1:3]
                        # A single frame passing through the target is not a
                        # settled view. Require consecutive sensor frames within
                        # one degree for 0.2 simulation seconds.
                        if ts != last_ts:
                            if float(r[:,0]@desired) >= math.cos(math.radians(1)):
                                if aligned_since is None or last_ts is None or ts-last_ts>.25:
                                    aligned_since=ts
                                if ts-aligned_since>=.2-1e-6:break
                            else:
                                aligned_since=None
                            last_ts=ts
                if time.monotonic()>deadline:raise RuntimeUnavailableError('Synchronized RGB-D and camera pose timed out')
                self._condition.wait(.03)
        else:raise RuntimeUnavailableError('Camera closed')
        if rgb.width!=depth.width or rgb.height!=depth.height:raise ValueError('RGB-D resolution mismatch')
        width,height=rgb.width,rgb.height
        calibration=self._calibration
        k=list(calibration.intrinsics.k)
        if (calibration.width!=width or calibration.height!=height or len(k)!=9
                or k[0]<=0 or k[4]<=0 or any(abs(v)>1e-9 for v in calibration.distortion.k)):
            raise ValueError('RGB-D needs matching rectified camera intrinsics')
        pixels=np.frombuffer(rgb.data,dtype=np.uint8).reshape(height,rgb.step)[:,:width*3].reshape(height,width,3)
        values=np.frombuffer(depth.data,dtype='<f4').reshape(height,depth.step//4)[:,:width]
        values=np.nan_to_num(values,nan=0.,posinf=0.,neginf=0.)
        values=np.where((values>0)&(values<request['far_m']),values,0.)
        native_vfov=math.degrees(2*math.atan(height/(2*k[4])))
        factor=min(1.,math.tan(math.radians(request['vertical_fov_deg']/2))/math.tan(math.radians(native_vfov/2)))
        ch=max(2,int(height*factor));cw=max(2,int(width*factor));top=(height-ch)//2;left=(width-cw)//2
        # Preserve native detail for model pixel grounding; UI display size is
        # independent of sensor resolution. Digital zoom still shares RGB/depth.
        out=(width,height)
        # Registered RGB and optical-axis depth use the same crop and sample locations.
        im=Image.fromarray(pixels[top:top+ch,left:left+cw]).resize(out,Image.Resampling.NEAREST)
        dm=Image.fromarray(values[top:top+ch,left:left+cw]).resize(out,Image.Resampling.NEAREST)
        buf=io.BytesIO();im.save(buf,format='PNG')
        meta=deepcopy(request)
        flip=np.diag([1.,-1.,1.]);pos=flip@p
        meta.update(source='gazebo_rgbd',simulation_time=ts-clock_origin,vehicle_id=self.vehicle,camera_id='rgbd',
                    width=out[0],height=out[1],
                    position_grid=(pos/np.array([3.,3.,5.])).tolist(),fx=k[0]*out[0]/cw,fy=k[4]*out[1]/ch,
                    vertical_fov_deg=math.degrees(2*math.atan(ch/(2*k[4]))),
                    cx=(k[2]-left)*out[0]/cw,cy=(k[5]-top)*out[1]/ch,
                    camera_basis=[(flip@r[:,0]).tolist(),(-flip@r[:,1]).tolist(),(flip@r[:,2]).tolist()],
                    pose_time_error_s=abs(pose[0]-ts),sensor_time=ts)
        forward=flip@r[:,0]
        meta.update(yaw_deg=math.degrees(math.atan2(forward[1],forward[0])),
                    pitch_deg=math.degrees(math.asin(float(np.clip(forward[2],-1,1)))),far_m=100.)
        return meta,dict(frame_id=meta['frame_id'],rgb='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),
                         depth_m=np.asarray(dm).reshape(-1).tolist())

    def close(self):
        with self._condition:
            self._closed=True
            self._condition.notify_all()
        with self._capture_lock:
            if self.node is None:return
            for topic in self._subscriptions:
                self.node.unsubscribe(topic)
            self._subscriptions.clear()
            # Release native transport threads while Python is still running;
            # bound callbacks otherwise retain a Node until interpreter teardown.
            self._yaw = self._pitch = None
            self.node = None
