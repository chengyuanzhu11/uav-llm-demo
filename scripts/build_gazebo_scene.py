"""Build a Gazebo world and an instrumented PX4 X500 from the shared scene."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uav_llm_demo.simulation_scene import campus_collision_geometry
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.scene_layout import scene_layout

ROOT = Path(__file__).resolve().parents[1]


def export_visual(scenario, objects, destination):
    directory = destination / 'models' / scenario.scenario_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(scenario=scenario.to_dict(), objects=objects), sort_keys=True)
    digest = hashlib.sha256(payload.encode())
    for source in ('scripts/export_gazebo_visual.cjs', 'package-lock.json',
                   'uav_llm_demo/static/semantic_campus.js', 'uav_llm_demo/static/campus_landscape.js'):
        digest.update((ROOT/source).read_bytes())
    fingerprint = digest.hexdigest()
    mesh, manifest = directory/'scenery.glb', directory/'export.json'
    if mesh.exists() and manifest.exists():
        try:
            previous = json.loads(manifest.read_text())
        except (OSError, ValueError):
            previous = None
        if (isinstance(previous, dict) and previous.get('source_sha256') == fingerprint
                and previous.get('mesh_sha256') == hashlib.sha256(mesh.read_bytes()).hexdigest()):
            return mesh
    source = directory/'scene.json'
    source.write_text(payload)
    result = subprocess.run(['node', str(ROOT/'scripts/export_gazebo_visual.cjs'), str(source), str(mesh)],
                            cwd=ROOT, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError('Scene export failed. Run npm ci and npx playwright install chromium --only-shell.\n'+result.stderr[-1500:])
    metadata = json.loads(result.stdout.strip().splitlines()[-1])
    metadata.update(source_sha256=fingerprint, mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest())
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata, indent=2)+'\n')
    temporary.replace(manifest)
    return mesh


def add(parent,tag,text=None,**attrs):
    e=ET.SubElement(parent,tag,attrs)
    if text is not None:e.text=str(text)
    return e


def plugin(parent,filename,name):
    return add(parent,'plugin',filename=filename,name='gz::sim::systems::'+name)


def build(scene_id, destination):
    scenario=load_scenario(scene_id)
    if scenario.gazebo_world != scene_id:
        raise ValueError('This scenario has no Gazebo world')
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    sdf=ET.Element('sdf',version='1.9');world=add(sdf,'world',name=scene_id)
    physics=add(world,'physics',name='physics',type='ignored');add(physics,'max_step_size',.004);add(physics,'real_time_factor',1)
    for suffix,name in [('physics','Physics'),('user-commands','UserCommands'),('scene-broadcaster','SceneBroadcaster'),
                        ('imu','Imu'),('magnetometer','Magnetometer'),('air-pressure','AirPressure'),('navsat','NavSat'),('contact','Contact')]:
        plugin(world,'gz-sim-'+suffix+'-system',name)
    sensors=plugin(world,'gz-sim-sensors-system','Sensors');add(sensors,'render_engine','ogre2')
    coords=add(world,'spherical_coordinates')
    for key,value in [('surface_model','EARTH_WGS84'),('world_frame_orientation','ENU'),('latitude_deg',47.397742),('longitude_deg',8.545594),('elevation',488),('heading_deg',0)]:add(coords,key,value)
    sc=add(world,'scene');add(sc,'ambient','0.5 0.5 0.5 1');add(sc,'background','0.65 0.78 0.9 1');add(sc,'shadows','true')
    light=add(world,'light',name='sun',type='directional');add(light,'pose','0 0 20 0 0 0');add(light,'direction','-0.5 0.3 -0.9');add(light,'diffuse','0.9 0.9 0.85 1');add(light,'cast_shadows','true')
    objects=scene_layout(scenario)
    if scenario.campus_layout:
        mesh=export_visual(scenario,objects,destination)
        model=add(world,'model',name='campus_visual');add(model,'static','true')
        visual=add(add(model,'link',name='body'),'visual',name='scenery')
        add(add(add(visual,'geometry'),'mesh'),'uri',mesh.as_uri())
        objects=[objects[0],*campus_collision_geometry(scenario.campus_layout)]
    for obj in objects:
        model=add(world,'model',name=obj['name']);add(model,'static','true')
        p=obj['position'];add(model,'pose',f"{p['x']*3} {-p['y']*3} {p['z']*5} 0 0 0")
        link=add(model,'link',name='body')
        for tag in (('collision',) if scenario.campus_layout else ('visual','collision')):
            shape=add(link,tag,name=tag);geo=add(shape,'geometry');size=obj['size']
            if obj.get('shape') == 'ellipsoid':
                ellipsoid=add(geo,'ellipsoid');add(ellipsoid,'radii',f"{size['x']/2} {size['y']/2} {size['z']/2}")
            elif obj.get('shape') == 'cylinder':
                cyl=add(geo,'cylinder');add(cyl,'radius',size['x']/2);add(cyl,'length',size['z'])
            else:
                box=add(geo,'box');add(box,'size',' '.join(str(size[k]) for k in 'xyz'))
            if tag=='visual':
                color=' '.join(str(int(obj['color'][i:i+2],16)/255) for i in (1,3,5))+' 1'
                material=add(shape,'material');add(material,'ambient',color);add(material,'diffuse',color)
    ET.indent(sdf);ET.ElementTree(sdf).write(destination/(scene_id+'.sdf'),encoding='unicode',xml_declaration=True)
    modeldir=destination/'models'/'x500_rgbd';modeldir.mkdir(parents=True,exist_ok=True)
    root=ET.Element('sdf',version='1.9');model=add(root,'model',name='x500_rgbd')
    inc=add(model,'include',merge='true');add(inc,'uri','model://x500')
    for name in ('gimbal_link','camera_link'):
        # Underside lens with ground clearance. The narrower optical field
        # keeps the landing gear outside the central oblique search view.
        # Localization always uses the measured joint/link pose.
        link=add(model,'link',name=name);add(link,'pose', '0 0 -.21 0 0 0',relative_to='base_link')
        inertial=add(link,'inertial');add(inertial,'mass',.01);inertia=add(inertial,'inertia')
        for axis in ('ixx','iyy','izz'):add(inertia,axis,.0001)
    for joint_name,parent,child,axis in [('gimbal_yaw','base_link','gimbal_link','0 0 1'),('gimbal_pitch','gimbal_link','camera_link','0 1 0')]:
        joint=add(model,'joint',name=joint_name,type='revolute');add(joint,'parent',parent);add(joint,'child',child)
        ax=add(joint,'axis');add(ax,'xyz',axis);limit=add(ax,'limit');add(limit,'lower',-math.pi);add(limit,'upper',math.pi)
        controller=plugin(model,'gz-sim-joint-position-controller-system','JointPositionController')
        add(controller,'joint_name',joint_name);add(controller,'topic',f'/model/x500_rgbd_0/{joint_name}/cmd_pos')
        # Gains match the light payload inertia and 4 ms physics step. Large
        # damping gains make this joint alternate every step, so equally stamped
        # sensor/pose samples can still disagree during the rendering update.
        add(controller,'p_gain',.02);add(controller,'d_gain',.004 if joint_name=='gimbal_yaw' else .003)
        add(controller,'cmd_max',.05);add(controller,'cmd_min',-.05)
    camera_link=next(e for e in model.findall('link') if e.attrib['name']=='camera_link')
    sensor=add(camera_link,'sensor',name='rgbd',type='rgbd_camera');add(sensor,'always_on','true');add(sensor,'update_rate',10)
    add(sensor,'topic','/model/x500_rgbd_0/rgbd');cam=add(sensor,'camera')
    add(cam,'horizontal_fov',2*math.atan(4/3*math.tan(math.radians(45/2))))
    image=add(cam,'image');add(image,'width',640);add(image,'height',480);add(image,'format','R8G8B8')
    clip=add(cam,'clip');add(clip,'near',.1);add(clip,'far',100)
    poses=plugin(model,'gz-sim-pose-publisher-system','PosePublisher')
    for k,v in [('publish_link_pose','true'),('publish_sensor_pose','true'),('publish_nested_model_pose','true'),('publish_model_pose','true'),('use_pose_vector_msg','true'),('update_frequency','100')]:add(poses,k,v)
    ET.indent(root);ET.ElementTree(root).write(modeldir/'model.sdf',encoding='unicode',xml_declaration=True)
    (modeldir/'model.config').write_text('<?xml version="1.0"?><model><name>x500_rgbd</name><version>1</version><sdf version="1.9">model.sdf</sdf><author><name>UAV LLM</name></author><description>PX4 X500 with measured RGB-D gimbal</description></model>')
    return destination/(scene_id+'.sdf')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scene',default='visual_garden');parser.add_argument('--output',default='.sim/worlds')
    args=parser.parse_args();print(build(args.scene,args.output))
