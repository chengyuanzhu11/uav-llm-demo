"""Independent geometry/provenance and asynchronous execution checks; no API calls."""
import base64
from concurrent.futures import Future
from copy import deepcopy
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib
import time
from threading import Event

from uav_llm_demo.onboard_observation import (accept_frame, camera_request, ground_pixel,
                                             validate_visual_evidence, WIDTH, HEIGHT)
from uav_llm_demo.visual_execution import VisualExecution, VisionAgent
from uav_llm_demo.semantic_tasks import task_catalog, semantic_plan_from_json_text
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.task_requirements import requirements_from_json_text
from uav_llm_demo.provider_usage import prefix_metrics
import json
import math


def png():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', WIDTH, HEIGHT, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress((b'\0' + bytes([50, 100, 150]) * WIDTH) * HEIGHT)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


def frame(now=10, yaw=0, pitch=0, revision=1, fov=75):
    request = camera_request([20, 30, 3], yaw, pitch, now, revision, 'execution-1', fov_deg=fov)
    return accept_frame(request, dict(frame_id=request['frame_id'], rgb=png(), depth_m=[10.]*(WIDTH*HEIGHT)))


class VisualFeedbackTests(unittest.TestCase):
    def observed_program(self):
        catalog = task_catalog(load_scenario('semantic_campus'))
        instruction = '在西侧花园找到目标并围绕它飞行一圈'
        requirement = dict(status='ready',summary='围绕实际观测目标',steps=[
            dict(id='detect',action='visual',target_id='west_garden',visual=dict(description='目标')),
            dict(id='orbit',action='motion',target_ref=dict(step_id='detect'),motion=dict(path=dict(
                kind='curve',x='cos(t)',y='sin(t)',t_start=0,t_end=2*math.pi,altitude_trend='level',
                flexible=dict(scale_groups=[['x','y']],offset_axes=['z']))))],constraints={},
            coverage=[dict(quote=instruction,requirement_ids=['detect','orbit'],constraint_keys=[])])
        plan = dict(status='ready',summary='使用观测目标坐标',constraints=dict(max_altitude_m=None,avoid_regions=[]),steps=[
            dict(id='camera',action='visual',target_id='west_garden',visual=dict(description='目标',max_distance_m=30),when=None),
            dict(id='flight',action='motion',target_id='west_garden',target_ref=dict(step_id='camera'),
                 motion=dict(path=dict(kind='curve',x='6*cos(t)',y='6*sin(t)',z='12',t_start=0,t_end=2*math.pi)),when=None)])
        return (catalog,requirements_from_json_text(json.dumps(requirement),catalog,instruction),
                semantic_plan_from_json_text(json.dumps(plan),catalog))

    def finish_detection(self, mission, target=(34.5,52.2,.2), now=10):
        position = [28,50,2]
        delta = [(target[i]-position[i])*(3,3,5)[i] for i in range(3)]
        yaw = math.degrees(math.atan2(delta[1],delta[0]))
        pitch = math.degrees(math.atan2(delta[2],math.hypot(*delta[:2])))
        step = mission.current_step
        views=[]
        for t in (now,now+1):
            request=camera_request(position,yaw,pitch,t,mission.revision,step['execution_id'],fov_deg=37.5)
            f=accept_frame(request,dict(frame_id=request['frame_id'],rgb=png(),depth_m=[math.hypot(*delta)]*(WIDTH*HEIGHT)))
            views.append(dict(ground_pixel(f,[.5,.5]),confidence=.99,revision=mission.revision,
                              execution_id=step['execution_id'],source=f['source'],simulation_time=t))
        mission.complete_visual(position,now+2,dict(observed=True,description='目标',views=views))
        return mission.evidence[-1]

    def test_observed_target_reference_survives_renaming_and_rejects_region_substitution(self):
        from uav_llm_demo.task_contract import TaskContractMonitor, TaskContractError
        from uav_llm_demo.observed_targets import UnresolvedTargetReference
        from uav_llm_demo.spatial_motion import compile_motion
        from uav_llm_demo.task_protocol import TaskProtocolError
        catalog, requirements, plan = self.observed_program()
        scenario = load_scenario('semantic_campus')
        TaskContractMonitor(requirements,'m',plan=plan,scenario=scenario)
        with self.assertRaises(UnresolvedTargetReference):
            compile_motion(plan.steps[1],scenario,scenario.base,plan.constraints)
        for change in ('remove','translate','forward','conditional'):
            bad=plan.to_dict()
            if change=='remove':bad['steps'][1].pop('target_ref')
            if change=='translate':bad['steps'][1]['motion']['path']['x']='20+6*cos(t)'
            if change=='forward':bad['steps'].reverse()
            if change=='conditional':bad['steps'][0]['when']=dict(source='telemetry',field='battery_percent',op='>',value=80)
            with self.subTest(change=change), self.assertRaises((TaskContractError,TaskProtocolError)):
                candidate=semantic_plan_from_json_text(json.dumps(bad),catalog)
                TaskContractMonitor(requirements,'m',plan=candidate,scenario=scenario)

    def test_bound_motion_and_control_use_observation_and_require_matching_execution_evidence(self):
        from uav_llm_demo.semantic_tasks import SemanticMission
        from uav_llm_demo.spatial_motion import compile_motion
        from uav_llm_demo.spatial_control import resolve_control_goal
        from uav_llm_demo.task_contract import TaskContractError
        from uav_llm_demo.observed_targets import resolve_target_reference
        catalog, requirements, plan = self.observed_program()
        scene=load_scenario('semantic_campus')
        data=plan.to_dict();data['steps'][1]['motion']['heading']=dict(mode='face_reference')
        plan=semantic_plan_from_json_text(json.dumps(data),catalog)
        mission=SemanticMission(scene,mission_id='observed')
        mission.accept(plan,requirements=requirements)
        visual=self.finish_detection(mission)
        step=mission.current_step
        compiled=compile_motion(step,scene,scene.base,plan.constraints,evidence=mission.evidence)
        binding=compiled.target_binding
        self.assertEqual(compiled.heading,dict(mode='face_point',point=tuple(binding['point_grid'])))
        self.assertEqual(binding['evidence_id'],visual['evidence_id'])
        self.assertGreater(math.hypot((binding['point_grid'][0]-28)*3,(binding['point_grid'][1]-50)*3),19)
        self.assertAlmostEqual((compiled.points[0][0]-binding['point_grid'][0])*3,6)
        self.assertAlmostEqual(compiled.points[0][2]*5-binding['point_grid'][2]*5,12)
        control=dict(id='nearby',action='control',target_id='west_garden',target_ref=step['target_ref'],
                     revision=step['revision'],control=dict(offset_m=[3,0,0],altitude_m=12))
        goal=resolve_control_goal(control,scene,plan.constraints,evidence=mission.evidence)
        self.assertAlmostEqual(goal['position'][0],binding['point_grid'][0]+1)
        self.assertEqual(goal['position'][2],12/5)
        # Same label from an older revision cannot substitute for this visual execution.
        with self.assertRaises(ValueError):
            resolve_target_reference({**step,'revision':2},mission.evidence)
        historical={**step,'revision':2,'target_ref':dict(evidence_id=visual['evidence_id'])}
        self.assertEqual(resolve_target_reference(historical,mission.evidence),binding)
        # Deferred geometry is not automatically safe: preview a recovery using
        # the completed observation, then change only unconstrained parameters.
        from uav_llm_demo.planning_tools import PlanningTools
        from uav_llm_demo.mission_service import MissionService
        service=MissionService(scenario=scene)
        self.addCleanup(service.close)
        catalog['mission_context']=mission.catalog_context()
        contract=deepcopy(mission.task_contract);contract.authorize_event_review()
        tools=PlanningTools(scene,service._context(),catalog,requirements=requirements,contract=contract)
        geometry,_=tools.call(dict(tool='query_geometry',arguments=dict(
            evidence_id=visual['evidence_id'])))
        self.assertEqual(geometry['object']['center_grid'],binding['point_grid'])
        self.assertEqual(geometry['trajectory_bounds_m'][0][2],10.8)
        minimum_z=geometry['relative_path_offset_bounds_m'][0][2]
        revised=plan.to_dict();revised['steps']=revised['steps'][1:]
        revised['steps'][0]['target_ref']=historical['target_ref']
        revised['steps'][0]['motion']['path']['z']='0'
        unsafe,_=tools.call(dict(tool='preview_plan',arguments=dict(plan=revised)))
        self.assertFalse(unsafe['ok'])
        self.assertEqual(unsafe['diagnostic']['code'],'motion_path_unsafe')
        self.assertIn('flight_bounds_grid',unsafe['diagnostic'])
        revised['steps'][0]['motion']['path']['z']=str(minimum_z-.005)
        unsafe,_=tools.call(dict(tool='preview_plan',arguments=dict(plan=revised)))
        violation=unsafe['diagnostic']['bound_violations'][0]
        self.assertAlmostEqual(violation['actual_m'],10.795)
        self.assertEqual(violation['minimum_m'],10.8)
        self.assertAlmostEqual(violation['required_shift_m'],.005)
        # Exhaustion preserves the final concrete reason, without restarting its
        # budget or turning a local geometric failure into an API failure.
        from uav_llm_demo.planning_tools import MAX_PLANNING_ROUNDS
        from uav_llm_demo.agents import TaskPlanningError
        tools.rounds=MAX_PLANNING_ROUNDS
        with self.assertRaises(ValueError) as stopped:tools.before_request()
        service.semantic=deepcopy(mission)
        service._semantic_review_pending=True;service._future_semantic_stage='plan'
        future=Future();future.set_exception(TaskPlanningError(stopped.exception.diagnostic))
        service._collect_semantic_plan(future)
        self.assertEqual(service.semantic.status,'safety_wait')
        self.assertEqual(service.semantic.evidence,mission.evidence)
        self.assertIn('10.795000',service.semantic.message)
        self.assertIn('10.800000',service.semantic.message)
        self.assertFalse(service._running)
        revised['steps'][0]['motion']['path']['z']=str(minimum_z+.1)
        safe,_=tools.call(dict(tool='preview_plan',arguments=dict(plan=revised)))
        self.assertTrue(safe['ok'],safe)
        self.assertEqual(safe['deferred_targets'],[])
        forged=dict(evidence_id='wrong-motion',mission_id=mission.mission_id,vehicle_id='uav-1',
            step_id=step['id'],execution_id=step['execution_id'],revision=1,action='motion',target_id='west_garden',
            source='runtime_trajectory_tracking',simulation_time=25,position=dict(zip('xyz',compiled.points[-1])),
            duration_seconds=10,result=dict(completed=True,distance_m=compiled.length_m,total_length_m=compiled.length_m,progress=1))
        with self.assertRaisesRegex(TaskContractError,'confirmed object'):
            deepcopy(mission.task_contract).record(step,forged)
        forged['result']['target_binding']=binding
        with self.assertRaisesRegex(TaskContractError,'heading'):
            deepcopy(mission.task_contract).record(step,forged)
        forged['result'].update(heading_samples=40,max_heading_error_deg=1)
        forged['position']=dict(x=28,y=50,z=3)
        with self.assertRaisesRegex(TaskContractError,'geometry'):
            deepcopy(mission.task_contract).record(step,forged)
        mission.complete_motion(compiled.points[-1],25,dict(execution_id=step['execution_id'],completed=True,
            covered_distance_m=compiled.length_m,total_length_m=compiled.length_m,duration_seconds=10,
            target_binding=binding,heading_samples=40,max_heading_error_deg=1))
        self.assertEqual(mission.status,'completed')
        self.assertTrue(mission.task_contract.verdict()['satisfied'])

    def test_repeated_detections_bind_latest_occurrence_and_followup_can_use_historical_target(self):
        from uav_llm_demo.semantic_tasks import SemanticMission
        from uav_llm_demo.spatial_motion import compile_motion
        catalog, requirements, plan = self.observed_program()
        instruction='重复两次：找到目标并围绕它飞行一圈'
        req=requirements.to_dict()
        req['steps']=[dict(id='twice',action='repeat',count=2,until=None,steps=req['steps'])]
        req['coverage']=[dict(quote=instruction,requirement_ids=['twice'],constraint_keys=[])]
        requirements=requirements_from_json_text(json.dumps(req),catalog,instruction)
        data=plan.to_dict()
        data['steps']=[dict(id='two_passes',action='repeat',count=2,until=None,steps=data['steps'],when=None)]
        plan=semantic_plan_from_json_text(json.dumps(data),catalog)
        scene=load_scenario('semantic_campus');mission=SemanticMission(scene,mission_id='repeat-target')
        mission.accept(plan,requirements=requirements)
        seen=[]
        for i in range(2):
            visual=self.finish_detection(mission,target=(34.5-i,52.2,.2),now=10+i*30)
            step=mission.current_step
            motion=compile_motion(step,scene,scene.base,plan.constraints,evidence=mission.evidence)
            self.assertEqual(motion.target_binding['evidence_id'],visual['evidence_id'])
            seen.append(motion.target_binding['evidence_id'])
            mission.complete_motion(motion.points[-1],25+i*30,dict(execution_id=step['execution_id'],completed=True,
                covered_distance_m=motion.length_m,total_length_m=motion.length_m,duration_seconds=10,
                target_binding=motion.target_binding))
        self.assertEqual(len(set(seen)),2)
        self.assertEqual(mission.status,'completed')
        # A subsequent user instruction may refer to that object without a new search.
        catalog['mission_context']=mission.catalog_context()
        instruction='在刚才目标的东侧3米、12米高悬停2秒'
        ref=dict(evidence_id=seen[-1])
        control=dict(state='airborne',offset_m=[3,0,0],altitude_m=12,duration_seconds=2)
        required=dict(status='ready',summary='相对历史目标悬停',steps=[dict(id='hold',action='control',target_ref=ref,control=control)],
                      constraints={},coverage=[dict(quote=instruction,requirement_ids=['hold'],constraint_keys=[])])
        requirements=requirements_from_json_text(json.dumps(required),catalog,instruction)
        candidate=dict(status='ready',summary='使用已确认的位置',steps=[dict(id='hover',action='control',target_id='west_garden',
                       target_ref=ref,control=control,when=None)],constraints=dict(max_altitude_m=None,avoid_regions=[]))
        next_plan=semantic_plan_from_json_text(json.dumps(candidate),catalog)
        mission.accept(next_plan,preserve_evidence=True,requirements=requirements)
        goal=mission.bind_current_target()
        mission.observe(goal['position'],0,2,60,enabled=True)
        self.assertEqual(mission.status,'completed')
        self.assertEqual(mission.evidence[-1]['result']['target_binding']['evidence_id'],seen[-1])
        foreign=deepcopy(catalog)
        for record in foreign['mission_context']['evidence']:record['mission_id']='another-mission'
        with self.assertRaises(ValueError):semantic_plan_from_json_text(json.dumps(candidate),foreign)

    def test_rgbd_calibration_and_packet_binding(self):
        for yaw, pitch, axis, sign in [(0, 0, 0, 1), (90, 0, 1, 1), (-180, 0, 0, -1), (0, -90, 2, -1)]:
            view = frame(yaw=yaw, pitch=pitch)
            p = ground_pixel(view, [.5, .5])['point_grid']
            self.assertAlmostEqual((p[axis]-view['position_grid'][axis])*(3,3,5)[axis], 10*sign,
                                   delta=.1)
        view = frame()
        view['depth_m'] = list(view['depth_m'])
        view['depth_m'][120*WIDTH+160] = 0
        with self.assertRaises(ValueError): ground_pixel(view, [.5,.5])
        with self.assertRaises(ValueError): accept_frame(view, dict(frame_id='old',rgb=png(),depth_m=[]))
        with self.assertRaises(ValueError): ground_pixel(frame(), [True,.5])
        wide, focused = frame(), frame(fov=37.5)
        self.assertGreater(focused['fx'], wide['fx'])
        self.assertLess(ground_pixel(focused,[.75,.5])['point_grid'][1], ground_pixel(wide,[.75,.5])['point_grid'][1])

    def test_coarse_localization_is_not_completion_and_requires_two_fresh_focused_views(self):
        loop=VisualExecution(Mock(model='offline'))
        self.addCleanup(loop.close)
        step=dict(execution_id='execution-1',visual=dict(description='目标外观',max_distance_m=15))
        service=SimpleNamespace(semantic=Mock(revision=1,evidence=[{}]),_time=12,
            _vehicle=SimpleNamespace(position=(20,30,3)),trace=Mock(),_semantic_telemetry=Mock(return_value={}))
        def ground(view, confidence=.95):
            return loop._apply(service,step,view,dict(tool='ground',arguments=dict(
                frame_id=view['frame_id'],pixel=[.5,.5],confidence=confidence),reason='确认可见表面'))
        step['visual']['max_distance_m']=9
        self.assertFalse(ground(frame()))
        self.assertEqual(loop.fov,37.5)
        self.assertEqual(loop.views,[])
        self.assertFalse(ground(frame(fov=loop.fov)))
        self.assertEqual(loop.feedback['code'],'viewing_distance_unmet')
        self.assertAlmostEqual(loop.feedback['measured']['current_range_m'],10,delta=.01)
        self.assertEqual(loop.fov,37.5)
        service._vehicle.position=(21,30,3)
        self.assertFalse(ground(frame(fov=loop.fov)))
        self.assertEqual(loop.feedback['code'],'view_refresh_required')
        self.assertEqual(loop.views,[])
        service._vehicle.position=(20,30,3)
        step['visual']['max_distance_m']=15
        self.assertFalse(ground(frame(),confidence=.5))
        self.assertEqual(loop.views,[])
        self.assertEqual(loop.fov,37.5)
        self.assertFalse(ground(frame(now=10.5,fov=loop.fov),confidence=.6))
        self.assertEqual(loop.fov,37.5)
        self.assertEqual(loop.views,[])
        service.semantic.complete_visual.assert_not_called()
        self.assertFalse(ground(frame(now=11,fov=loop.fov)))
        service.semantic.complete_visual.assert_not_called()
        self.assertTrue(ground(frame(now=12,fov=loop.fov)))
        service.semantic.complete_visual.assert_called_once()
        evidence=service.semantic.complete_visual.call_args.args[2]['views']
        self.assertEqual([v['simulation_time'] for v in evidence],[11,12])

    def test_completion_requires_fresh_consistent_execution_bound_images(self):
        node = dict(visual=dict(description='带白色圆形的蓝色标记',max_distance_m=15))
        views = []
        for now in (10,11):
            f = frame(now)
            views.append(dict(ground_pixel(f,[.5,.5]), confidence=.9, revision=1, execution_id='execution-1',
                              source=f['source'],simulation_time=now))
        evidence = dict(result=dict(observed=True,description=node['visual']['description'],views=views),
                        position=[20,30,3],simulation_time=12,revision=1,execution_id='execution-1')
        validate_visual_evidence(node,evidence)
        changes = [('revision',2),('simulation_time',110),('position',[50,30,3])]
        for key,value in changes:
            bad=deepcopy(evidence); bad[key]=value
            with self.assertRaises(ValueError): validate_visual_evidence(node,bad)
        for key,value in [('frame_id',views[0]['frame_id']),('confidence',.5),('point_grid',[50,40,3])]:
            bad=deepcopy(evidence); bad['result']['views'][1][key]=value
            with self.assertRaises(ValueError): validate_visual_evidence(node,bad)

    def test_requirements_freeze_description_but_not_unrequested_viewing_distance(self):
        catalog=task_catalog(load_scenario('semantic_campus'))
        requirement=dict(status='ready',summary='寻找外观目标',steps=[dict(id='find',action='visual',target_id='stadium',
            visual=dict(description='红色三角标记'))],constraints={},coverage=[dict(quote='寻找红色三角标记',requirement_ids=['find'],constraint_keys=[])])
        req=requirements_from_json_text(json.dumps(requirement),catalog,'寻找红色三角标记').to_dict()
        unspecified=deepcopy(requirement)
        unspecified['steps'][0].pop('target_id')
        self.assertNotIn('target_id',requirements_from_json_text(json.dumps(unspecified),catalog,'寻找红色三角标记').steps[0])
        concrete=dict(status='ready',summary='视觉确认',steps=[dict(id='find',action='visual',target_id='stadium',
            visual=dict(description='红色三角标记',max_distance_m=20),when=None)],constraints=dict(max_altitude_m=None,avoid_regions=[]))
        plan=semantic_plan_from_json_text(json.dumps(concrete),catalog)
        # Independent specifications are sparse; the program still must carry a distance.
        self.assertNotIn('max_distance_m',req['steps'][0]['visual'])
        self.assertEqual(plan.steps[0]['visual']['max_distance_m'],20)
        from uav_llm_demo.task_contract import _requirement_matches
        self.assertTrue(_requirement_matches(req['steps'][0],plan.steps[0]))
        req['steps'][0]['visual']['max_distance_m']=25
        self.assertTrue(_requirement_matches(req['steps'][0],plan.steps[0]))
        req['steps'][0]['visual']['max_distance_m']=10
        self.assertFalse(_requirement_matches(req['steps'][0],plan.steps[0]))
        concrete['steps'][0]['visual']['description']='蓝色圆形'
        self.assertFalse(_requirement_matches(req['steps'][0],concrete['steps'][0]))

    def test_delayed_reply_cannot_interrupt_segment_and_old_revision_is_discarded(self):
        loop=VisualExecution(Mock(model='offline'))
        self.addCleanup(loop.close)
        step=dict(execution_id='execution-1',visual=dict(description='目标',max_distance_m=15))
        service=SimpleNamespace(semantic=SimpleNamespace(revision=1),_vehicle=SimpleNamespace(position=(20,30,3),velocity=(1,0,0)),
            _time=12,waypoints=SimpleNamespace(current=(21,30,3)),avoidance=SimpleNamespace(active=False),
            _plan_requested=False,trace=Mock(),runtime=SimpleNamespace())
        loop.identity=(1,'execution-1');loop.status='reasoning';loop.future_identity=loop.identity
        loop.inference_started=time.monotonic()
        loop.future=Future();loop.future.set_result((loop.identity,frame(),dict(tool='look',arguments=dict(yaw_deg=90,pitch_deg=-30),reason='查看侧面')))
        self.assertFalse(loop.tick(service,step,False))
        self.assertIsNotNone(loop.pending);self.assertEqual(loop.yaw,0)
        self.assertEqual(service.waypoints.current,(21,30,3))
        loop.pending=None;loop.future=Future();loop.future_identity=(0,'old')
        loop.future.set_result(((0,'old'),frame(),{}))
        loop.frame=None
        loop.tick(service,step,False)
        self.assertEqual(loop.discarded,1)
        self.assertIsNone(loop.pending)

    def test_multimodal_metrics_do_not_count_images_as_cached_text(self):
        metrics=prefix_metrics([dict(role='user',content=[dict(type='text',text='query'),dict(type='image_url',image_url=dict(url=png()))])],None)
        self.assertEqual(metrics['image_count'],1)
        self.assertIsNone(metrics['prompt_content_chars'])
        self.assertNotIn('base64',json.dumps(metrics))

    def test_image_reports_are_bound_before_text_planning_and_cancellation_prevents_next_call(self):
        from concurrent.futures import CancelledError
        agent=VisionAgent(dict(api_key='fixture-secret',model='fixture-language',base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'))
        observation=frame()
        agent.client._request_json_content=Mock(return_value=json.dumps(dict(frames=[
            dict(frame_index=0,scene='有遮挡',targets=[],occlusions=[dict(description='遮挡面',point=[500,400])])])) )
        agent.planner._request_json_content=Mock(return_value=json.dumps(dict(tool='look',arguments=dict(yaw_deg=270,pitch_deg=-90),reason='观测新方向')))
        started=Mock()
        context=dict(visual_goal=dict(description='目标外观',max_distance_m=20),known_starting_area=dict(center_grid=[10,20,3]))
        decision=agent.decide(context,[observation],additional_request=started)
        self.assertEqual(decision['tool'],'look');started.assert_called_once()
        sent=agent.planner._request_json_content.call_args.args[0][1]['content']
        self.assertNotIn('base64',sent);self.assertNotIn('depth_m',sent)
        self.assertEqual(json.loads(sent)['image_perception'][0]['frame_id'],observation['frame_id'])
        self.assertEqual(json.loads(sent)['image_perception'][0]['occlusions'][0]['pixel'],[500/999,400/999])
        image_text=agent.client._request_json_content.call_args.args[0][1]['content'][0]['text']
        self.assertNotIn('known_starting_area',image_text)
        context['feedback']=dict(candidate_id='safe-view',ok=True,tool='preview_move')
        agent.decide(context,[frame(now=9),observation],additional_request=started)
        agent.client._request_json_content.assert_called_once()
        self.assertEqual(agent.planner._request_json_content.call_count,2)
        messages=agent.planner._request_json_content.call_args.args[0]
        self.assertEqual([m['role'] for m in messages],['system','user','assistant','user'])
        self.assertEqual(json.loads(messages[-1]['content'])['feedback']['candidate_id'],'safe-view')
        started.assert_called_once()
        agent.perception[0]['targets']=[dict(target_index=0,description='目标',pixel=[.62,.4],confidence=.85)]
        agent.planner._request_json_content.return_value=json.dumps(dict(tool='ground',arguments=dict(target_index=0),reason='选择当前候选'))
        grounded=agent.decide(context,[observation],additional_request=started)
        self.assertEqual(grounded['arguments'],dict(frame_id=observation['frame_id'],pixel=[.62,.4],confidence=.85))
        agent.planner._request_json_content.return_value=json.dumps(dict(tool='ground',arguments=dict(pixel=[.1,.1],confidence=1),reason='无来源的选点'))
        with self.assertRaises(ValueError):agent.decide(context,[observation])
        agent.planner._request_json_content.reset_mock();started.reset_mock()
        cancellation=iter([False,True])
        with self.assertRaises(CancelledError):
            agent.decide(context,[frame(now=11)],additional_request=started,cancelled=lambda:next(cancellation))
        agent.planner._request_json_content.assert_not_called();started.assert_not_called()
        with patch.dict('os.environ',dict(DEEPSEEK_API_KEY='language-only-fixture',
                VISION_BASE_URL='https://vision.invalid/v1',VISION_MODEL='image-fixture'),clear=True):
            separate=VisionAgent(dict(api_key='language-only-fixture',model='text',base_url='https://language.invalid/v1'))
            self.assertIsNone(separate.client.api_key)
            self.assertFalse(separate.configured)
            with self.assertRaises(ValueError):separate.decide(context,[observation])

    def test_waiting_for_vision_holds_position_without_completing_task(self):
        from dataclasses import replace
        from tests.semantic_fixtures import fixture_requirements
        from uav_llm_demo.agents import DeepSeekAgent
        from uav_llm_demo.mission_service import MissionService
        from uav_llm_demo.html_runtime import HtmlRuntime
        from uav_llm_demo.motion import distance
        scene=replace(load_scenario('semantic_campus'),dynamic_objects=())
        position=(26,84,2.2)
        runtime=HtmlRuntime()
        runtime.reset(position)
        entered,release=Event(),Event()
        def decide(*args,**kwargs):
            entered.set();release.wait(10)
            return dict(tool='look',arguments=dict(yaw_deg=-90,pitch_deg=-45),reason='待视角观测')
        service=MissionService(agent=DeepSeekAgent(api_key='offline'),scenario=scene,runtime=runtime,
                               vision_agent=SimpleNamespace(model='blocking-fixture',decide=decide))
        self.addCleanup(service.close);self.addCleanup(release.set)
        raw=dict(status='ready',summary='视觉定位',steps=[dict(id='find',action='visual',target_id='stadium',
            visual=dict(description='红色三角标记',max_distance_m=16),when=None)],constraints=dict(max_altitude_m=None,avoid_regions=[]))
        plan=semantic_plan_from_json_text(json.dumps(raw),task_catalog(scene))
        service.semantic.accept(plan,requirements=fixture_requirements(raw))
        service._running=service._safety_armed=True
        service._environment.phase='fly_to_target';service._environment.target=position
        service._commit_vehicle(runtime.read_state(),0)
        service.waypoints.load((position,),position);service.waypoints.index=1
        for _ in range(100):
            service.advance(.05)
            if service.visual.request is not None:
                break
        request=service.visual.request
        self.assertIsNotNone(request)
        service.receive_camera_frame(dict(frame_id=request['frame_id'],rgb=png(),depth_m=[10.]*(WIDTH*HEIGHT)))
        service.advance(.05)
        self.assertTrue(entered.wait(1))
        for _ in range(100):service.advance(.05)
        self.assertLess(distance(service._vehicle.position,position),.15)
        self.assertTrue(service._safety_armed)
        self.assertEqual(service.semantic.status,'running')
        self.assertEqual(service.semantic.evidence,[])
        self.assertEqual(service.visual.calls,1)


if __name__=='__main__': unittest.main()
