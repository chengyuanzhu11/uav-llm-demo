"""Isolated browser + mission + RGB-D feedback acceptance.

Default uses a pixel-based offline detector to test execution, not language/VLM
quality. --live uses both configured language and vision providers. Browser-only
fixtures and the independent oracle are never supplied to either model.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from flask import jsonify, request, Response
from werkzeug.serving import make_server, WSGIRequestHandler
from uav_llm_demo import create_app
from uav_llm_demo.agents import DeepSeekAgent
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_tasks import semantic_plan_from_json_text
from uav_llm_demo.task_requirements import requirements_from_json_text
from uav_llm_demo.run_log import JournalWriter

INSTRUCTION = '飞到田径场，从机载画面寻找红色三角标记，在距离16米内确认后返航降落。'
SPEC = dict(status='ready',summary='查找外观目标并返航', steps=[
    dict(id='find',action='visual',target_id='stadium',visual=dict(description='红色三角标记',max_distance_m=16)),
    dict(id='home',action='return_home',target_id='home')], constraints={},
    coverage=[dict(quote=INSTRUCTION,requirement_ids=['find','home'],constraint_keys=[])])


class OfflineLanguage(DeepSeekAgent):
    def __init__(self): super().__init__(api_key='offline-fixture-no-network')
    def extract_task_requirements(self,catalog,instruction,feedback='',**kwargs):
        return requirements_from_json_text(json.dumps(SPEC),catalog,instruction)
    def audit_task_requirements(self,catalog,instruction,requirements,*args,**kwargs):
        # Match the same successful audit return shape as the real provider.
        from uav_llm_demo.task_requirements import audit_requirements_from_json_text
        from tests.semantic_fixtures import fixture_audit
        return audit_requirements_from_json_text(json.dumps(fixture_audit(requirements, instruction)),requirements,instruction)
    def interpret_task(self,catalog,*args,**kwargs):
        plan=deepcopy(SPEC);plan.pop('coverage');plan['constraints']=dict(max_altitude_m=None,avoid_regions=[])
        for node in plan['steps']:
            node['when']=None
            if node['action']=='return_home':node['duration_seconds']=0
        return semantic_plan_from_json_text(json.dumps(plan),catalog)


class BrowserDetector:
    """Browser decodes RGB and returns a colored-pixel centroid, without object IDs.

    Only used for deterministic camera/execution verification, NEVER for live
    perception or claims about general language/image understanding.
    """
    model='offline-pixel-fixture'
    def __init__(self):
        from threading import Event
        self.request=None;self.reply=None;self.ready=Event()
        self.frame_id=None
    def decide(self,context,frames,**kwargs):
        if self.frame_id!=frames[-1]['frame_id']:
            self.ready.clear();self.reply=None
            self.request=dict(context=context,frame=frames[-1])
            if not self.ready.wait(15):raise ValueError('Offline pixel decoder did not respond.')
            self.request=None;self.frame_id=frames[-1]['frame_id']
        pixel=self.reply
        feedback=context['feedback']
        def result(tool,args):return dict(tool=tool,arguments=args,reason='离线像素夹具验证')
        if feedback.get('candidate_id'):return result('submit_move',dict(candidate_id=feedback['candidate_id']))
        measured=feedback.get('measured')
        current_range=(sum(((measured['point_grid'][i]-context['current_position_grid'][i])*(3,3,5)[i])**2 for i in range(3))**.5
                       if measured else 0)
        if measured and current_range>context['visual_goal']['max_distance_m']:
            delta=[(measured['point_grid'][i]-context['current_position_grid'][i])*3 for i in range(2)]
            length=sum(v*v for v in delta)**.5
            return result('preview_move',dict(offset_m=[v/length*min(8,length) for v in delta]+[0]))
        if pixel:return result('ground',dict(frame_id=frames[-1]['frame_id'],pixel=pixel,confidence=.95))
        if any(item.get('feedback',{}).get('measured') for item in context['history']):
            return result('look',dict(yaw_deg=-90,pitch_deg=-80))
        if frames[-1]['yaw_deg']==-90:
            return result('preview_move',dict(offset_m=[0,0,4]))
        return result('look',dict(yaw_deg=-90,pitch_deg=-40))


class Quiet(WSGIRequestHandler):
    def log_request(self,*args,**kwargs):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--variant',type=int,choices=[0,1],default=0)
    parser.add_argument('--scene',choices=['markers','tree','sculpture','bin_orbit','landmark'],default='markers',
                        help='Use a private marker fixture or actual rendered campus scenery.')
    parser.add_argument('--instruction',help='Override the live instruction while retaining the selected independent target oracle.')
    parser.add_argument('--target-landmark',help='Independent scoring only; never sent to the agent.')
    parser.add_argument('--planner-timeout',type=float,default=90)
    parser.add_argument('--wall-budget',type=float,default=780)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if (args.scene != 'markers' or args.instruction) and not args.live:
        parser.error('Actual-campus semantic recognition requires --live; the offline detector only checks marker pixels.')
    if args.scene == 'landmark' and not (args.instruction and args.target_landmark):
        parser.error('landmark requires an instruction and a separate scoring target')
    instruction = args.instruction or {'markers': INSTRUCTION,
                   'tree': '到南广场寻找一棵树，从机载画面确认后返航。',
                   'sculpture': '在南广场寻找红色圆环雕塑，接近到15米内确认后返航。',
                   'bin_orbit': '去西侧花园找到垃圾桶，围绕其旋转一周后返航。'}[args.scene]
    output=Path(args.output).resolve()
    if output.exists():raise SystemExit('Choose a new output path; existing evidence is not overwritten.')
    load_dotenv(ROOT/'.env')
    language=DeepSeekAgent(timeout_seconds=args.planner_timeout) if args.live else OfflineLanguage()
    detector=None if args.live else BrowserDetector()
    scenario=replace(load_scenario('semantic_campus'),dynamic_objects=())
    journal=JournalWriter(output.with_suffix('.events.jsonl'))
    service=MissionService(agent=language,scenario=scenario,scenario_id='semantic_campus',journal=journal,
                           vision_agent=detector,target_seed=91,traffic_seed=0)
    app=create_app(service=service)
    app.logger.disabled=True

    @app.get('/__verify__/fixture')
    def fixture():return jsonify(variant=args.variant,scene=args.scene,instruction=instruction,live=args.live,target_landmark=args.target_landmark,wall_budget=args.wall_budget)

    @app.get('/__verify__/app.js')
    def instrument():
        source=(ROOT/'uav_llm_demo/static/app.js').read_text()
        source=source.replace('renderer.setAnimationLoop(animate);','renderer.setAnimationLoop(null);')
        return Response(source+'\nwindow.visualProbe={THREE,scene,renderer,onboardCamera,acceptState,captureOnboardObservation,get state(){return state;}};\n',mimetype='text/javascript')

    @app.post('/__verify__/start')
    def start():return jsonify(service.submit_task(instruction,background=False))

    @app.post('/__verify__/advance')
    def advance():
        # Fast-forward deterministic control. While a remote inference is pending,
        # follow wall time, preserving real asynchronous age/latency behaviour.
        active=service.visual.future is not None and not service.visual.future.done()
        for _ in range(1 if active else 20):service.advance(.05)
        return jsonify(service.state())

    @app.get('/__verify__/decode')
    def decode():return jsonify(detector.request if detector else None)

    @app.post('/__verify__/decode')
    def decoded():
        if detector:
            detector.reply=request.get_json()['pixel'];detector.ready.set()
        return jsonify(ok=True)

    server=make_server('127.0.0.1',0,app,threaded=True,request_handler=Quiet)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    env=dict(os.environ,UAV_VISUAL_URL=f'http://127.0.0.1:{server.server_port}',UAV_VISUAL_OUTPUT=str(output))
    try:
        process=subprocess.Popen(['node',str(ROOT/'scripts/verify_visual_feedback.cjs')],env=env,cwd=ROOT,start_new_session=True)
        try:
            return process.wait(timeout=args.wall_budget+70)
        except subprocess.TimeoutExpired:
            if not output.exists():
                state=service.state()
                output.parent.mkdir(parents=True,exist_ok=True)
                output.write_text(json.dumps(dict(error='browser_acceptance_timeout',live=args.live,scene=args.scene,
                    semantic=state.get('semantic_mission'),vision=state.get('onboard')),ensure_ascii=False,indent=2))
            print('Browser acceptance timed out; partial evidence saved.',flush=True)
            return 1
        finally:
            # A timed-out Node process must not leave its Chromium children running.
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
    finally:
        server.shutdown();service.close();thread.join(timeout=3);journal.close()


if __name__=='__main__':sys.exit(main())
