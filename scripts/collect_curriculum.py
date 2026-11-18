"""Collect frozen tasks with the configured VLM, then independently review RGB-D labels.

High-level demonstrations come from the disclosed source standards. The vision
policy sees images and tool feedback only; scene truth is used after execution.
"""
from pathlib import Path
import argparse, gzip, json, sys, time
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from dotenv import load_dotenv
from collect_demonstrations import collect_case
from collect_visual_demonstration import score_visual
from verify_gazebo import target_error
from prepare_demonstration_curriculum import prepare
from uav_llm_demo.direct_vision import DirectVisionAgent
from uav_llm_demo.provider_config import load_planner_config
from uav_llm_demo.scenarios import load_asset_scenario
from uav_llm_demo.simulation_scene import load_scene,scene_boxes
from uav_llm_demo.onboard_observation import ground_pixel
from uav_llm_demo.demonstration_data import decision_candidates,build_review,audit_episode,_sample


def review_regions(episode,scenario,target):
    parts=scene_boxes(load_scene(scenario.asset_scene));regions=[]
    object_ids=[o['id'] for o in load_scene(scenario.asset_scene)['objects']]
    for c in decision_candidates(episode):
        terminal=(c.get('result') or {}).get('payload',{});action=terminal.get('action',{})
        if terminal.get('status') not in {'applied','completed'} or action.get('tool') not in {'ground','probe'}:continue
        args=action['arguments'];f=next(v for v in c['input']['payload']['frames'] if v['frame_id']==args['frame_id'])
        frame=deepcopy(f)
        frame['depth_m']=np.frombuffer(gzip.decompress((episode/f['depth']['path']).read_bytes()),dtype='<f4').tolist()
        point=ground_pixel(frame,args['pixel'])['point_grid']
        ids=object_ids if action['tool']=='probe' else [target]
        obj=min(ids,key=lambda key:target_error(point,parts,key))
        # A small visible-interior polygon must consist entirely of measured
        # target surface. Ground truth is never passed back to the policy.
        x,y=args['pixel'];dx,dy=1.5/f['width'],1.5/f['height']
        polygon=[[x-dx,y-dy],[x+dx,y-dy],[x+dx,y+dy],[x-dx,y+dy]]
        if any(not 0<u<1 or not 0<v<1 for u,v in polygon):continue
        try:
            errors=[target_error(ground_pixel(frame,p)['point_grid'],parts,obj) for p in [args['pixel'],*polygon]]
        except ValueError:continue
        if max(errors)>.05:continue
        regions.append(dict(frame_id=f['frame_id'],rgb_sha256=f['rgb_sha256'],polygon=polygon,
            object_id=obj,reviewer='gazebo_rgbd_surface_patch_v1',annotation_source='simulator_geometry',human_reviewed=False))
    return regions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='benchmarks/demonstration_curriculum_v2.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--limit',type=int,default=42)
    parser.add_argument('--resume',action='store_true',help='Reuse closed executions from the identical frozen manifest')
    parser.add_argument('--retry-failed',action='store_true',help='Retain failed/interrupted attempts and rerun their frozen task')
    parser.add_argument('--case-ids',nargs='+')
    parser.add_argument('--wall-budget',type=float,default=360)
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    manifest=json.loads(Path(args.manifest).read_text());cases=prepare(manifest)
    if args.case_ids:cases=[c for c in cases if c['standard']['id'] in args.case_ids]
    cases=cases[:args.limit]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=args.resume)
    saved=out/'frozen-manifest.json'
    if saved.exists():
        if json.loads(saved.read_text())!=manifest:raise ValueError('Frozen manifest changed')
    else:saved.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    rows=[]
    for case in cases:
        standard=case['standard'];cid=standard['id'];directory=out/cid
        scene_file=ROOT/'simulation/scenes'/(standard['scenario']+'.json');scenario=load_asset_scenario(scene_file)
        target=case['oracle_target']
        if args.retry_failed and directory.exists():
            prior=json.loads((directory/'execution.json').read_text()) if (directory/'execution.json').exists() else {}
            if not prior.get('passed'):
                archive=out/'attempts'/(cid+'-'+str(time.time_ns()))
                archive.parent.mkdir(exist_ok=True)
                if not directory.resolve().is_relative_to(out.resolve()) or not archive.resolve().is_relative_to(out.resolve()):raise ValueError('Attempt path escaped output')
                directory.rename(archive)
        vision=DirectVisionAgent(load_planner_config())
        if args.resume and (directory/'execution.json').exists():
            result=json.loads((directory/'execution.json').read_text())
            annotations=json.loads((directory/'visual_annotations.json').read_text())
            # Older measured records did not distinguish probe from confirmation.
            tools={c['decision_id']:(c.get('result') or {}).get('payload',{}).get('action',{}).get('tool') for c in decision_candidates(directory)}
            missing=[loc for loc in result.get('localizations',[]) if 'tool' not in loc]
            if missing:
                if (directory/'review.json').exists():raise ValueError('Reviewed legacy execution requires an explicit migration')
                for loc in missing:loc['tool']=tools[loc['decision_id']]
                (directory/'execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        else:
            result=collect_case(case,directory,seed=case['seed'],simulation_budget=600,
                wall_budget=args.wall_budget,runtime_backend='gazebo',vision_agent=vision,
                scenario_override=scenario,scene_file=scene_file)
            annotations=review_regions(directory,scenario,target)
            result=score_visual(directory,result,target,annotations,scene_file=scene_file)
        approved=[];excluded=[]
        if result.get('passed'):
            for c in decision_candidates(directory):
                target_review=dict(response_sequence=c['response_sequence'],decision_id=c['decision_id'],
                    reason='Source task standard and native execution checked; visual responses additionally checked against exact RGB-D frame and independent scene surface.')
                try:_sample(directory,c,standard,result,target_review,annotations)
                except (ValueError,KeyError,TypeError) as e:
                    excluded.append(dict(sequence=c['response_sequence'],reason=str(e)));continue
                approved.append(c['response_sequence'])
            if approved and not (directory/'review.json').exists():
                build_review(directory,standard=standard,reviewer='codex_source_and_rgbd_review_v1',
                    response_sequences=approved,target_reviews={n:'Matches frozen source standard and validated execution; exact frame surface and causal continuation checked for visual tools.' for n in approved},visual_annotations=annotations)
            audit=audit_episode(directory) if approved else {}
            audit={k:v for k,v in audit.items() if k not in {'samples','standard'}}
        else:audit={}
        (directory/'excluded-labels.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
        row=dict(task_id=cid,split=standard['frozen_split'],passed=bool(result.get('passed')),
            approved_decisions=len(approved),audit=audit,wall_seconds=result.get('wall_seconds'),episode=str(directory))
        rows.append(row)
        summary=dict(requested=len(cases),attempted=len(rows),successful=sum(r['passed'] for r in rows),
            approved_decisions=sum(r['approved_decisions'] for r in rows),training_started=False,episodes=rows)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='audit'},ensure_ascii=False),flush=True)
    return 0 if all(r['passed'] and r['approved_decisions'] for r in rows) else 1

if __name__=='__main__':raise SystemExit(main())
