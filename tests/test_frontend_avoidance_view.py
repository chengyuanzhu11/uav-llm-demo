"""Exercise diagnostic overlays with real Three.js and the render clock."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required for Three.js checks")
class AvoidanceViewTest(unittest.TestCase):
    def run_js(self, test):
        vendor = (ROOT / "uav_llm_demo/static/vendor/three.module.js").as_uri()
        module = (ROOT / "uav_llm_demo/static/avoidance_view.js").as_uri()
        script = "\n".join([
            'import assert from "node:assert/strict";',
            f'import * as THREE from {json.dumps(vendor)};',
            f'import {{AvoidanceView}} from {json.dumps(module)};',
            'const toWorld = p => new THREE.Vector3(p.x * 3, p.z * 5, p.y * 3);',
            'const view = new AvoidanceView(THREE, toWorld);',
            'const p = x => ({x,y:0,z:2});',
            '''const active = (sequence, time, x = 0, id = 1) => ({sequence, simulation_time: time,
              active:true,limited:false,nominal_path:[p(x),p(x+1),p(x+2)],
              selected_path:[p(x),{x:x+1,y:1,z:2},{x:x+2,y:2,z:2}],
              threats:[{id:"bird",predicted_clearance_m:-.2}],
              episode:{id,status:"active",started_at:time,ended_at:null,
                min_separation_clearance_m:null,object_ids:["bird"]}});''',
            test,
        ])
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_delayed_clock_interpolates_only_within_one_active_episode(self):
        self.run_js('''
          view.ingest(active(1, 1, 0)); view.ingest(active(2, 1.2, 2));
          assert.equal(view.sample(.99), null); // Future decisions never precede the UAV.
          assert.equal(view.sample(1).nominal_path[0].x, 0);
          assert.ok(Math.abs(view.sample(1.1).nominal_path[0].x - 1) < 1e-9);
          assert.ok(Math.abs(view.sample(1.1).selected_path[1].y - 1) < 1e-9);
          view.ingest(active(3, 1.4, 20, 2));
          assert.equal(view.sample(1.3).nominal_path[0].x, 2); // No blend across episodes.
          assert.equal(view.sample(1.4).nominal_path[0].x, 20);
          view.ingest(active(4, 3, 30, 2));
          assert.equal(view.sample(2).nominal_path[0].x, 20); // Long packet gap is not fabricated.
        ''')

    def test_ended_episode_lingers_three_simulation_seconds_then_clears(self):
        self.run_js('''
          view.ingest(active(1, 1));
          const ended = {...active(2, 2),active:false,nominal_path:[],selected_path:[]};
          ended.episode = {...ended.episode,status:"cleared",ended_at:2,min_separation_clearance_m:.73};
          view.ingest(ended);
          assert.equal(view.sample(1.99).active, true);
          assert.equal(view.sample(2).active, false);
          assert.equal(view.sample(4.99).selected_path.length, 3);
          assert.equal(view.sample(4.99).episode.min_separation_clearance_m, .73);
          assert.equal(view.sample(5.01), null);
          // Pause/stall does not run out a simulation-clock display deadline.
          assert.equal(view.sample(4.99).episode.id, 1);
          view.ingest({sequence:0,simulation_time:0,episode:null});
          assert.equal(view.samples.length,0); assert.equal(view.object.visible,false);
        ''')

    def test_geometry_is_sampled_curves_with_stable_gpu_buffers_and_displayed_origin(self):
        self.run_js('''
          const shown = new THREE.Vector3(1.2, 10, 0);
          view.update(1,shown,p(10),true);
          assert.equal(view.executionLine.geometry.attributes.position.getX(0),Math.fround(1.2));
          assert.equal(view.executionLine.geometry.attributes.position.getX(1),30);
          view.ingest(active(1,1)); view.update(1,shown,p(10),true);
          assert.equal(view.executionLine.visible,false);
          assert.equal(view.selectedLine.userData.sampleCount,3);
          assert.equal(view.selectedLine.geometry.drawRange.count,72);
          // Each six-sided ring is centred exactly on the sampled rollout.
          const tube=view.selectedLine.geometry.attributes.position;
          const centre=[0,0,0];
          for(let i=6;i<12;i++){centre[0]+=tube.getX(i)/6;centre[1]+=tube.getY(i)/6;centre[2]+=tube.getZ(i)/6;}
          assert.ok(Math.abs(centre[0]-3)<1e-6);assert.ok(Math.abs(centre[1]-10)<1e-6);assert.ok(Math.abs(centre[2]-3)<1e-6);
          const positions = view.selectedLine.geometry.attributes.position;
          const distances = view.executionLine.geometry.attributes.lineDistance;
          for(let frame=0;frame<60;frame++) view.update(1,shown,p(10),true);
          assert.equal(view.selectedLine.geometry.attributes.position,positions);
          assert.equal(view.executionLine.geometry.attributes.lineDistance,distances);
          assert.equal(view.object.children.length,3);
          view.update(1,shown,null,false,false);
          assert.equal(view.object.visible,false);
          view.reset(); view.update(1,shown,p(10),true);
          assert.equal(view.nominalLine.visible,false);assert.equal(view.selectedLine.visible,false);
          assert.equal(view.executionLine.visible,true);
          view.dispose(); assert.equal(view.samples.length,0);
        ''')

    def test_normal_execution_line_is_hidden_during_planning_and_active_avoidance(self):
        self.run_js("""
          const shown=new THREE.Vector3();
          view.update(1,shown,p(10),false);assert.equal(view.executionLine.visible,false);
          view.update(1,shown,p(10),true);assert.equal(view.executionLine.visible,true);
          view.ingest(active(1,1));view.update(1,shown,p(10),true);
          assert.equal(view.executionLine.visible,false);assert.equal(view.selectedLine.visible,true);
          const ended={...active(2,2),active:false};
          ended.episode={...ended.episode,status:"cleared",ended_at:2};
          view.ingest(ended);view.update(2,shown,p(10),true);
          assert.equal(view.executionLine.visible,true);
          view.update(2,shown,p(10),false);assert.equal(view.executionLine.visible,false);
        """)

    def test_malformed_and_reordered_forecasts_do_not_draw_invented_paths(self):
        self.run_js('''
          view.ingest(active(2,1)); view.ingest(active(1,.9,100));
          assert.equal(view.samples.length,1);
          const bad = active(3,1.2); bad.episode.object_ids = "malformed";
          bad.nominal_path=[p(0),{x:NaN,y:0,z:2}]; bad.selected_path=[p(0)];
          view.ingest(bad);view.update(1.2,new THREE.Vector3(),null,false);
          assert.equal(view.nominalLine.visible,false); assert.equal(view.selectedLine.visible,false);
          assert.equal(view.sample(1.2).episode.min_separation_clearance_m,null);
          const ended={...active(4,1.4),active:false,episode:null};
          view.ingest(ended);assert.equal(view.sample(1.4),null);
        ''')

    def test_status_distinguishes_measured_clearance_and_interrupted_episode(self):
        from tests.test_frontend_contract import FrontendContractTest
        FrontendContractTest.run_frontend_functions(self, ["renderAvoidanceStatus"], '''
          const avoidanceStatus={hidden:true,dataset:{}};
          const nodes={avoidanceStatusTitle:{},avoidanceClearance:{}};
          const document={getElementById:id=>nodes[id]};
          const setText=(node,text)=>{node.textContent=text;};
          renderAvoidanceStatus({active:true,episode:{min_separation_clearance_m:null}});
          assert.equal(nodes.avoidanceClearance.textContent,"最小安全净空 未测");
          renderAvoidanceStatus({active:false,episode:{status:"cleared",object_ids:["bird"],min_separation_clearance_m:.321}});
          assert.equal(nodes.avoidanceStatusTitle.textContent,"恢复航行");
          assert.equal(nodes.avoidanceClearance.textContent,"最小安全净空 0.32 m");
          renderAvoidanceStatus({active:false,episode:{status:"interrupted",object_ids:["bird"],min_separation_clearance_m:-.02}});
          assert.equal(nodes.avoidanceStatusTitle.textContent,"避障结束");
          assert.equal(avoidanceStatus.dataset.violation,"true");
          renderAvoidanceStatus({active:true,episode:{object_ids:[],min_separation_clearance_m:null}});
          assert.equal(nodes.avoidanceStatusTitle.textContent,"航迹调整");
          renderAvoidanceStatus({active:false,episode:{status:"cleared",object_ids:[]}});
          assert.equal(nodes.avoidanceStatusTitle.textContent,"调整结束");
          renderAvoidanceStatus(null);assert.equal(avoidanceStatus.hidden,true);
        ''')


if __name__ == "__main__":
    unittest.main()
