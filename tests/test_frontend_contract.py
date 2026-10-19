from pathlib import Path
import json
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTest(unittest.TestCase):
    def test_scenario_catalog_arriving_after_state_keeps_current_map_selected(self):
        self.run_frontend_functions(["loadScenarios"], r'''
let scenariosLoaded = false;
const state = {scenario_id: "semantic_campus"};
const document = {createElement: () => ({})};
const scenarioSelect = {
  value: "", options: [],
  replaceChildren() { this.options = []; this.value = ""; },
  appendChild(option) { this.options.push(option); if (!this.value) this.value = option.value; },
};
let requests = 0;
async function requestJson() {
  requests += 1;
  return {scenarios: [{id: "urban_crossing", name: "Urban Crossing"},
    {id: "semantic_campus", name: "北航杭州国际校园"}]};
}
await loadScenarios();
assert.equal(scenarioSelect.value, "semantic_campus");
assert.equal(scenarioSelect.options.length, 2);
await loadScenarios();
assert.equal(requests, 1);
''')

    def test_runtime_selector_and_conditional_airsim_panel_exist(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('data-runtime="html"', html)
        self.assertIn('data-runtime="airsim"', html)
        self.assertIn('id="airsimPanel"', html)
        self.assertIn('role="radiogroup"', html)

    def test_javascript_uses_only_the_runtime_api(self):
        javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('"/api/runtime"', javascript)
        self.assertIn('"/api/runtime/stop"', javascript)
        self.assertIn('"/api/runtime/wind"', javascript)
        self.assertNotIn("/api/simulator/", javascript)
        self.assertNotIn("simulator_backend", javascript)
        self.assertNotIn("state.simulator", javascript)
        self.assertNotIn("new THREE.Clock", javascript)
        self.assertNotIn("PCFSoftShadowMap", javascript)
        self.assertIn("new THREE.Timer", javascript)

    def test_styles_include_pending_and_hidden_runtime_states(self):
        styles = (ROOT / "uav_llm_demo/static/styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn(".runtime-selector", styles)
        self.assertIn(".runtime-option[aria-checked=\"true\"]", styles)
        self.assertIn(".is-hidden", styles)

    def test_plan_b_scenario_and_metrics_controls_exist(self):
        html = (ROOT / "uav_llm_demo" / "templates" / "index.html").read_text(
            encoding="utf-8"
        )
        javascript = (ROOT / "uav_llm_demo" / "static" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="scenarioSelect"', html)
        self.assertIn("/api/scenarios", javascript)
        self.assertIn("/api/scenario", javascript)

    def test_scene_first_workbench_structure_exists(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )

        required = (
            'class="app-toolbar"',
            'id="decisionInspector"',
            'data-agent="mock"',
            'data-agent="deepseek"',
            'id="agentDialog"',
            'id="airsimDialog"',
            'id="airsimConfigBtn"',
            'data-stage="observation"',
            'data-stage="agent"',
            'data-stage="execution"',
            'class="stage-header stage-title-row"',
            'Reasoning &amp; Decision',
            '>Flight<',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, html)
        self.assertEqual(html.count('class="trace-stage"'), 3)
        self.assertNotIn('id="settingsBtn"', html)
        self.assertNotIn('id="settingsDialog"', html)
        self.assertNotIn('id="operationBanner"', html)
        self.assertNotIn('id="operationStatus"', html)
        self.assertNotIn('id="sceneExecutionStatus"', html)
        self.assertNotIn('id="inspectorToggleBtn"', html)
        self.assertNotIn('id="historyView"', html)
        self.assertNotIn('class="recent-history"', html)

    def test_icons_are_local_and_accessible(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )
        icon_bundle = ROOT / "uav_llm_demo/static/vendor/lucide.min.js"

        self.assertTrue(icon_bundle.is_file())
        self.assertIn("vendor/lucide.min.js", html)
        self.assertNotIn("unpkg.com", html)
        self.assertNotIn("cdn.jsdelivr.net", html)
        self.assertIn('aria-label="Configure DeepSeek API key"', html)
        self.assertIn('aria-label="Configure AirSim"', html)
        self.assertNotIn("Expand decision inspector", html)

    def test_javascript_exposes_focused_workbench_renderers(self):
        javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(
            encoding="utf-8"
        )

        for function_name in (
            "renderToolbarState",
            "renderTelemetry",
            "renderDecisionTrace",
            "renderRuntimeControls",
        ):
            self.assertIn(f"function {function_name}", javascript)
        self.assertNotIn("inspectorExpanded", javascript)
        self.assertNotIn("showHistoryView", javascript)
        self.assertNotIn("renderHistory", javascript)
        self.assertNotIn("settingsBtn", javascript)
        self.assertNotIn("settingsDialog", javascript)
        self.assertNotIn("operationBanner", javascript)
        self.assertNotIn("sceneExecutionStatus", javascript)
        self.assertIn('postState("/api/config"', javascript)
        self.assertIn("switchAgent", javascript)
        self.assertNotIn("localStorage", javascript)
        self.assertNotIn("sessionStorage", javascript)
        self.assertIn('"/api/runtime/stop"', javascript)
        self.assertNotIn("setInterval(", javascript)

    def test_styles_define_two_width_inspector_and_operation_states(self):
        styles = (ROOT / "uav_llm_demo/static/styles.css").read_text(
            encoding="utf-8"
        )

        for selector in (
            ".decision-inspector",
            ".decision-guard",
            ".simulator",
            ".control-dialog",
            ".agent-selector",
            ".trace-stage > .stage-title-row",
        ):
            self.assertIn(selector, styles)
        self.assertIn("minmax(240px, 19%)", styles)
        self.assertNotIn("inspector-expanded", styles)
        self.assertNotIn(".operation-banner", styles)
        self.assertNotIn(".scene-execution-status", styles)

    def test_scene_telemetry_includes_current_phase(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )
        javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="telemetryPhaseValue"', html)
        self.assertIn(
            'const telemetryPhaseValue = document.getElementById("telemetryPhaseValue")',
            javascript,
        )
        self.assertIn("setText(telemetryPhaseValue, state.flight?.mode)", javascript)

    def run_frontend_functions(self, names, test):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is required for frontend behavior checks")
        source = (ROOT / "uav_llm_demo/static/app.js").read_text(encoding="utf-8")
        functions = []
        for name in names:
            match = re.search(
                rf"^(?:async )?function {name}\([^\n]*\) \{{.*?^\}}",
                source, re.M | re.S,
            )
            self.assertIsNotNone(match, name)
            functions.append(match.group())
        script = "const assert = require('node:assert/strict');\n(async () => {\n"
        script += "\n".join(functions) + "\n" + test
        script += "\n})().catch(e => {console.error(e); process.exit(1)});"
        result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_star_reward_follows_milestone_once_and_resets(self):
        self.run_frontend_functions(["updateGoalStar"], r"""
          const object = () => ({position:{set(){}},rotation:{y:0},scale:{setScalar(){}},material:{opacity:1}});
          const reward = () => ({star:object(),ring:object(),sparks:[object()],collected:false,age:null});
          let goalStar = {userData:{reward:reward()},visible:true};
          const goalReward = {hidden:true,textContent:''};
          const state = {milestones:{target_reached:false}};
          updateGoalStar(.05,0,true);
          assert.equal(goalStar.visible,true); assert.equal(goalReward.hidden,true);
          state.milestones.target_reached = true;
          state.milestones.target_reached_at = 1;
          // The visual reward waits for the delayed vehicle to reach the goal.
          updateGoalStar(.05,.8,true,.65);
          assert.equal(goalStar.userData.reward.collected,false);
          assert.equal(goalReward.hidden,true);
          updateGoalStar(.05,1,true,1);
          assert.equal(goalReward.hidden,false);
          assert.match(goalReward.textContent,/\+1/);
          updateGoalStar(2,3,true);
          assert.equal(goalStar.visible,false); assert.equal(goalReward.hidden,true);
          updateGoalStar(.05,4,true);
          assert.equal(goalReward.hidden,true);
          // A refresh after collection must not replay the reward.
          goalStar = {userData:{reward:{...reward(),collected:true}},visible:true};
          updateGoalStar(.05,5,true);
          assert.equal(goalStar.visible,false); assert.equal(goalReward.hidden,true);
          // Reset makes the goal collectible again.
          state.milestones.target_reached = false;
          goalStar = {userData:{reward:reward()},visible:false};
          updateGoalStar(.05,6,true); assert.equal(goalStar.visible,true);
          updateGoalStar(.05,7,false); assert.equal(goalStar.visible,false);
        """)

    def test_planned_route_includes_fixed_origin_even_with_one_waypoint(self):
        vendor = (ROOT / "uav_llm_demo/static/vendor/three.module.js").as_uri()
        trail = (ROOT / "uav_llm_demo/static/flight_trail.js").as_uri()
        self.run_frontend_functions(["updateWaypoints", "disposeRoute"],
            "const THREE = await import(" + json.dumps(vendor) + ");\n"
            + "const {FlightTrail} = await import(" + json.dumps(trail) + ");\n" + r"""
          let plannedRoute = null, plannedTrail = null, plannedRouteKey = '';
          const state = {flight:{plan_id:1, plan_start:{x:1,y:2,z:3},
            waypoints:[{x:8,y:9,z:3}]}, uav:{x:1,y:2,z:3}};
          const toWorld = p => new THREE.Vector3(p.x,p.y,p.z);
          const updateAvoidanceRoute = () => {};
          const scene = new THREE.Scene();
          const ringCenter = (mesh, ring) => {
            const points=mesh.geometry.getAttribute('position');
            const center=new THREE.Vector3();
            for(let index=ring*6; index<ring*6+6; index++) {
              center.add(new THREE.Vector3().fromBufferAttribute(points,index));
            }
            return center.divideScalar(6);
          };
          const same = (actual, expected) => assert.ok(actual.distanceTo(new THREE.Vector3(...expected))<1e-6);
          updateWaypoints();
          const tube = plannedTrail.object.children[0];
          same(ringCenter(tube,0), [1,2,3]); same(ringCenter(tube,1), [8,9,3]);
          assert.equal(plannedRoute.children.length, 2); // Only real waypoints get markers.
          assert.equal(plannedTrail.stats.segments,1);
          assert.equal(tube.isMesh,true); // Width comes from geometry, not unsupported WebGL linewidth.
          const center=ringCenter(tube,0);
          const vertex=new THREE.Vector3().fromBufferAttribute(tube.geometry.getAttribute('position'),0);
          assert.ok(Math.abs(vertex.distanceTo(center)-.095)<1e-6);
          assert.equal(tube.material.opacity,.92);
          assert.equal(tube.material.color.getHex(),0xffc857);
          assert.equal(tube.material.depthTest,false);
          assert.equal(tube.renderOrder,4); // Actual flight history remains above the plan.
          assert.equal(plannedTrail.object.renderOrder,0); // Group order must not override the mesh order.
          const hits=[];tube.raycast(null,hits);plannedRoute.children[1].raycast(null,hits);
          assert.deepEqual(hits,[]); // Diagnostic overlays must not intercept bird hover.
          const original=plannedRoute, geometry=tube.geometry;
          const version=geometry.getAttribute('position').version;
          state.uav = {x:4,y:5,z:3};
          updateWaypoints();
          assert.equal(plannedRoute,original); assert.equal(plannedTrail.object.children[0].geometry,geometry);
          assert.equal(geometry.getAttribute('position').version,version);
          let disposedGeometry=0, disposedMaterial=0, disposedMarker=0;
          geometry.addEventListener('dispose',()=>disposedGeometry++);
          tube.material.addEventListener('dispose',()=>disposedMaterial++);
          plannedRoute.children[1].geometry.addEventListener('dispose',()=>disposedMarker++);
          state.flight = {plan_id:2, plan_start:{x:8,y:9,z:3},
            waypoints:[{x:4,y:4,z:3},{x:1,y:2,z:3}]};
          updateWaypoints();
          assert.equal(disposedGeometry,1);assert.equal(disposedMaterial,1);assert.equal(disposedMarker,1);
          assert.equal(plannedTrail.stats.segments,2);
          same(ringCenter(plannedTrail.object.children[0],0), [8,9,3]);
          same(ringCenter(plannedTrail.object.children[0],3), [1,2,3]);
          disposeRoute(); updateWaypoints(); // Refresh/rebuild preserves the origin.
          same(ringCenter(plannedTrail.object.children[0],0), [8,9,3]);
          state.flight = {plan_id:0, plan_start:null, waypoints:[]};
          updateWaypoints();assert.equal(plannedRoute,null);assert.equal(plannedTrail,null);
          assert.equal(scene.children.length,0);
          state.flight = {plan_id:3,plan_start:null,waypoints:[{x:1,y:2,z:3}]};
          updateWaypoints();assert.equal(plannedTrail.stats.segments,0); // No fabricated origin.
          assert.equal(plannedRoute.children.length,2);
          disposeRoute();disposeRoute();assert.equal(scene.children.length,0);
        """)

    def test_state_rejects_late_poll_and_control_responses_before_scene_updates(self):
        self.run_frontend_functions(["acceptState"], """
          let state = null, lastSequence = -1, trafficMotion;
          let renders = 0;
          const hasOwn = (o, k) => Object.hasOwn(o, k);
          const clearEnvironment = () => {}, syncScene = () => {};
          const renderUi = () => renders++;
          const snapshot = n => ({sequence:n, flight:{plan_id:1}, traffic_motion:{elapsed:n}});
          assert.equal(acceptState(snapshot(12)), true);
          assert.equal(acceptState(snapshot(11)), false);
          assert.equal(acceptState(snapshot(12)), false);
          assert.equal(acceptState({}), false);
          assert.equal(state.sequence, 12);
          assert.equal(trafficMotion.elapsed, 12);
          assert.equal(renders, 1);
        """)

    def test_changed_target_rebuilds_scene_even_when_both_states_are_idle(self):
        self.run_frontend_functions(["acceptState"], """
          let state = {sequence:1, scenario_id:'urban_crossing', runtime_backend:'html',
            target:{x:3,y:8,z:2}, flight:{plan_id:0}, path:[{}]};
          let lastSequence = 1, clears = 0, initial = false;
          const hasOwn = (object, key) => Object.hasOwn(object, key);
          const clearEnvironment = () => clears++;
          const syncScene = value => initial = value;
          const renderUi = () => {};
          assert.equal(acceptState({...state,sequence:2,target:{x:16,y:3,z:2}}), true);
          assert.equal(clears, 1); assert.equal(initial, true);
          assert.equal(acceptState({...state,sequence:3}), true);
          assert.equal(clears, 1); assert.equal(initial, false);
        """)

    def test_semantic_step_target_preserves_scene_but_mission_reset_rebuilds_it(self):
        self.run_frontend_functions(["acceptState"], """
          let state = {sequence:1, scenario_id:'semantic_campus', runtime_backend:'html',
            scenario:{mission_mode:'semantic'}, diagnostics:{run_id:'mission-1'},
            target:{x:23,y:12,z:3}, flight:{plan_id:2}, path:[{},{}]};
          let lastSequence = 1, clears = 0, initial = false, commandError = '';
          const hasOwn = (object, key) => Object.hasOwn(object, key);
          const clearEnvironment = () => clears++;
          const syncScene = value => initial = value;
          const renderUi = () => {};
          const next = {...state,sequence:2,target:{x:2,y:27,z:0},flight:{plan_id:0}};
          assert.equal(acceptState(next), true);
          assert.equal(clears, 0); assert.equal(initial, false);
          assert.equal(acceptState({...state,sequence:3,
            diagnostics:{run_id:'mission-2'},path:[{}]}), true);
          assert.equal(clears, 1); assert.equal(initial, true);
        """)

    def test_trace_title_uses_recorded_planner_after_agent_switch(self):
        self.run_frontend_functions(["renderToolbarState", "isSemanticScenario"], """
          let state = {
            status:'active', phase:'returning_home', agent_backend:'deepseek',
            agent_model:'deepseek-v4-flash', flight:{mode:'running'},
            history:[{decision:{agent_backend:'mock', agent_model:null}}],
          };
          const commandError = '', commandPending = false;
          const backendValue = {}, phaseValue = {};
          const setText = (node, value) => { node.textContent = value; };
          const simulator = {setAttribute() {}};
          const getHistoryEntries = () => state.history;
          const button = () => ({setAttribute() {}, classList:{toggle() {}}, dataset:{agent:'deepseek'}});
          const planBtn = button(), startBtn = button(), resetBtn = button();
          const scenarioSelect = button(), agentButtons = [button()], agentKeyBtn = null;
          const updateCommandButton = () => false;
          renderToolbarState();
          assert.equal(backendValue.textContent, 'mock');
          state.history = [];
          renderToolbarState();
          assert.equal(backendValue.textContent, 'deepseek:deepseek-v4-flash');
          assert.equal(planBtn.hidden, false);
          assert.equal(agentButtons[0].hidden, false);
          // Semantic decisions are always attributed to DeepSeek, while the
          // most recent geometry plan may have come from the local planner.
          state.scenario = {mission_mode:'semantic'};
          state.semantic_mission = {status:'interpreting'};
          state.flight = {running:true, planning:true, mode:'planning'};
          state.history = [{decision:{agent_backend:'local',agent_model:null}}];
          const mockButton = button(); mockButton.dataset.agent = 'mock';
          agentButtons.push(mockButton);
          renderToolbarState();
          assert.equal(backendValue.textContent, 'DeepSeek');
          assert.equal(planBtn.hidden, true);
          assert.equal(mockButton.hidden, true);
          assert.equal(startBtn.disabled, false); // Interpretation can pause.
          state.flight.running = false;
          renderToolbarState();
          assert.equal(startBtn.disabled, false); // And resume while a request is pending.
          state.semantic_mission.status = 'error';
          state.flight = {running:false, planning:false, mode:'error'};
          renderToolbarState();
          assert.equal(startBtn.disabled, false); // Reconfigure a key, then retry.
          for (const status of ['idle','needs_clarification','completed','safety_wait']) {
            state.semantic_mission.status = status;
            renderToolbarState();
            assert.equal(startBtn.disabled, true);
          }
        """)

    def test_poll_is_serial_read_only_and_retries_without_mutating(self):
        self.run_frontend_functions(["pollState"], """
          let pollActive = false, reads = 0, inFlight = 0, accepted = 0;
          const feedbackText = {}, setText = () => {};
          const acceptState = () => accepted++;
          const requestJson = async (url, options) => {
            assert.equal(url, '/api/state'); assert.equal(options, undefined);
            assert.equal(++inFlight, 1); reads++;
            await Promise.resolve(); inFlight--;
            if (reads === 1) throw Error('offline');
            return {};
          };
          const sleep = async ms => {
            assert.ok(ms >= 100 && ms <= 150);
            if (reads === 3) pollActive = false;
          };
          await Promise.all([pollState(), pollState()]);
          assert.equal(reads, 3); assert.equal(accepted, 2);
        """)

    def test_mission_commands_follow_backend_state_and_are_serial(self):
        self.run_frontend_functions(
            ["runCommand", "postState", "toggleMission", "planMission", "switchAgent", "isSemanticScenario"], """
          let state = {flight:{running:false}, agent_backend:'mock'}, commandPending = false;
          let commandError = '', release;
          const urls = [], renderUi = () => {}, acceptState = () => {};
          const agentDialog = {}, closeDialog = () => {}, agentDialogError = null;
          const requestJson = async (url, options) => {
            assert.equal(options.method, 'POST'); urls.push(url);
            await new Promise(resolve => { release = resolve; }); return {};
          };
          let pending = toggleMission();
          assert.equal(commandPending, true);
          assert.equal(await planMission(), false);
          release(); await pending;
          state.flight.running = true;
          pending = toggleMission(); release(); await pending;
          state.flight.running = false;
          pending = planMission(); release(); await pending;
          pending = switchAgent('deepseek', true, {api_key:'test-only'});
          assert.equal(await toggleMission(), false);
          release(); await pending;
          assert.deepEqual(urls, ['/api/mission/start', '/api/runtime/stop', '/api/mission/plan', '/api/config']);
          // Urban retains its existing planning-is-active pause behavior.
          state.flight = {running:false, planning:true};
          pending = toggleMission(); release(); await pending;
          assert.equal(urls.at(-1), '/api/runtime/stop');
          // A paused semantic interpretation can resume even if its provider
          // request is still pending. It cannot issue a route-only Plan or switch to Mock.
          state.scenario = {mission_mode:'semantic'};
          state.semantic_mission = {status:'interpreting'};
          const before = urls.length;
          assert.equal(await planMission(), false);
          assert.equal(await switchAgent('mock', true), false);
          assert.equal(urls.length, before);
          pending = toggleMission(); release(); await pending;
          assert.equal(urls.at(-1), '/api/mission/start');
          state.flight.running = true;
          pending = toggleMission(); release(); await pending;
          assert.equal(urls.at(-1), '/api/runtime/stop');
        """)

    def test_continuous_contract_has_no_legacy_step_or_replay_loop(self):
        javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(encoding="utf-8")
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(encoding="utf-8")
        for old in ("/api/step", "autoRunning", "stepInFlight", "currentStepSettled",
                    "trajectory_duration", "activeTrajectory", "runAutoLoop"):
            self.assertNotIn(old, javascript)
        self.assertIn('id="planBtn"', html)
        self.assertIn('id="startBtn"', html)
        self.assertIn('id="plannerProgress"', html)
        self.assertRegex(html, r'id="deepseekApiKey"[^>]*type="password"')
        self.assertIn("void pollState()", javascript)
        self.assertIn("new FlightTrail(THREE, toWorld)", javascript)
        self.assertIn("disposeRoute()", javascript)

    def test_position_telemetry_reserves_space_for_full_coordinates(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )
        styles = (ROOT / "uav_llm_demo/static/styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn('class="telemetry-item position"', html)
        self.assertIn("minmax(150px, 1.35fr)", styles)

    def test_trajectory_is_a_persistent_thick_three_dimensional_path(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )
        javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(
            encoding="utf-8"
        )

        self.assertNotIn('id="trajectoryBtn"', html)
        trail = (ROOT / "uav_llm_demo/static/flight_trail.js").read_text(encoding="utf-8")
        self.assertIn("new FlightTrail(THREE, toWorld)", javascript)
        self.assertNotIn("new THREE.TubeGeometry", javascript)
        self.assertIn("depthTest: false", trail)
        self.assertIn("mesh.renderOrder = 5", trail)

    def test_airsim_runtime_does_not_replay_html_scene_motion(self):
        javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'const htmlRuntimeActive = state.runtime_backend === "html";',
            javascript,
        )
        self.assertIn("uavGroup.visible = htmlRuntimeActive", javascript)
        self.assertIn("bird.visible = htmlRuntimeActive", javascript)
        self.assertIn('if (state.runtime_backend !== "html")', javascript)
        self.assertIn('const frame = htmlRuntimeActive ? motionTimeline.sample(', javascript)
        self.assertIn("if (uavGroup && frame)", javascript)


if __name__ == "__main__":
    unittest.main()
