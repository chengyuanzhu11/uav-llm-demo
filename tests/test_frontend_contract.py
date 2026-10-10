from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTest(unittest.TestCase):
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

    def test_planned_route_includes_fixed_origin_even_with_one_waypoint(self):
        self.run_frontend_functions(["updateWaypoints", "disposeRoute"], """
          let plannedRoute = null, plannedRouteKey = '';
          const state = {flight:{plan_id:1, plan_start:{x:1,y:2,z:3},
            waypoints:[{x:8,y:9,z:3}]}, uav:{x:1,y:2,z:3}};
          const toWorld = p => [p.x,p.y,p.z];
          const updateAvoidanceRoute = () => {};
          const scene = {add() {}, remove() {}};
          class Geometry {
            setFromPoints(points) { this.points = points; return this; }
            dispose() {}
          }
          class Material { dispose() {} }
          const THREE = {
            Group: class {
              constructor() { this.children = []; }
              add(o) { this.children.push(o); }
              traverse(fn) { this.children.forEach(fn); }
            },
            BufferGeometry: Geometry, SphereGeometry: Geometry,
            LineDashedMaterial: Material, MeshBasicMaterial: Material,
            Line: class {
              constructor(geometry, material) { Object.assign(this,{geometry,material}); }
              computeLineDistances() {}
            },
            Mesh: class {
              constructor(geometry, material) {
                Object.assign(this,{geometry,material}); this.position = {copy() {}};
              }
            }
          };
          updateWaypoints();
          assert.deepEqual(plannedRoute.children[0].geometry.points, [[1,2,3],[8,9,3]]);
          assert.equal(plannedRoute.children.length, 2); // Only real waypoints get markers.
          const original = plannedRoute;
          state.uav = {x:4,y:5,z:3};
          updateWaypoints();
          assert.equal(plannedRoute, original); // Origin must not follow the UAV.
          state.flight = {plan_id:2, plan_start:{x:8,y:9,z:3},
            waypoints:[{x:4,y:4,z:3},{x:1,y:2,z:3}]};
          updateWaypoints();
          assert.deepEqual(plannedRoute.children[0].geometry.points,
            [[8,9,3],[4,4,3],[1,2,3]]);
          disposeRoute(); updateWaypoints(); // Refresh/rebuild preserves the origin.
          assert.deepEqual(plannedRoute.children[0].geometry.points[0], [8,9,3]);
          state.flight = {plan_id:0, plan_start:null, waypoints:[]};
          updateWaypoints(); assert.equal(plannedRoute, null);
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

    def test_trace_title_uses_recorded_planner_after_agent_switch(self):
        self.run_frontend_functions(["renderToolbarState"], """
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
            ["runCommand", "postState", "toggleMission", "planMission", "switchAgent"], """
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
        self.assertIn("LineDashedMaterial", javascript)
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
        self.assertIn("new THREE.TubeGeometry", javascript)
        self.assertIn("depthTest: false", javascript)
        self.assertIn("pathLine.renderOrder = 5", javascript)

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
        self.assertIn("if (uavGroup && htmlRuntimeActive)", javascript)


if __name__ == "__main__":
    unittest.main()
