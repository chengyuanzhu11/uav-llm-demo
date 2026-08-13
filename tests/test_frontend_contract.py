from pathlib import Path
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
        self.assertIn('id="metricsValue"', html)
        self.assertIn("/api/scenarios", javascript)
        self.assertIn("/api/scenario", javascript)

    def test_scene_first_workbench_structure_exists(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )

        required = (
            'class="app-toolbar"',
            'id="decisionInspector"',
            'id="inspectorToggleBtn"',
            'id="settingsBtn"',
            'id="settingsDialog"',
            'id="trajectoryBtn"',
            'id="operationStatus"',
            'data-stage="observation"',
            'data-stage="agent"',
            'data-stage="guard"',
            'data-stage="execution"',
            'id="historyTableBody"',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, html)

    def test_icons_are_local_and_accessible(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )
        icon_bundle = ROOT / "uav_llm_demo/static/vendor/lucide.min.js"

        self.assertTrue(icon_bundle.is_file())
        self.assertIn("vendor/lucide.min.js", html)
        self.assertNotIn("unpkg.com", html)
        self.assertNotIn("cdn.jsdelivr.net", html)
        self.assertIn('aria-label="Settings"', html)
        self.assertIn('aria-label="Expand decision inspector"', html)

    def test_javascript_exposes_focused_workbench_renderers(self):
        javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(
            encoding="utf-8"
        )

        for function_name in (
            "renderToolbarState",
            "renderTelemetry",
            "renderDecisionTrace",
            "renderHistory",
            "renderRuntimeSettings",
            "setOperationState",
            "toggleInspectorWidth",
            "showHistoryView",
        ):
            self.assertIn(f"function {function_name}", javascript)
        self.assertIn(
            'sessionStorage.getItem("uav-inspector-expanded")', javascript
        )
        self.assertIn('requestJson("/api/runtime/stop"', javascript)
        self.assertNotIn("setInterval(", javascript)

    def test_styles_define_two_width_inspector_and_operation_states(self):
        styles = (ROOT / "uav_llm_demo/static/styles.css").read_text(
            encoding="utf-8"
        )

        for selector in (
            ".simulator.inspector-expanded",
            ".decision-inspector",
            '.simulator[data-operation="executing"]',
            '.simulator[data-operation="paused"]',
            '.simulator[data-operation="error"]',
            ".settings-dialog",
            ".history-table",
        ):
            self.assertIn(selector, styles)
        self.assertIn("minmax(340px, 28%)", styles)
        self.assertIn("minmax(400px, 34%)", styles)

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
        self.assertIn("setText(telemetryPhaseValue, state.phase)", javascript)

    def test_pause_keeps_a_concurrently_committed_step_snapshot(self):
        javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(
            encoding="utf-8"
        )
        stop_start = javascript.index("async function stopAuto")
        stop_end = javascript.index("\n\nfunction stopAutoWhenDone", stop_start)
        stop_function = javascript[stop_start:stop_end]

        guard = 'if (stepOutcome?.status !== "committed")'
        assignment = "state = stoppedState;"
        self.assertIn(guard, stop_function)
        self.assertIn(assignment, stop_function)
        self.assertLess(stop_function.index(guard), stop_function.index(assignment))

    def test_position_telemetry_reserves_space_for_full_coordinates(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )
        styles = (ROOT / "uav_llm_demo/static/styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn('class="telemetry-item position"', html)
        self.assertIn("minmax(150px, 1.35fr)", styles)


if __name__ == "__main__":
    unittest.main()
