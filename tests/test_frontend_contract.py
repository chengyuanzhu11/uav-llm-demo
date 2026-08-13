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
        self.assertIn("/api/scenarios", javascript)
        self.assertIn("/api/scenario", javascript)

    def test_scene_first_workbench_structure_exists(self):
        html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
            encoding="utf-8"
        )

        required = (
            'class="app-toolbar"',
            'id="decisionInspector"',
            'id="settingsBtn"',
            'id="settingsDialog"',
            'id="operationStatus"',
            'data-stage="observation"',
            'data-stage="agent"',
            'data-stage="execution"',
            'Reasoning &amp; Decision',
            '>Result<',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, html)
        self.assertEqual(html.count('class="trace-stage"'), 3)
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
        self.assertIn('aria-label="Settings"', html)
        self.assertNotIn("Expand decision inspector", html)

    def test_javascript_exposes_focused_workbench_renderers(self):
        javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(
            encoding="utf-8"
        )

        for function_name in (
            "renderToolbarState",
            "renderTelemetry",
            "renderDecisionTrace",
            "renderRuntimeSettings",
            "setOperationState",
        ):
            self.assertIn(f"function {function_name}", javascript)
        self.assertNotIn("inspectorExpanded", javascript)
        self.assertNotIn("showHistoryView", javascript)
        self.assertNotIn("renderHistory", javascript)
        self.assertIn('requestJson("/api/runtime/stop"', javascript)
        self.assertNotIn("setInterval(", javascript)

    def test_styles_define_two_width_inspector_and_operation_states(self):
        styles = (ROOT / "uav_llm_demo/static/styles.css").read_text(
            encoding="utf-8"
        )

        for selector in (
            ".decision-inspector",
            ".decision-guard",
            '.simulator[data-operation="executing"]',
            '.simulator[data-operation="paused"]',
            '.simulator[data-operation="error"]',
            ".settings-dialog",
        ):
            self.assertIn(selector, styles)
        self.assertIn("minmax(240px, 19%)", styles)
        self.assertNotIn("inspector-expanded", styles)

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


if __name__ == "__main__":
    unittest.main()
