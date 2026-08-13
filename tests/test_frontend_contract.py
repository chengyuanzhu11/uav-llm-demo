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


if __name__ == "__main__":
    unittest.main()
