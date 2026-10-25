from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StartupContractTest(unittest.TestCase):
    def test_optional_airsim_tools_are_retained(self):
        self.assertTrue((ROOT / "scripts/start_airsim.ps1").is_file())
        self.assertTrue((ROOT / "scripts/verify_airsim.py").is_file())

    def test_readme_uses_browser_controls_for_agent_switching(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("工具栏切换 `Mock / DeepSeek API`", readme)
        self.assertIn("bash scripts/start_demo.sh", readme)
        self.assertIn(".venv/bin/python", readme)
        self.assertNotIn("Invoke-RestMethod -Method Post", readme)

    def test_airsim_client_is_an_optional_dependency(self):
        default_requirements = (ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        )
        airsim_requirements = (ROOT / "requirements-airsim.txt").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("cosysairsim", default_requirements)
        self.assertIn("-r requirements.txt", airsim_requirements)
        self.assertIn("cosysairsim==3.3.0", airsim_requirements)


if __name__ == "__main__":
    unittest.main()
