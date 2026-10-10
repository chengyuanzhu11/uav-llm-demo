from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StartupContractTest(unittest.TestCase):
    def test_start_demo_only_starts_flask(self):
        script = (ROOT / "scripts/start_demo.ps1").read_text(encoding="utf-8")

        self.assertNotIn("start_airsim.ps1", script)
        self.assertNotIn("/api/simulator/config", script)
        self.assertNotIn("41451", script)
        self.assertIn("http://127.0.0.1:5000", script)

    def test_start_demo_restarts_only_its_tracked_listener(self):
        script = (ROOT / "scripts/start_demo.ps1").read_text(encoding="utf-8")

        self.assertIn("$flaskListener.OwningProcess -eq $trackedPid", script)
        self.assertIn("taskkill.exe /PID $trackedPid /T /F", script)
        self.assertIn("Port 5000 is already in use by process", script)

    def test_start_demo_skips_broken_python_candidates(self):
        script = (ROOT / "scripts/start_demo.ps1").read_text(encoding="utf-8")

        self.assertIn("function Test-PythonRuntime", script)
        self.assertIn('import flask', script)
        self.assertIn("Test-PythonRuntime -Candidate $_", script)

    def test_optional_airsim_tools_are_retained(self):
        self.assertTrue((ROOT / "scripts/start_airsim.ps1").is_file())
        self.assertTrue((ROOT / "scripts/verify_airsim.py").is_file())

    def test_readme_uses_browser_controls_for_agent_switching(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("工具栏切换 `Mock / DeepSeek API`", readme)
        self.assertIn("bash scripts/start_demo.sh", readme)
        self.assertIn(".\\scripts\\start_demo.ps1", readme)
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
