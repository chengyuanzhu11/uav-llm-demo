from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StartupContractTest(unittest.TestCase):

    def test_readme_uses_browser_controls_for_agent_switching(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("工具栏切换 `Mock / DeepSeek API`", readme)
        self.assertIn("bash scripts/start_demo.sh", readme)
        self.assertIn(".venv/bin/python", readme)
        self.assertNotIn("Invoke-RestMethod -Method Post", readme)



if __name__ == "__main__":
    unittest.main()
