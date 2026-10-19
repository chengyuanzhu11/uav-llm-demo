"""Run deterministic production-JavaScript motion and resource regressions."""
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required for frontend motion checks")
class FrontendMotionTests(unittest.TestCase):
    def run_script(self, name):
        result = subprocess.run([shutil.which("node"), str(ROOT / "scripts" / name)],
                                capture_output=True, text=True, timeout=45, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_delayed_motion_uses_confirmed_shared_samples(self):
        self.run_script("test_render_motion.mjs")

    def test_flight_trail_appends_and_bounds_resources(self):
        self.run_script("test_flight_trail.mjs")

    def test_follow_camera_previews_corners_without_clipping(self):
        self.run_script("test_follow_camera.mjs")
