"""Exercise real process shutdown in a disposable checkout, never live services."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.stop_services import snapshot, send_signal, stop


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux", "Uses Linux process identities")
class StopServicesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="uav stop ")
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        (self.root / "scripts").mkdir()
        for name in ("stop_demo.sh", "stop_services.py"):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        self.children = []
        self.addCleanup(self.cleanup_processes)
        (self.root / "flask.py").write_text(
            "import signal, sys, time\n"
            "from pathlib import Path\n"
            "if '--ignore-term' in sys.argv: signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "Path(str(__import__('os').getpid())+'.ready').touch()\n"
            "time.sleep(600)\n")
        (self.root / "scripts/start_gazebo.sh").write_text(
            "#!/usr/bin/env bash\nsleep 600 &\necho $! > gazebo-child\nwait\n")

    def launch(self, args, *, cwd=None, env=None):
        process = subprocess.Popen(args, cwd=cwd or self.root, env=env,
                                   start_new_session=True, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        self.children.append(process)
        return process

    def cleanup_processes(self):
        for process in self.children:
            # The test owns these sessions, including orphaned fixture children.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def wait_file(self, path):
        deadline = time.monotonic() + 5
        while not path.exists():
            if time.monotonic() > deadline:
                self.fail(f"Fixture did not start: {path}")
            time.sleep(.02)

    def command(self, *args):
        result = subprocess.run(["bash", str(self.root / "scripts/stop_demo.sh"), *args],
                                cwd="/tmp", capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_scoped_shutdown_children_orphans_and_repeat(self):
        web_args = [sys.executable, "-m", "flask", "--app", "app", "run", "--port", "5019"]
        web = self.launch(web_args)
        gazebo = self.launch(["bash", "scripts/start_gazebo.sh"])
        # Same executable/arguments elsewhere, and a different app in this cwd.
        other = self.root / "other"
        other.mkdir()
        shutil.copyfile(self.root / "flask.py", other / "flask.py")
        guards = [self.launch(web_args, cwd=other),
                  self.launch([sys.executable, "-m", "flask", "--app", "other", "run"])]
        orphan_args = ["bash", "-c", "exec -a 'gz sim gui' sleep 600"]
        orphan = self.launch(orphan_args, env=dict(os.environ, PX4_GZ_WORLDS=str(self.root / ".sim/worlds")))
        guards.append(self.launch(orphan_args, env=dict(os.environ, PX4_GZ_WORLDS=str(other / ".sim/worlds"))))
        self.wait_file(self.root / f"{web.pid}.ready")
        self.wait_file(self.root / "gazebo-child")
        self.wait_file(other / f"{guards[0].pid}.ready")
        self.wait_file(self.root / f"{guards[1].pid}.ready")
        child_pid = int((self.root / "gazebo-child").read_text())

        preview = self.command("--dry-run")
        for process in (web, gazebo, orphan):
            self.assertIn(f"PID {process.pid}\n", preview)
            self.assertIsNone(process.poll())
        for process in guards:
            self.assertNotIn(f"PID {process.pid}\n", preview)
        self.command("--web-only")
        self.assertEqual(web.wait(timeout=2), -signal.SIGTERM)
        self.assertIsNone(gazebo.poll())
        self.assertIsNone(orphan.poll())

        web = self.launch(web_args)
        self.wait_file(self.root / f"{web.pid}.ready")
        self.command("--gazebo-only")
        gazebo.wait(timeout=2)
        orphan.wait(timeout=2)
        self.assertNotIn(child_pid, snapshot())
        self.assertIsNone(web.poll())
        self.command()
        web.wait(timeout=2)
        self.assertIn("没有正在运行", self.command())
        for process in guards:
            self.assertIsNone(process.poll())

    def test_reused_identity_is_ignored_and_unresponsive_service_is_stopped(self):
        web = self.launch([sys.executable, "-m", "flask", "--app=app", "run", "--ignore-term"])
        self.wait_file(self.root / f"{web.pid}.ready")
        process = snapshot()[web.pid]
        send_signal(replace(process, started=str(int(process.started) - 1)), signal.SIGKILL)
        self.assertIsNone(web.poll())
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(stop(self.root, {"web"}, timeout=.2), 0)
        self.assertEqual(web.wait(timeout=2), -signal.SIGKILL)
        self.assertIn("正常退出超时", output.getvalue())


if __name__ == "__main__":
    unittest.main()
