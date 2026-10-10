"""Browser acceptance against an isolated Mock server, never the user's task."""
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from werkzeug.serving import WSGIRequestHandler, make_server
from uav_llm_demo import create_app
from uav_llm_demo.mission_service import MissionService


class QuietHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        pass


def main():
    service = MissionService(target_seed=11)
    server = make_server("127.0.0.1", 0, create_app(service=service), request_handler=QuietHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        env = dict(os.environ, UAV_DEMO_URL=f"http://127.0.0.1:{server.server_port}")
        return subprocess.run(["node", str(ROOT / "scripts/verify_random_targets.cjs")],
                              cwd=ROOT, env=env, timeout=90).returncode
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
