"""Isolated offline browser acceptance; never touches the user's port 5000."""
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
import time
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from flask import jsonify
from werkzeug.serving import WSGIRequestHandler, make_server
from test_avoidance import encounter_service
from uav_llm_demo import create_app


class QuietHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        pass


def main():
    service = encounter_service()
    app = create_app(service=service)

    @app.post("/__test__/advance")
    def advance():
        for _ in range(10):
            service.advance(.05)
        return jsonify(service.state())

    @app.post("/__test__/provider_failure")
    def provider_failure():
        service.reset()
        service.configure_agent({"backend": "deepseek", "api_key": "dummy-test-credential"})
        service.plan(background=False)
        for _ in range(100):
            if service.state()["flight"]["error"]:
                break
            time.sleep(.001)
            service.advance(.05)
        return jsonify(service.state())

    server = make_server("127.0.0.1", 0, app, request_handler=QuietHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        env = dict(os.environ, UAV_DEMO_URL=f"http://127.0.0.1:{server.server_port}")
        with patch("uav_llm_demo.agents._post_deepseek",
                   return_value=Mock(status_code=401, text="dummy-test-credential")):
            return subprocess.run(["node", str(ROOT / "scripts/verify_avoidance.cjs")],
                                  cwd=ROOT, env=env, timeout=90).returncode
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
