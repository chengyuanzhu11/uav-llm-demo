"""Offline real-page/WebGL traffic-envelope acceptance on a temporary local port."""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask import abort, jsonify, request, send_file
from werkzeug.serving import WSGIRequestHandler, make_server
from uav_llm_demo import create_app
from uav_llm_demo.agents import DeepSeekAgent, MockLLMAgent
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import distance


class QuietHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/traffic-upgrade")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with patch.dict(os.environ), patch.object(DeepSeekAgent, "_request_json_content",
                                             side_effect=AssertionError("Provider access is forbidden")) as provider:
        for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
            os.environ.pop(name, None)
        service = MissionService(agent=MockLLMAgent(), scenario_id="urban_crossing",
                                 target_seed=9, traffic_seed=17, traffic_policy="fixed")
        server = worker = None
        try:
            initial = deepcopy(service.state())
            states = [initial]
            service.start(background=False)
            for tick in range(200):
                if service._future is not None:
                    service._future.result(timeout=30)
                service.advance(.05)
                if tick % 2 == 1:
                    states.append(deepcopy(service.state()))
            final = states[-1]
            assert service._thread is None, "Fixture generation must not start a service loop"
            assert not final["deepseek_key_configured"]
            assert final["diagnostics"]["usage"]["requests"] == 0
            assert len(final["dynamic_objects"]) == 8
            assert distance(service._vehicle.position, service._environment.base) > 5
            fixture = dict(states=states, provider_calls=provider.call_count,
                           observation="Actual offline Mock control samples; replay does not change navigation.")
            with gzip.open(output / "browser-fixture.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(fixture, stream)
            print(json.dumps(dict(stage="fixture_ready", simulated_seconds=service._time,
                                  packets=len(states), birds=8, provider_calls=provider.call_count)), flush=True)
            app = create_app(service=service)
            app.view_functions["api_state"] = lambda: jsonify(initial)
            requests = []

            @app.before_request
            def read_only_fixture():
                requests.append(dict(method=request.method, path=request.path))
                if request.method != "GET":
                    abort(405)

            @app.get("/__fixture")
            def captured_fixture():
                return jsonify(fixture)

            font = os.environ.get("UAV_TEST_FONT")
            if font:
                @app.get("/__font.ttf")
                def local_font():
                    return send_file(font, mimetype="font/ttf")

            server = make_server("127.0.0.1", 0, app, request_handler=QuietHandler)
            assert server.server_port != 5000
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            env = dict(os.environ, UAV_TRAFFIC_TEST_URL=f"http://127.0.0.1:{server.server_port}",
                       UAV_CAPTURE_DIR=str(output))
            result = subprocess.run(["node", str(ROOT / "scripts/verify_traffic_upgrade.cjs")],
                                    cwd=ROOT, env=env, timeout=240)
            assert all(item["method"] == "GET" for item in requests)
            assert provider.call_count == 0
            (output / "server-check.json").write_text(json.dumps(dict(
                port=server.server_port, read_only=True, provider_calls=0,
                service_thread_started=False, requests=requests, browser_exit_code=result.returncode), indent=2))
            return result.returncode
        finally:
            if server is not None:
                server.shutdown()
                worker.join(timeout=3)
                server.server_close()
            service.close()


if __name__ == "__main__":
    raise SystemExit(main())
