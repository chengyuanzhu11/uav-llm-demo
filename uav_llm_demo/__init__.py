from __future__ import annotations

import base64
from collections.abc import Mapping

from flask import Flask, Response, jsonify, render_template, request

from .gazebo_runtime import GazeboRuntime, GazeboRuntimeConfig
from .mission_service import MissionService
from .runtime import RuntimeStateError, RuntimeUnavailableError
from .scenarios import list_scenarios
from .provider_config import ProviderConfigError

def create_app(
    *,
    service: MissionService | None = None,
    runtime_factory=None,
) -> Flask:
    app = Flask(__name__)
    mission_service = service or MissionService(
        runtime_factory=runtime_factory or _build_runtime
    )
    app.extensions["mission_service"] = mission_service
    if service is None:
        mission_service._ensure_loop()

    @app.errorhandler(RuntimeStateError)
    def runtime_state_conflict(exc):
        return jsonify(error=str(exc), error_code=exc.error_code), 409

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.get("/api/state")
    def api_state():
        return jsonify(mission_service.state())

    @app.get("/api/scenarios")
    def api_scenarios():
        return jsonify(scenarios=list_scenarios())

    @app.post("/api/onboard/frame")
    def api_onboard_frame():
        if request.content_length is None or request.content_length > 4_000_000:
            return jsonify(error="Camera packet exceeds the supported size."), 413
        try:
            return jsonify(mission_service.receive_camera_frame(request.get_json(silent=True)))
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/onboard/image")
    def api_onboard_image():
        with mission_service._lock:
            frame = mission_service.visual.frame
            if frame is None or request.args.get("frame_id") != frame["frame_id"]:
                return "", 404
            return Response(base64.b64decode(frame["rgb"].split(",", 1)[1]), mimetype="image/png",
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/mission/start")
    def api_start():
        try:
            return jsonify(mission_service.start())
        except RuntimeUnavailableError as exc:
            return _runtime_error_response(mission_service, exc, 503)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/mission/plan")
    def api_plan():
        try:
            return jsonify(mission_service.plan())
        except RuntimeUnavailableError as exc:
            return _runtime_error_response(mission_service, exc, 503)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/mission/task")
    def api_task():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"instruction"}:
            return jsonify(error="请提交包含 instruction 的任务指令。"), 400
        try:
            return jsonify(mission_service.submit_task(data["instruction"]))
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeUnavailableError as exc:
            return _runtime_error_response(mission_service, exc, 503)

    @app.post("/api/mission/goal")
    def api_goal():
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(mission_service.update_goal(data.get("target"), data.get("task")))
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/reset")
    def api_reset():
        try:
            return jsonify(mission_service.reset())
        except RuntimeUnavailableError as exc:
            return _runtime_error_response(mission_service, exc, 503)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/scenario")
    def api_scenario():
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(
                mission_service.switch_scenario(
                    str(data.get("scenario_id", "urban_crossing"))
                )
            )
        except RuntimeUnavailableError as exc:
            return _runtime_error_response(mission_service, exc, 503)
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/runtime")
    def api_runtime():
        data = request.get_json(silent=True) or {}
        config = data.get("config") or {}
        if not isinstance(config, Mapping):
            return jsonify(error="Runtime config must be an object."), 400
        try:
            return jsonify(
                mission_service.switch_runtime(
                    str(data.get("backend", "html")),
                    config,
                )
            )
        except RuntimeUnavailableError as exc:
            return _runtime_error_response(mission_service, exc, 503)
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/runtime/stop")
    def api_runtime_stop():
        try:
            return jsonify(mission_service.stop_runtime())
        except RuntimeUnavailableError as exc:
            return _runtime_error_response(mission_service, exc, 503)

    @app.post("/api/runtime/wind")
    def api_runtime_wind():
        # Published endpoint: reject old clients explicitly without changing a flight.
        return jsonify(error="本地动力学模式及风速接口已移除。物理实验请使用 Gazebo＋PX4，并在 Gazebo 场景中配置风场。",
                       error_code="runtime_removed"), 410

    @app.get("/api/config")
    def api_read_config():
        return jsonify(mission_service.provider_config())

    @app.post("/api/config/models")
    def api_models():
        try:
            return jsonify(mission_service.provider_models(request.get_json(silent=True)))
        except ProviderConfigError as exc:
            return jsonify(error=str(exc), error_code=exc.error_code), 400

    @app.post("/api/config")
    def api_config():
        data = request.get_json(silent=True)
        try:
            return jsonify(mission_service.configure_agent(data))
        except ProviderConfigError as exc:
            return jsonify(error=str(exc), error_code=exc.error_code), 400
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    return app


def _runtime_error_response(
    service: MissionService,
    exc: Exception,
    status_code: int,
):
    message = f"Runtime error: {exc}"
    payload = service.state(feedback=message)
    payload["error"] = message
    payload["error_code"] = "runtime_unavailable"
    return jsonify(payload), status_code


def _build_runtime(
    backend: str,
    config: Mapping[str, object],
) -> GazeboRuntime:
    if backend != "gazebo":
        raise ValueError(f"Unsupported external runtime '{backend}'.")
    return GazeboRuntime(GazeboRuntimeConfig(**dict(config)))
