from __future__ import annotations

import os
from collections.abc import Mapping

from flask import Flask, jsonify, render_template, request

from .airsim_runtime import AirSimRuntime, AirSimRuntimeConfig
from .mission_service import MissionService
from .runtime import RuntimeCapabilityError, RuntimeUnavailableError
from .scenarios import list_scenarios

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
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(mission_service.set_wind(data))
        except RuntimeCapabilityError as exc:
            return _runtime_error_response(mission_service, exc, 409)
        except RuntimeUnavailableError as exc:
            return _runtime_error_response(mission_service, exc, 503)
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/config")
    def api_config():
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(mission_service.configure_agent(data))
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
    return jsonify(payload), status_code


def _build_runtime(
    backend: str,
    config: Mapping[str, object],
) -> AirSimRuntime:
    if backend != "airsim":
        raise ValueError(f"Unsupported external runtime '{backend}'.")
    return AirSimRuntime(
        AirSimRuntimeConfig(
            host=str(config.get("host") or os.getenv("AIRSIM_HOST", "127.0.0.1")),
            port=_bounded_int(
                config.get("port"),
                int(os.getenv("AIRSIM_PORT", "41451")),
                1,
                65535,
            ),
            vehicle_name=str(
                config.get("vehicle_name")
                or os.getenv("AIRSIM_VEHICLE_NAME", "Drone1")
            ),
            velocity_mps=_bounded_float(
                config.get("velocity_mps"),
                float(os.getenv("AIRSIM_VELOCITY_MPS", "4.0")),
                0.5,
                20.0,
            ),
            meters_per_cell=_bounded_float(
                config.get("meters_per_cell"),
                3.0,
                0.5,
                20.0,
            ),
            meters_per_altitude_level=_bounded_float(
                config.get("meters_per_altitude_level"),
                5.0,
                0.5,
                20.0,
            ),
        )
    )


def _bounded_float(
    value: object,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    number = default if value in (None, "") else float(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"Value {number} must be between {minimum} and {maximum}.")
    return number


def _bounded_int(
    value: object,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    number = default if value in (None, "") else int(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"Value {number} must be between {minimum} and {maximum}.")
    return number
