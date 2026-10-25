"""Read-only Blocks RPC check. This script never arms, resets or flies a vehicle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_llm_demo import _airsim_defaults
from uav_llm_demo.airsim_runtime import AirSimRuntime, AirSimRuntimeConfig


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    defaults = _airsim_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=defaults["host"])
    parser.add_argument("--port", type=int, default=int(defaults["port"]))
    parser.add_argument("--vehicle", default=defaults["vehicle_name"])
    args = parser.parse_args()
    runtime = AirSimRuntime(AirSimRuntimeConfig(host=args.host, port=args.port, vehicle_name=args.vehicle))
    try:
        state = runtime.connect()
        details = runtime._worker.call(lambda: dict(
            vehicles=runtime.client.listVehicles(),
            cube_asset="Cube" in runtime.client.simListAssets(),
            api_control=runtime.client.isApiControlEnabled(vehicle_name=args.vehicle),
            settings=json.loads(runtime.client.getSettingsString()),
        ))
        print(json.dumps(dict(rpc="ok", mode="read-only", host=args.host, port=args.port,
                             vehicle=args.vehicle, vehicles=details["vehicles"],
                             cube_asset=details["cube_asset"], api_control=details["api_control"],
                             sim_mode=details["settings"].get("SimMode"),
                             vehicle_type=details["settings"].get("Vehicles", {}).get(args.vehicle, {}).get("VehicleType"),
                             position_ned=state["telemetry"]["position_ned"],
                             landed_state=state["telemetry"]["landed_state"]), indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
