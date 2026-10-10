"""Read-only geometry audit against a running Flask/Blocks pair."""
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_llm_demo.airsim_runtime import import_airsim_client, grid_to_ned


def main():
    with urlopen("http://127.0.0.1:5000/api/state", timeout=5) as response:
        state = json.load(response)
    assert state["runtime_backend"] == "airsim", "Select AirSim first"
    api, _ = import_airsim_client()
    runtime = state["runtime"]
    client = api.MultirotorClient(ip=runtime["host"], port=runtime["port"], timeout_value=10)
    origin = runtime["scene_origin_ned"]
    objects = client.simListSceneObjects("UAVDemo_.*")
    expected = {"UAVDemo_" + item["name"] for item in state["scene_layout"]}
    assert set(objects) == expected, "Scene object set differs from shared layout"
    for item in state["scene_layout"]:
        name = "UAVDemo_" + item["name"]
        pose = client.simGetObjectPose(name)
        scale = client.simGetObjectScale(name)
        grid = tuple(item["position"][axis] for axis in "xyz")
        ned = grid_to_ned(grid)
        for i, axis in enumerate("xyz"):
            assert abs(getattr(pose.position, axis + "_val") - ned[i] - origin[axis]) < 0.001, name
        for axis, size_axis in zip("xyz", "yxz"):
            assert abs(getattr(scale, axis + "_val") - item["size"][size_axis]) < 0.001, name
    images = client.simGetImages([api.ImageRequest("0", api.ImageType.Scene, False, True)],
                                vehicle_name=runtime["vehicle_name"])
    if images and images[0].image_data_uint8:
        path = Path(__file__).resolve().parents[1] / "artifacts" / "airsim-shared-scene.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(bytes(images[0].image_data_uint8))
        print(f"Camera image: {path}")
    print(f"Verified {len(expected)} object positions and scales: {runtime['scene_id']}")


if __name__ == "__main__":
    main()
