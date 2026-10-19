"""Capture a complete offline Mock flight for browser motion regression checks."""

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_llm_demo import create_app
from uav_llm_demo.agents import DeepSeekAgent, MockLLMAgent
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.scenarios import list_scenarios


def capture(output):
    # This fixture must neither read a user's key nor contact a provider.
    with patch.dict(os.environ), patch.object(
        DeepSeekAgent, "_request_json_content", side_effect=AssertionError("Provider calls are forbidden")
    ):
        os.environ.pop("DEEPSEEK_API_KEY", None)
        service = MissionService(agent=MockLLMAgent(), target_seed=9, traffic_seed=17)
        try:
            initial = service.state()
            samples = [initial["motion_samples"][-1]]
            html = create_app(service=service).test_client().get("/").get_data(as_text=True)
            service.start(background=False)
            for _ in range(3600):
                service.advance(0.05)
                samples.append(service._snapshot["motion_samples"][-1])
                if service._future is not None:
                    time.sleep(0.001)
                if service._snapshot["milestones"]["task_completed"]:
                    break
            if not service.state()["milestones"]["task_completed"]:
                raise RuntimeError("Offline flight did not complete within 180 simulated seconds")
            for _ in range(20):
                service.advance(0.05)
                samples.append(service._snapshot["motion_samples"][-1])
            final = service.state()
            assert not final["deepseek_key_configured"]
            assert final["diagnostics"]["usage"]["requests"] == 0
            assert service._thread is None
            fixture = dict(initial=initial, final=final, samples=samples, html=html,
                           catalog={"scenarios": list_scenarios()})
            output.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(output, "wt", encoding="utf-8") as handle:
                json.dump(fixture, handle)
            print(json.dumps(dict(completed=True, simulation_seconds=service._time,
                                  samples=len(samples), path_points=len(final["continuous_path"]),
                                  provider_calls=0, fixture=str(output))))
        finally:
            service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/follow-camera/flight-fixture.json.gz")
    capture(parser.parse_args().output)
