"""Small, explicit geometry for controller tests; never part of the map registry."""
from uav_llm_demo.scenarios import DynamicObjectSpec, ScenarioDefinition


def empty_airspace():
    return ScenarioDefinition(
        scenario_id="test_airspace", name="Controller test airspace", grid_size=20,
        min_altitude=2, max_altitude=10, base=(1, 18, 0), target=(18, 2, 5),
        buildings=(), dynamic_objects=(
            DynamicObjectSpec("bird-1", "bird", (9.5, 10.5, 5.0), (.35, -.2, 0)),), seed=11)
