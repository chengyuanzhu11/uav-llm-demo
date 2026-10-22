"""Offline language replies with production campus geometry and control."""
from dataclasses import replace
from math import atan2, pi
from threading import Event
import unittest
from unittest.mock import patch

from tests.test_semantic_updates import OfflineSemanticAgent, compact_campus, home, program, pump, skill
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import distance, norm
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.spatial_motion import spatial_catalog


def motion(path, target="stadium", heading=None, name="flight", speed=2.5):
    return dict(id=name, action="motion", target_id=target, when=None,
                motion=dict(path=path, speed_mps=speed, heading=heading))


def advance(service):
    if service._future is not None:
        try:
            service._future.result(timeout=15)
        except Exception:
            pass
    service.advance(.1)
    return service.state()


class SemanticMotionServiceTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch("uav_llm_demo.agents._post_deepseek",
                           side_effect=AssertionError("Remote API forbidden"))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def service(self, *replies, actual=False):
        scenario = replace(load_scenario("semantic_campus"), dynamic_objects=()) if actual else compact_campus()
        service = MissionService(scenario=scenario, agent=OfflineSemanticAgent(replies),
                                 planner_latency_seconds=0, traffic_seed=17)
        self.addCleanup(service.close)
        return service

    def test_actual_campus_two_clockwise_laps_then_returns_home(self):
        node = motion(dict(kind="boundary", direction="clockwise", laps=2),
                      heading=dict(mode="face_target", target_id="stadium"))
        service = self.service(program([node, home()]), actual=True)
        service.submit_task("去田径场顺时针飞两圈，始终面向场地中心，随后返航。", background=False)
        center = next(o["center_grid"] for o in spatial_catalog(service._environment.scenario)["objects"]
                      if o["id"] == "stadium")
        angles, progresses, speeds, heading_errors = [], [], [], []
        for _ in range(5000):
            state = advance(service)
            current = state["semantic_mission"].get("motion_progress")
            if current and current["phase"] == "tracking":
                p = service._vehicle.position
                angles.append(atan2(p[1]-center[1], p[0]-center[0]))
                progresses.append(current["distance_m"])
                speeds.append(norm(service._vehicle.velocity))
                desired = atan2(center[0]-p[0], -(center[1]-p[1]))
                heading_errors.append(abs((service._vehicle.yaw-desired+pi) % (2*pi)-pi))
            if state["status"] == "completed":
                break
        self.assertEqual(state["status"], "completed", service.semantic.message)
        self.assertGreater(len(angles), 100)
        winding = sum((b-a+pi) % (2*pi)-pi for a,b in zip(angles,angles[1:]))
        self.assertGreater(winding, 3.9*pi)
        self.assertLess(winding, 4.1*pi)
        self.assertEqual(progresses, sorted(progresses))
        self.assertLess(sum(v < .25 for v in speeds[30:-30]) / max(1,len(speeds)-60), .05)
        self.assertLess(max(heading_errors[50:]), .15)
        evidence = state["semantic_mission"]["evidence"]
        self.assertEqual([e["action"] for e in evidence], ["motion", "return_home"])
        self.assertTrue(evidence[0]["result"]["completed"])
        self.assertAlmostEqual(evidence[0]["result"]["distance_m"], evidence[0]["result"]["total_length_m"])
        self.assertTrue(state["milestones"]["safe_success"])
        self.assertLess(distance(service._vehicle.position, service._environment.base), .18)
        self.assertEqual(len(service.agent.inputs), 1)

    def test_curve_and_polyline_compose_height_and_independent_heading(self):
        curve = motion(dict(kind="curve", x="6*sin(t)", y="3*sin(2*t)", z="t",
                            t_start=0, t_end=2*pi), "library",
                       dict(mode="fixed", yaw_deg=90), name="eight", speed=1.5)
        line = motion(dict(kind="polyline", points_m=[[0,0,2],[3,0,2],[3,3,2],[0,3,2]]),
                      "library", dict(mode="fixed", yaw_deg=0), name="sideways", speed=1)
        service = self.service(program([curve, line, home()]))
        service.submit_task("飞一个逐渐升高的八字，再沿折线侧飞，最后返航。", background=False)
        top, headings = 0, []
        for _ in range(3500):
            state = advance(service)
            step = service.semantic.current_step
            if service._motion_phase == "tracking" and step and step["id"] == "eight":
                top = max(top, service._vehicle.position[2] * 5)
            if service._motion_phase == "tracking" and step and step["id"] == "sideways":
                headings.append(service._vehicle.yaw)
            if state["status"] == "completed":
                break
        self.assertEqual(state["status"], "completed", service.semantic.message)
        self.assertGreater(top, 17)
        self.assertTrue(headings)
        self.assertLess(max(abs(y) for y in headings[30:]), .05)
        self.assertEqual([e["action"] for e in service.semantic.evidence], ["motion","motion","return_home"])

    def test_pause_resume_preserves_path_progress_and_user_update_cancels_remainder(self):
        path = motion(dict(kind="curve", x="6*cos(t)", y="6*sin(t)", z=0,
                           t_start=0, t_end=4*pi), "library", speed=1.5)
        service = self.service(program([path,home()]), program([home()]))
        service.submit_task("绕行两圈。", background=False)
        pump(service, lambda s: (s["semantic_mission"].get("motion_progress") or {}).get("distance_m",0)>5)
        follower = service._motion_follower
        before = follower.progress_m
        service.stop_runtime()
        for _ in range(20):
            advance(service)
        self.assertAlmostEqual(follower.progress_m, before)
        service.start(background=False)
        pump(service, lambda s: service._motion_follower.progress_m > before+3)
        self.assertIs(service._motion_follower, follower)
        service.submit_task("取消剩余绕行，现在返航。", background=False)
        final = pump(service, lambda s:s["status"]=="completed")
        self.assertFalse(any(e["action"]=="motion" for e in service.semantic.evidence))
        self.assertEqual(final["semantic_mission"]["evidence"][-1]["action"], "return_home")

    def test_unsafe_motion_curve_rejected_before_takeoff(self):
        node = motion(dict(kind="polyline",points_m=[[0,0,0],[0,0,-20]]), "library")
        service = self.service(program([node,home()]))
        service.submit_task("执行超出飞行范围的路径。",background=False)
        for _ in range(4):
            state=advance(service)
        self.assertEqual(service._vehicle.position, service._environment.base)
        self.assertNotEqual(state["status"], "completed")
        self.assertEqual(service.semantic.evidence, [])


if __name__ == "__main__":
    unittest.main()
