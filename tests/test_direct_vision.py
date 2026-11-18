"""Protocol tests use a real local HTTP fixture, not a model-success substitute."""
from concurrent.futures import CancelledError
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch

from uav_llm_demo.agents import PlannerProviderError
from uav_llm_demo.direct_vision import DirectVisionAgent
from uav_llm_demo.provider_usage import request_trace


PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII="


def frame(identity="current"):
    return dict(frame_id=identity, rgb=PNG, depth_m=(4.0,), width=1, height=1,
                simulation_time=1, position_grid=[0, 0, 3], yaw_deg=0, pitch_deg=-45, vertical_fov_deg=75, cx=.5, cy=.5, fx=1, fy=1, rgb_sha256="a"*64, depth_sha256="b"*64)


def response_for(name="ground", arguments=None):
    arguments = arguments if arguments is not None else dict(
        frame_id="current", box=[100, 500, 400, 900], point=[250, 750], confidence=.9, reason="visible target")
    return dict(choices=[dict(finish_reason="tool_calls", message=dict(role="assistant",
        content=None, tool_calls=[dict(id="call_1", type="function", function=dict(
            name=name, arguments=json.dumps(arguments)))]))],
        usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120))


@contextmanager
def provider(body=None, status=200, *, hold_body=False, chunk_delay=0, targets=None):
    requests_seen, release = [], Event()
    encoded = json.dumps(body if body is not None else response_for()).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.loads(received)
            requests_seen.append(dict(path=self.path, body=payload,
                                      authorization=self.headers.get("Authorization")))
            outgoing = encoded
            if body is None and "tools" not in payload:
                report = dict(targets=targets if targets is not None else [dict(box=[100,500,400,900],point=[250,750],confidence=.95)])
                outgoing = json.dumps(dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(report)))])).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(outgoing)))
            self.end_headers()
            self.wfile.flush()
            if hold_body:
                release.wait(3)
            elif chunk_delay:
                try:
                    for byte in outgoing:
                        if release.wait(chunk_delay):
                            break
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # A cancelled/deadline-limited client closes the response.
            else:
                self.wfile.write(outgoing)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    worker.start()
    try:
        env = dict(VISION_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                   VISION_MODEL="Qwen/Qwen3-VL-4B-Instruct", NO_PROXY="127.0.0.1",
                   no_proxy="127.0.0.1")
        with patch.dict(os.environ, env, clear=True):
            yield requests_seen, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(1)


class DirectVisionTests(unittest.TestCase):
    def test_real_multimodal_single_tool_request_and_recording(self):
        events, usage, unused = [], [], Mock()
        constructor_observer = Mock()
        context = dict(visual_goal=dict(description="find the target"),
            feedback=dict(ok=False, code="occluded", measured=dict(
                frame_id="current", pixel=[250/999, 750/999], point_grid=[2.5, 3.0, .4])),
            history=[dict(tool="probe", feedback=dict(measured=dict(
                frame_id="previous", pixel=[.5, 1.0], point_grid=[4.0, 5.5, .8])))])
        original_context = deepcopy(context)
        with provider() as (seen, _), request_trace("run", "decision-1", usage.append):
            agent = DirectVisionAgent(dict(api_key="unrelated-language-secret",
                base_url="https://language.invalid/v1"), constructor_observer)
            decision = agent.decide(context, [frame("previous"), frame()],
                additional_request=unused, interaction_observer=events.append)
        self.assertEqual(decision, dict(tool="ground",
            arguments=dict(frame_id="current", pixel=[250/999, 750/999], confidence=.9),
            reason="visible target"))
        self.assertEqual(len(seen), 2)
        self.assertIsNone(seen[0]["authorization"])
        self.assertEqual(seen[0]["path"], "/v1/chat/completions")
        payload = seen[0]["body"]
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(len(payload["tools"]), 6)
        self.assertEqual({tool["function"]["name"] for tool in payload["tools"]},
                         {"look", "probe", "preview_move", "submit_move", "ground", "wait"})
        self.assertIs(payload["parallel_tool_calls"], False)
        self.assertFalse({"thinking", "enable_thinking", "response_format"} & payload.keys())
        content = payload["messages"][-1]["content"]
        self.assertEqual([x["image_url"]["url"] for x in content if x["type"] == "image_url"], [PNG])
        self.assertNotIn("depth_m", json.dumps(content))
        sent_context = json.loads(content[0]["text"])
        self.assertEqual(sent_context["current_frame_id"], "current")
        self.assertNotIn('"point":', json.dumps(sent_context))
        self.assertNotIn('"point_grid":', json.dumps(sent_context))
        self.assertNotIn("previous", json.dumps(sent_context))
        self.assertEqual(sent_context["feedback"]["code"], "occluded")
        self.assertEqual(sent_context["history"][0]["tool"], "probe")
        self.assertEqual(context, original_context)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1]["purpose"], "grounding_verification")
        self.assertEqual(events[0]["request"], payload)
        self.assertEqual(events[0]["outcome"], "ok")
        self.assertEqual(events[0]["response"]["choices"][0]["message"]["tool_calls"][0]["id"], "call_1")
        self.assertNotIn("headers", events[0])
        self.assertNotIn("unrelated-language-secret", json.dumps(events))
        self.assertEqual(usage[0]["image_count"], 1)
        self.assertEqual(usage[0]["total_tokens"], 120)
        unused.assert_called_once()
        constructor_observer.assert_not_called()

    def test_image_verification_rejects_absence_or_disagreement_and_reuses_only_same_goal_frame(self):
        for targets in ([], [dict(box=[100,100,400,400],point=[250,250],confidence=.95)]):
            with self.subTest(targets=targets), provider(targets=targets) as (seen,_):
                agent=DirectVisionAgent({})
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError,'does not support'):
                        agent.decide(dict(visual_goal=dict(description='target')),[frame()])
                self.assertEqual(len(seen),3)  # Two proposals, one immutable image check.
                with self.assertRaises(ValueError):
                    agent.decide(dict(visual_goal=dict(description='different target')),[frame()])
                self.assertEqual(len(seen),5)  # A different goal must be checked again.

    def test_uncertain_candidate_can_be_acquired_but_not_confirmed(self):
        target=dict(box=[100,500,400,900],point=[250,750],confidence=.5)
        with provider(targets=[target]):
            agent=DirectVisionAgent({})
            self.assertEqual(agent.decide(dict(visual_goal=dict(description='target')),[frame()])["tool"],"ground")
            with self.assertRaisesRegex(ValueError,'does not support'):
                agent.decide(dict(visual_goal=dict(description='target')),[dict(frame(),vertical_fov_deg=37.5)])

    def test_matching_model_points_on_background_are_rejected_without_snapping_the_action(self):
        # Both model calls select the same background pixel; agreement alone
        # must not count as foreground evidence. No object IDs or colour rules.
        f=dict(frame(), width=12, height=12, cx=6, cy=6, fx=12, fy=12,
               depth_m=[4. if 3 <= x <= 5 and 2 <= y <= 9 else 10.
                        for y in range(12) for x in range(12)])
        report=dict(box=[150,150,850,850],point=[750,750],confidence=.95)
        bad=dict(tool="ground",arguments=dict(frame_id="current",pixel=[750/999,750/999],confidence=.95),reason="candidate")
        good=deepcopy(bad);good["arguments"]["pixel"]=[375/999,500/999]
        with provider():
            agent=DirectVisionAgent({})
            with patch.object(agent,"_call",side_effect=[bad,[report],good]) as request:
                with self.assertRaisesRegex(ValueError,"behind the foreground"):
                    agent.decide(dict(visual_goal=dict(description="target")),[f])
                accepted=agent.decide(dict(visual_goal=dict(description="target")),[f])
                self.assertEqual(accepted,good)
                self.assertEqual(request.call_count,3)
        self.assertEqual(report["point"],[750,750])
        uniform=dict(f,depth_m=[4.]*144)
        self.assertGreaterEqual(agent._foreground_surface(uniform,report["box"])["depth_limit_m"],4.)
        # An unusable depth at the verifier's point does not rewrite a valid
        # policy point; measured foreground is only a cross-check reference.
        missing=deepcopy(f);missing["depth_m"][9*12+9]=0.
        with provider():
            agent=DirectVisionAgent({})
            with patch.object(agent,"_call",side_effect=[good,[report]]):
                self.assertEqual(agent.decide(dict(visual_goal=dict(description="target")),[missing]),good)

    def test_current_frame_and_strict_arguments_reject_unusable_decisions(self):
        cases = [
            response_for(arguments=dict(frame_id="current", box=[100,100,400,400], point=[500,500], confidence=.9, reason="outside target")),
            response_for(arguments=dict(frame_id="previous", box=[0,0,500,500], point=[1, 1], confidence=.9, reason="old")),
            response_for(arguments=dict(frame_id="current", box=[0,0,999,999], point=[1000, 1], confidence=.9, reason="outside")),
            response_for(arguments=dict(frame_id="current", box=[0,0,500,500], point=[True, 1], confidence=.9, reason="boolean")),
            response_for(arguments=dict(frame_id="current", box=[0,0,999,999], point=[250/999, 750/999], confidence=.9,
                                        reason="copied normalized feedback")),
            response_for(arguments=dict(target_index=0, confidence=.9, reason="two stage only")),
            response_for("preview_move", dict(offset_m=[12, 12, 0], reason="too far")),
            response_for("complete", dict(reason="invented completion")),
            response_for("wait", dict(yaw_deg=0, pitch_deg=-45, reason="wrong action arguments")),
        ]
        multiple = response_for()
        multiple["choices"][0]["message"]["tool_calls"] *= 2
        cases.append(multiple)
        wrong_function = response_for()
        wrong_function["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "select_visual_action"
        cases.append(wrong_function)
        duplicate = response_for("wait", {})
        duplicate["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            '{"reason":"one","reason":"two"}')
        cases.append(duplicate)
        for body in cases:
            with self.subTest(body=body), provider(body):
                recorded = []
                with self.assertRaises(ValueError):
                    DirectVisionAgent({}).decide({}, [frame()], interaction_observer=recorded.append)
                self.assertEqual(recorded[0]["outcome"], "error")

    def test_independent_remote_endpoint_requires_its_own_key(self):
        with patch.dict(os.environ, dict(VISION_BASE_URL="https://vision.invalid/v1",
                VISION_MODEL="image-model", DEEPSEEK_API_KEY="language-env-secret"), clear=True):
            agent = DirectVisionAgent(dict(base_url="https://language.invalid/v1", api_key="language-key"))
            self.assertIsNone(agent.api_key)
            with self.assertRaisesRegex(ValueError, "require their own key"):
                agent.decide({}, [frame()])

    def test_explicit_key_is_used_only_as_auth_and_observer_is_redacted(self):
        body = response_for()
        body["provider_debug"] = "fixture-private-key"
        with provider(body) as (seen, _), patch.dict(os.environ, VISION_API_KEY="fixture-private-key"):
            recorded = []
            with self.assertRaisesRegex(ValueError, "protected credential"):
                DirectVisionAgent({}).decide({}, [frame()], interaction_observer=recorded.append)
        self.assertEqual(seen[0]["authorization"], "Bearer fixture-private-key")
        self.assertNotIn("fixture-private-key", json.dumps(recorded))
        self.assertEqual(recorded[0]["response"]["provider_debug"], "[redacted]")

    def test_http_error_does_not_read_body_and_is_recorded(self):
        with provider(status=503, hold_body=True) as (seen, release):
            events = []
            agent = DirectVisionAgent({}, events.append)
            agent.timeout_seconds = .05
            with self.assertRaisesRegex(PlannerProviderError, "HTTP 503") as caught:
                agent.decide({}, [frame()])
            self.assertTrue(caught.exception.retryable)
            self.assertFalse(release.is_set())
        self.assertEqual(len(seen), 1)
        self.assertEqual(events[0]["http_status"], 503)
        self.assertIsNone(events[0]["response"])
        self.assertEqual(events[0]["outcome"], "error")

    def test_slow_response_cancellation_and_total_deadline(self):
        for stop in ("cancel", "deadline"):
            with self.subTest(stop=stop), provider(chunk_delay=.01):
                events = []
                agent = DirectVisionAgent({}, events.append)
                agent.timeout_seconds = .5 if stop == "cancel" else .08
                started = time.monotonic()
                cancelled = lambda: stop == "cancel" and time.monotonic()-started >= .04
                expected = CancelledError if stop == "cancel" else PlannerProviderError
                with self.assertRaises(expected):
                    agent.decide({}, [frame()], cancelled=cancelled)
                self.assertLess(time.monotonic()-started, .4)
                self.assertEqual(events[0]["outcome"], "cancelled" if stop == "cancel" else "error")
                self.assertIsNone(events[0]["response"])
                if stop == "deadline":
                    self.assertEqual(events[0]["error"], "network_timeout")

    def test_response_body_size_is_bounded(self):
        with provider(), patch("uav_llm_demo.direct_vision.MAX_RESPONSE_BYTES", 32):
            events = []
            with self.assertRaisesRegex(ValueError, "maximum body size"):
                DirectVisionAgent({}, events.append).decide({}, [frame()])
            self.assertEqual(events[0]["outcome"], "error")
            self.assertIsNone(events[0]["response"])

    def test_read_timeout_and_post_inference_cancellation_are_distinct(self):
        with provider(hold_body=True):
            events = []
            agent = DirectVisionAgent({}, events.append)
            agent.timeout_seconds = .05
            with self.assertRaisesRegex(PlannerProviderError, "timeout"):
                agent.decide({}, [frame()])
            self.assertEqual(events[0]["error"], "network_timeout")
        with provider() as (seen, _):
            events = []
            checks = iter([False, True])
            with self.assertRaises(CancelledError):
                DirectVisionAgent({}).decide({}, [frame()], cancelled=lambda: next(checks),
                                             interaction_observer=events.append)
            self.assertEqual(len(seen), 1)
            self.assertEqual(events[0]["outcome"], "cancelled")
        with provider() as (seen, _):
            with self.assertRaises(CancelledError):
                DirectVisionAgent({}).decide({}, [frame()], cancelled=lambda: True)
            self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
