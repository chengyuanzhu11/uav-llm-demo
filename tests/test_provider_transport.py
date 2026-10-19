"""Exercise real local response reads without contacting the provider."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError, ReadTimeoutError

from uav_llm_demo import agents
from uav_llm_demo.provider_usage import LOGGER


@contextmanager
def local_provider(status, body=b"", *, hold_body=False):
    release_body = Event()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body) or 100))
            self.end_headers()
            self.wfile.flush()
            if hold_body:
                release_body.wait(3)
            elif body:
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    worker.start()
    try:
        with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
            yield f"http://127.0.0.1:{server.server_port}", calls, release_body
    finally:
        release_body.set()
        server.shutdown()
        server.server_close()
        worker.join(1)


class ProviderTransportTests(unittest.TestCase):
    def capture_response(self):
        original_post = requests.post
        responses = []

        def send(*args, **kwargs):
            response = original_post(*args, **kwargs)
            response.close = Mock(wraps=response.close)
            responses.append(response)
            return response

        return patch.object(agents.requests, "post", side_effect=send), responses

    def test_real_503_does_not_wait_for_body_and_closes_response(self):
        with local_provider(503, hold_body=True) as (url, calls, release), \
                patch.object(agents, "record_request") as usage:
            transport, responses = self.capture_response()
            with transport as post:
                agent = agents.DeepSeekAgent(api_key="offline-test-key", base_url=url, timeout_seconds=.1)
                with self.assertRaisesRegex(agents.PlannerProviderError, "HTTP 503.*overloaded") as caught:
                    agent._request_json_content([], "route plan")
            self.assertTrue(caught.exception.retryable)
            self.assertFalse(release.is_set())
            self.assertEqual(calls, ["/chat/completions"])
            self.assertEqual(post.call_count, 1)
            self.assertTrue(post.call_args.kwargs["stream"])
            responses[0].close.assert_called_once_with()
            self.assertTrue(responses[0].raw.closed)
            self.assertEqual(usage.call_count, 1)
            self.assertEqual(usage.call_args.kwargs["http_status"], 503)

    def test_real_success_body_timeout_is_sanitized_and_response_is_closed(self):
        with local_provider(200, hold_body=True) as (url, calls, _), \
                patch.object(agents, "record_request") as usage:
            transport, responses = self.capture_response()
            with transport:
                agent = agents.DeepSeekAgent(api_key="offline-test-key", base_url=url, timeout_seconds=.1)
                with self.assertRaisesRegex(agents.PlannerProviderError, r"network request failed \(timeout\)") as caught:
                    agent._request_json_content([], "semantic task plan")
            self.assertTrue(caught.exception.retryable)
            self.assertTrue(caught.exception.__suppress_context__)
            self.assertEqual(calls, ["/chat/completions"])
            responses[0].close.assert_called_once_with()
            self.assertTrue(responses[0].raw.closed)
            self.assertEqual(usage.call_count, 1)
            self.assertEqual(usage.call_args.kwargs["http_status"], 200)
            self.assertNotIn(url, str(caught.exception))

    def test_real_success_json_is_read_once_and_connection_is_closed(self):
        body = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()
        with local_provider(200, body) as (url, calls, _), \
                patch.object(agents, "record_request") as usage:
            transport, responses = self.capture_response()
            with transport:
                content = agents.DeepSeekAgent(api_key="offline-test-key", base_url=url)._request_json_content(
                    [], "route plan")
            self.assertEqual(content, "{}")
            self.assertEqual(calls, ["/chat/completions"])
            responses[0].close.assert_called_once_with()
            self.assertTrue(responses[0].raw.closed)
            self.assertEqual(usage.call_count, 1)
            self.assertEqual(usage.call_args.kwargs["http_status"], 200)

    def test_http_error_skips_json_and_never_exposes_body(self):
        response = Mock(status_code=503, text="private-test-key private-response")
        with patch.object(agents, "_post_deepseek", return_value=response), \
                self.assertLogs(LOGGER, level="INFO") as captured:
            with self.assertRaises(agents.PlannerProviderError) as caught:
                agents.DeepSeekAgent(api_key="private-test-key")._request_json_content([], "route plan")
        response.json.assert_not_called()
        response.close.assert_called_once_with()
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(json.loads(captured.records[0].getMessage())["http_status"], 503)
        for private in ("private-test-key", "private-response"):
            self.assertNotIn(private, str(caught.exception) + str(captured.output))

    def test_wrapped_read_timeout_is_recognized_without_rendering_private_text(self):
        marker = "private-transport-test-key"
        inner = ReadTimeoutError(None, marker, marker)
        response = Mock(status_code=200)
        response.json.side_effect = requests.ConnectionError(inner)
        with patch.object(agents, "_post_deepseek", return_value=response), \
                self.assertLogs(LOGGER, level="INFO") as captured:
            with self.assertRaisesRegex(agents.PlannerProviderError, r"\(timeout\)") as caught:
                agents.DeepSeekAgent(api_key=marker)._request_json_content([], "route plan")
        self.assertTrue(caught.exception.retryable)
        response.close.assert_called_once_with()
        self.assertEqual(len(captured.records), 1)
        self.assertNotIn(marker, str(caught.exception) + str(captured.output))

    def test_timeout_chain_handles_causes_context_reason_and_cycles(self):
        for link in ("__cause__", "__context__", "reason"):
            with self.subTest(link=link):
                outer = requests.ConnectionError("private diagnostic")
                inner = ReadTimeoutError(None, "private endpoint", "private message")
                setattr(outer, link, inner)
                inner.__context__ = outer
                self.assertEqual(agents._network_error_kind(outer), "timeout")
        cyclic = requests.ConnectionError("private diagnostic")
        cyclic.__context__ = cyclic
        self.assertEqual(agents._network_error_kind(cyclic), "connection")

    def test_refused_connection_is_not_misclassified_as_timeout(self):
        # urllib3 NewConnectionError inherits its ConnectTimeoutError class even
        # when the connection was refused immediately.
        refused = NewConnectionError(None, "private refused endpoint")
        wrapped = requests.ConnectionError(MaxRetryError(None, "private URL", refused))
        self.assertEqual(agents._network_error_kind(wrapped), "connection")

    def test_invalid_json_keeps_payload_classification_and_closes_response(self):
        response = Mock(status_code=200)
        response.json.side_effect = ValueError("private malformed response")
        with patch.object(agents, "_post_deepseek", return_value=response), \
                patch.object(agents, "record_request") as usage:
            with self.assertRaisesRegex(agents.PlannerProviderError, "invalid semantic task plan") as caught:
                agents.DeepSeekAgent(api_key="offline-test-key")._request_json_content([], "semantic task plan")
        response.close.assert_called_once_with()
        self.assertEqual(usage.call_count, 1)
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn("private malformed response", str(caught.exception))

    def test_windows_buffered_bridge_stays_closeable_on_success_and_error(self):
        for status in (200, 503):
            with self.subTest(status=status):
                agent = agents.DeepSeekAgent(api_key="offline-test-key")
                payload = {"status": status, "body": json.dumps({
                    "choices": [{"message": {"content": "{}"}}]})}
                result = Mock(returncode=0, stdout=json.dumps(payload), stderr="private diagnostic")
                with patch.object(agents, "os", SimpleNamespace(name="nt")), \
                        patch.object(agents.subprocess, "CREATE_NO_WINDOW", 0, create=True), \
                        patch.object(agents.subprocess, "run", return_value=result) as run, \
                        patch.object(agents, "record_request") as usage:
                    if status == 200:
                        self.assertEqual(agent._request_json_content([], "route plan"), "{}")
                    else:
                        with self.assertRaisesRegex(agents.PlannerProviderError, "HTTP 503"):
                            agent._request_json_content([], "route plan")
                self.assertEqual(run.call_count, 1)
                self.assertNotIn("offline-test-key", " ".join(run.call_args.args[0]))
                self.assertEqual(json.loads(run.call_args.kwargs["input"])["timeout"], 25)
                self.assertEqual(usage.call_count, 1)
                self.assertEqual(usage.call_args.kwargs["http_status"], status)


if __name__ == "__main__":
    unittest.main()
