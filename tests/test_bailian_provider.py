"""Regression checks for DeepSeek over Alibaba Cloud Model Studio."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from uav_llm_demo import agents


class BailianProviderTests(unittest.TestCase):
    def test_bailian_json_requests_disable_thinking(self):
        endpoints = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope.aliyuncs.com:443/compatible-mode/v1",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "https://test-space.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/",
            "https://test-space.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        )
        messages = [{"role": "user", "content": "Return JSON."}]
        for endpoint in endpoints:
            for kind, limit in (("route", 2000), ("semantic", 12000)):
                with self.subTest(endpoint=endpoint, kind=kind):
                    response = Mock(status_code=200)
                    response.json.return_value = {
                        "choices": [{"message": {"content": '{"ok": true}'}}]
                    }
                    agent = agents.DeepSeekAgent(
                        api_key="test-key", model="deepseek-v4.1-flash-1017",
                        base_url=endpoint,
                    )
                    with patch.object(agents, "_post_deepseek", return_value=response) as post:
                        result = agent._request_json_content(
                            messages, kind, max_tokens=limit,
                        )
                    self.assertEqual(json.loads(result), {"ok": True})
                    self.assertEqual(post.call_args.args[0], endpoint.rstrip("/") + "/chat/completions")
                    body = post.call_args.kwargs["json"]
                    self.assertIs(body["enable_thinking"], False)
                    self.assertNotIn("thinking", body)
                    self.assertEqual(body["model"], "deepseek-v4.1-flash-1017")
                    self.assertEqual(body["messages"], messages)
                    self.assertEqual(body["response_format"], {"type": "json_object"})
                    self.assertEqual(body["max_tokens"], limit)
                    self.assertIs(body["stream"], False)
                    response.close.assert_called_once_with()

    def test_other_hosts_keep_existing_deepseek_parameters(self):
        for endpoint in (
            "https://api.deepseek.com",
            "https://proxy.example/v1",
            "https://dashscope.aliyuncs.com.example/v1",
            "https://example.test/v1?upstream=dashscope.aliyuncs.com",
        ):
            with self.subTest(endpoint=endpoint):
                response = Mock(status_code=200)
                response.json.return_value = {
                    "choices": [{"message": {"content": '{"ok": true}'}}]
                }
                with patch.object(agents, "_post_deepseek", return_value=response) as post:
                    agents.DeepSeekAgent(
                        api_key="test-key", base_url=endpoint,
                    )._request_json_content([], "route")
                body = post.call_args.kwargs["json"]
                self.assertEqual(body["thinking"], {"type": "disabled"})
                self.assertNotIn("enable_thinking", body)

    def test_windows_bridge_receives_bailian_parameters(self):
        reply = {
            "status": 200,
            "body": json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}),
        }
        process = Mock(returncode=0, stdout=json.dumps(reply), stderr="")
        with (
            patch.object(agents, "os", SimpleNamespace(name="nt")),
            patch.object(agents.subprocess, "CREATE_NO_WINDOW", 0, create=True),
            patch.object(agents.subprocess, "run", return_value=process) as run,
        ):
            agent = agents.DeepSeekAgent(
                api_key="test-key", model="deepseek-v4.1-flash-1017",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self.assertEqual(json.loads(agent._request_json_content([], "route")), {"ok": True})
        payload = json.loads(run.call_args.kwargs["input"])
        body = json.loads(payload["body"])
        self.assertIs(body["enable_thinking"], False)
        self.assertNotIn("thinking", body)


if __name__ == "__main__":
    unittest.main()
