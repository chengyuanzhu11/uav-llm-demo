"""Provider configuration is editable, isolated, and verified before activation.

All provider traffic in this module is mocked. These tests never contact the
running Flask server or load a real key from the environment.
"""
from copy import deepcopy
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch

import requests

from uav_llm_demo import create_app
from uav_llm_demo.agents import DeepSeekAgent, MockLLMAgent
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.scenarios import load_scenario


OFFICIAL = "https://api.deepseek.com"
BAILIAN = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BAILIAN_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
MODEL = "deepseek-v4-flash"
BAILIAN_MODEL = "deepseek-v4.1-flash-1017"


def http_response(body, status=200):
    response = Mock(status_code=status, headers={})
    response.json.return_value = body
    response.text = "untrusted provider response"
    return response


def official_models(*models):
    return http_response({"data": [{"id": name} for name in models or (MODEL,)]})


def bailian_models(*models, total=None, page_no=1):
    names = models or (BAILIAN_MODEL,)
    return http_response({"success": True, "output": {
        "total": len(names) if total is None else total,
        "page_no": page_no, "page_size": 200,
        "models": [{"model": name, "provider": "deepseek", "capabilities": ["TG"],
                    "features": ["structured-outputs"]} for name in names],
    }})


class ProviderConfigTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {
            "DEEPSEEK_API_KEY": "", "DEEPSEEK_BASE_URL": OFFICIAL,
            "DEEPSEEK_MODEL": MODEL,
        })
        env.start()
        self.addCleanup(env.stop)
        network = patch("uav_llm_demo.provider_config.requests.get")
        self.get = network.start()
        self.get.return_value = official_models()
        self.addCleanup(network.stop)
        paid = patch("uav_llm_demo.agents._post_deepseek",
                     side_effect=AssertionError("Configuration must not create paid completions."))
        self.completions = paid.start()
        self.addCleanup(paid.stop)
        self.service = MissionService(agent=MockLLMAgent(), scenario=load_scenario("urban_crossing"))
        self.addCleanup(self.service.close)
        self.app = create_app(service=self.service)
        self.app.testing = True
        self.client = self.app.test_client()

    def save(self, *, provider="deepseek", api_key="official-test-secret", model=MODEL,
             base_url=None, verify=True):
        payload = dict(backend="deepseek", provider=provider, model=model, verify=verify)
        if api_key is not None:
            payload["api_key"] = api_key
        if base_url is not None:
            payload["base_url"] = base_url
        return self.client.post("/api/config", json=payload)

    def config(self):
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_successful_configuration_can_be_replaced_without_restart(self):
        first = self.save()
        self.assertEqual(first.status_code, 200)
        old_agent = self.service.agent
        run_id = self.service.trace.run_id
        position = self.service.state()["uav"]
        second = self.save(api_key="replacement-test-secret")
        self.assertEqual(second.status_code, 200)
        self.assertIsNot(self.service.agent, old_agent)
        self.assertEqual(self.service.agent.api_key, "replacement-test-secret")
        self.assertEqual(self.service.trace.run_id, run_id)
        self.assertEqual(second.get_json()["uav"], position)
        self.assertEqual(second.get_json()["agent_backend"], "deepseek")
        self.assertNotIn("replacement-test-secret", str(second.get_json()) + str(self.config()))
        self.assertEqual(self.get.call_count, 2)
        self.completions.assert_not_called()

    def test_invalid_initial_configuration_can_be_corrected_in_same_process(self):
        self.get.return_value = http_response({"error": {"message": "bad model"}}, 401)
        bad = self.save(api_key="wrong-test-secret")
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.service.agent.backend_name, "mock")
        self.assertFalse(any(item["key_configured"] for item in self.config()["profiles"]))
        self.get.return_value = official_models()
        good = self.save(api_key="valid-test-secret")
        self.assertEqual(good.status_code, 200)
        self.assertEqual(self.service.agent.api_key, "valid-test-secret")

    def test_key_is_reused_only_for_same_provider_and_endpoint(self):
        self.assertEqual(self.save().status_code, 200)
        self.get.reset_mock()
        reused = self.save(api_key="")
        self.assertEqual(reused.status_code, 200)
        self.assertEqual(self.service.agent.api_key, "official-test-secret")
        self.assertEqual(self.get.call_args.kwargs["headers"]["Authorization"],
                         "Bearer official-test-secret")
        self.get.reset_mock()
        cross_provider = self.save(provider="bailian", api_key=None, model=BAILIAN_MODEL)
        changed_endpoint = self.save(provider="custom", api_key=None,
                                     base_url="https://another.example/v1")
        self.assertEqual(cross_provider.status_code, 400)
        self.assertEqual(changed_endpoint.status_code, 400)
        self.get.assert_not_called()
        self.assertEqual(self.service.agent.api_key, "official-test-secret")

    def test_official_and_bailian_saved_profiles_keep_independent_keys_and_models(self):
        self.assertEqual(self.save().status_code, 200)
        self.get.return_value = bailian_models()
        selected = self.save(provider="bailian", api_key="bailian-test-secret", model=BAILIAN_MODEL)
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(self.service.agent.api_key, "bailian-test-secret")
        self.assertEqual(self.service.agent.base_url, BAILIAN)
        self.assertEqual(self.service.agent.model, BAILIAN_MODEL)
        self.get.return_value = official_models()
        self.assertEqual(self.save(api_key=None).status_code, 200)
        self.assertEqual(self.service.agent.api_key, "official-test-secret")
        self.get.return_value = bailian_models()
        self.assertEqual(self.save(provider="bailian", api_key=None,
                                   model=BAILIAN_MODEL).status_code, 200)
        self.assertEqual(self.service.agent.api_key, "bailian-test-secret")
        config = self.config()
        profiles = {item["id"]: item for item in config["profiles"]}
        self.assertTrue(profiles["deepseek"]["key_configured"])
        self.assertTrue(profiles["bailian"]["key_configured"])
        self.assertEqual(config["active_provider"], "bailian")
        for secret in ("official-test-secret", "bailian-test-secret"):
            self.assertNotIn(secret, str(config))

    def test_custom_endpoint_change_does_not_reuse_old_endpoint_key(self):
        first = self.save(provider="custom", base_url="https://first.example/v1")
        self.assertEqual(first.status_code, 200)
        self.get.reset_mock()
        changed = self.save(provider="custom", base_url="https://second.example/v1", api_key="")
        self.assertEqual(changed.status_code, 400)
        self.get.assert_not_called()
        self.assertEqual(self.service.agent.base_url, "https://first.example/v1")

    def test_model_discovery_does_not_save_key_or_change_active_agent(self):
        before = deepcopy(self.config())
        old_agent = self.service.agent
        self.get.return_value = official_models(MODEL, "deepseek-v4-pro")
        discovered = self.client.post("/api/config/models", json={
            "provider": "deepseek", "api_key": "discovery-only-secret"})
        self.assertEqual(discovered.status_code, 200)
        body = discovered.get_json()
        self.assertEqual(body["provider"], "deepseek")
        self.assertEqual(body["base_url"], OFFICIAL)
        self.assertEqual(set(body["models"]), {MODEL, "deepseek-v4-pro"})
        self.assertIn(body["recommended_model"], body["models"])
        self.assertIn("current_model", body)
        self.assertEqual(self.config(), before)
        self.assertIs(self.service.agent, old_agent)
        self.assertNotIn("discovery-only-secret", str(body))
        self.assertEqual(self.get.call_args.args[0], OFFICIAL + "/models")
        self.assertIs(self.get.call_args.kwargs["allow_redirects"], False)
        self.assertIn("timeout", self.get.call_args.kwargs)

    def test_bailian_model_discovery_filters_non_deepseek_and_reads_all_pages(self):
        first = bailian_models(BAILIAN_MODEL, total=3)
        first.json.return_value["output"]["models"].append({
            "model": "qwen-test", "provider": "qwen", "capabilities": ["TG"],
            "features": ["structured-outputs"]})
        second = bailian_models("deepseek-v4-pro", total=3, page_no=2)
        self.get.side_effect = [first, second]
        response = self.client.post("/api/config/models", json={
            "provider": "bailian", "api_key": "discovery-only-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.get_json()["models"]), {BAILIAN_MODEL, "deepseek-v4-pro"})
        self.assertEqual(self.get.call_count, 2)
        for call in self.get.call_args_list:
            self.assertEqual(call.args[0], "https://dashscope.aliyuncs.com/api/v1/models")
            self.assertIs(call.kwargs["allow_redirects"], False)
        self.assertEqual(self.get.call_args_list[1].kwargs["params"]["page_no"], 2)

    def test_bailian_international_profile_uses_its_own_discovery_host(self):
        self.get.return_value = bailian_models()
        response = self.client.post("/api/config/models", json={
            "provider": "bailian-intl", "api_key": "intl-test-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["base_url"], BAILIAN_INTL)
        self.assertEqual(self.get.call_args.args[0],
                         "https://dashscope-intl.aliyuncs.com/api/v1/models")

    def test_catalog_pagination_is_bounded_and_never_saves_partial_results(self):
        self.get.side_effect = [bailian_models(f"deepseek-page-{i}", total=2000, page_no=i)
                                for i in range(1, 12)]
        before = self.config()
        response = self.client.post("/api/config/models", json={
            "provider": "bailian", "api_key": "discovery-only-secret"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.get.call_count, 10)
        self.assertEqual(self.config(), before)

    def test_malformed_catalogs_fail_safely_without_activating_profile(self):
        cases = [
            ("deepseek", http_response([])),
            ("deepseek", http_response({"data": None})),
            ("deepseek", http_response({"data": []})),
            ("bailian", http_response({"success": False, "output": {}})),
            ("bailian", http_response({"success": True, "output": {"models": [], "total": True}})),
        ]
        malformed_features = bailian_models()
        malformed_features.json.return_value["output"]["models"][0]["features"] = 42
        cases.append(("bailian", malformed_features))
        for provider, response in cases:
            with self.subTest(provider=provider, data=response.json.return_value):
                self.get.return_value = response
                result = self.client.post("/api/config/models", json={
                    "provider": provider, "api_key": "test-secret"})
                self.assertEqual(result.status_code, 400)
                self.assertTrue(result.get_json()["error"])
                self.assertEqual(self.service.agent.backend_name, "mock")
                self.assertFalse(any(item["key_configured"] for item in self.config()["profiles"]))
                response.close.assert_called()

    def test_catalog_rejects_credential_echoes_as_model_ids(self):
        self.get.return_value = official_models(MODEL, "prefix-discovery-secret-suffix")
        result = self.client.post("/api/config/models", json={
            "provider": "deepseek", "api_key": "discovery-secret"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.get_json()["models"], [MODEL])
        self.assertNotIn("discovery-secret", str(result.get_json()))

    def test_nonexistent_selected_model_is_rejected_before_configuration_mutation(self):
        self.assertEqual(self.save().status_code, 200)
        before = self.config()
        old_agent = self.service.agent
        generation = self.service._generation
        response = self.save(api_key="new-secret", model="deepseek-misspelled")
        self.assertEqual(response.status_code, 400)
        self.assertIs(self.service.agent, old_agent)
        self.assertEqual(self.service._generation, generation)
        self.assertEqual(self.config(), before)

    def test_reconfiguration_preserves_paused_campus_task_on_failure_and_success(self):
        service = MissionService(scenario=load_scenario("semantic_campus"),
                                 agent=DeepSeekAgent(api_key="old-test-secret"))
        self.addCleanup(service.close)
        service.semantic.instruction = "巡检图书馆后返航"
        service.semantic.status = "interpreting"
        service._plan_requested = False
        service._publish_state()
        client = create_app(service=service).test_client()
        before = service.state()
        old_agent = service.agent
        generation = service._generation
        self.get.return_value = http_response({"error": "invalid credential"}, 401)
        response = client.post("/api/config", json={
            "backend": "deepseek", "provider": "deepseek", "api_key": "bad-test-secret",
            "model": MODEL, "verify": True})
        self.assertEqual(response.status_code, 400)
        self.assertIs(service.agent, old_agent)
        self.assertEqual(service.state()["semantic_mission"], before["semantic_mission"])
        self.assertEqual(service.state()["uav"], before["uav"])
        self.assertEqual(service._generation, generation)
        self.assertFalse(service._plan_requested)
        self.get.return_value = official_models()
        response = client.post("/api/config", json={
            "backend": "deepseek", "provider": "deepseek", "api_key": "new-test-secret",
            "model": MODEL, "verify": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.agent.api_key, "new-test-secret")
        self.assertEqual(service.semantic.instruction, "巡检图书馆后返航")
        self.assertFalse(service._plan_requested)
        self.completions.assert_not_called()

    def test_provider_failures_and_network_errors_do_not_echo_untrusted_text_or_secrets(self):
        failures = [http_response({"error": {"message": "leaked-secret response-body-token"}}, 401),
                    http_response({"error": "leaked-secret response-body-token"}, 503),
                    requests.exceptions.Timeout("Bearer leaked-secret response-body-token"),
                    requests.exceptions.ConnectionError("https://leaked-secret@host.invalid")]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.get.side_effect = failure if isinstance(failure, Exception) else None
                self.get.return_value = failure
                result = self.save(api_key="leaked-secret")
                self.assertEqual(result.status_code, 400)
                payload = result.get_json()
                self.assertTrue(payload["error"])
                self.assertNotIn("leaked-secret", str(payload))
                self.assertNotIn("response-body-token", str(payload))
                self.assertEqual(self.service.agent.backend_name, "mock")

    def test_redirects_are_rejected_without_following_or_credential_forwarding(self):
        response = http_response({}, 302)
        response.headers = {"Location": "https://different.example/models"}
        self.get.return_value = response
        rejected = self.save()
        self.assertEqual(rejected.status_code, 400)
        self.get.assert_called_once()
        self.assertIs(self.get.call_args.kwargs["allow_redirects"], False)
        self.assertEqual(self.service.agent.backend_name, "mock")

    def test_invalid_endpoint_forms_are_rejected_before_sending_credentials(self):
        endpoints = ("http://public.example/v1", "https://user:pass@example.com/v1",
                     "https://example.com/v1?api_key=secret", "https://example.com/v1#fragment",
                     "ftp://example.com/v1", "//example.com/v1", "https://", "https://@example.com/v1",
                     "http://127.0.0.1.example.com/v1")
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                self.get.reset_mock()
                result = self.save(provider="custom", base_url=endpoint)
                self.assertEqual(result.status_code, 400)
                self.get.assert_not_called()

    def test_local_http_endpoint_is_accepted_for_local_model_servers(self):
        for endpoint in ("http://127.0.0.1:8000/v1", "http://localhost:8000/v1", "http://[::1]:8000/v1"):
            with self.subTest(endpoint=endpoint):
                result = self.save(provider="custom", base_url=endpoint)
                self.assertEqual(result.status_code, 200)
                self.assertEqual(self.service.agent.base_url, endpoint)

    def test_key_cannot_be_embedded_in_public_endpoint_or_model_fields(self):
        for fields in ({"base_url": "https://example.com/accidental-secret/v1"},
                       {"base_url": "https://example.com/v1", "model": "accidental-secret"}):
            with self.subTest(field=next(reversed(fields))):
                result = self.save(provider="custom", api_key="accidental-secret", **fields)
                self.assertEqual(result.status_code, 400)
                self.assertNotIn("accidental-secret", str(result.get_json()))
                self.get.assert_not_called()
                self.assertEqual(self.service.agent.backend_name, "mock")

    def test_unknown_provider_and_non_object_payloads_fail_without_network(self):
        for payload in ([], "text", {"backend": "deepseek", "provider": "unknown",
                                      "api_key": "test-secret", "model": MODEL, "verify": True}):
            with self.subTest(payload=payload):
                response = self.client.post("/api/config", json=payload)
                self.assertEqual(response.status_code, 400)
        self.get.assert_not_called()

    def test_legacy_configuration_remains_offline_and_reuses_key(self):
        first = self.client.post("/api/config", json={
            "backend": "deepseek", "api_key": "legacy-test-secret", "model": MODEL})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(self.client.post("/api/config", json={"backend": "mock"}).status_code, 200)
        reused = self.client.post("/api/config", json={"backend": "deepseek"})
        self.assertEqual(reused.status_code, 200)
        self.assertEqual(self.service.agent.api_key, "legacy-test-secret")
        self.get.assert_not_called()

    def test_network_wait_releases_flight_lock_and_cannot_overwrite_newer_configuration(self):
        entered, release = Event(), Event()
        results, errors = [], []

        def network(*args, **kwargs):
            if kwargs["headers"]["Authorization"] == "Bearer slow-secret":
                entered.set()
                if not release.wait(3):
                    raise AssertionError("Test did not release older verification.")
            return official_models()

        def save_older():
            try:
                with self.app.test_client() as client:
                    results.append(client.post("/api/config", json={
                        "backend": "deepseek", "provider": "deepseek", "api_key": "slow-secret",
                        "model": MODEL, "verify": True}))
            except Exception as exc:
                errors.append(exc)

        self.get.side_effect = network
        thread = Thread(target=save_older)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            acquired = self.service._lock.acquire(timeout=.5)
            try:
                self.assertTrue(acquired, "Provider network I/O held the flight control lock.")
                self.assertEqual(self.service.agent.backend_name, "mock")
            finally:
                if acquired:
                    self.service._lock.release()
            newest = self.save(api_key="newest-secret")
            self.assertEqual(newest.status_code, 200)
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertIn(results[0].status_code, (400, 409))
        self.assertEqual(self.service.agent.api_key, "newest-secret")
        self.assertNotIn("slow-secret", str(results[0].get_json()))


if __name__ == "__main__":
    unittest.main()
