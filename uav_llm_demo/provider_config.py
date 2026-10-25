"""In-memory provider profiles and credential-safe model discovery."""
import ipaddress
import re
import time
from urllib.parse import urlsplit, urlunsplit

import requests


PROVIDERS = (
    dict(id="deepseek", label="DeepSeek 官方", default_base_url="https://api.deepseek.com"),
    dict(id="bailian", label="阿里云百炼（北京）", default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    dict(id="bailian-intl", label="阿里云百炼（新加坡）", default_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    dict(id="custom", label="自定义兼容接口", default_base_url=""),
)
_PRESETS = {item["id"]: item for item in PROVIDERS}
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,159}\Z")


class ProviderConfigError(ValueError):
    def __init__(self, error_code, message):
        self.error_code = error_code
        super().__init__(message)


def text_field(data, name):
    value = data.get(name)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProviderConfigError("configuration_invalid", "配置字段必须为文本。")
    value = value.strip()
    if len(value) > 4096 or any(ord(char) < 32 for char in value):
        raise ProviderConfigError("configuration_invalid", "配置字段格式无效。")
    return value


def normalize_endpoint(value):
    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname
        parsed.port  # Reject invalid ports before storing the endpoint.
        if (not host or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
                or parsed.scheme not in {"https", "http"}):
            raise ValueError
        if parsed.scheme == "http":
            try:
                local = ipaddress.ip_address(host).is_loopback
            except ValueError:
                local = host == "localhost"
            if not local:
                raise ValueError
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            path = path[:-len("/chat/completions")]
        return urlunsplit((parsed.scheme, parsed.netloc.lower(), path, "", ""))
    except (ValueError, AttributeError):
        raise ProviderConfigError("configuration_invalid", "接口地址需为 HTTPS 基础地址，不能包含账号、查询参数或片段。") from None


def identify_provider(base_url):
    host = urlsplit(base_url).hostname or ""
    if host == "api.deepseek.com":
        return "deepseek"
    if host == "dashscope.aliyuncs.com" or host.endswith(".cn-beijing.maas.aliyuncs.com"):
        return "bailian"
    if host == "dashscope-intl.aliyuncs.com" or host.endswith(".ap-southeast-1.maas.aliyuncs.com"):
        return "bailian-intl"
    return "custom"


class ProviderProfiles:
    """Keys are scoped to a provider and exact endpoint, never sent to the UI."""
    def __init__(self, initial):
        self.profiles = {item["id"]: dict(provider=item["id"], base_url=item["default_base_url"],
                                        api_key=None, model="") for item in PROVIDERS}
        self.revision = 0
        endpoint = normalize_endpoint(initial.get("base_url") or _PRESETS["deepseek"]["default_base_url"])
        key = initial.get("api_key")
        if key and (key in endpoint or key in (initial.get("model") or "")):
            raise ProviderConfigError("configuration_invalid", "密钥只能填写在 API Key 字段中。")
        self.active = identify_provider(endpoint)
        self.profiles[self.active].update(initial, provider=self.active, base_url=endpoint)

    def prepare(self, data):
        explicit = text_field(data, "provider")
        endpoint = text_field(data, "base_url")
        provider = explicit or (identify_provider(normalize_endpoint(endpoint)) if endpoint else self.active)
        if provider not in self.profiles:
            raise ProviderConfigError("configuration_invalid", "请选择有效的 API 服务商。")
        stored = self.profiles[provider]
        endpoint = normalize_endpoint(endpoint or stored["base_url"])
        if provider != "custom" and identify_provider(endpoint) != provider:
            raise ProviderConfigError("configuration_invalid", "接口地址与所选服务商不一致，请调整服务商或高级地址。")
        same_endpoint = endpoint == stored["base_url"]
        key = text_field(data, "api_key") or (stored.get("api_key") if same_endpoint else None)
        if not key:
            raise ProviderConfigError("missing_key", "请填写该服务商及接口对应的 API Key。")
        if key in endpoint:
            raise ProviderConfigError("configuration_invalid", "密钥只能填写在 API Key 字段中。")
        model = text_field(data, "model") or (stored.get("model") if same_endpoint else "") or ""
        if model and (not _MODEL_ID.fullmatch(model) or key in model):
            raise ProviderConfigError("configuration_invalid", "模型名称格式无效，请从模型列表选择。")
        return dict(provider=provider, base_url=endpoint, api_key=key, model=model)

    def commit(self, candidate):
        self.profiles[candidate["provider"]] = dict(candidate)
        self.active = candidate["provider"]
        self.revision += 1

    def public(self, backend):
        return dict(backend=backend, active_provider=self.active, providers=[dict(item) for item in PROVIDERS],
                    profiles=[dict(id=item["id"], label=item["label"], base_url=self.profiles[item["id"]]["base_url"],
                                   model=self.profiles[item["id"]].get("model") or "",
                                   key_configured=bool(self.profiles[item["id"]].get("api_key"))) for item in PROVIDERS])


def _get_json(url, key, params, timeout):
    response = None
    try:
        response = requests.get(url, headers={"Authorization": "Bearer " + key}, params=params,
                                timeout=timeout, allow_redirects=False)
        status = response.status_code
        if status in (401, 403):
            raise ProviderConfigError("authentication_failed", "认证或权限校验失败，请检查服务商、地域及 API Key。")
        if status == 404:
            raise ProviderConfigError("endpoint_unreachable", "模型列表接口不存在，请检查服务商或高级接口地址。")
        if status == 429:
            raise ProviderConfigError("models_unavailable", "服务商暂时限流，请稍后重新读取模型。")
        if status != 200:
            raise ProviderConfigError("endpoint_unreachable", "服务商接口暂时不可用，请稍后重试。")
        return response.json()
    except (requests.RequestException, OSError):
        raise ProviderConfigError("endpoint_unreachable", "无法连接服务商，请检查网络或代理后重新读取模型。") from None
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ProviderConfigError):
            raise
        raise ProviderConfigError("models_unavailable", "服务商未返回有效模型列表。") from None
    finally:
        if response is not None:
            response.close()


def discover_models(candidate):
    """Read metadata only; discovery does not prove inference permissions or quality."""
    endpoint, key = candidate["base_url"], candidate["api_key"]
    bailian = identify_provider(endpoint) in {"bailian", "bailian-intl"}
    deadline = time.monotonic() + 30
    models, received = set(), 0
    for page in range(1, 11):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderConfigError("endpoint_unreachable", "读取模型列表超时，请重试。")
        if bailian:
            parsed = urlsplit(endpoint)
            url = urlunsplit((parsed.scheme, parsed.netloc, "/api/v1/models", "", ""))
            params = dict(providers="deepseek", capabilities="TG", features="structured-outputs",
                          page_no=page, page_size=200, language="zh-CN")
        else:
            url, params = endpoint + "/models", None
        data = _get_json(url, key, params, min(10, remaining))
        if not isinstance(data, dict):
            raise ProviderConfigError("models_unavailable", "服务商模型列表格式无效。")
        if bailian:
            output = data.get("output")
            if data.get("success") is not True or not isinstance(output, dict):
                raise ProviderConfigError("models_unavailable", "百炼未能返回模型目录，请检查 Key 的地域和权限。")
            rows, total = output.get("models"), output.get("total")
            if type(total) is not int or total < 0 or total > 2000:
                raise ProviderConfigError("models_unavailable", "模型列表分页信息无效。")
        else:
            rows, total = data.get("data"), 0
        if not isinstance(rows, list) or len(rows) > 2000:
            raise ProviderConfigError("models_unavailable", "服务商模型列表格式无效。")
        for row in rows:
            if not isinstance(row, dict):
                continue
            model = row.get("model" if bailian else "id")
            if not isinstance(model, str) or not _MODEL_ID.fullmatch(model) or key in model:
                continue
            if bailian:
                features = row.get("features")
                if (row.get("provider") != "deepseek" or not isinstance(features, list)
                        or "structured-outputs" not in features
                        or row.get("inference_provider") not in (None, "aliyun-bailian", "alibaba-cloud-modelstudio")):
                    continue
            models.add(model)
        received += len(rows)
        if not bailian or received >= total:
            break
        if not rows or page == 10:
            raise ProviderConfigError("models_unavailable", "模型列表不完整，请重新读取。")
    if not models:
        raise ProviderConfigError("models_unavailable", "没有找到适用模型，请检查地域、权限或高级接口地址。")
    ordered = sorted(models, key=lambda value: ("flash" not in value.lower(), "chat" not in value.lower(), value))
    current = candidate.get("model") or ""
    return dict(provider=candidate["provider"], base_url=endpoint, models=ordered,
                recommended_model=current if current in models else ordered[0], current_model=current)
