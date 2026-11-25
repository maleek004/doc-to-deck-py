"""The plan and generate endpoints over HTTP, as the page calls them."""

import pytest
from fastapi.testclient import TestClient

from doctodeck.app import create_app, get_llm_registry
from doctodeck.llm.contracts import LlmErrorCode
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.llm.registry import LlmRegistry
from doctodeck.pipeline.models import INVALID_REQUEST
from tests.layer2.data import plan_request


def as_json(model) -> dict:
    return model.model_dump(mode="json", by_alias=True)  # what the page would send


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def client_with(scripted: FakeLlmClient) -> TestClient:
    """An app with one extra provider, "scripted", whose answers the test controls."""
    app = create_app()
    app.dependency_overrides[get_llm_registry] = lambda: LlmRegistry([FakeLlmClient(), scripted])
    return TestClient(app)


def test_providers_lists_the_fake(client):
    assert client.get("/api/providers").json() == [{"provider": "fake", "model": "fake-demo", "isReal": False}]


def test_plan_then_generate_works_end_to_end_on_the_fake(client):
    plan = client.post("/api/plan", json=as_json(plan_request()))
    assert plan.status_code == 200

    request = as_json(plan_request()) | {"cleanText": "Async basics\nAn async function returns a coroutine."}
    request |= {"modules": plan.json()["modules"]}
    generate = client.post("/api/generate", json=request)

    assert generate.status_code == 200
    body = generate.json()
    assert body["cards"]
    assert all(card["qualityScore"] for card in body["cards"])
    assert set(body["usage"]) == {"provider", "model", "promptTokens", "completionTokens", "latencyMs"}


def test_unknown_provider_is_a_400(client):
    response = client.post("/api/plan", json=as_json(plan_request(provider="nope")))

    assert response.status_code == 400
    assert response.json()["error"] == LlmErrorCode.UNKNOWN_PROVIDER


def test_empty_outline_is_a_400_invalid_request(client):
    response = client.post("/api/plan", json=as_json(plan_request()) | {"sections": []})

    assert response.status_code == 400
    assert response.json()["error"] == INVALID_REQUEST


def test_a_body_missing_fields_is_a_400_in_the_usual_error_shape(client):
    response = client.post("/api/plan", json={"provider": "fake"})

    assert response.status_code == 400
    assert response.json()["error"] == INVALID_REQUEST
    assert "sourceTitle" in response.json()["message"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (LlmErrorCode.RATE_LIMITED, 503),
        (LlmErrorCode.TIMEOUT, 504),
        (LlmErrorCode.REFUSED, 422),
        (LlmErrorCode.PROVIDER_ERROR, 502),
        (LlmErrorCode.INVALID_OUTPUT, 502),
        (LlmErrorCode.TRUNCATED, 502),
    ],
)
def test_provider_failures_map_to_the_right_status(code, expected):
    with client_with(FakeLlmClient("scripted").fails(code, "boom")) as client:
        response = client.post("/api/plan", json=as_json(plan_request(provider="scripted")))

    assert response.status_code == expected
    assert response.json() == {"error": code, "message": "boom"}
