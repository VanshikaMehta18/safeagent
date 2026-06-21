"""Tests for FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.schemas import FinalResponse, Verdict
from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_query_returns_final_response(client):
    mock_final = FinalResponse(
        content="Test response content.",
        safety_verdict=Verdict.PASS.value,
        was_rewritten=False,
        referral_suggestion=None,
        safety_note="Passed all checks.",
    )
    mock_state = {
        "query_id": "test-uuid",
        "final_response": mock_final,
    }

    with patch("api.routes.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.return_value = mock_state
        response = client.post("/query", json={"query": "What is compound interest?"})

    assert response.status_code == 200
    data = response.json()
    assert data["query_id"] == "test-uuid"
    assert "latency_ms" in data
    assert data["response"]["content"] == "Test response content."
    assert data["response"]["safety_verdict"] == "PASS"


def test_safety_report_structure(client):
    mock_report = {
        "total_queries": 10,
        "pass_count": 7,
        "warn_count": 2,
        "block_count": 1,
        "most_flagged_principles": [{"principle": "non_maleficence", "count": 3}],
        "average_safety_score": 0.82,
        "average_latency_ms": 1200.0,
    }

    with patch("api.routes.aggregate_report", return_value=mock_report):
        response = client.get("/safety-report")

    assert response.status_code == 200
    data = response.json()
    assert data["total_queries"] == 10
    assert data["pass_count"] == 7
    assert "most_flagged_principles" in data


def test_safety_log_endpoint(client):
    with patch("api.routes.read_log_entries", return_value=[]):
        response = client.get("/safety-log")
    assert response.status_code == 200
    assert response.json() == []
