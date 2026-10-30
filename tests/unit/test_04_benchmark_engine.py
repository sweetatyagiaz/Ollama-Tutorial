import json
from unittest.mock import Mock, patch

import pytest

from benchmarks.benchmark_models import benchmark_once, summarize


def make_result(category, status="success", **metrics):
    result = {
        "category": category,
        "status": status,
        "total_duration_ms": 1000,
        "load_duration_ms": 100,
        "prompt_eval_duration_ms": 200,
        "eval_duration_ms": 700,
        "tokens_per_second": 20,
        "prompt_tokens": 10,
        "generated_tokens": 20,
    }
    result.update(metrics)
    return result


def test_summarize_calculates_average_and_median():
    results = [
        make_result("coding", total_duration_ms=1000, tokens_per_second=20),
        make_result("coding", total_duration_ms=3000, tokens_per_second=30),
    ]

    summary = summarize(results)

    assert summary["successful_runs"] == 2
    assert summary["failed_runs"] == 0
    assert summary["average_total_duration_ms"] == 2000
    assert summary["median_total_duration_ms"] == 2000
    assert summary["average_tokens_per_second"] == 25
    assert summary["median_tokens_per_second"] == 25


def test_summarize_groups_results_by_category():
    results = [
        make_result("coding", total_duration_ms=1000),
        make_result("coding", total_duration_ms=3000),
        make_result("reasoning", total_duration_ms=5000),
    ]

    summary = summarize(results)

    assert summary["categories"]["coding"]["successful_runs"] == 2
    assert summary["categories"]["coding"]["average_total_duration_ms"] == 2000
    assert summary["categories"]["reasoning"]["successful_runs"] == 1
    assert summary["categories"]["reasoning"]["average_total_duration_ms"] == 5000


def test_summarize_counts_failed_runs():
    results = [
        make_result("coding"),
        make_result("coding", status="error"),
    ]

    summary = summarize(results)

    assert summary["successful_runs"] == 1
    assert summary["failed_runs"] == 1
    assert summary["categories"]["coding"]["failed_runs"] == 1


def test_summarize_handles_no_successful_runs():
    results = [make_result("coding", status="error")]

    summary = summarize(results)

    assert summary["successful_runs"] == 0
    assert summary["failed_runs"] == 1
    assert summary["average_total_duration_ms"] is None
    assert summary["median_total_duration_ms"] is None


@patch("benchmarks.benchmark_models.time.perf_counter", side_effect=[10.0, 12.0])
def test_benchmark_once_converts_ollama_metrics(mock_timer):
    session = Mock()
    response = Mock()
    response.json.return_value = {
        "message": {"content": "Hello"},
        "prompt_eval_count": 10,
        "eval_count": 20,
        "load_duration": 100000000,
        "prompt_eval_duration": 200000000,
        "eval_duration": 4000000000,
    }
    session.post.return_value = response

    result = benchmark_once(
        session=session,
        base_url="http://localhost:11434",
        model="llama3.2:latest",
        prompt="Hello",
        temperature=0.0,
        timeout=30,
    )

    assert result["response"] == "Hello"
    assert result["total_duration_ms"] == 2000
    assert result["load_duration_ms"] == 100
    assert result["prompt_eval_duration_ms"] == 200
    assert result["eval_duration_ms"] == 4000
    assert result["prompt_tokens"] == 10
    assert result["generated_tokens"] == 20
    assert result["tokens_per_second"] == 5
    session.post.assert_called_once()


def test_benchmark_once_rejects_invalid_message():
    session = Mock()
    response = Mock()
    response.json.return_value = {"message": {"content": None}}
    session.post.return_value = response

    with pytest.raises(ValueError, match="invalid message"):
        benchmark_once(
            session=session,
            base_url="http://localhost:11434",
            model="llama3.2:latest",
            prompt="Hello",
            temperature=0.0,
            timeout=30,
        )


def test_benchmark_once_handles_zero_generation_duration():
    session = Mock()
    response = Mock()
    response.json.return_value = {
        "message": {"content": "Hello"},
        "prompt_eval_count": 5,
        "eval_count": 10,
        "eval_duration": 0,
    }
    session.post.return_value = response

    with patch(
        "benchmarks.benchmark_models.time.perf_counter",
        side_effect=[10.0, 11.0],
    ):
        result = benchmark_once(
            session=session,
            base_url="http://localhost:11434",
            model="llama3.2:latest",
            prompt="Hello",
            temperature=0.0,
            timeout=30,
        )

    assert result["tokens_per_second"] == 0.0
