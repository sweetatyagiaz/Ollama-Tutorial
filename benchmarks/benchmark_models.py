"""Benchmark Ollama models for latency, throughput, and resource usage."""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "benchmarks" / "benchmark_config.json"


def load_config() -> dict[str, Any]:
    """Load and validate benchmark configuration."""
    config = json.loads(CONFIG_PATH.read_text())
    required = {"base_url", "models", "repetitions", "temperature", "timeout", "output_file"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if not config["models"] or config["repetitions"] < 1:
        raise ValueError("At least one model and one repetition are required.")
    return config


def load_prompts() -> list[dict[str, str]]:
    """Load prompts from the individual category files."""
    prompts = []
    prompt_dir = ROOT / "benchmarks" / "prompts"

    for path in sorted(prompt_dir.glob("*.json")):
        content = json.loads(path.read_text())
        category = content.get("category", path.stem)

        for prompt in content.get("prompts", []):
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Invalid prompt in {path}")
            prompts.append({"category": category, "prompt": prompt})

    if not prompts:
        raise ValueError("No benchmark prompts were found.")

    return prompts


def benchmark_once(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    """Execute one inference request and collect Ollama metrics."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }

    started = time.perf_counter()

    response = session.post(
        f"{base_url.rstrip(chr(39) + chr(47) + chr(39))}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    elapsed = time.perf_counter() - started
    data = response.json()
    message = data.get("message", {}).get("content")

    if not isinstance(message, str):
        raise ValueError("Ollama returned an invalid message.")

    prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
    generated_tokens = int(data.get("eval_count", 0) or 0)

    load_ns = int(data.get("load_duration", 0) or 0)
    prompt_ns = int(data.get("prompt_eval_duration", 0) or 0)
    eval_ns = int(data.get("eval_duration", 0) or 0)

    return {
        "response": message,
        "total_duration_ms": round(elapsed * 1000, 2),
        "load_duration_ms": round(load_ns / 1_000_000, 2),
        "prompt_eval_duration_ms": round(prompt_ns / 1_000_000, 2),
        "eval_duration_ms": round(eval_ns / 1_000_000, 2),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(
            generated_tokens / (eval_ns / 1_000_000_000), 2
        ) if eval_ns > 0 else 0.0,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate overall and category-wise benchmark statistics."""
    def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [item for item in records if item["status"] == "success"]

        metric_names = [
            "total_duration_ms",
            "load_duration_ms",
            "prompt_eval_duration_ms",
            "eval_duration_ms",
            "tokens_per_second",
            "prompt_tokens",
            "generated_tokens",
        ]

        summary = {
            "successful_runs": len(successful),
            "failed_runs": len(records) - len(successful),
        }

        for metric in metric_names:
            values = [item[metric] for item in successful]
            summary[f"average_{metric}"] = (
                round(statistics.mean(values), 2) if values else None
            )
            summary[f"median_{metric}"] = (
                round(statistics.median(values), 2) if values else None
            )

        return summary

    overall = calculate_metrics(results)
    categories = sorted({item["category"] for item in results})

    overall["categories"] = {
        category: calculate_metrics(
            [item for item in results if item["category"] == category]
        )
        for category in categories
    }

    return overall


def run_benchmarks() -> dict[str, Any]:
    """Run all configured prompts against all configured models."""
    config = load_config()
    prompts = load_prompts()
    all_results = []
    summaries = {}

    with requests.Session() as session:
        for model in config["models"]:
            model_results = []
            print(f"\\nBenchmarking model: {model}")

            for prompt_data in prompts:
                for repetition in range(1, config["repetitions"] + 1):
                    print(
                        f"  {prompt_data['category']} "
                        f"run {repetition}/{config['repetitions']}",
                        end=" ... ",
                        flush=True,
                    )

                    record = {
                        "model": model,
                        "category": prompt_data["category"],
                        "prompt": prompt_data["prompt"],
                        "repetition": repetition,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    try:
                        metrics = benchmark_once(
                            session=session,
                            base_url=config["base_url"],
                            model=model,
                            prompt=prompt_data["prompt"],
                            temperature=config["temperature"],
                            timeout=config["timeout"],
                        )
                        record.update(metrics)
                        record["status"] = "success"
                        print(f"{metrics['tokens_per_second']} tokens/sec")
                    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                        record.update({"status": "error", "error": str(exc)})
                        print(f"FAILED: {exc}")

                    model_results.append(record)
                    all_results.append(record)

            summaries[model] = summarize(model_results)

    output = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "prompt_count": len(prompts),
        "results": all_results,
        "summaries": summaries,
    }

    output_path = ROOT / config["output_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    print(f"\\nResults saved to: {output_path}")
    print(json.dumps(summaries, indent=2))

    return output


if __name__ == "__main__":
    run_benchmarks()
