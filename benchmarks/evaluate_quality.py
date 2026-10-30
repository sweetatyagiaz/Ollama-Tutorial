"""Quality evaluation utilities for Ollama model responses."""

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent.parent


CODING_TEST_CASES = {
    "palindrome": {
        "function": "is_palindrome",
        "tests": [
            (("madam",), True),
            (("hello",), False),
            (("A man, a plan, a canal: Panama",), True),
            (("",), True),
        ],
    },
    "duplicates": {
        "function": "find_duplicates",
        "tests": [
            (([1, 2, 2, 3, 3, 3],), [2, 3]),
            (([1, 2, 3],), []),
            (([],), []),
            (([4, 4, 4],), [4]),
        ],
    },
}


def extract_python_code(response: str) -> str:
    """Extract Python code from a response, including Markdown fences."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Model response is empty.")

    lines = response.splitlines()
    code_blocks = []
    current_block = []
    inside_block = False

    for line in lines:
        if line.strip().startswith("```"):
            if inside_block:
                code_blocks.append("\n".join(current_block))
                current_block = []
                inside_block = False
            else:
                inside_block = True
            continue

        if inside_block:
            current_block.append(line)

    if code_blocks:
        return max(code_blocks, key=len).strip()

    return response.strip()


def evaluate_python_code(
    response: str,
    function_name: str,
    test_cases: list[tuple[tuple[Any, ...], Any]],
    timeout: int = 5,
) -> dict[str, Any]:
    """Evaluate generated Python code against predefined test cases."""
    try:
        code = extract_python_code(response)
        tree = ast.parse(code)

        functions = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if function_name not in functions:
            return {
                "passed": 0,
                "failed": len(test_cases),
                "total": len(test_cases),
                "score": 0.0,
                "error": f"Function {function_name!r} not found.",
            }

        test_script = (
            code
            + "\nimport json\n"
            + f"function = {function_name}\n"
            + f"test_cases = {test_cases!r}\n"
            + "results = []\n"
            + "for args, expected in test_cases:\n"
            + "    try:\n"
            + "        actual = function(*args)\n"
            + "        results.append(actual == expected)\n"
            + "    except Exception:\n"
            + "        results.append(False)\n"
            + "print(json.dumps(results))\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "generated_test.py"
            script_path.write_text(test_script)

            process = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir,
                check=False,
            )

        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or "Generated code failed.")

        results = json.loads(process.stdout.strip().splitlines()[-1])
        passed = sum(results)
        total = len(test_cases)

        return {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "score": round(passed / total, 4) if total else 0.0,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": 0,
            "failed": len(test_cases),
            "total": len(test_cases),
            "score": 0.0,
            "error": "Execution timed out.",
        }
    except Exception as exc:
        return {
            "passed": 0,
            "failed": len(test_cases),
            "total": len(test_cases),
            "score": 0.0,
            "error": str(exc),
        }

def evaluate_coding_response(response: str, prompt: str) -> dict[str, Any]:
    """Select coding test cases based on the benchmark prompt."""
    prompt_lower = prompt.lower()

    if "palindrome" in prompt_lower:
        test = CODING_TEST_CASES["palindrome"]
    elif "duplicate" in prompt_lower:
        test = CODING_TEST_CASES["duplicates"]
    else:
        return {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "score": None,
            "error": "No coding evaluation test is configured for this prompt.",
        }

    return evaluate_python_code(
        response=response,
        function_name=test["function"],
        test_cases=test["tests"],
    )

def run_quality_evaluation() -> dict[str, Any]:
    """Evaluate saved coding benchmark responses and write a quality report."""
    input_path = ROOT / "benchmarks" / "results" / "model_benchmarks.json"
    output_path = ROOT / "benchmarks" / "results" / "quality_results.json"

    benchmark_data = json.loads(input_path.read_text())
    evaluation_results = []

    for item in benchmark_data["results"]:
        if item.get("category") != "coding":
            continue

        if item.get("status") != "success":
            continue

        evaluation = evaluate_coding_response(
            response=item["response"],
            prompt=item["prompt"],
        )

        evaluation_results.append({
            "model": item["model"],
            "category": item["category"],
            "prompt": item["prompt"],
            "repetition": item["repetition"],
            "evaluation": evaluation,
        })

    successful_scores = [
        item["evaluation"]["score"]
        for item in evaluation_results
        if item["evaluation"]["score"] is not None
    ]

    report = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluated_responses": len(evaluation_results),
        "average_quality_score": (
            round(sum(successful_scores) / len(successful_scores), 4)
            if successful_scores else None
        ),
        "results": evaluation_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({
        "evaluated_responses": report["evaluated_responses"],
        "average_quality_score": report["average_quality_score"],
        "output_file": str(output_path),
    }, indent=2))

    return report


if __name__ == "__main__":
    run_quality_evaluation()

