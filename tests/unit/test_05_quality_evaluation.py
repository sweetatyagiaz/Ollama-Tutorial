import pytest

from benchmarks.evaluate_quality import (
    evaluate_coding_response,
    evaluate_python_code,
    extract_python_code,
)


def test_extract_python_code_from_markdown():
    response = "```python\ndef add(a, b):\n    return a + b\n```"
    code = extract_python_code(response)
    assert "def add(a, b):" in code


def test_extract_python_code_rejects_empty_response():
    with pytest.raises(ValueError, match="empty"):
        extract_python_code("")


def test_palindrome_evaluation():
    response = """
def is_palindrome(s):
    s = "".join(c for c in s if c.isalnum()).lower()
    return s == s[::-1]
"""
    result = evaluate_coding_response(
        response,
        "Write a Python function that checks whether a string is a palindrome."
    )
    assert result["passed"] == 4
    assert result["failed"] == 0
    assert result["score"] == 1.0


def test_duplicate_evaluation():
    response = """
def find_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)
"""
    result = evaluate_coding_response(
        response,
        "Write a Python function to find duplicate elements in a list."
    )
    assert result["passed"] == 4
    assert result["score"] == 1.0


def test_missing_function_returns_zero_score():
    result = evaluate_python_code(
        response="def another_function():\n    return True",
        function_name="is_palindrome",
        test_cases=[(("madam",), True)],
    )
    assert result["passed"] == 0
    assert result["score"] == 0.0
    assert "not found" in result["error"]


def test_invalid_python_returns_error():
    result = evaluate_python_code(
        response="def broken(:",
        function_name="broken",
        test_cases=[((), True)],
    )
    assert result["score"] == 0.0
    assert result["error"] is not None


def test_unsupported_coding_prompt():
    result = evaluate_coding_response(
        "print('hello')",
        "Write a Python sorting algorithm."
    )
    assert result["score"] is None
    assert result["total"] == 0


def test_execution_timeout():
    result = evaluate_python_code(
        response="def slow():\n    while True:\n        pass",
        function_name="slow",
        test_cases=[((), True)],
        timeout=1,
    )
    assert result["score"] == 0.0
    assert result["error"] == "Execution timed out."
