"""LLM provider abstraction for the NTM-BP project (stdlib-only).

Three roles are covered:

* ``LLMInterface``: the abstract contract any provider must implement.
* ``MockLLMClient``: deterministic responses for tests and CI (never
  scientific evidence -- use it only to test plumbing).
* ``DeepSeekClient``: a minimal OpenAI-compatible chat client built on
  :mod:`urllib` so the package stays dependency-free.  The API key is
  read from the ``DEEPSEEK_API_KEY`` environment variable (or passed to
  the constructor); the repository never hard-codes or commits keys.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMInterface(ABC):
    """Abstract interface for LLM providers used by the NTM system.

    Implementations: :class:`DeepSeekClient` (real API),
    :class:`MockLLMClient` (tests only).
    """

    @abstractmethod
    def evaluate_confidence(self, evidence: str, value: str) -> float:
        """Ask the LLM to assess the confidence of a fact given evidence.

        Args:
            evidence: The supporting or contradicting evidence text.
            value: The factual value to assess.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        ...

    @abstractmethod
    def extract_conditional_prob(
        self,
        premise_fact_id: str,
        premise_value: str,
        conclusion_fact_id: str,
        conclusion_value: str,
    ) -> float:
        """Ask the LLM for P(conclusion_value | premise_value).

        Args:
            premise_fact_id: The premise fact identifier.
            premise_value: The value of the premise fact.
            conclusion_fact_id: The conclusion fact identifier.
            conclusion_value: The value of the conclusion fact.

        Returns:
            Conditional probability in [0.0, 1.0].
        """
        ...

    @abstractmethod
    def extract_cpt(
        self,
        premise_fact_id: str,
        premise_values: List[str],
        conclusion_fact_id: str,
        conclusion_values: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Extract a full Conditional Probability Table for an edge.

        Args:
            premise_fact_id: The premise fact identifier.
            premise_values: All possible values of the premise fact.
            conclusion_fact_id: The conclusion fact identifier.
            conclusion_values: All possible values of the conclusion fact.

        Returns:
            CPT dict as {premise_value: {conclusion_value: probability, ...}, ...}.
        """
        ...

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Basic text completion.

        Args:
            prompt: The prompt text.

        Returns:
            The LLM's response text.
        """
        ...


class MockLLMClient(LLMInterface):
    """Mock LLM client for testing and development.

    Returns deterministic values without making any API calls.
    """

    def __init__(
        self,
        default_confidence: float = 0.8,
        default_conditional_prob: float = 0.7,
    ) -> None:
        self.default_confidence = default_confidence
        self.default_conditional_prob = default_conditional_prob

    def evaluate_confidence(self, evidence: str, value: str) -> float:
        return self.default_confidence

    def extract_conditional_prob(
        self,
        premise_fact_id: str,
        premise_value: str,
        conclusion_fact_id: str,
        conclusion_value: str,
    ) -> float:
        return self.default_conditional_prob

    def extract_cpt(
        self,
        premise_fact_id: str,
        premise_values: List[str],
        conclusion_fact_id: str,
        conclusion_values: List[str],
    ) -> Dict[str, Dict[str, float]]:
        cpt = {}
        for pv in premise_values:
            cpt[pv] = {}
            remaining = 1.0
            for i, cv in enumerate(conclusion_values):
                if i == len(conclusion_values) - 1:
                    cpt[pv][cv] = round(remaining, 2)
                else:
                    cpt[pv][cv] = round(self.default_conditional_prob / len(conclusion_values), 2)
                    remaining -= cpt[pv][cv]
        return cpt

    def complete(self, prompt: str) -> str:
        return f"Mock response to: {prompt[:50]}..."


class DeepSeekClient(LLMInterface):
    """Minimal OpenAI-compatible client for DeepSeek (stdlib urllib).

    Configuration (in priority order):
    1. Constructor arguments
    2. Environment variables (``DEEPSEEK_API_KEY``, ``DEEPSEEK_BASE_URL``,
       ``DEEPSEEK_MODEL``)
    3. Defaults

    No API key is ever embedded in this repository; the client raises
    ``ValueError`` when no key is available.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> None:
        """
        Args:
            api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
            base_url: API base URL. Falls back to DEEPSEEK_BASE_URL env var,
                then to https://api.deepseek.com.
            model: Model name. Falls back to DEEPSEEK_MODEL env var,
                then to "deepseek-chat".
            temperature: Generation temperature (default 0.1 for
                deterministic output).
            timeout: HTTP timeout in seconds.
        """
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        resolved_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", self.DEFAULT_BASE_URL)
        resolved_model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

        if not resolved_key:
            raise ValueError(
                "DeepSeek API key is required. Set the DEEPSEEK_API_KEY "
                "environment variable or pass api_key to the constructor. "
                "Keys are never hard-coded in this repository."
            )

        self._api_key = resolved_key
        self._base_url = resolved_url.rstrip("/")
        self._model = resolved_model
        self._temperature = temperature
        self._timeout = timeout
        # Cumulative usage counters (for cost accounting in experiments)
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ------------------------------------------------------------------
    # LLMInterface implementation
    # ------------------------------------------------------------------

    def evaluate_confidence(self, evidence: str, value: str) -> float:
        """Evaluate the confidence of a fact given evidence."""
        prompt = (
            f"已知证据：{evidence}\n"
            f"事实值：{value}\n\n"
            "请评估给定证据对该事实值的支持程度，输出一个 0.0 到 1.0 之间的置信度分数。\n"
            "只输出数字，不要输出其他内容。\n"
            "置信度："
        )
        response = self.complete(prompt)
        return self._parse_float(response, default=0.5)

    def extract_conditional_prob(
        self,
        premise_fact_id: str,
        premise_value: str,
        conclusion_fact_id: str,
        conclusion_value: str,
    ) -> float:
        """Extract P(conclusion_value | premise_value)."""
        prompt = (
            f"已知前提 [{premise_fact_id}] = {premise_value}\n"
            f"结论 [{conclusion_fact_id}] 的值是：{conclusion_value}\n\n"
            "请估计在给定前提的情况下，结论成立的条件概率 P(结论 | 前提)。\n"
            "只输出一个 0.0 到 1.0 之间的数字，不要输出其他内容。\n"
            "概率："
        )
        response = self.complete(prompt)
        return self._parse_float(response, default=0.5)

    def extract_cpt(
        self,
        premise_fact_id: str,
        premise_values: List[str],
        conclusion_fact_id: str,
        conclusion_values: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Extract a full CPT for an edge using the LLM."""
        prompt = (
            f"已知前提事实 [{premise_fact_id}] 的可能值：{premise_values}\n"
            f"结论事实 [{conclusion_fact_id}] 的可能值：{conclusion_values}\n\n"
            "请为每个前提值，估计结论取各个值的条件概率。\n"
            "输出格式为 JSON：\n"
            '{{\n'
            '  "<前提值>": {{"<结论值1>": 概率, "<结论值2>": 概率, ...}},\n'
            '  ...\n'
            '}}\n\n'
            '要求：每个前提值对应的概率之和为 1.0，保留 2 位小数。\n'
            "只输出 JSON，不要输出其他内容。\n"
            "CPT："
        )
        response = self.complete(prompt)
        try:
            cpt = json.loads(response)
            # Validate structure
            for pv in cpt:
                probs = cpt[pv]
                total = sum(probs.get(cv, 0.0) for cv in conclusion_values)
                if total > 0 and abs(total - 1.0) > 0.05:
                    # Renormalize
                    for cv in conclusion_values:
                        if cv in probs:
                            probs[cv] = round(probs[cv] / total, 2)
            return cpt
        except (json.JSONDecodeError, TypeError):
            # Fallback: uniform distribution
            return {
                pv: {cv: round(1.0 / len(conclusion_values), 2) for cv in conclusion_values}
                for pv in premise_values
            }

    def complete(self, prompt: str) -> str:
        """Send a completion request to the DeepSeek chat API."""
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": 256,
        }
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        self.total_calls += 1
        usage = body.get("usage") or {}
        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.total_completion_tokens += int(usage.get("completion_tokens", 0))

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"DeepSeek API returned no choices: {body}")
        return str(choices[0].get("message", {}).get("content", "")).strip()

    @property
    def usage_summary(self) -> Dict[str, int]:
        """Cumulative API usage since client creation."""
        return {
            "calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }

    @staticmethod
    def _parse_float(text: str, default: float = 0.5) -> float:
        """Parse a float from LLM text response."""
        try:
            # Look for explicit float patterns like "0.75", "1.0", "0.0"
            # Avoid matching numbers embedded in Chinese text
            match = re.search(r"(?:^|\s)(0\.\d+|1\.0|1\.00|0\.0|1\.00)(?:\s|$|[.,;])", text)
            if match:
                value = float(match.group(1))
                return max(0.0, min(1.0, value))
            # Fallback: try to find any float-like number
            match = re.search(r"0\.\d+|1\.0", text)
            if match:
                value = float(match.group())
                return max(0.0, min(1.0, value))
        except (ValueError, TypeError):
            pass
        return default
