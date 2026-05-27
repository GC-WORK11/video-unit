"""EnsembleVLMJudgeEvaluator - Multi-model ensemble evaluation using voting/consensus."""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ._base import Evaluator
from ._context import EvaluationContext
from ._registry import register_evaluator
from ._result import EvaluationResult

logger = logging.getLogger(__name__)


class VoteDecision(Enum):
    """Voting decision from a single VLM model."""

    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    ERROR = "error"


class DisagreementLevel(Enum):
    """Level of agreement among ensemble members."""

    UNANIMOUS_PASS = "unanimous_pass"
    UNANIMOUS_FAIL = "unanimous_fail"
    CONSENSUS_PASS = "consensus_pass"
    CONSENSUS_FAIL = "consensus_fail"
    FULL_AGREEMENT = "full_agreement"
    DISAGREEMENT = "disagreement"
    MODEL_ERRORS = "model_errors"


@dataclass
class VLMMemberResult:
    """Result from a single VLM model in the ensemble."""

    model_name: str
    provider: str
    vote: VoteDecision
    answer: str
    confidence: float
    reasoning: str
    latency_ms: float
    error: str | None = None


@dataclass
class EnsembleScore:
    """Aggregated ensemble evaluation result."""

    ensemble_score: float
    vote_summary: dict
    disagreement_level: DisagreementLevel
    agreement_ratio: float
    confidence_estimate: float
    member_results: list[VLMMemberResult] = field(default_factory=list)


# Provider configuration for VLM models
VLM_PROVIDER_CONFIG = {
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
    },
    "claude-3-5-sonnet": {
        "provider": "openrouter",
        "model": "anthropic/claude-3.5-sonnet",
    },
    "claude-3-opus": {
        "provider": "openrouter",
        "model": "anthropic/claude-3-opus",
    },
    "gemini-1-5-pro": {
        "provider": "gemini",
        "model": "gemini-1.5-pro",
    },
    "gemini-1-5-flash": {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
    },
    "gemma-2-9b-it": {
        "provider": "ollama",
        "model": "gemma2:9b",
        "base_url": "http://localhost:11434/v1",
    },
    "llama-3-1-8b-instruct": {
        "provider": "ollama",
        "model": "llama3.1:8b",
        "base_url": "http://localhost:11434/v1",
    },
}

# Fallback defaults
DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@register_evaluator
class EnsembleVLMJudgeEvaluator(Evaluator):
    """Ensemble VLM evaluator using multi-model voting/consensus.

    This evaluator queries multiple VLM models in parallel and aggregates
    their responses using configurable strategies (majority, unanimous,
    confidence_weighted). It provides robust semantic evaluation with
    disagreement detection.

    Contract schema:
        vlm_ensemble:
          - question: "Is the ball continuously visible?"
            expected_answer: "yes"
            models:
              - gpt-4o
              - claude-3-5-sonnet
            strategy: "majority"  # majority, unanimous, confidence_weighted
            disagreement_threshold: 0.5  # flag for human review if below this

    Supports:
        - OpenAI models (GPT-4o, GPT-4o-mini)
        - OpenRouter models (Claude, Gemini via OpenRouter)
        - Google Gemini models
        - Local models via Ollama (Gemma, Llama)
    """

    name = "vlm_ensemble"
    required_inputs = ["frames"]

    VLM_PROMPT_TEMPLATE = """You are evaluating a video. Answer the following question about the video frames provided.

Question: {question}

Examine the frames carefully and provide your answer. Respond with ONLY a JSON object in this exact format:
{{"answer": "your answer here", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Be honest about uncertainty. If you cannot determine the answer from the frames, set confidence below 0.5 and answer "uncertain"."""

    DEFAULT_MODELS = ["gpt-4o", "claude-3-5-sonnet", "gemini-1-5-pro"]

    def __init__(self):
        self.default_strategy = "majority"
        self.default_disagreement_threshold = 0.5
        self.default_timeout = 60.0  # seconds per model
        self.max_frames_per_model = 8

    async def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run the ensemble VLM evaluation.

        Args:
            context: Evaluation context with frames and contract.

        Returns:
            EvaluationResult with ensemble evaluation results.
        """
        missing = self.get_missing_inputs(context)
        if missing:
            result = EvaluationResult(passed=False, score=0.0)
            result.add_failure(
                timestamp=context.frame_to_timestamp(0),
                frame_number=0,
                failure_type="perception_unavailable",
                severity="warning",
                message=f"Evaluator '{self.name}' could not run: missing inputs {missing}. "
                        "Perception pipeline may not have produced required outputs.",
                object=None,
                suggested_fix="Check that perception pipeline ran successfully and produced tracks/masks/depth.",
            )
            return result
        self.validate_inputs(context)

        result = EvaluationResult(passed=True, score=100.0)
        vlm_ensemble_checks = context.contract.get("vlm_ensemble", [])

        if not vlm_ensemble_checks:
            logger.info("No vlm_ensemble checks specified in contract")
            return result

        total_frames = context.video_metadata.get("total_frames", 0)
        if total_frames == 0:
            logger.warning("Cannot determine total frames for VLM ensemble check")
            result.add_failure(
                timestamp=context.frame_to_timestamp(0),
                frame_number=0,
                failure_type="no_video_frames",
                severity="warning",
                message="No frames available for VLM ensemble evaluation",
            )
            return result

        for check_spec in vlm_ensemble_checks:
            ensemble_result = await self._evaluate_check(check_spec, context, total_frames)

            if ensemble_result is None:
                continue

            # Determine pass/fail based on ensemble score
            check_passed = self._determine_pass_fail(ensemble_result, check_spec)

            if not check_passed:
                severity = self._severity_from_disagreement(ensemble_result.disagreement_level)
                expected = check_spec.get("expected_answer", "")
                result.add_failure(
                    timestamp=context.frame_to_timestamp(0),
                    frame_number=0,
                    failure_type="vlm_ensemble_mismatch",
                    severity=severity,
                    message=f"Ensemble VLM evaluation failed: expected '{expected}'. "
                    f"Agreement: {ensemble_result.agreement_ratio:.0%}, "
                    f"Level: {ensemble_result.disagreement_level.value}",
                    suggested_fix="Review video content or adjust ensemble models",
                )
                result.score = min(
                    result.score, self._score_from_disagreement(ensemble_result.disagreement_level)
                )
                result.passed = False

            # Add evidence with ensemble summary
            if ensemble_result.member_results:
                result.add_evidence(
                    timestamp=context.frame_to_timestamp(0),
                    frame_number=0,
                    thumbnail_path=str(context.get_thumbnail_path(0, "ensemble_sample")),
                    explanation=self._build_ensemble_explanation(ensemble_result),
                    confidence=ensemble_result.confidence_estimate,
                )

            # Flag for human review if disagreement is high
            if ensemble_result.agreement_ratio < check_spec.get(
                "disagreement_threshold", self.default_disagreement_threshold
            ):
                result.add_failure(
                    timestamp=context.frame_to_timestamp(0),
                    frame_number=0,
                    failure_type="requires_human_review",
                    severity="warning",
                    message=f"High disagreement among models "
                    f"(agreement: {ensemble_result.agreement_ratio:.0%}). "
                    f"Manual review recommended.",
                )

        return result

    async def _evaluate_check(
        self, check_spec: dict, context: EvaluationContext, total_frames: int
    ) -> EnsembleScore | None:
        """Evaluate a single vlm_ensemble check.

        Args:
            check_spec: Check specification from contract.
            context: Evaluation context.
            total_frames: Total frames in video.

        Returns:
            EnsembleScore with aggregated results.
        """
        question = check_spec.get("question", "")
        expected_answer = check_spec.get("expected_answer", "").lower().strip()
        models = check_spec.get("models", self.DEFAULT_MODELS)
        strategy = check_spec.get("strategy", self.default_strategy)

        if not question:
            logger.warning("Empty question in vlm_ensemble check")
            return None

        # Select sample frames
        sample_frames = self._select_sample_frames(total_frames, len(models))
        frame_paths = [context.get_frame_path(f) for f in sample_frames]
        frame_paths = [p for p in frame_paths if p.exists()]

        if not frame_paths:
            logger.warning("No valid frame paths for ensemble evaluation")
            return None

        # Load and encode frames
        frame_images = self._load_frames(frame_paths[: self.max_frames_per_model])

        # Query all models in parallel
        member_results = await self._query_all_models(
            question, frame_images, models, expected_answer, context
        )

        if not member_results:
            logger.error("All VLM models failed in ensemble")
            return None

        # Aggregate results
        return self._aggregate_results(member_results, strategy)

    async def _query_all_models(
        self,
        question: str,
        frame_images: list[bytes],
        models: list[str],
        expected_answer: str,
        context: EvaluationContext,
    ) -> list[VLMMemberResult]:
        """Query all VLM models in parallel.

        Args:
            question: Question to ask models.
            frame_images: List of encoded frame images.
            models: List of model names to query.
            expected_answer: Expected answer for comparison.
            context: Evaluation context.

        Returns:
            List of VLMMemberResult from each model.
        """
        tasks = [
            self._query_single_model(model, question, frame_images, expected_answer)
            for model in models
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        member_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Model {models[i]} raised exception: {result}")
                member_results.append(
                    VLMMemberResult(
                        model_name=models[i],
                        provider="unknown",
                        vote=VoteDecision.ERROR,
                        answer="error",
                        confidence=0.0,
                        reasoning=str(result),
                        latency_ms=0.0,
                        error=str(result),
                    )
                )
            else:
                member_results.append(result)

        return member_results

    async def _query_single_model(
        self, model_name: str, question: str, frame_images: list[bytes], expected_answer: str
    ) -> VLMMemberResult:
        """Query a single VLM model.

        Args:
            model_name: Name of the model to query.
            question: Question to ask.
            frame_images: List of encoded frame images.
            expected_answer: Expected answer for comparison.

        Returns:
            VLMMemberResult from the model.
        """
        config = VLM_PROVIDER_CONFIG.get(model_name, {})
        provider = config.get("provider", DEFAULT_PROVIDER)
        model_id = config.get("model", model_name)
        base_url = config.get("base_url", DEFAULT_BASE_URL)

        prompt = self.VLM_PROMPT_TEMPLATE.format(question=question)

        start_time = time.perf_counter()

        try:
            if provider == "ollama" or "localhost" in base_url:
                response_text = await self._query_ollama(prompt, frame_images, model_id, base_url)
            elif provider == "gemini":
                response_text = await self._query_gemini(prompt, frame_images, model_id)
            else:
                response_text = await self._query_openrouter_compatible(
                    prompt, frame_images, model_id, base_url, provider
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            parsed = self._parse_vlm_response(response_text)
            answer = parsed.get("answer", "")
            confidence = parsed.get("confidence", 0.5)
            reasoning = parsed.get("reasoning", "")

            vote = self._determine_vote(answer, expected_answer, confidence)

            return VLMMemberResult(
                model_name=model_name,
                provider=provider,
                vote=vote,
                answer=answer,
                confidence=confidence,
                reasoning=reasoning,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Model {model_name} failed: {e}")
            return VLMMemberResult(
                model_name=model_name,
                provider=provider,
                vote=VoteDecision.ERROR,
                answer="error",
                confidence=0.0,
                reasoning=str(e),
                latency_ms=latency_ms,
                error=str(e),
            )

    async def _query_openrouter_compatible(
        self, prompt: str, frame_images: list[bytes], model: str, base_url: str, provider: str
    ) -> str:
        """Query OpenRouter-compatible API endpoint.

        Args:
            prompt: Text prompt.
            frame_images: List of encoded frame images.
            model: Model name.
            base_url: API base URL.
            provider: Provider name for headers.

        Returns:
            Response text from the model.
        """
        import httpx

        api_key = os.getenv("AETHER_LLM_API_KEY") or os.getenv("MINIMAX_API_KEY", "")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://video-unit.io"
            headers["X-Title"] = "VideoUnit Ensemble"

        # Build multi-modal message with images
        content = [{"type": "text", "text": prompt}]

        for img_bytes in frame_images[:4]:  # Limit to 4 images
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        messages = [{"role": "user", "content": content}]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.default_timeout)) as client:
            url = f"{base_url.rstrip('/')}/chat/completions"
            resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                error_data = resp.text
                logger.error(f"OpenRouter API error: {resp.status_code} - {error_data}")
                raise RuntimeError(f"API error {resp.status_code}: {error_data[:200]}")

            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _query_gemini(self, prompt: str, frame_images: list[bytes], model: str) -> str:
        """Query Google Gemini API.

        Args:
            prompt: Text prompt.
            frame_images: List of encoded frame images.
            model: Model name.

        Returns:
            Response text from the model.
        """
        import httpx

        api_key = os.getenv("AETHER_LLM_API_KEY") or os.getenv("GEMINI_API_KEY", "")

        if not api_key:
            raise RuntimeError("No API key for Gemini")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Build multi-modal content
        content = [{"type": "text", "text": prompt}]

        for img_bytes in frame_images[:4]:
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        messages = [{"role": "user", "content": content}]

        # Gemini uses a different endpoint format
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        url = f"{base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.default_timeout)) as client:
            resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                error_data = resp.text
                logger.error(f"Gemini API error: {resp.status_code} - {error_data}")
                raise RuntimeError(f"Gemini API error {resp.status_code}: {error_data[:200]}")

            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _query_ollama(
        self, prompt: str, frame_images: list[bytes], model: str, base_url: str
    ) -> str:
        """Query local Ollama model.

        Args:
            prompt: Text prompt.
            frame_images: List of encoded frame images.
            model: Model name.
            base_url: Ollama base URL.

        Returns:
            Response text from the model.
        """
        import httpx

        # Ollama uses a different API format
        url = f"{base_url.rstrip('/')}/chat"

        content = [{"type": "text", "text": prompt}]

        # Ollama natively supports image URLs
        for img_bytes in frame_images[:4]:
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.default_timeout * 2)) as client:
            resp = await client.post(url, json=payload)

            if resp.status_code != 200:
                error_data = resp.text
                logger.error(f"Ollama API error: {resp.status_code} - {error_data}")
                raise RuntimeError(f"Ollama API error {resp.status_code}: {error_data[:200]}")

            data = resp.json()
            return data["message"]["content"]

    def _parse_vlm_response(self, response_text: str) -> dict[str, Any]:
        """Parse VLM response text into structured dict.

        Args:
            response_text: Raw response text.

        Returns:
            Parsed response dict with answer, confidence, reasoning.
        """
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                return {
                    "answer": str(parsed.get("answer", "")).lower().strip(),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "reasoning": str(parsed.get("reasoning", "")),
                }
        except json.JSONDecodeError:
            pass

        # Fallback parsing
        cleaned = response_text.strip().lower()

        if any(yes in cleaned for yes in ["yes", "true", "correct", "pass", "visible", "present"]):
            answer = "yes"
        elif any(
            no in cleaned for no in ["no", "false", "incorrect", "fail", "not visible", "missing"]
        ):
            answer = "no"
        elif "uncertain" in cleaned:
            answer = "uncertain"
        else:
            answer = cleaned[:50] if cleaned else "unknown"

        return {"answer": answer, "confidence": 0.5, "reasoning": response_text[:200]}

    def _determine_vote(self, answer: str, expected_answer: str, confidence: float) -> VoteDecision:
        """Determine vote based on answer and expected answer.

        Args:
            answer: Model's actual answer.
            expected_answer: Expected answer from contract.
            confidence: Model's confidence.

        Returns:
            VoteDecision enum value.
        """
        expected_lower = expected_answer.lower().strip()
        answer_lower = answer.lower().strip()

        if answer_lower == "uncertain" or confidence < 0.3:
            return VoteDecision.UNCERTAIN

        if answer_lower == "error":
            return VoteDecision.ERROR

        # Check for match
        if self._check_answer_match(answer_lower, expected_lower):
            return VoteDecision.PASS
        else:
            return VoteDecision.FAIL

    def _check_answer_match(self, actual_answer: str, expected_answer: str) -> bool:
        """Check if actual answer matches expected answer.

        Args:
            actual_answer: Model's actual answer.
            expected_answer: Expected answer from contract.

        Returns:
            True if answers match.
        """
        if actual_answer == expected_answer:
            return True

        positive_indicators = {"yes", "true", "correct", "pass", "good", "ok", "visible", "present"}
        negative_indicators = {"no", "false", "incorrect", "fail", "bad", "missing", "not visible"}

        if expected_answer in positive_indicators and actual_answer in positive_indicators:
            return True
        if expected_answer in negative_indicators and actual_answer in negative_indicators:
            return True

        return expected_answer in actual_answer or actual_answer in expected_answer

    def _aggregate_results(
        self, member_results: list[VLMMemberResult], strategy: str
    ) -> EnsembleScore:
        """Aggregate individual model results into ensemble score.

        Args:
            member_results: List of results from each model.
            strategy: Aggregation strategy (majority, unanimous, confidence_weighted).

        Returns:
            EnsembleScore with aggregated results.
        """
        votes = [r.vote for r in member_results]
        errors = [r for r in member_results if r.vote == VoteDecision.ERROR]

        # Count votes
        pass_count = sum(1 for v in votes if v == VoteDecision.PASS)
        fail_count = sum(1 for v in votes if v == VoteDecision.FAIL)
        uncertain_count = sum(1 for v in votes if v == VoteDecision.UNCERTAIN)

        total_valid = pass_count + fail_count + uncertain_count
        total_members = len(member_results)

        # Calculate agreement ratio
        if total_valid > 0:
            max_agreement = max(pass_count, fail_count)
            agreement_ratio = max_agreement / total_valid
        else:
            agreement_ratio = 0.0

        # Determine disagreement level
        disagreement_level = self._determine_disagreement_level(
            pass_count, fail_count, uncertain_count, total_members, len(errors)
        )

        # Calculate ensemble score based on strategy
        if strategy == "unanimous":
            ensemble_score = self._calculate_unanimous_score(pass_count, total_valid)
        elif strategy == "confidence_weighted":
            ensemble_score = self._calculate_confidence_weighted_score(member_results)
        else:  # majority
            ensemble_score = self._calculate_majority_score(pass_count, total_valid)

        # Calculate confidence estimate
        avg_confidence = np.mean([r.confidence for r in member_results if r.confidence > 0])
        confidence_estimate = float(avg_confidence) * agreement_ratio

        vote_summary = {
            "pass": pass_count,
            "fail": fail_count,
            "uncertain": uncertain_count,
            "errors": len(errors),
            "total": total_members,
        }

        return EnsembleScore(
            ensemble_score=ensemble_score,
            vote_summary=vote_summary,
            disagreement_level=disagreement_level,
            agreement_ratio=agreement_ratio,
            confidence_estimate=confidence_estimate,
            member_results=member_results,
        )

    def _determine_disagreement_level(
        self, pass_count: int, fail_count: int, uncertain_count: int, total: int, error_count: int
    ) -> DisagreementLevel:
        """Determine the level of disagreement among models.

        Args:
            pass_count: Number of PASS votes.
            fail_count: Number of FAIL votes.
            uncertain_count: Number of UNCERTAIN votes.
            total: Total number of models.
            error_count: Number of model errors.

        Returns:
            DisagreementLevel enum value.
        """
        if error_count > 0 and error_count == total:
            return DisagreementLevel.MODEL_ERRORS

        if error_count > 0:
            return DisagreementLevel.MODEL_ERRORS

        if pass_count == total:
            return DisagreementLevel.UNANIMOUS_PASS
        if fail_count == total:
            return DisagreementLevel.UNANIMOUS_FAIL

        if uncertain_count == total:
            return DisagreementLevel.DISAGREEMENT

        if pass_count > fail_count and pass_count > uncertain_count:
            if pass_count > total / 2:
                return DisagreementLevel.CONSENSUS_PASS
        if fail_count > pass_count and fail_count > uncertain_count:
            if fail_count > total / 2:
                return DisagreementLevel.CONSENSUS_FAIL

        if uncertain_count > 0:
            return DisagreementLevel.DISAGREEMENT

        if pass_count + fail_count == total and pass_count > 0 and fail_count > 0:
            return DisagreementLevel.DISAGREEMENT

        return DisagreementLevel.DISAGREEMENT

    def _calculate_majority_score(self, pass_count: int, total: int) -> float:
        """Calculate score using majority voting.

        Args:
            pass_count: Number of PASS votes.
            total: Total valid votes.

        Returns:
            Score from 0-100.
        """
        if total == 0:
            return 0.0
        return (pass_count / total) * 100.0

    def _calculate_unanimous_score(self, pass_count: int, total: int) -> float:
        """Calculate score requiring unanimous agreement.

        Args:
            pass_count: Number of PASS votes.
            total: Total valid votes.

        Returns:
            Score from 0-100.
        """
        if total == 0:
            return 0.0
        if pass_count == total:
            return 100.0
        # Partial score based on how close to unanimous
        return (pass_count / total) * 50.0

    def _calculate_confidence_weighted_score(self, member_results: list[VLMMemberResult]) -> float:
        """Calculate score using confidence-weighted voting.

        Args:
            member_results: Results from each model.

        Returns:
            Score from 0-100.
        """
        valid_results = [
            r
            for r in member_results
            if r.vote != VoteDecision.ERROR and r.vote != VoteDecision.UNCERTAIN
        ]

        if not valid_results:
            return 0.0

        total_confidence = sum(r.confidence for r in valid_results)
        if total_confidence == 0:
            return 0.0

        weighted_score = 0.0
        for r in valid_results:
            weight = r.confidence / total_confidence
            vote_value = 1.0 if r.vote == VoteDecision.PASS else 0.0
            weighted_score += vote_value * weight

        return weighted_score * 100.0

    def _determine_pass_fail(self, ensemble_result: EnsembleScore, check_spec: dict) -> bool:
        """Determine if the check passed based on ensemble result.

        Args:
            ensemble_result: Aggregated ensemble result.
            check_spec: Check specification.

        Returns:
            True if check passes.
        """
        strategy = check_spec.get("strategy", self.default_strategy)

        if strategy == "unanimous":
            return ensemble_result.vote_summary["pass"] == len(ensemble_result.member_results)
        elif strategy == "confidence_weighted":
            return ensemble_result.ensemble_score >= 70.0
        else:  # majority
            return ensemble_result.vote_summary["pass"] > ensemble_result.vote_summary["fail"]

    def _severity_from_disagreement(self, disagreement_level: DisagreementLevel) -> str:
        """Map disagreement level to severity.

        Args:
            disagreement_level: Level of disagreement.

        Returns:
            Severity string.
        """
        severity_map = {
            DisagreementLevel.UNANIMOUS_PASS: "info",
            DisagreementLevel.UNANIMOUS_FAIL: "critical",
            DisagreementLevel.CONSENSUS_PASS: "info",
            DisagreementLevel.CONSENSUS_FAIL: "fail",
            DisagreementLevel.FULL_AGREEMENT: "info",
            DisagreementLevel.DISAGREEMENT: "warning",
            DisagreementLevel.MODEL_ERRORS: "warning",
        }
        return severity_map.get(disagreement_level, "warning")

    def _score_from_disagreement(self, disagreement_level: DisagreementLevel) -> float:
        """Map disagreement level to score penalty.

        Args:
            disagreement_level: Level of disagreement.

        Returns:
            Score value.
        """
        score_map = {
            DisagreementLevel.UNANIMOUS_PASS: 100.0,
            DisagreementLevel.UNANIMOUS_FAIL: 0.0,
            DisagreementLevel.CONSENSUS_PASS: 80.0,
            DisagreementLevel.CONSENSUS_FAIL: 30.0,
            DisagreementLevel.FULL_AGREEMENT: 100.0,
            DisagreementLevel.DISAGREEMENT: 50.0,
            DisagreementLevel.MODEL_ERRORS: 40.0,
        }
        return score_map.get(disagreement_level, 50.0)

    def _build_ensemble_explanation(self, ensemble_result: EnsembleScore) -> str:
        """Build human-readable explanation of ensemble result.

        Args:
            ensemble_result: Ensemble result to explain.

        Returns:
            Explanation string.
        """
        summary = ensemble_result.vote_summary
        parts = []

        for result in ensemble_result.member_results:
            status = (
                "PASS"
                if result.vote == VoteDecision.PASS
                else (
                    "FAIL"
                    if result.vote == VoteDecision.FAIL
                    else "UNCERTAIN" if result.vote == VoteDecision.UNCERTAIN else "ERROR"
                )
            )
            parts.append(f"{result.model_name}: {status} ({result.confidence:.0%})")

        return (
            f"Ensemble [{summary['pass']}P/{summary['fail']}F/{summary['uncertain']}U/{summary['errors']}E] - "
            + "; ".join(parts[:3])
        )

    def _select_sample_frames(self, total_frames: int, num_models: int) -> list[int]:
        """Select representative frames for ensemble evaluation.

        Args:
            total_frames: Total frames in video.
            num_models: Number of models being evaluated.

        Returns:
            List of frame numbers to sample.
        """
        max_frames_total = 16  # Limit total frames across all models
        num_samples = min(num_models * 2, max_frames_total)

        if total_frames <= num_samples:
            return list(range(total_frames))

        step = total_frames // num_samples
        return [i * step for i in range(num_samples)]

    def _load_frames(self, frame_paths: list[Path]) -> list[bytes]:
        """Load and encode frames as JPEG bytes.

        Args:
            frame_paths: List of frame file paths.

        Returns:
            List of JPEG-encoded frame bytes.
        """
        frames = []
        for path in frame_paths:
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frames.append(buffer.tobytes())
            except Exception as e:
                logger.warning(f"Failed to load frame {path}: {e}")
                continue
        return frames
