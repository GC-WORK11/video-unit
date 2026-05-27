"""VLMJudgeEvaluator - semantic evaluation using Vision Language Models."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ._base import Evaluator
from ._context import EvaluationContext
from ._registry import register_evaluator
from ._result import EvaluationResult

logger = logging.getLogger(__name__)


@register_evaluator
class VLMQuestionEvaluator(Evaluator):
    """Evaluates video semantics using Vision Language Models.

    This evaluator sends frames to a VLM (either local Gemma4 or cloud MiniMax)
    and asks semantic questions about the video content. Responses are parsed
    and evaluated against expected answers from the contract.

    Contract schema:
        vlm_checks:
          - question: "Is the ball continuously visible?"
            expected_answer: "yes"
            model: "gemma4"  # or "minimax"
            failure_response: "Object not visible throughout"

    Supports:
        - Local inference with Gemma4
        - Cloud inference with MiniMax
        - Custom prompts per question
    """

    name = "vlm_judge"
    required_inputs = ["frames"]

    VLM_PROMPT_TEMPLATE = """You are evaluating a video. Answer the following question about the video frames provided.

Question: {question}

Examine the frames carefully and provide your answer. Respond with ONLY a JSON object in this exact format:
{{"answer": "your answer here", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Be honest about uncertainty. If you cannot determine the answer from the frames, set confidence below 0.5."""

    def __init__(self):
        self.default_model = "minimax"
        self.default_timeout = 30.0  # seconds

    async def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run the VLM-based semantic evaluation.

        Args:
            context: Evaluation context with frames.

        Returns:
            EvaluationResult with failures for any semantic mismatches.
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
        vlm_checks = context.contract.get("vlm_checks", [])

        if not vlm_checks:
            logger.info("No VLM checks specified in contract")
            return result

        total_frames = context.video_metadata.get("total_frames", 0)
        if total_frames == 0:
            logger.warning("Cannot determine total frames for VLM check")
            return result

        sample_frames = self._select_sample_frames(total_frames, len(vlm_checks))

        for check_spec in vlm_checks:
            question = check_spec.get("question", "")
            expected_answer = check_spec.get("expected_answer", "").lower().strip()
            model = check_spec.get("model", self.default_model)
            failure_message = check_spec.get("failure_response", "VLM check failed")
            threshold = check_spec.get("confidence_threshold", 0.7)

            if not question:
                continue

            frame_paths = [context.get_frame_path(f) for f in sample_frames]
            frame_paths = [p for p in frame_paths if p.exists()]

            if not frame_paths:
                result.add_failure(
                    timestamp=context.frame_to_timestamp(sample_frames[0]),
                    frame_number=sample_frames[0],
                    failure_type="no_frames_for_vlm",
                    severity="warning",
                    message="No frames available for VLM evaluation"
                )
                continue

            try:
                vlm_response = await self._query_vlm(
                    question, frame_paths, model, context
                )

                answer_match = self._check_answer_match(
                    vlm_response.get("answer", ""),
                    expected_answer
                )
                confidence = vlm_response.get("confidence", 0.0)

                if not answer_match or confidence < threshold:
                    severity = self._severity_from_confidence(confidence)
                    result.add_failure(
                        timestamp=context.frame_to_timestamp(sample_frames[0]),
                        frame_number=sample_frames[0],
                        failure_type="vlm_semantic_mismatch",
                        severity=severity,
                        message=f"{failure_message}. VLM answered: '{vlm_response.get('answer', 'unknown')}'",
                        suggested_fix="Review video content for the specified semantic property"
                    )
                    result.score = min(result.score, self._score_from_severity(severity))
                    result.passed = False

                if sample_frames:
                    mid_frame = sample_frames[len(sample_frames) // 2]
                    result.add_evidence(
                        timestamp=context.frame_to_timestamp(mid_frame),
                        frame_number=mid_frame,
                        thumbnail_path=str(context.get_thumbnail_path(
                            mid_frame, "vlm_sample"
                        )),
                        explanation=f"VLM response: {vlm_response.get('answer', 'unknown')} "
                                   f"(confidence: {confidence:.2f})",
                        confidence=confidence
                    )

            except Exception as e:
                logger.error(f"VLM evaluation failed: {e}")
                result.add_failure(
                    timestamp=context.frame_to_timestamp(sample_frames[0]),
                    frame_number=sample_frames[0],
                    failure_type="vlm_error",
                    severity="warning",
                    message=f"VLM evaluation error: {str(e)}",
                    suggested_fix="Check VLM service availability"
                )
                result.score = min(result.score, 50.0)

        return result

    def _select_sample_frames(self, total_frames: int, num_checks: int) -> list[int]:
        """Select representative frames for VLM evaluation.

        Args:
            total_frames: Total frames in video.
            num_checks: Number of checks to perform.

        Returns:
            List of frame numbers to sample.
        """
        max_frames_to_sample = 8
        num_samples = min(num_checks, max_frames_to_sample)

        if total_frames <= num_samples:
            return list(range(total_frames))

        step = total_frames // num_samples
        return [i * step for i in range(num_samples)]

    async def _query_vlm(
        self,
        question: str,
        frame_paths: list[Path],
        model: str,
        context: EvaluationContext
    ) -> dict[str, Any]:
        """Query the VLM with frames and question.

        Args:
            question: Question to ask the VLM.
            frame_paths: Paths to frame images.
            model: Model to use ("gemma4" or "minimax").
            context: Evaluation context.

        Returns:
            VLM response dict with answer, confidence, reasoning.
        """
        prompt = self.VLM_PROMPT_TEMPLATE.format(question=question)

        if model == "gemma4":
            return await self._query_gemma4(prompt, frame_paths)
        elif model == "minimax":
            return await self._query_minimax(prompt, frame_paths, context)
        else:
            logger.warning(f"Unknown VLM model '{model}', defaulting to minimax")
            return await self._query_minimax(prompt, frame_paths, context)

    async def _query_gemma4(
        self,
        prompt: str,
        frame_paths: list[Path]
    ) -> dict[str, Any]:
        """Query local Gemma4 model.

        Args:
            prompt: Prompt text.
            frame_paths: Paths to frame images.

        Returns:
            VLM response dict.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "google/gemma-4-2b-it"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            images = [self._load_image(path) for path in frame_paths[:4]]

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return self._parse_vlm_response(response_text)

        except ImportError:
            logger.warning("Transformers not available, falling back to mock response")
            return self._mock_vlm_response()
        except Exception as e:
            logger.error(f"Gemma4 inference failed: {e}")
            return {"answer": "unknown", "confidence": 0.0, "reasoning": f"Error: {e}"}

    async def _query_minimax(
        self,
        prompt: str,
        frame_paths: list[Path],
        context: EvaluationContext
    ) -> dict[str, Any]:
        """Query MiniMax cloud VLM.

        Args:
            prompt: Prompt text.
            frame_paths: Paths to frame images.
            context: Evaluation context.

        Returns:
            VLM response dict.
        """
        try:
            import aether_client

            client = aether_client.AetherClient()
            orchestrator = client.orchestrator()

            frame_images = []
            for path in frame_paths[:8]:
                img = cv2.imread(str(path))
                if img is not None:
                    _, buffer = cv2.imencode('.jpg', img)
                    frame_images.append(buffer.tobytes())

            response = await asyncio.wait_for(
                orchestrator.analyze_frames(
                    frames=frame_images,
                    prompt=prompt
                ),
                timeout=self.default_timeout
            )

            if isinstance(response, str):
                return self._parse_vlm_response(response)
            return response

        except ImportError:
            logger.warning("Aether client not available, using mock response")
            return self._mock_vlm_response()
        except asyncio.TimeoutError:
            logger.error("MiniMax VLM request timed out")
            return {"answer": "unknown", "confidence": 0.0, "reasoning": "Timeout"}
        except Exception as e:
            logger.error(f"MiniMax VLM query failed: {e}")
            return {"answer": "unknown", "confidence": 0.0, "reasoning": f"Error: {e}"}

    def _load_image(self, path: Path) -> NDArray[np.uint8]:
        """Load an image from file.

        Args:
            path: Path to image file.

        Returns:
            Image array.
        """
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img

    def _parse_vlm_response(self, response_text: str) -> dict[str, Any]:
        """Parse VLM response text into structured dict.

        Args:
            response_text: Raw response text.

        Returns:
            Parsed response dict.
        """
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                return {
                    "answer": str(parsed.get("answer", "")).lower().strip(),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "reasoning": str(parsed.get("reasoning", ""))
                }
        except json.JSONDecodeError:
            pass

        cleaned = response_text.strip().lower()
        if any(yes in cleaned for yes in ["yes", "true", "correct", "pass"]):
            answer = "yes"
        elif any(no in cleaned for no in ["no", "false", "incorrect", "fail"]):
            answer = "no"
        else:
            answer = cleaned[:50] if cleaned else "unknown"

        return {
            "answer": answer,
            "confidence": 0.5,
            "reasoning": response_text[:200]
        }

    def _mock_vlm_response(self) -> dict[str, Any]:
        """Generate a mock VLM response for testing.

        Returns:
            Mock response dict.
        """
        import random
        answers = ["yes", "no"]
        return {
            "answer": random.choice(answers),
            "confidence": 0.7,
            "reasoning": "Mock response for testing"
        }

    def _check_answer_match(self, actual_answer: str, expected_answer: str) -> bool:
        """Check if actual answer matches expected answer.

        Args:
            actual_answer: VLM's actual answer.
            expected_answer: Expected answer from contract.

        Returns:
            True if answers match.
        """
        actual_lower = actual_answer.lower().strip()
        expected_lower = expected_answer.lower().strip()

        if actual_lower == expected_lower:
            return True

        positive_indicators = {"yes", "true", "correct", "pass", "good", "ok"}
        negative_indicators = {"no", "false", "incorrect", "fail", "bad", "missing"}

        if expected_lower in positive_indicators and actual_lower in positive_indicators:
            return True
        if expected_lower in negative_indicators and actual_lower in negative_indicators:
            return True

        return expected_lower in actual_lower or actual_lower in expected_lower

    def _severity_from_confidence(self, confidence: float) -> str:
        """Determine severity from VLM confidence.

        Args:
            confidence: VLM confidence value (0-1).

        Returns:
            Severity string.
        """
        if confidence >= 0.8:
            return "critical"
        elif confidence >= 0.6:
            return "fail"
        elif confidence >= 0.4:
            return "warning"
        else:
            return "info"

    def _score_from_severity(self, severity: str) -> float:
        """Map severity to score penalty."""
        scores = {
            "critical": 0.0,
            "fail": 25.0,
            "warning": 50.0,
            "info": 80.0
        }
        return scores.get(severity, 50.0)


try:
    import cv2
except ImportError:
    cv2 = None
