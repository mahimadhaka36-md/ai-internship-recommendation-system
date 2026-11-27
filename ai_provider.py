"""
ai_provider.py
--------------
Centralized Generative AI Provider for the AI Internship Recommendation System.

Verified Covexcode Managed LLM Endpoint (HolboxAI / Covexcode):
- Primary Base URL: https://llm.covexcode.com/v1
- Backup Base URL (DNS resilience): https://llm.boxcode.sh/v1
- Authentication: Authorization: Bearer <COVEXCODE_API_KEY>
- Protocol: OpenAI-compatible Chat Completions (/chat/completions) and Models (/models)
- Verified Models: deepseek-v4-flash (fast/cost-effective), deepseek-v4-pro (flagship)

Security & Resilience:
- Reads COVEXCODE_API_KEY strictly from environment variables or Streamlit secrets.
- Never hardcodes, logs, or prints raw API keys.
- Gracefully handles expired credits (402/429), rate limits, invalid keys (401),
  unavailable models (400/404), and network timeouts so the app remains 100%
  functional with local ML models (all-MiniLM-L6-v2 / TF-IDF / KNN) when offline
  or after event credits expire.
- Exposes reusable methods for AI Mock Interview, AI Resume Analyzer,
  Personalized Career Roadmap, and AI Career Assistant.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_COVEXCODE_ENDPOINT = "https://llm.covexcode.com/v1"
BACKUP_COVEXCODE_ENDPOINT = "https://llm.boxcode.sh/v1"
DEFAULT_COVEXCODE_MODEL = "deepseek-v4-flash"
SUPPORTED_COVEXCODE_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")


def _mask_key(api_key: str) -> str:
    """Return a safe masked prefix for UI display without exposing the secret."""
    clean = str(api_key or "").strip()
    if not clean:
        return ""
    if len(clean) <= 10:
        return "***"
    return f"{clean[:8]}...{clean[-4:]}"


def _read_config_value(key_name: str, default: str = "") -> str:
    """Read a configuration value from environment variables or Streamlit secrets."""
    env_val = os.environ.get(key_name, "").strip()
    if env_val:
        return env_val
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key_name in st.secrets:
            sec_val = str(st.secrets[key_name]).strip()
            if sec_val:
                return sec_val
    except Exception:
        pass
    return default


def _extract_error_message(response: requests.Response) -> str:
    """Extract a concise human-readable message from an OpenAI-style error response."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"]).strip()
            if payload.get("message"):
                return str(payload["message"]).strip()
    except Exception:
        pass
    text = (response.text or "").strip()
    return text[:240] if text else f"HTTP {response.status_code}"


def _classify_http_error(status_code: int, message: str) -> Tuple[str, str]:
    """Classify HTTP errors into predictable categories for graceful fallback messaging."""
    lower = message.lower()
    if status_code == 401:
        return "auth_error", "Invalid or expired Covexcode API key (HTTP 401)."
    if status_code == 402 or any(
        marker in lower
        for marker in ("out of credit", "credit", "event allowance", "event ended", "insufficient")
    ):
        return "credits_exhausted", f"Covexcode credits exhausted or event ended: {message}"
    if status_code == 429 or "rate limit" in lower or "daily spend" in lower:
        return "rate_limit", f"Covexcode rate or daily spend limit reached: {message}"
    if status_code in (400, 404) and "model" in lower:
        return "model_unavailable", f"Requested Covexcode model unavailable: {message}"
    if status_code >= 500:
        return "server_error", f"Covexcode server temporarily unavailable (HTTP {status_code})."
    return "api_error", f"Covexcode API error (HTTP {status_code}): {message}"


def _strip_json_fences(raw_text: str) -> str:
    """Remove optional markdown code fences around JSON output."""
    cleaned = str(raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


class CovexcodeProvider:
    """
    Centralized client for Covexcode's OpenAI-compatible inference proxy.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self._explicit_api_key = api_key
        self._explicit_endpoint = endpoint
        self._explicit_model = default_model

    @property
    def api_key(self) -> str:
        if self._explicit_api_key is not None:
            return str(self._explicit_api_key).strip()
        return _read_config_value("COVEXCODE_API_KEY", "")

    @property
    def endpoint(self) -> str:
        base = (
            self._explicit_endpoint
            if self._explicit_endpoint is not None
            else _read_config_value("COVEXCODE_ENDPOINT", DEFAULT_COVEXCODE_ENDPOINT)
        )
        return base.strip().rstrip("/") or DEFAULT_COVEXCODE_ENDPOINT

    @property
    def default_model(self) -> str:
        model = (
            self._explicit_model
            if self._explicit_model is not None
            else _read_config_value("COVEXCODE_MODEL", DEFAULT_COVEXCODE_MODEL)
        )
        return model.strip() or DEFAULT_COVEXCODE_MODEL

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_status(self) -> Dict[str, Any]:
        """Return non-sensitive provider configuration status for UI display."""
        key = self.api_key
        return {
            "configured": bool(key),
            "provider_name": "Covexcode",
            "endpoint": self.endpoint,
            "default_model": self.default_model,
            "supported_models": list(SUPPORTED_COVEXCODE_MODELS),
            "key_masked": _mask_key(key),
        }

    def _chat_completions_url(self, base_endpoint: str) -> str:
        base = base_endpoint.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _models_url(self, base_endpoint: str) -> str:
        base = base_endpoint.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/models"
        return f"{base}/v1/models"

    def _post_chat(
        self,
        payload: Dict[str, Any],
        timeout: int = 25,
    ) -> Tuple[Optional[requests.Response], Optional[str], Optional[str]]:
        """Send chat completion POST with optional backup-host DNS fallback."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoints_to_try = [self.endpoint]
        if "llm.covexcode.com" in self.endpoint:
            endpoints_to_try.append(
                self.endpoint.replace("llm.covexcode.com", "llm.boxcode.sh")
            )

        last_err = None
        for idx, base in enumerate(endpoints_to_try):
            url = self._chat_completions_url(base)
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                return resp, None, None
            except requests.Timeout:
                last_err = ("timeout", "Covexcode API request timed out.")
            except requests.ConnectionError as exc:
                last_err = ("connection_error", f"Could not reach Covexcode endpoint: {exc}")
                # Try backup host on connection/DNS failure when primary fails
                if idx == 0 and len(endpoints_to_try) > 1:
                    continue
            except requests.RequestException as exc:
                last_err = ("network_error", f"Network error calling Covexcode: {exc}")
            break

        return None, last_err[0], last_err[1]

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
        timeout: int = 25,
    ) -> Dict[str, Any]:
        """Execute an OpenAI-compatible chat completion call against Covexcode."""
        if not self.is_configured():
            return {
                "ok": False,
                "content": "",
                "model_used": model or self.default_model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "error_kind": "not_configured",
                "error": "COVEXCODE_API_KEY is not configured.",
            }

        chosen_model = (model or self.default_model).strip()
        payload = {
            "model": chosen_model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }

        resp, err_kind, err_msg = self._post_chat(payload, timeout=timeout)
        if resp is None:
            return {
                "ok": False,
                "content": "",
                "model_used": chosen_model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "error_kind": err_kind,
                "error": err_msg,
            }

        if not (200 <= resp.status_code < 300):
            raw_msg = _extract_error_message(resp)
            kind, friendly = _classify_http_error(resp.status_code, raw_msg)
            return {
                "ok": False,
                "content": "",
                "model_used": chosen_model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "error_kind": kind,
                "error": friendly,
            }

        try:
            data = resp.json()
            choices = data.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            content = str(message.get("content") or "").strip()
            usage_raw = data.get("usage") or {}
            usage = {
                "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
                "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
                "total_tokens": int(usage_raw.get("total_tokens", 0)),
            }
            if not content:
                return {
                    "ok": False,
                    "content": "",
                    "model_used": str(data.get("model") or chosen_model),
                    "usage": usage,
                    "error_kind": "empty_response",
                    "error": "Covexcode returned an empty completion content (token budget may have been exhausted by reasoning).",
                }
            return {
                "ok": True,
                "content": content,
                "model_used": str(data.get("model") or chosen_model),
                "usage": usage,
                "error_kind": None,
                "error": None,
            }
        except Exception as exc:
            return {
                "ok": False,
                "content": "",
                "model_used": chosen_model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "error_kind": "parse_error",
                "error": f"Failed to parse Covexcode JSON response: {exc}",
            }

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Run chat completion and parse the assistant response as JSON."""
        res = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if not res["ok"]:
            return {**res, "data": None}

        cleaned = _strip_json_fences(res["content"])
        try:
            parsed = json.loads(cleaned)
            return {**res, "data": parsed}
        except json.JSONDecodeError:
            # Try extracting first JSON array/object block if prose surrounded it
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    return {**res, "data": parsed}
                except json.JSONDecodeError:
                    pass
            return {
                **res,
                "ok": False,
                "data": None,
                "error_kind": "invalid_json",
                "error": "Model response was not valid JSON.",
            }

    def test_connection(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Perform a minimal connection check against Covexcode."""
        return self.chat_completion(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            model=model or self.default_model,
            max_tokens=20,
            temperature=0.0,
            timeout=15,
        )

    # -------------------------------------------------------------------------
    # High-level Reusable Feature Methods
    # -------------------------------------------------------------------------

    def generate_interview_questions(
        self,
        role: str,
        domain: str,
        difficulty: str,
        interview_type: str,
        count: int = 2,
        skills: Optional[List[str]] = None,
        internship_context: str = "",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate personalized mock interview questions grounded in student skills and role."""
        skills_str = ", ".join(skills[:12]) if skills else "general role fundamentals"
        ctx_line = f"Target internship listing: {internship_context}." if internship_context else ""
        prompt = (
            f"Generate {count} unique {difficulty}-level {interview_type} internship interview questions "
            f"for the role '{role}' ({domain}). Student skills: {skills_str}. {ctx_line}\n"
            "Return ONLY a JSON array of objects, each with keys:\n"
            "- topic (short string)\n"
            "- question_type ('Technical', 'HR', or 'Behavioral')\n"
            "- question_text (clear interview question)\n"
            "- reference_answer (3-4 sentence accurate model answer)\n"
            "- expected_concepts (array of 4-5 short lowercase key concepts)\n"
            "- evaluation_rubric (object with technical_correctness, concept_coverage, completeness, clarity)"
        )
        res = self.chat_json(
            messages=[
                {"role": "system", "content": "You are a technical internship interviewer. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=1400,
            temperature=0.3,
        )
        if not res["ok"] or not isinstance(res["data"], list):
            return {**res, "questions": []}

        questions: List[Dict[str, Any]] = []
        for idx, item in enumerate(res["data"][:count], start=1):
            if not isinstance(item, dict):
                continue
            if all(k in item for k in ("question_text", "reference_answer", "expected_concepts")):
                qtype = str(item.get("question_type", "Technical" if interview_type == "Mixed" else interview_type))
                if qtype not in ("Technical", "HR", "Behavioral"):
                    qtype = "Technical" if interview_type == "Mixed" else interview_type
                questions.append({
                    "question_id": f"CX-AI-{role[:3].upper()}-{abs(hash(item['question_text'])) % 10000:04d}-{idx}",
                    "role": role,
                    "domain": domain,
                    "difficulty": difficulty,
                    "question_type": qtype,
                    "topic": str(item.get("topic") or f"{role} Practice").strip(),
                    "question_text": str(item["question_text"]).strip(),
                    "reference_answer": str(item["reference_answer"]).strip(),
                    "expected_concepts": [
                        str(c).lower().strip()
                        for c in item.get("expected_concepts", [])
                        if str(c).strip()
                    ][:6],
                    "evaluation_rubric": item.get("evaluation_rubric")
                    if isinstance(item.get("evaluation_rubric"), dict)
                    else {
                        "technical_correctness": "Accurate explanation aligned with role fundamentals.",
                        "concept_coverage": "Addresses expected domain concepts.",
                        "completeness": "Covers definition and practical reasoning.",
                        "clarity": "Clear and structured.",
                    },
                    "source": f"Covexcode AI ({res['model_used']})",
                })
        return {**res, "questions": questions}

    def generate_followup_question(
        self,
        question: Dict[str, Any],
        student_answer: str,
        covered_concepts: List[str],
        missing_concepts: List[str],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate one contextual follow-up question based on the student's actual answer."""
        if not str(student_answer or "").strip():
            return {
                "ok": False,
                "followup_question": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "error_kind": "empty_answer",
                "error": "No answer submitted to follow up on.",
            }

        prompt = (
            f"Original question ({question.get('role')} - {question.get('topic')}): {question.get('question_text')}\n"
            f"Student's answer: {student_answer}\n"
            f"Concepts student mentioned: {', '.join(covered_concepts) or 'none'}\n"
            f"Missing expected concepts: {', '.join(missing_concepts) or 'none'}\n\n"
            "Write ONE concise, realistic interviewer follow-up question that probes deeper into what the student said "
            "or nudges them on a missed concept. Do not invent claims the student did not make. "
            "Return ONLY a JSON object with keys: followup_question, focus_area, reference_hint."
        )
        res = self.chat_json(
            messages=[
                {"role": "system", "content": "You are a supportive technical interviewer. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=450,
            temperature=0.3,
        )
        if not res["ok"] or not isinstance(res["data"], dict):
            return {**res, "followup_question": "", "focus_area": "", "reference_hint": ""}

        return {
            **res,
            "followup_question": str(res["data"].get("followup_question", "")).strip(),
            "focus_area": str(res["data"].get("focus_area", "")).strip(),
            "reference_hint": str(res["data"].get("reference_hint", "")).strip(),
        }

    def evaluate_interview_answer(
        self,
        question: Dict[str, Any],
        student_answer: str,
        semantic_score_baseline: float,
        covered_concepts: List[str],
        missing_concepts: List[str],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a student's interview response with Covexcode LLM grounded strictly
        in the submitted answer, reference answer, and rubric.
        """
        prompt = (
            f"Role: {question.get('role')} | Difficulty: {question.get('difficulty')} | Topic: {question.get('topic')}\n"
            f"Question: {question.get('question_text')}\n"
            f"Reference Answer: {question.get('reference_answer')}\n"
            f"Expected Concepts: {', '.join(question.get('expected_concepts', []))}\n"
            f"Student Answer: {student_answer}\n"
            f"Observed Covered Concepts: {', '.join(covered_concepts) or 'none'}\n"
            f"Observed Missing Concepts: {', '.join(missing_concepts) or 'none'}\n"
            f"Embedding Similarity Baseline (0-100): {semantic_score_baseline:.1f}\n\n"
            "Evaluate strictly based on what the student actually wrote—do NOT fabricate strengths or claim facts "
            "they did not write. Return ONLY a JSON object with keys:\n"
            "- technical_correctness_score (number 0-100)\n"
            "- completeness_score (number 0-100)\n"
            "- clarity_score (number 0-100)\n"
            "- strengths (array of 1-3 short factual bullets grounded in the student's text)\n"
            "- areas_for_improvement (array of 1-3 short actionable bullets)\n"
            "- suggested_improved_answer (concise improved version building on what the student wrote)"
        )
        res = self.chat_json(
            messages=[
                {"role": "system", "content": "You are an objective interview evaluator. Never fabricate details not in the student's answer. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=850,
            temperature=0.1,
        )
        if not res["ok"] or not isinstance(res["data"], dict):
            return {**res, "evaluation": None}
        return {**res, "evaluation": res["data"]}

    def analyze_resume_text(
        self,
        resume_text: str,
        target_role: str = "",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reusable interface for the upcoming AI Resume Analyzer feature."""
        snippet = str(resume_text or "").strip()[:5000]
        if not snippet:
            return {"ok": False, "data": None, "error_kind": "empty_input", "error": "No resume text provided."}
        prompt = (
            f"Analyze this resume snippet for a student targeting '{target_role or 'Technical Internship'}'.\n"
            f"Resume:\n{snippet}\n\n"
            "Return ONLY a JSON object with keys: strengths (list), skill_gaps (list), bullet_improvements (list)."
        )
        return self.chat_json(
            messages=[
                {"role": "system", "content": "You are a concise technical resume reviewer. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=900,
            temperature=0.2,
        )

    def generate_career_roadmap(
        self,
        current_skills: List[str],
        target_role: str,
        weeks: int = 8,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reusable interface for the upcoming Personalized Career Roadmap feature."""
        prompt = (
            f"Build a practical {weeks}-week learning roadmap for a student with skills [{', '.join(current_skills)}] "
            f"targeting '{target_role}'. Return ONLY a JSON object with keys: milestones (list of week, focus, project_idea)."
        )
        return self.chat_json(
            messages=[
                {"role": "system", "content": "You are an academic career mentor. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=1000,
            temperature=0.2,
        )

    def career_assistant_reply(
        self,
        user_question: str,
        context_summary: str = "",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reusable interface for the upcoming AI Career Assistant feature."""
        if not str(user_question or "").strip():
            return {"ok": False, "content": "", "error_kind": "empty_input", "error": "Question is empty."}
        return self.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful, honest AI Career Assistant for college internship applicants. "
                    "Never claim guaranteed hiring outcomes.",
                },
                {
                    "role": "user",
                    "content": (f"Student Context: {context_summary}\n\nQuestion: {user_question}" if context_summary else user_question),
                },
            ],
            model=model,
            max_tokens=700,
            temperature=0.3,
        )


_provider_instance: Optional[CovexcodeProvider] = None


def get_ai_provider() -> CovexcodeProvider:
    """Return a shared CovexcodeProvider singleton."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = CovexcodeProvider()
    return _provider_instance
