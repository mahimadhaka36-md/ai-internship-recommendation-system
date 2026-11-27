"""
interview_evaluator.py
----------------------
ML-based Answer Evaluation Engine for AI Mock Interviews.

Primary Semantic Model:
- Lazy-loaded Sentence Transformers (`all-MiniLM-L6-v2`) cached via Streamlit `@st.cache_resource`
  to minimize memory consumption and startup overhead.

Fallback Engine:
- Automatic TF-IDF (unigrams + bigrams) + Concept-Matching fallback if `sentence-transformers`
  or the `all-MiniLM-L6-v2` model cannot be loaded (e.g., offline environments or memory limits).

Transparent Scoring Rubric (0-100):
- Semantic Relevance (35%): Cosine similarity between student answer and reference answer embeddings.
- Concept Coverage (30%): Deterministic presence of expected key concepts/terms.
- Technical Correctness (15%): Grounded blend of concept alignment, semantic alignment, and uncertainty checks
  (semantic similarity alone never guarantees factual correctness).
- Completeness (10%): Concept coverage paired with substantive response length relative to the reference.
- Answer Clarity (10%): Observable structure, sentence flow, and explanatory connectors.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ai_provider import CovexcodeProvider, get_ai_provider

try:
    import streamlit as st
except ImportError:  # pragma: no cover - allows standalone unit testing without streamlit runtime
    st = None


MODEL_NAME = "all-MiniLM-L6-v2"

RUBRIC_WEIGHTS: Dict[str, float] = {
    "semantic_relevance": 0.35,
    "concept_coverage": 0.30,
    "technical_correctness": 0.15,
    "completeness": 0.10,
    "answer_clarity": 0.10,
}

_EXPLANATORY_CONNECTORS = {
    "because", "therefore", "since", "however", "whereas", "while", "instead",
    "for example", "such as", "specifically", "first", "second", "finally",
    "so that", "which means", "this prevents", "this ensures", "by using",
    "in contrast", "on the other hand", "when", "then",
}

_UNCERTAINTY_ONLY_PHRASES = {
    "i don't know", "i do not know", "dont know", "no idea", "not sure",
    "skip", "pass", "na", "n/a", "idk",
}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "and",
    "or", "but", "it", "this", "that", "these", "those", "can", "will",
    "you", "your", "we", "they", "their", "into", "about", "than", "when",
}

_STANDALONE_MODEL_CACHE: Dict[str, Tuple[Optional[Any], Optional[str]]] = {}


def _load_sentence_transformer_uncached(
    model_name: str = MODEL_NAME,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Attempt to import and load SentenceTransformer without hanging or crashing:
    1. Try local cached model first (`local_files_only=True`) so cached models load instantly offline.
    2. If not cached locally, do a fast DNS check before downloading so offline environments
       fall back immediately to TF-IDF without multi-minute hub retries.
    """
    try:
        import os
        import socket
        from sentence_transformers import SentenceTransformer

        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            model = SentenceTransformer(model_name, local_files_only=True)
            return model, None
        except Exception:
            pass

        # Fast DNS probe before triggering huggingface_hub multi-retry downloads
        socket.getaddrinfo("huggingface.co", 443)
        model = SentenceTransformer(model_name)
        return model, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


if st is not None:

    @st.cache_resource(show_spinner=False)
    def get_cached_sentence_transformer(
        model_name: str = MODEL_NAME,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Lazily load and cache SentenceTransformer in Streamlit resource cache."""
        return _load_sentence_transformer_uncached(model_name)

else:

    def get_cached_sentence_transformer(
        model_name: str = MODEL_NAME,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Module-level cache when running outside Streamlit."""
        if model_name not in _STANDALONE_MODEL_CACHE:
            _STANDALONE_MODEL_CACHE[model_name] = _load_sentence_transformer_uncached(model_name)
        return _STANDALONE_MODEL_CACHE[model_name]


def _normalize_text(text: str) -> str:
    """Lowercase and normalize punctuation/whitespace for deterministic matching."""
    lowered = text.lower().strip()
    lowered = lowered.replace("-", " ").replace("_", " ").replace("/", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _extract_content_words(text: str) -> List[str]:
    """Return meaningful non-stopword tokens from text."""
    tokens = re.findall(r"[a-z0-9+#.]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _concept_present(concept: str, normalized_answer: str, raw_lower_answer: str) -> bool:
    """
    Check whether an expected concept appears in the student's answer.
    Supports exact boundary matching plus multi-word root/token matching
    without fabricating matches for unrelated words.
    """
    c_raw = concept.lower().strip()
    c_norm = _normalize_text(concept)
    if not c_norm:
        return False

    # Direct match on raw lowercase (preserves symbols like c++, o(1), .venv)
    if len(c_raw) <= 3:
        if re.search(rf"(?<![a-z0-9+#.]){re.escape(c_raw)}(?![a-z0-9+#.])", raw_lower_answer):
            return True
    elif re.search(rf"\b{re.escape(c_raw)}\b", raw_lower_answer):
        return True

    # Normalized phrase match (e.g. 'cross-validation' -> 'cross validation')
    if re.search(rf"\b{re.escape(c_norm)}\b", normalized_answer):
        return True

    # Multi-word concept: require all meaningful sub-tokens (or 4+ char prefix stems) to appear
    sub_tokens = [tok for tok in c_norm.split() if tok not in _STOPWORDS and len(tok) > 1]
    if len(sub_tokens) >= 2:
        answer_words = set(re.findall(r"[a-z0-9+#.]+", normalized_answer))
        matched_tokens = 0
        for tok in sub_tokens:
            stem = tok[:4] if len(tok) >= 5 else tok
            if tok in answer_words or any(w.startswith(stem) for w in answer_words):
                matched_tokens += 1
        if matched_tokens == len(sub_tokens):
            return True
    elif len(sub_tokens) == 1 and len(sub_tokens[0]) >= 5:
        stem = sub_tokens[0][:5]
        answer_words = re.findall(r"[a-z0-9+#.]+", normalized_answer)
        if any(w.startswith(stem) for w in answer_words):
            return True

    return False


def _compute_tfidf_similarity(student_answer: str, reference_answer: str) -> float:
    """Fallback cosine similarity using unigram+bigram TF-IDF vectors."""
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        mat = vectorizer.fit_transform([reference_answer, student_answer])
        sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        return float(np.clip(sim, 0.0, 1.0))
    except ValueError:
        return 0.0


class InterviewAnswerEvaluator:
    """
    Evaluates interview responses using all-MiniLM-L6-v2 semantic embeddings
    combined with concept coverage, completeness, and clarity heuristics,
    with automatic TF-IDF fallback.
    """

    def __init__(self, model_name: str = MODEL_NAME, force_fallback: bool = False):
        self.model_name = model_name
        self.force_fallback = force_fallback
        self._model = None
        self._load_error: Optional[str] = None
        self._load_attempted = False

    def _ensure_semantic_backend(self) -> Tuple[Optional[Any], str, Optional[str]]:
        """
        Lazily obtain the SentenceTransformer model if allowed and available.
        Returns (model_or_none, backend_label, fallback_reason_or_none).
        """
        if self.force_fallback:
            return None, "TF-IDF + Concept Matching (Fallback)", "Fallback mode explicitly enabled."

        if not self._load_attempted:
            self._load_attempted = True
            self._model, self._load_error = get_cached_sentence_transformer(self.model_name)

        if self._model is not None:
            return self._model, f"Sentence Transformers ({self.model_name})", None
        return (
            None,
            "TF-IDF + Concept Matching (Fallback)",
            self._load_error or "Sentence Transformer model unavailable.",
        )

    def get_backend_status(self) -> Dict[str, Any]:
        """Inspect evaluator backend status without crashing if model is unavailable."""
        model, label, reason = self._ensure_semantic_backend()
        return {
            "using_transformer": model is not None,
            "backend_label": label,
            "model_name": self.model_name,
            "fallback_reason": reason,
        }

    def evaluate_answer(
        self,
        question: Dict[str, Any],
        student_answer: str,
        skipped: bool = False,
        use_ai: bool = False,
        ai_model: Optional[str] = None,
        ai_provider: Optional[CovexcodeProvider] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a student's interview response against the question's reference answer,
        expected concepts, and rubric.
        """
        expected_concepts: List[str] = list(question.get("expected_concepts", []))
        reference_answer: str = str(question.get("reference_answer", "")).strip()
        rubric: Dict[str, str] = dict(question.get("evaluation_rubric", {}))

        # 1. Handle skipped questions explicitly
        if skipped:
            return {
                "status": "skipped",
                "skipped": True,
                "overall_score": None,
                "dimensions": {
                    "semantic_relevance": 0.0,
                    "concept_coverage": 0.0,
                    "technical_correctness": 0.0,
                    "completeness": 0.0,
                    "answer_clarity": 0.0,
                },
                "score_contributions": {k: 0.0 for k in RUBRIC_WEIGHTS},
                "semantic_similarity_raw": 0.0,
                "backend_used": "Not evaluated (Question skipped)",
                "fallback_reason": None,
                "covered_concepts": [],
                "missing_concepts": expected_concepts,
                "word_count": 0,
                "areas_for_improvement": [
                    "You skipped this question. Review the reference answer and expected concepts below before retrying."
                ],
                "reference_answer": reference_answer,
                "suggested_improved_answer": reference_answer,
                "scoring_explanation": "Question was skipped, so no practice score was recorded toward your attempted average.",
                "rubric": rubric,
            }

        raw_answer = str(student_answer or "").strip()
        content_words = _extract_content_words(raw_answer)
        total_words = len(re.findall(r"\S+", raw_answer))

        # 2. Handle empty submissions gracefully
        if not raw_answer or total_words == 0:
            return {
                "status": "empty",
                "skipped": False,
                "overall_score": 0.0,
                "dimensions": {
                    "semantic_relevance": 0.0,
                    "concept_coverage": 0.0,
                    "technical_correctness": 0.0,
                    "completeness": 0.0,
                    "answer_clarity": 0.0,
                },
                "score_contributions": {k: 0.0 for k in RUBRIC_WEIGHTS},
                "semantic_similarity_raw": 0.0,
                "backend_used": "No text submitted",
                "fallback_reason": None,
                "covered_concepts": [],
                "missing_concepts": expected_concepts,
                "word_count": 0,
                "areas_for_improvement": [
                    "No answer was submitted. Aim for a 3-5 sentence explanation covering the expected concepts below."
                ],
                "reference_answer": reference_answer,
                "suggested_improved_answer": reference_answer,
                "scoring_explanation": (
                    "Empty answers automatically receive 0/100 across all rubric dimensions "
                    "(Semantic Relevance, Concept Coverage, Technical Correctness, Completeness, and Clarity)."
                ),
                "rubric": rubric,
            }

        # 3. Concept coverage (deterministic verification against student's exact words)
        norm_answer = _normalize_text(raw_answer)
        raw_lower = raw_answer.lower()
        covered_concepts: List[str] = []
        missing_concepts: List[str] = []
        for concept in expected_concepts:
            if _concept_present(concept, norm_answer, raw_lower):
                covered_concepts.append(concept)
            else:
                missing_concepts.append(concept)

        concept_ratio = (
            len(covered_concepts) / len(expected_concepts)
            if expected_concepts
            else 0.0
        )
        concept_score = round(concept_ratio * 100.0, 1)

        # 4. Semantic similarity via all-MiniLM-L6-v2 or TF-IDF fallback
        model, backend_used, fallback_reason = self._ensure_semantic_backend()
        if model is not None:
            try:
                embeddings = model.encode(
                    [reference_answer, raw_answer],
                    normalize_embeddings=True,
                )
                raw_sim = float(np.dot(embeddings[0], embeddings[1]))
                semantic_sim = float(np.clip(raw_sim, 0.0, 1.0))
            except Exception as exc:
                backend_used = "TF-IDF + Concept Matching (Fallback)"
                fallback_reason = f"Embedding inference failed ({type(exc).__name__}: {exc}); used TF-IDF fallback."
                semantic_sim = _compute_tfidf_similarity(raw_answer, reference_answer)
        else:
            semantic_sim = _compute_tfidf_similarity(raw_answer, reference_answer)

        # Dampen purely trivial 1-3 word answers so short fragments don't inflate similarity
        length_gate = min(1.0, len(content_words) / 6.0)
        semantic_score = round(semantic_sim * (0.45 + 0.55 * length_gate) * 100.0, 1)

        # 5. Technical correctness estimate (grounded in concept overlap + semantic alignment)
        is_uncertainty_only = (
            norm_answer in _UNCERTAINTY_ONLY_PHRASES
            or (
                any(p in raw_lower for p in ("i don't know", "i do not know", "no idea"))
                and len(content_words) < 10
            )
        )
        if is_uncertainty_only:
            correctness_score = 5.0
        else:
            raw_correctness = (0.60 * concept_score) + (0.40 * semantic_score)
            if len(covered_concepts) == 0 and semantic_sim < 0.25:
                raw_correctness = min(raw_correctness, 25.0)
            correctness_score = round(float(np.clip(raw_correctness * (0.5 + 0.5 * length_gate), 0.0, 100.0)), 1)

        # 6. Completeness (concept coverage + substantive content length vs reference)
        ref_content_count = max(12, len(_extract_content_words(reference_answer)))
        target_min_content = max(14, int(ref_content_count * 0.55))
        length_ratio = min(1.0, len(content_words) / float(target_min_content))
        completeness_score = round(
            float(np.clip((0.60 * concept_ratio + 0.40 * length_ratio) * 100.0, 0.0, 100.0)),
            1,
        )

        # 7. Answer clarity (observable sentence flow, structure, and explanatory connectors)
        sentences = [s.strip() for s in re.split(r"[.!?]+", raw_answer) if s.strip()]
        num_sentences = max(1, len(sentences))
        avg_sentence_len = total_words / float(num_sentences)
        connector_hits = sum(1 for c in _EXPLANATORY_CONNECTORS if c in norm_answer)

        if total_words < 6:
            structure_score = 25.0
        elif 8.0 <= avg_sentence_len <= 32.0:
            structure_score = 80.0
        elif avg_sentence_len < 8.0:
            structure_score = 55.0
        else:
            structure_score = 62.0

        connector_bonus = min(20.0, connector_hits * 7.0)
        unique_ratio = len(set(content_words)) / float(max(1, len(content_words)))
        repetition_penalty = 15.0 if (len(content_words) >= 8 and unique_ratio < 0.45) else 0.0
        clarity_score = round(
            float(np.clip((structure_score + connector_bonus - repetition_penalty) * min(1.0, total_words / 10.0), 0.0, 100.0)),
            1,
        )

        dimensions = {
            "semantic_relevance": semantic_score,
            "concept_coverage": concept_score,
            "technical_correctness": correctness_score,
            "completeness": completeness_score,
            "answer_clarity": clarity_score,
        }

        contributions = {
            key: round(dimensions[key] * weight, 2)
            for key, weight in RUBRIC_WEIGHTS.items()
        }
        overall_score = round(float(np.clip(sum(contributions.values()), 0.0, 100.0)), 1)

        # 8. Grounded areas for improvement (never fabricated)
        areas_for_improvement: List[str] = []
        if missing_concepts:
            areas_for_improvement.append(
                f"Address missing key concepts: {', '.join(missing_concepts)}."
            )
        if total_words < 20:
            areas_for_improvement.append(
                f"Your response is brief ({total_words} words). Expand with definitions, reasoning, and a concrete example."
            )
        elif completeness_score < 65.0 and rubric.get("completeness"):
            areas_for_improvement.append(
                f"Completeness check: {rubric['completeness']}"
            )
        if connector_hits == 0 and total_words >= 8:
            areas_for_improvement.append(
                "Add causal or contrasting phrasing (e.g., 'because', 'for example', 'whereas') to make your reasoning easier to follow."
            )
        if semantic_score < 45.0 and rubric.get("technical_correctness"):
            areas_for_improvement.append(
                f"Alignment check: {rubric['technical_correctness']}"
            )
        if not areas_for_improvement:
            areas_for_improvement.append(
                "Strong response covering the core concepts clearly. Keep practicing concise real-world examples under timed conditions."
            )

        suggested_improved_answer = self._build_suggested_improved_answer(
            covered_concepts=covered_concepts,
            missing_concepts=missing_concepts,
            reference_answer=reference_answer,
            total_words=total_words,
        )

        # 9. Optional Covexcode AI Evaluation & Detailed Feedback (with seamless local fallback)
        ai_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ai_evaluation_used = False
        ai_strengths: List[str] = []
        ai_fallback_note: Optional[str] = None

        if use_ai and not self.force_fallback and not is_uncertainty_only and total_words >= 4:
            provider = ai_provider or get_ai_provider()
            if provider.is_configured():
                ai_res = provider.evaluate_interview_answer(
                    question=question,
                    student_answer=raw_answer,
                    semantic_score_baseline=semantic_score,
                    covered_concepts=covered_concepts,
                    missing_concepts=missing_concepts,
                    model=ai_model,
                )
                ai_usage = ai_res.get("usage") or ai_usage
                if ai_res.get("ok") and isinstance(ai_res.get("evaluation"), dict):
                    ev = ai_res["evaluation"]
                    ai_evaluation_used = True
                    # Blend Covexcode rubric assessment (50%) with grounded local baseline (50%)
                    for ev_key, dim_key in [
                        ("technical_correctness_score", "technical_correctness"),
                        ("completeness_score", "completeness"),
                        ("clarity_score", "answer_clarity"),
                    ]:
                        try:
                            ai_val = float(np.clip(float(ev.get(ev_key, dimensions[dim_key])), 0.0, 100.0))
                            dimensions[dim_key] = round(0.5 * dimensions[dim_key] + 0.5 * ai_val, 1)
                        except (TypeError, ValueError):
                            pass

                    contributions = {
                        key: round(dimensions[key] * weight, 2)
                        for key, weight in RUBRIC_WEIGHTS.items()
                    }
                    overall_score = round(float(np.clip(sum(contributions.values()), 0.0, 100.0)), 1)

                    raw_strengths = ev.get("strengths") or []
                    ai_strengths = [str(s).strip() for s in raw_strengths if str(s).strip()][:3]

                    raw_imp = ev.get("areas_for_improvement") or []
                    ai_imp = [str(s).strip() for s in raw_imp if str(s).strip()][:3]
                    if ai_imp:
                        if missing_concepts:
                            areas_for_improvement = [
                                f"Address missing key concepts: {', '.join(missing_concepts)}."
                            ] + [a for a in ai_imp if not all(m in a.lower() for m in missing_concepts)][:2]
                        else:
                            areas_for_improvement = ai_imp

                    if str(ev.get("suggested_improved_answer") or "").strip():
                        suggested_improved_answer = str(ev["suggested_improved_answer"]).strip()

                    backend_used = f"{backend_used} + Covexcode ({ai_res.get('model_used', provider.default_model)})"
                else:
                    ai_fallback_note = ai_res.get("error") or "Covexcode evaluation unavailable; used local ML evaluator."

        scoring_explanation = (
            f"Practice score ({overall_score}/100) combines Semantic Relevance via {backend_used} "
            f"(35% weight, raw similarity = {semantic_sim:.3f}), Expected Concept Coverage "
            f"(30% weight, {len(covered_concepts)}/{len(expected_concepts)} concepts matched), "
            "Technical Correctness alignment (15%), Completeness (10%), and Answer Clarity (10%). "
            "Note: Semantic similarity measures closeness to the reference wording and does not by itself verify factual correctness."
        )

        return {
            "status": "evaluated",
            "skipped": False,
            "overall_score": overall_score,
            "dimensions": dimensions,
            "score_contributions": contributions,
            "semantic_similarity_raw": round(semantic_sim, 4),
            "backend_used": backend_used,
            "fallback_reason": fallback_reason,
            "ai_evaluation_used": ai_evaluation_used,
            "ai_strengths": ai_strengths,
            "ai_usage": ai_usage,
            "ai_fallback_note": ai_fallback_note,
            "covered_concepts": covered_concepts,
            "missing_concepts": missing_concepts,
            "word_count": total_words,
            "areas_for_improvement": areas_for_improvement,
            "reference_answer": reference_answer,
            "suggested_improved_answer": suggested_improved_answer,
            "scoring_explanation": scoring_explanation,
            "rubric": rubric,
        }

    def _build_suggested_improved_answer(
        self,
        covered_concepts: List[str],
        missing_concepts: List[str],
        reference_answer: str,
        total_words: int,
    ) -> str:
        """
        Build an honest improvement guide anchored strictly to observed covered/missing concepts
        and the curated reference answer.
        """
        if not missing_concepts and total_words >= 20:
            return (
                f"Your answer already includes all expected concepts ({', '.join(covered_concepts)}). "
                f"To polish delivery even further, compare your structure against this concise target framing:\n\n{reference_answer}"
            )

        parts: List[str] = []
        if covered_concepts:
            parts.append(
                f"Keep the strong points you already mentioned ({', '.join(covered_concepts)}), "
            )
        else:
            parts.append("Anchor your opening sentence directly to the core definition, ")

        if missing_concepts:
            parts.append(
                f"and explicitly weave in: {', '.join(missing_concepts)}. "
            )
        parts.append(f"A complete, interview-ready model framing is:\n\n{reference_answer}")
        return "".join(parts)
