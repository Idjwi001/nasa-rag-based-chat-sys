"""Real-time RAGAS evaluation of (question, retrieved contexts, answer) triples.

Metrics always computed (no reference answer needed):
    - response_relevancy  (RAGAS ResponseRelevancy, LLM + embeddings)
    - faithfulness        (RAGAS Faithfulness, LLM)

Extra metrics computed when a reference answer is supplied (e.g. batch evaluation):
    - bleu_score          (RAGAS BleuScore, needs sacrebleu)
    - rouge_score         (RAGAS RougeScore, ROUGE-L F1, needs rouge_score)

Extra metric computed when reference contexts are supplied:
    - context_precision   (RAGAS NonLLMContextPrecisionWithReference, needs rapidfuzz)
"""

import math
import os
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Optional

# Opt out of RAGAS usage analytics unless the user has configured it explicitly.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from llm_client import configure_openai_endpoint

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import (BleuScore, Faithfulness, NonLLMContextPrecisionWithReference,
                               ResponseRelevancy, RougeScore)
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# The legacy metric classes used here emit deprecation warnings in ragas 0.4.x.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ragas(\.|$)")

EVALUATOR_MODEL = "gpt-4.1-mini"
EVALUATOR_EMBEDDING_MODEL = "text-embedding-3-small"

CORE_METRICS = ("response_relevancy", "faithfulness")
REFERENCE_METRICS = ("bleu_score", "rouge_score")
CONTEXT_METRICS = ("context_precision",)


@lru_cache(maxsize=4)
def _evaluator_models(api_key: str):
    """Create (and cache) the evaluator LLM and embeddings for an API key."""
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=EVALUATOR_MODEL, api_key=api_key, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        # Inputs are short, so skip the tiktoken length check (avoids a tokenizer download).
        OpenAIEmbeddings(model=EVALUATOR_EMBEDDING_MODEL, api_key=api_key,
                         check_embedding_ctx_length=False)
    )
    return evaluator_llm, evaluator_embeddings


def _validate_inputs(question: Any, answer: Any, contexts: Any) -> Optional[str]:
    """Return an error message for malformed inputs, or None if they are usable."""
    if not isinstance(question, str) or not question.strip():
        return "Question is empty or not a string."
    if not isinstance(answer, str) or not answer.strip():
        return "Answer is empty or not a string."
    if answer.strip().lower().startswith("error generating response"):
        return "The answer is an error message, so it was not evaluated."
    if not isinstance(contexts, list) or not contexts:
        return "No retrieved context was provided, so faithfulness cannot be measured."
    if not all(isinstance(c, str) for c in contexts):
        return "Every context must be a string."
    if not any(c.strip() for c in contexts):
        return "All retrieved contexts are empty."
    return None


def _clean_score(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else round(value, 4)


def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                              reference: Optional[str] = None,
                              reference_contexts: Optional[List[str]] = None,
                              api_key: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate response quality using RAGAS metrics.

    Args:
        question: The user's question.
        answer: The model's answer.
        contexts: Retrieved context chunks (list of strings; a single string is accepted).
        reference: Optional expected answer, enables BLEU and ROUGE.
        reference_contexts: Optional ground-truth contexts, enables context precision.
        api_key: OpenAI key for the evaluator; defaults to OPENAI_API_KEY.

    Returns:
        {metric_name: score} with scores in [0, 1]. A metric that fails is reported as
        None and explained under "metric_errors". Unusable input returns {"error": message}.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available. Install with: pip install ragas"}

    if isinstance(contexts, str):
        contexts = [contexts]
    problem = _validate_inputs(question, answer, contexts)
    if problem:
        return {"error": problem}
    contexts = [c for c in contexts if c.strip()]

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "An OpenAI API key is required for RAGAS evaluation."}

    configure_openai_endpoint(api_key)
    try:
        evaluator_llm, evaluator_embeddings = _evaluator_models(api_key)
    except Exception as e:
        return {"error": f"Could not create evaluator models: {e}"}

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference or None,
        reference_contexts=reference_contexts or None,
    )

    metrics = {
        "response_relevancy": ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "faithfulness": Faithfulness(llm=evaluator_llm),
    }
    if reference:
        metrics["bleu_score"] = BleuScore()
        metrics["rouge_score"] = RougeScore()
    if reference_contexts:
        metrics["context_precision"] = NonLLMContextPrecisionWithReference()

    scores: Dict[str, Any] = {}
    metric_errors: Dict[str, str] = {}
    for name, metric in metrics.items():
        try:
            scores[name] = _clean_score(metric.single_turn_score(sample))
            if scores[name] is None:
                metric_errors[name] = "metric returned no score"
        except Exception as e:
            scores[name] = None
            metric_errors[name] = f"{type(e).__name__}: {e}"

    if metric_errors:
        scores["metric_errors"] = metric_errors
    return scores
