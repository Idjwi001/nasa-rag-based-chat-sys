"""OpenAI chat client for the NASA mission intelligence assistant."""

import os
from typing import Dict, List, Optional

from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1-mini"
VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"

# Only the most recent turns are sent to keep each request focused and cheap.
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = """You are a NASA mission operations specialist and space-history expert \
with deep knowledge of the Apollo 11, Apollo 13 and Space Shuttle Challenger (STS-51L) missions.

You answer questions using the excerpts from NASA archives (air-to-ground transcripts, \
public affairs commentary, technical debriefs, flight plans and mission reports) that are \
supplied to you as context.

Rules:
1. Ground every factual claim in the provided context. Do not invent names, times, \
numbers or events that the context does not support.
2. Cite your sources inline using the bracketed source numbers from the context, \
for example [Source 2]. When helpful, mention the mission and document type.
3. If the context is missing, incomplete or does not answer the question, say so \
plainly (for example: "The retrieved documents do not cover this"). You may then add \
widely known background, but label it clearly as general knowledge, not archive content.
4. Transcripts are noisy OCR/speech-to-text output. Interpret them carefully, and quote \
short key phrases (such as crew call-outs) when they make the answer clearer.
5. Be accurate, clear and concise. Use mission elapsed times (GET) or dates when the \
context provides them. Use short paragraphs or bullet points for multi-part answers."""


def configure_openai_endpoint(api_key: Optional[str]) -> None:
    """Point OpenAI clients at the Vocareum proxy when a Udacity ``voc-`` key is used.

    Sets OPENAI_BASE_URL, which the openai SDK, langchain-openai and ChromaDB's
    OpenAI embedding function all honour. An explicitly set OPENAI_BASE_URL wins.
    """
    if api_key and api_key.startswith("voc-") and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = VOCAREUM_BASE_URL


def _trim_history(conversation_history: Optional[List[Dict]],
                  max_messages: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
    """Keep only valid user/assistant turns (role + content) from the recent history."""
    if not conversation_history:
        return []

    cleaned = []
    for message in conversation_history:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})

    return cleaned[-max_messages:] if max_messages > 0 else []


def build_messages(user_message: str, context: str,
                   conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
    """Assemble the message list: system prompt, recent history, then context + question.

    The retrieved context is attached only to the current question, so earlier
    turns' context is never re-sent.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_trim_history(conversation_history))

    if context and context.strip():
        user_content = (
            f"{context.strip()}\n\n"
            "Using the context above, answer the following question. "
            "Cite sources as [Source N].\n\n"
            f"Question: {user_message}"
        )
    else:
        user_content = (
            "No relevant documents were retrieved from the NASA archive for this question. "
            "Say that the archive context is insufficient, then answer only with clearly "
            "labelled general knowledge if you can.\n\n"
            f"Question: {user_message}"
        )

    messages.append({"role": "user", "content": user_content})
    return messages


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = DEFAULT_MODEL,
                      temperature: float = 0.2, max_tokens: int = 800) -> str:
    """Generate a grounded answer with OpenAI using the retrieved context.

    Args:
        openai_key: OpenAI API key.
        user_message: The user's current question.
        context: Formatted context string from the RAG client.
        conversation_history: Prior turns as dicts with "role" and "content".
        model: OpenAI chat model name.
        temperature: Sampling temperature (low values favour factual answers).
        max_tokens: Maximum tokens in the answer.

    Returns:
        The model's answer text.
    """
    if not openai_key:
        raise ValueError("An OpenAI API key is required.")
    if not user_message or not user_message.strip():
        raise ValueError("The question must not be empty.")

    messages = build_messages(user_message, context, conversation_history)

    configure_openai_endpoint(openai_key)
    client = OpenAI(api_key=openai_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return (completion.choices[0].message.content or "").strip()
