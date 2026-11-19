#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Streamlit chat app that retrieves context from ChromaDB, answers with OpenAI and
scores every answer in real time with RAGAS.

Run with:  streamlit run chat.py
"""

import os
from typing import Dict, List, Optional

import streamlit as st

import llm_client
import rag_client
import ragas_evaluator

RAGAS_AVAILABLE = ragas_evaluator.RAGAS_AVAILABLE

MISSION_OPTIONS = {
    "All missions": None,
    "Apollo 11": "apollo_11",
    "Apollo 13": "apollo_13",
    "Challenger (STS-51L)": "challenger",
}

METRIC_LABELS = {
    "response_relevancy": "Response Relevancy",
    "faithfulness": "Faithfulness",
    "bleu_score": "BLEU",
    "rouge_score": "ROUGE-L",
    "context_precision": "Context Precision",
}

# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)

if not RAGAS_AVAILABLE:
    st.warning("RAGAS not available. Install with: pip install ragas")


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    return rag_client.discover_chroma_backends()


@st.cache_resource(show_spinner=False)
def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        return rag_client.initialize_rag_system(chroma_dir, collection_name)
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    try:
        return rag_client.retrieve_documents(collection, query, n_results, mission_filter)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None


def format_context(documents: List[str], metadatas: List[Dict],
                   distances: Optional[List[float]] = None) -> str:
    """Format retrieved documents into context"""
    return rag_client.format_context(documents, metadatas, distances)


def generate_response(openai_key, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-4.1-mini") -> str:
    """Generate response using OpenAI with context"""
    try:
        return llm_client.generate_response(openai_key, user_message, context,
                                            conversation_history, model)
    except Exception as e:
        return f"Error generating response: {e}"


def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                              openai_key: Optional[str] = None) -> Dict:
    """Evaluate response quality using RAGAS metrics"""
    try:
        return ragas_evaluator.evaluate_response_quality(question, answer, contexts,
                                                         api_key=openai_key)
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


def display_evaluation_metrics(scores: Dict):
    """Display evaluation metrics for the latest answer in the sidebar"""
    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return

    st.sidebar.subheader("📊 Response Quality (last answer)")

    for metric_name, score in scores.items():
        if metric_name == "metric_errors":
            continue
        label = METRIC_LABELS.get(metric_name, metric_name.replace('_', ' ').title())
        if not isinstance(score, (int, float)):
            st.sidebar.metric(label=label, value="n/a")
            continue

        if score >= 0.8:
            status = "🟢"
        elif score >= 0.6:
            status = "🟠"
        else:
            status = "🔴"

        st.sidebar.metric(label=f"{status} {label}", value=f"{score:.3f}")
        st.sidebar.progress(min(max(float(score), 0.0), 1.0))

    for metric_name, message in scores.get("metric_errors", {}).items():
        st.sidebar.caption(f"⚠️ {METRIC_LABELS.get(metric_name, metric_name)}: {message[:200]}")


def render_scores_inline(scores: Optional[Dict]):
    """Compact one-line score summary shown under each answer."""
    if not scores:
        return
    if "error" in scores:
        st.caption(f"Evaluation: {scores['error']}")
        return
    parts = [f"{METRIC_LABELS.get(k, k)}: {v:.2f}" for k, v in scores.items()
             if isinstance(v, (int, float))]
    if parts:
        st.caption("📊 " + " · ".join(parts))


def render_sources(sources: Optional[List[Dict]]):
    """Expandable list of the retrieved chunks used for an answer."""
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, source in enumerate(sources, start=1):
            meta = source.get("metadata", {})
            mission = str(meta.get("mission", "unknown")).replace("_", " ").title()
            similarity = source.get("similarity")
            sim_text = f" · similarity {similarity:.3f}" if similarity is not None else ""
            st.markdown(f"**[Source {i}] {mission}** · `{meta.get('source', 'unknown')}` "
                        f"· chunk {meta.get('chunk_index', '?')}{sim_text}")
            text = source.get("text", "")
            st.text(text[:600] + ("..." if len(text) > 600 else ""))


def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Ask about **Apollo 11**, **Apollo 13** and **Challenger (STS-51L)**. "
                "Answers are grounded in NASA archive documents and scored in real time with RAGAS.")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()

        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Please run the embedding pipeline first:\n"
                    "`python embedding_pipeline.py --data-path ./data_text`")
            st.stop()

        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}
        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Choose which document collection to use for retrieval"
        )
        selected_backend = available_backends[selected_backend_key]

        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )

        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()
        else:
            # Used by the collection's persisted OpenAI embedding function for queries.
            os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

        model_choice = st.selectbox(
            "OpenAI Model",
            options=["gpt-4.1-mini", "gpt-4.1", "gpt-4o"],
            help="Choose the OpenAI model for responses"
        )

        st.subheader("🔍 Retrieval Settings")
        mission_label = st.selectbox(
            "Mission focus",
            options=list(MISSION_OPTIONS.keys()),
            help="Restrict retrieval to one mission using metadata filtering"
        )
        mission_filter = MISSION_OPTIONS[mission_label]
        n_docs = st.slider("Documents to retrieve (top-k)", 1, 10, 3)

        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=RAGAS_AVAILABLE,
                                        disabled=not RAGAS_AVAILABLE)

        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.last_evaluation = None
            st.session_state.last_contexts = []
            st.rerun()

        if st.session_state.current_backend != selected_backend_key:
            st.session_state.current_backend = selected_backend_key
            st.cache_resource.clear()

    # Initialize RAG system
    with st.spinner("Initializing RAG system..."):
        collection, success, error = initialize_rag_system(
            selected_backend["directory"],
            selected_backend["collection_name"]
        )

    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()

    if st.session_state.last_evaluation and enable_evaluation:
        display_evaluation_metrics(st.session_state.last_evaluation)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources"))
                render_scores_inline(message.get("evaluation"))

    # Chat input
    if prompt := st.chat_input("Ask about NASA space missions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        evaluation_scores = None
        sources: List[Dict] = []

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                docs_result = retrieve_documents(collection, prompt, n_docs, mission_filter)

                context = ""
                contexts_list: List[str] = []
                if docs_result and docs_result.get("documents") and docs_result["documents"][0]:
                    documents = docs_result["documents"][0]
                    metadatas = docs_result["metadatas"][0]
                    distances = docs_result.get("distances", [[]])[0]
                    context = format_context(documents, metadatas, distances)
                    contexts_list = documents
                    sources = [
                        {"text": doc, "metadata": meta,
                         "similarity": (1 - dist) if dist is not None else None}
                        for doc, meta, dist in zip(documents, metadatas,
                                                   distances or [None] * len(documents))
                    ]
                st.session_state.last_contexts = contexts_list

                # History excludes the current question, which is sent with its context.
                response = generate_response(
                    openai_key,
                    prompt,
                    context,
                    st.session_state.messages[:-1],
                    model_choice
                )
            st.markdown(response)
            render_sources(sources)

            if enable_evaluation and RAGAS_AVAILABLE:
                with st.spinner("Evaluating response quality..."):
                    evaluation_scores = evaluate_response_quality(
                        prompt, response, contexts_list, openai_key)
                    st.session_state.last_evaluation = evaluation_scores

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "evaluation": evaluation_scores,
        })
        st.rerun()


if __name__ == "__main__":
    main()
