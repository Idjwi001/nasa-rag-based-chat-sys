#!/usr/bin/env python3
"""
Batch evaluation of the NASA RAG system with RAGAS.

For each question in the dataset this script runs the full pipeline (retrieve from
ChromaDB, then generate with OpenAI, then score with RAGAS). It prints a summary per
question and aggregate statistics (mean / min / max / std) for every metric.

Usage:
    python batch_evaluate.py --dataset evaluation_dataset.txt
    python batch_evaluate.py --dataset test_questions.json --top-k 5 --output results.json
"""

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import llm_client
import rag_client
import ragas_evaluator

FIELD_PATTERN = re.compile(r"^(Category|Mission|Question|Expected):\s*(.*)$", re.IGNORECASE)


def load_evaluation_dataset(path: str) -> List[Dict[str, str]]:
    """Load test questions from evaluation_dataset.txt (see its header) or a JSON list.

    Each item has: id, category, mission, question, expected.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if file_path.suffix.lower() == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        items = data.get("questions", data) if isinstance(data, dict) else data
        dataset = [{
            "id": str(item.get("id", f"Q{i}")),
            "category": item.get("category", "general"),
            "mission": item.get("mission", "all"),
            "question": item.get("question", ""),
            "expected": item.get("expected") or item.get("expected_answer") or item.get("reference", ""),
        } for i, item in enumerate(items, start=1)]
    else:
        dataset = []
        current = None
        field = None
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("### "):
                current = {"id": line[4:].strip(), "category": "general", "mission": "all",
                           "question": "", "expected": ""}
                dataset.append(current)
                field = None
                continue
            if current is None or (line.startswith("#") and field != "expected"):
                continue
            match = FIELD_PATTERN.match(line)
            if match:
                field = match.group(1).lower()
                current[field] = match.group(2).strip()
            elif field == "expected" and line.strip():
                current["expected"] = f"{current['expected']} {line.strip()}".strip()

    dataset = [item for item in dataset if item["question"].strip()]
    if not dataset:
        raise ValueError(f"No questions found in {path}")
    return dataset


def aggregate_scores(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute mean / min / max / std and the count of scored questions per metric."""
    values: Dict[str, List[float]] = {}
    for result in results:
        for name, score in result.get("scores", {}).items():
            if isinstance(score, (int, float)):
                values.setdefault(name, []).append(float(score))

    summary = {}
    for name, scores in values.items():
        summary[name] = {
            "mean": round(statistics.mean(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "std": round(statistics.pstdev(scores), 4),
            "n": len(scores),
        }
    return summary


def evaluate_item(item: Dict[str, str], collection, openai_key: str, model: str,
                  top_k: int, use_mission_filter: bool) -> Dict[str, Any]:
    """Run retrieval, generation and RAGAS scoring for one dataset item."""
    mission_filter = item["mission"] if use_mission_filter else None
    result: Dict[str, Any] = {**item, "answer": "", "sources": [], "scores": {}}

    try:
        docs = rag_client.retrieve_documents(collection, item["question"], top_k, mission_filter)
        documents = docs["documents"][0] if docs else []
        metadatas = docs["metadatas"][0] if docs else []
        distances = docs["distances"][0] if docs else []
        context = rag_client.format_context(documents, metadatas, distances)
        result["sources"] = [f"{m.get('mission')}/{m.get('source')}#{m.get('chunk_index')}"
                             for m in metadatas]

        result["answer"] = llm_client.generate_response(
            openai_key, item["question"], context, [], model)

        result["scores"] = ragas_evaluator.evaluate_response_quality(
            item["question"], result["answer"], documents,
            reference=item.get("expected") or None, api_key=openai_key)
    except Exception as e:
        result["scores"] = {"error": f"{type(e).__name__}: {e}"}

    return result


def print_report(results: List[Dict[str, Any]], summary: Dict[str, Dict[str, float]]) -> None:
    metric_names = list(summary.keys())

    print("\n" + "=" * 80)
    print("PER-QUESTION RESULTS")
    print("=" * 80)
    for r in results:
        print(f"\n[{r['id']}] ({r['category']}, {r['mission']}) {r['question']}")
        print(f"  Sources: {', '.join(r['sources']) or 'none'}")
        answer = " ".join(r["answer"].split())
        print(f"  Answer:  {answer[:300]}{'...' if len(answer) > 300 else ''}")
        scores = r["scores"]
        if "error" in scores:
            print(f"  Scores:  ERROR - {scores['error']}")
            continue
        shown = [f"{m}={scores[m]:.3f}" if isinstance(scores.get(m), (int, float)) else f"{m}=n/a"
                 for m in metric_names if m in scores]
        print(f"  Scores:  {', '.join(shown)}")
        for m, err in scores.get("metric_errors", {}).items():
            print(f"  Warning: {m} failed: {err[:200]}")

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)
    if not summary:
        print("No metrics were computed.")
        return
    print(f"{'metric':<22}{'mean':>8}{'min':>8}{'max':>8}{'std':>8}{'n':>5}")
    for name, s in summary.items():
        print(f"{name:<22}{s['mean']:>8.3f}{s['min']:>8.3f}{s['max']:>8.3f}{s['std']:>8.3f}{s['n']:>5}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch RAGAS evaluation for the NASA RAG system")
    parser.add_argument("--dataset", default="evaluation_dataset.txt",
                        help="evaluation_dataset.txt or a JSON list of questions")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key (defaults to OPENAI_API_KEY)")
    parser.add_argument("--model", default=llm_client.DEFAULT_MODEL, help="Chat model for answers")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks retrieved per question")
    parser.add_argument("--no-mission-filter", action="store_true",
                        help="Ignore each question's Mission field and search all missions")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="Where to write the detailed JSON results")
    args = parser.parse_args()

    if not args.openai_key:
        parser.error("an OpenAI API key is required: pass --openai-key or set OPENAI_API_KEY")
    os.environ.setdefault("CHROMA_OPENAI_API_KEY", args.openai_key)

    try:
        dataset = load_evaluation_dataset(args.dataset)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        sys.exit(f"Could not load dataset: {e}")
    print(f"Loaded {len(dataset)} questions from {args.dataset}")

    collection, ok, error = rag_client.initialize_rag_system(args.chroma_dir, args.collection_name)
    if not ok:
        sys.exit(f"Could not open collection '{args.collection_name}' in {args.chroma_dir}: {error}\n"
                 "Run the embedding pipeline first: python embedding_pipeline.py")

    results = []
    for i, item in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] {item['question']}")
        results.append(evaluate_item(item, collection, args.openai_key, args.model,
                                     args.top_k, not args.no_mission_filter))

    summary = aggregate_scores(results)
    print_report(results, summary)

    output = {
        "config": {"dataset": args.dataset, "model": args.model, "top_k": args.top_k,
                   "mission_filter": not args.no_mission_filter,
                   "collection": args.collection_name},
        "results": results,
        "aggregate": summary,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
