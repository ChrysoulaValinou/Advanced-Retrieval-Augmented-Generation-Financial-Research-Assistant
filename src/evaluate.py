"""
src/evaluate.py
===============
Task 4 – LLM-as-a-Judge Evaluation

Implements an automated evaluation loop over eval_dataset.jsonl:

  4.1  Evaluation dataset  – 15 QA pairs (5+ from JSON/Excel sources)
  4.2  Automated grading loop:
       • For each question, run the full RAG pipeline
       • Call a Judge LLM (temperature=0) to score three dimensions:
           – Faithfulness     : is the answer grounded in retrieved context?
           – Answer Relevance : does it address the question?
           – Context Precision: was the right context retrieved?
       • Judge returns {"score": float[0,1], "reason": str}
       • Report mean score across all 15 questions

Usage
-----
    python -m src.evaluate                  # run full evaluation
    python -m src.evaluate --sample 5       # run on first 5 questions only
    python -m src.evaluate --strategy dense # override retrieval strategy
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)

EVAL_DATASET_PATH = Path(__file__).parent.parent / "eval_dataset.jsonl"
RESULTS_PATH      = Path(__file__).parent.parent / "eval_results.json"

# ── Judge prompt ──────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are a strict and impartial evaluation judge for a financial research RAG system.

You will be given:
  1. A QUESTION asked by the user
  2. The RETRIEVED CONTEXT chunks that the RAG system used
  3. The GENERATED ANSWER produced by the RAG system
  4. The GROUND TRUTH ANSWER (the correct answer)

Evaluate the generated answer on THREE dimensions:

  A. FAITHFULNESS (0.0 – 1.0)
     Is every claim in the generated answer supported by the retrieved context?
     Score 1.0 if all claims are grounded. Score 0.0 if the answer contains
     hallucinated facts not found in the context. Score in between proportionally.

  B. ANSWER RELEVANCE (0.0 – 1.0)
     Does the generated answer actually address the question asked?
     Compare with the ground truth. Score 1.0 if it fully answers the question
     with correct key facts. Score 0.0 if it is off-topic or misses the point.

  C. CONTEXT PRECISION (0.0 – 1.0)
     Was the right context retrieved? Does the retrieved context contain the
     information needed to answer the question?
     Score 1.0 if the relevant facts are clearly present in the context.
     Score 0.0 if the context is entirely irrelevant to the question.

Compute the FINAL SCORE as the average of the three dimension scores:
    final_score = (faithfulness + answer_relevance + context_precision) / 3

You MUST respond with ONLY a valid JSON object in this exact format (no other text):
{{
  "faithfulness": <float between 0.0 and 1.0>,
  "answer_relevance": <float between 0.0 and 1.0>,
  "context_precision": <float between 0.0 and 1.0>,
  "score": <float between 0.0 and 1.0, the average of the three>,
  "reason": "<one concise sentence explaining the main strength or weakness>"
}}

--- QUESTION ---
{question}

--- RETRIEVED CONTEXT ---
{context}

--- GENERATED ANSWER ---
{generated_answer}

--- GROUND TRUTH ANSWER ---
{ground_truth_answer}
"""

# ── RAG prompt (used during answer generation) ────────────────────────────────────

RAG_PROMPT = """\
You are a financial research assistant. Answer the user's question using ONLY
the provided context. If the context does not contain enough information to
answer, say "I don't have enough information in the retrieved context."

Be concise and accurate. Cite specific numbers and facts when available.

--- CONTEXT ---
{context}

--- QUESTION ---
{question}

Answer:"""


# ══════════════════════════════════════════════════════════════════════════════════
# Core evaluation classes
# ══════════════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    """
    Runs the full LLM-as-a-Judge evaluation loop.

    Parameters
    ----------
    retriever        : Retriever instance from src.retrieval
    judge_model      : OpenAI model used for judging (temperature ALWAYS 0)
    generator_model  : OpenAI model used for answer generation
    retrieval_strategy : "hybrid" | "dense" | "sparse"
    top_k            : number of chunks to retrieve per question
    """

    def __init__(
        self,
        retriever,
        judge_model:         str = "gemini-2.5-flash",
        generator_model:     str = "gemini-2.5-flash",
        retrieval_strategy:  str = "hybrid",
        top_k:               int = 4,
    ) -> None:
        self._retriever   = retriever
        # Αρχικοποιούμε τα μοντέλα της Google
        self._judge_llm   = ChatGoogleGenerativeAI(model=judge_model, temperature=0.0)
        self._gen_llm     = ChatGoogleGenerativeAI(model=generator_model, temperature=0.2)
        self._strategy    = retrieval_strategy
        self._top_k       = top_k

    # ── Step 1: retrieve context ──────────────────────────────────────────────────

    def retrieve(self, question: str) -> tuple[list, str]:
        """
        Run the configured retrieval strategy and return
        (list_of_RetrievedDoc, formatted_context_string).
        """
        strategy = self._strategy.lower()
        if strategy == "hybrid":
            docs = self._retriever.hybrid(question, k=self._top_k)
        elif strategy == "dense":
            docs = self._retriever.dense(question,  k=self._top_k)
        elif strategy == "sparse":
            docs = self._retriever.sparse(question, k=self._top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        context_parts = []
        for d in docs:
            text_content = getattr(d, 'text', getattr(d, 'content', getattr(d, 'preview', '')))
            context_parts.append(text_content)
        context = "\n\n".join(context_parts)
        return docs, context

    # ── Step 2: generate answer ───────────────────────────────────────────────────

    def generate_answer(self, question: str, context: str) -> str:
        """Call the generator LLM with the RAG prompt."""
        prompt = RAG_PROMPT.format(context=context, question=question)
        messages = [HumanMessage(content=prompt)]
        
        response = self._gen_llm.invoke(messages)
        return response.content.strip()

    # ── Step 3: judge the answer ──────────────────────────────────────────────────

    def judge(
        self,
        question:           str,
        context:            str,
        generated_answer:   str,
        ground_truth:       str,
    ) -> dict:
        prompt = JUDGE_PROMPT.format(
            question=question,
            context=context,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth,
        )

        messages = [HumanMessage(content=prompt)]
        response = self._judge_llm.invoke(messages)
        raw = response.content.strip()

        # Parse JSON response — strip markdown fences if present
        raw_clean = raw.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(raw_clean)
        except json.JSONDecodeError as exc:
            log.error("Judge returned invalid JSON: %s\nRaw: %s", exc, raw)
            result = {
                "faithfulness":      0.0,
                "answer_relevance":  0.0,
                "context_precision": 0.0,
                "score":             0.0,
                "reason":            f"Judge parsing error: {exc}",
            }

        # Clamp all scores to [0, 1]
        for key in ("faithfulness", "answer_relevance", "context_precision", "score"):
            if key in result:
                result[key] = max(0.0, min(1.0, float(result[key])))

        return result

    # ── Full evaluation loop ──────────────────────────────────────────────────────

    def run(
        self,
        dataset_path: Path = EVAL_DATASET_PATH,
        sample:       Optional[int] = None,
        delay_secs:   float = 0.5,
    ) -> dict:
        """
        Run the complete evaluation loop.

        Parameters
        ----------
        dataset_path : path to eval_dataset.jsonl
        sample       : if set, evaluate only the first N questions
        delay_secs   : sleep between API calls to avoid rate-limiting

        Returns
        -------
        dict with keys:
          results       – list of per-question result dicts
          mean_score    – float, average score across all questions
          accuracy_pct  – mean_score * 100, formatted for README
          low_scorers   – questions with score < 0.5
        """
        # Load dataset
        with dataset_path.open(encoding="utf-8") as fh:
            items = [json.loads(line) for line in fh if line.strip()]

        if sample:
            items = items[:sample]

        log.info("Running evaluation on %d questions (strategy=%s, top_k=%d) …",
                 len(items), self._strategy, self._top_k)

        results = []
        for i, item in enumerate(items, start=1):
            question     = item["question"]
            ground_truth = item["ground_truth_answer"]
            source_file  = item.get("source_file", "unknown")

            log.info("[%2d/%d] %s", i, len(items), question[:70])

            # Step 1: retrieve
            retrieved_docs, context = self.retrieve(question)
            retrieved_sources = [
                {"source": d.metadata.get("source","?"),
                 "category": d.metadata.get("document_category","?")}
                for d in retrieved_docs
            ]

            # Step 2: generate
            generated_answer = self.generate_answer(question, context)

            # Step 3: judge
            judgment = self.judge(question, context, generated_answer, ground_truth)

            result = {
                "question_id":       i,
                "question":          question,
                "source_file":       source_file,
                "ground_truth":      ground_truth,
                "generated_answer":  generated_answer,
                "retrieved_sources": retrieved_sources,
                "faithfulness":      judgment.get("faithfulness", 0.0),
                "answer_relevance":  judgment.get("answer_relevance", 0.0),
                "context_precision": judgment.get("context_precision", 0.0),
                "score":             judgment.get("score", 0.0),
                "reason":            judgment.get("reason", ""),
            }
            results.append(result)

            log.info("    → score=%.2f | F=%.2f AR=%.2f CP=%.2f | %s",
                     result["score"],
                     result["faithfulness"],
                     result["answer_relevance"],
                     result["context_precision"],
                     result["reason"][:80])

            if delay_secs > 0:
                time.sleep(delay_secs)

        # Aggregate
        mean_score   = sum(r["score"] for r in results) / len(results)
        accuracy_pct = mean_score * 100
        low_scorers  = [r for r in results if r["score"] < 0.5]

        summary = {
            "results":         results,
            "mean_score":      round(mean_score, 4),
            "accuracy_pct":    round(accuracy_pct, 1),
            "total_questions": len(results),
            "low_scorers":     low_scorers,
            "strategy":        self._strategy,
            "judge_model":     self._judge_model,
            "generator_model": self._gen_model,
            "top_k":           self._top_k,
        }

        # Persist results
        with RESULTS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        log.info("Results saved → %s", RESULTS_PATH)

        return summary


# ══════════════════════════════════════════════════════════════════════════════════
# Report printer
# ══════════════════════════════════════════════════════════════════════════════════

def print_report(summary: dict) -> None:
    """Pretty-print evaluation results to stdout."""
    sep  = "=" * 72
    sep2 = "-" * 72

    print(f"\n{sep}")
    print("  LLM-AS-A-JUDGE EVALUATION REPORT")
    print(sep)
    print(f"  Strategy      : {summary['strategy']}")
    print(f"  Judge model   : {summary['judge_model']} (temperature=0)")
    print(f"  Generator     : {summary['generator_model']}")
    print(f"  Questions     : {summary['total_questions']}")
    print(f"  Top-K chunks  : {summary['top_k']}")
    print(sep2)

    # Per-question table
    print(f"  {'#':>2}  {'Score':>6}  {'F':>5}  {'AR':>5}  {'CP':>5}  Source / Reason")
    print(sep2)
    for r in summary["results"]:
        flag = " ⚠" if r["score"] < 0.5 else ""
        print(f"  {r['question_id']:>2}  {r['score']:>6.3f}  "
              f"{r['faithfulness']:>5.2f}  {r['answer_relevance']:>5.2f}  "
              f"{r['context_precision']:>5.2f}  "
              f"[{r['source_file']}]{flag}")
        print(f"        Reason: {r['reason'][:65]}")
    print(sep2)

    # Overall
    print(f"\n  MEAN SCORE   : {summary['mean_score']:.4f}")
    print(f"  ACCURACY     : {summary['accuracy_pct']:.1f}%")

    # Low scorers diagnosis
    if summary["low_scorers"]:
        print(f"\n{sep2}")
        print("  LOW SCORERS (score < 0.5) — Failure Diagnosis")
        print(sep2)
        for r in summary["low_scorers"]:
            print(f"\n  Q{r['question_id']}: {r['question'][:70]}")
            print(f"  Score: {r['score']:.3f} | Source: {r['source_file']}")
            print(f"  Retrieved from: {[s['source'] for s in r['retrieved_sources']]}")
            print(f"  Reason: {r['reason']}")

            # Diagnose: retrieval failure vs hallucination
            correct_src  = r["source_file"]
            retrieved_srcs = [s["source"] for s in r["retrieved_sources"]]
            if not any(correct_src in s for s in retrieved_srcs):
                diagnosis = ("RETRIEVAL FAILURE — the correct source document was NOT "
                             "in the retrieved chunks. Fix: improve chunking strategy "
                             "or add more specific metadata filtering.")
            elif r["faithfulness"] < 0.4:
                diagnosis = ("LLM HALLUCINATION — the correct context WAS retrieved but "
                             "the generator model added claims not supported by the context. "
                             "Fix: lower generator temperature or add stricter RAG prompt.")
            else:
                diagnosis = ("PARTIAL FAILURE — context was retrieved but answer was incomplete "
                             "or imprecise. Fix: increase top_k or refine the answer prompt.")
            print(f"  Diagnosis: {diagnosis}")
    else:
        print("\n  No questions scored below 0.5 ✓")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from src.ingestion  import build_vectorstore, build_documents, DATA_DIR
    from src.retrieval  import Retriever

    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge evaluation")
    parser.add_argument("--sample",   type=int, default=None,
                        help="Evaluate only the first N questions")
    parser.add_argument("--strategy", default="hybrid",
                        choices=["hybrid", "dense", "sparse"],
                        help="Retrieval strategy (default: hybrid)")
    parser.add_argument("--top-k",   type=int, default=4,
                        help="Number of chunks to retrieve per question (default: 4)")
    parser.add_argument("--judge",   default="gemini-2.5-flash",
                        help="Judge LLM model (default: gemini-2.5-flash)")
    parser.add_argument("--no-delay", action="store_true",
                        help="Skip delay between API calls")
    args = parser.parse_args()

    # Build retriever
    vs        = build_vectorstore()
    docs      = build_documents(DATA_DIR)
    retriever = Retriever(vs, docs)

    # Run evaluation
    evaluator = RAGEvaluator(
        retriever=retriever,
        judge_model=args.judge,
        retrieval_strategy=args.strategy,
        top_k=args.top_k,
    )
    summary = evaluator.run(
        sample=args.sample,
        delay_secs=5.0
    )
    print_report(summary)