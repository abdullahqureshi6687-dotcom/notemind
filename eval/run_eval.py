"""Measure NoteMind's retrieval quality and refusal behaviour.

Usage (from the project root):

    python eval/run_eval.py                       # retrieval + threshold refusal (no API calls)
    python eval/run_eval.py --with-llm            # also measure LLM refusals and citations
    python eval/run_eval.py --with-llm --llm-sample 18   # balanced subset that fits a free tier
    python eval/run_eval.py --chunk-sizes 500 800 --modes faiss hybrid --k 1 3 5

What it measures
----------------
Retrieval (answerable questions only). A retrieved chunk is relevant if its
(file, page) is one of the question's expected pages.

* Recall@k - fraction of questions with at least one relevant chunk in the top k.
* MRR      - mean of 1 / (rank of the first relevant chunk), 0 if none in the top k.

These are computed for every combination of chunk size and search mode, so
embeddings-only vs BM25 vs hybrid and different chunk sizes can be compared on
the same questions. Each score comes with a 95% bootstrap confidence interval:
with ~40 questions, small differences are noise, and the interval shows that.

Refusal. NoteMind refuses without calling the LLM when the best cosine
similarity is below ``retrieval.min_similarity``. The script sweeps that
threshold and reports, for each value, how many unanswerable questions are
correctly refused and how many answerable questions are wrongly refused.
With ``--with-llm`` it also runs the full pipeline (threshold + LLM) and
checks whether answers cite an expected page.

Outputs (in eval/results/): results.md, results.json, recall_by_chunk_size.png,
refusal_threshold.png.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from notemind.config import PROJECT_ROOT, AppConfig, load_config, setup_logging
from notemind.embedder import Embedder
from notemind.generator import extract_citations
from notemind.llm_client import LLMError, LLMRateLimitError, format_wait
from notemind.pipeline import RAGPipeline

logger = logging.getLogger("run_eval")

MODE_LABELS = {"heap": "Embeddings (heap)", "faiss": "Embeddings (FAISS)", "bm25": "BM25", "hybrid": "Hybrid (RRF)"}
# Validated categorical palette (slots 1-3 pass all-pairs CVD checks); see README.
MODE_COLORS = {"faiss": "#2a78d6", "heap": "#2a78d6", "bm25": "#eb6834", "hybrid": "#1baf7a"}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MAX_CONSECUTIVE_LLM_ERRORS = 3  # stop the LLM run instead of burning quota on a dead provider


# --------------------------------------------------------------------------- data


@dataclass
class EvalQuestion:
    id: str
    question: str
    answerable: bool
    expected: set[tuple[str, int]] = field(default_factory=set)


def load_eval_set(path: Path) -> tuple[list[EvalQuestion], Path]:
    """Read and validate eval_set.json. Returns questions and the PDF folder."""
    data = json.loads(path.read_text(encoding="utf-8"))
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for item in data["questions"]:
        qid = item["id"]
        if qid in seen:
            raise ValueError(f"Duplicate question id: {qid}")
        seen.add(qid)
        expected = {(e["file"], int(e["page"])) for e in item.get("expected", [])}
        answerable = bool(item.get("answerable", True))
        if answerable and not expected:
            raise ValueError(f"{qid}: answerable questions need at least one expected page")
        if not answerable and expected:
            raise ValueError(f"{qid}: unanswerable questions must have an empty 'expected' list")
        questions.append(EvalQuestion(qid, item["question"], answerable, expected))
    pdf_dir = PROJECT_ROOT / data.get("pdf_dir", "data/sample_pdfs")
    return questions, pdf_dir


def check_labels(questions: list[EvalQuestion], pipeline: RAGPipeline) -> None:
    """Fail early if a label points at a file or page that doesn't exist."""
    pages = {d.source: d.num_pages for d in pipeline.documents}
    problems = [
        f"{q.id}: {file} p.{page}"
        for q in questions
        for file, page in q.expected
        if file not in pages or not 1 <= page <= pages[file]
    ]
    if problems:
        raise ValueError("Eval labels point at missing files/pages:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- metrics


def first_relevant_rank(retrieved: list[tuple[str, int]], expected: set[tuple[str, int]]) -> int | None:
    """1-based rank of the first relevant chunk, or None."""
    for rank, item in enumerate(retrieved, start=1):
        if item in expected:
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks) if ranks else 0.0


def bootstrap_ci(values: list[float], iterations: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean of ``values``."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iterations))
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


# --------------------------------------------------------------------------- experiment


def build_pipeline(config: AppConfig, embedder: Embedder, pdf_dir: Path, chunk_size: int, overlap: int,
                   max_k: int) -> RAGPipeline:
    cfg = config.with_overrides(
        chunking={"chunk_size": chunk_size, "chunk_overlap": overlap},
        retrieval={"top_k": max_k, "candidate_k": max(config.retrieval.candidate_k, max_k)},
    )
    pipeline = RAGPipeline(cfg, embedder=embedder)
    reports = pipeline.add_folder(pdf_dir)
    if not reports:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    stats = pipeline.build_index()
    if stats is None:
        raise RuntimeError(f"No text could be indexed from {pdf_dir}")
    logger.info("chunk_size=%d overlap=%d -> %d chunks (%s)", chunk_size, overlap, stats.num_chunks,
                "cache" if stats.from_cache else "built")
    return pipeline


def evaluate_retrieval(pipeline: RAGPipeline, questions: list[EvalQuestion], modes: list[str],
                       ks: list[int]) -> tuple[dict, dict[str, float]]:
    """Run every answerable question in every mode. Also returns max cosine per question."""
    max_k = max(ks)
    answerable = [q for q in questions if q.answerable]
    results: dict[str, dict] = {}
    max_sim: dict[str, float] = {}

    for q in questions:  # best cosine is mode-independent; compute once per question
        output = pipeline.retrieve(q.question, top_k=1, mode="faiss")
        max_sim[q.id] = float(output.max_similarity or 0.0)

    for mode in modes:
        ranks: list[int | None] = []
        latencies: list[float] = []
        per_question = []
        for q in answerable:
            started = time.perf_counter()
            output = pipeline.retrieve(q.question, top_k=max_k, mode=mode)
            latencies.append((time.perf_counter() - started) * 1000)
            retrieved = [(r.chunk.source, r.chunk.page_number) for r in output.results]
            rank = first_relevant_rank(retrieved, q.expected)
            ranks.append(rank)
            per_question.append({"id": q.id, "rank": rank, "retrieved": [f"{s} p.{p}" for s, p in retrieved]})

        hits = {k: [1.0 if r is not None and r <= k else 0.0 for r in ranks] for k in ks}
        reciprocal = [1.0 / r if r is not None else 0.0 for r in ranks]
        results[mode] = {
            "recall": {k: recall_at_k(ranks, k) for k in ks},
            "recall_ci": bootstrap_ci(hits[max_k]),
            "mrr": mrr(ranks),
            "mrr_ci": bootstrap_ci(reciprocal),
            "latency_ms": statistics.median(latencies) if latencies else 0.0,
            "questions": per_question,
        }
    return results, max_sim


def threshold_sweep(questions: list[EvalQuestion], max_sim: dict[str, float],
                    thresholds: list[float]) -> list[dict]:
    """Correct-refusal and false-refusal rates for each similarity threshold."""
    answerable = [max_sim[q.id] for q in questions if q.answerable]
    unanswerable = [max_sim[q.id] for q in questions if not q.answerable]
    rows = []
    for t in thresholds:
        correct = sum(s < t for s in unanswerable) / len(unanswerable) if unanswerable else 0.0
        false = sum(s < t for s in answerable) / len(answerable) if answerable else 0.0
        rows.append({"threshold": round(t, 3), "correct_refusal": correct, "false_refusal": false})
    return rows


def stratified_sample(questions: list[EvalQuestion], n: int, seed: int = 0) -> list[EvalQuestion]:
    """Up to n questions, keeping the answerable/unanswerable ratio (at least 2 of each kind)."""
    if n >= len(questions):
        return list(questions)
    rng = random.Random(seed)
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    n_unans = min(len(unanswerable), max(2, round(n * len(unanswerable) / len(questions))))
    n_ans = min(len(answerable), n - n_unans)
    return rng.sample(answerable, n_ans) + rng.sample(unanswerable, n_unans)


def recommend_threshold(sweep: list[dict], max_false_refusal: float = 0.0) -> float | None:
    """Highest threshold whose false-refusal rate is at most ``max_false_refusal``.

    False refusals cost more than missed refusals: a falsely refused question
    never reaches the LLM, while an unanswerable one that passes the threshold
    can still be refused by the LLM. So the threshold is chosen for safety, not
    for the best balance.
    """
    allowed = [row["threshold"] for row in sweep if row["false_refusal"] <= max_false_refusal + 1e-12]
    return max(allowed) if allowed else None


def evaluate_with_llm(pipeline: RAGPipeline, questions: list[EvalQuestion], delay: float) -> dict:
    """Full pipeline (threshold + LLM) on every question. Costs up to one API call per question.

    Stops early when the provider reports an exhausted quota (for example a
    daily free-tier limit), because every further call would fail too.
    """
    rows = []
    stopped_reason = None
    consecutive_errors = 0
    for i, q in enumerate(questions, start=1):
        row: dict = {"id": q.id, "answerable": q.answerable}
        try:
            answer = pipeline.ask(q.question)
            cited = set(extract_citations(answer.text))
            row.update(
                refused=answer.refused,
                reason=answer.refusal_reason,
                cites_expected=bool(cited & q.expected) if q.answerable and not answer.refused else None,
                has_citation=bool(cited),
                invalid_citations=len(answer.invalid_citations),
            )
        except LLMError as exc:
            row.update(error=f"{type(exc).__name__}: {exc}")
            logger.warning("%s failed: %s", q.id, str(exc)[:200])
            consecutive_errors += 1
            if isinstance(exc, LLMRateLimitError) and exc.quota_exhausted:
                stopped_reason = (f"Stopped after {i} of {len(questions)} questions: the provider's quota "
                                  f"is used up (resets in about {format_wait(exc.retry_after)}).")
            elif consecutive_errors >= MAX_CONSECUTIVE_LLM_ERRORS:
                stopped_reason = (f"Stopped after {i} of {len(questions)} questions: {consecutive_errors} calls "
                                  f"in a row failed ({type(exc).__name__}). The provider or model is probably "
                                  "overloaded or unavailable; try another model or try later.")
            if stopped_reason:
                rows.append(row)
                print(f"  {stopped_reason}")
                break
        else:
            consecutive_errors = 0
        rows.append(row)
        print(f"  LLM eval {i}/{len(questions)} {q.id}: "
              f"{'error' if 'error' in row else ('refused' if row['refused'] else 'answered')}")
        if delay and i < len(questions):
            time.sleep(delay)  # stay under free-tier rate limits

    ok = [r for r in rows if "error" not in r]
    unans = [r for r in ok if not r["answerable"]]
    ans = [r for r in ok if r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    return {
        "n_ok": len(ok),
        "n_errors": len(rows) - len(ok),
        "n_planned": len(questions),
        "stopped_reason": stopped_reason,
        "unanswerable_refused": sum(r["refused"] for r in unans) / len(unans) if unans else 0.0,
        "refused_by_threshold": sum(r["reason"] == "low_similarity" for r in unans),
        "refused_by_llm": sum(r["reason"] == "llm" for r in unans),
        "answerable_wrongly_refused": sum(r["refused"] for r in ans) / len(ans) if ans else 0.0,
        "answers_citing_expected_page": (sum(bool(r["cites_expected"]) for r in answered) / len(answered)
                                         if answered else 0.0),
        "answers_with_invalid_citations": (sum(r["invalid_citations"] > 0 for r in answered) / len(answered)
                                           if answered else 0.0),
        "rows": rows,
    }


# --------------------------------------------------------------------------- reporting


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_markdown(path: Path, meta: dict, retrieval: dict, ks: list[int], sweep: list[dict],
                   configured_threshold: float, llm: dict | None, default_size: int,
                   questions: list[EvalQuestion]) -> str:
    max_k = max(ks)
    lines = [
        "# NoteMind evaluation results",
        "",
        f"- Date: {meta['date']}",
        f"- Embedding model: `{meta['model']}`",
        f"- PDFs: {', '.join(meta['pdfs'])}",
        f"- Questions: {meta['n_answerable']} answerable, {meta['n_unanswerable']} unanswerable",
        "- Chunk overlap: min(150, chunk_size / 3) characters",
        "",
        f"## Retrieval (Recall@k and MRR@{max_k}, answerable questions)",
        "",
        "| Chunk size | Mode | " + " | ".join(f"Recall@{k}" for k in ks) + f" | 95% CI (R@{max_k}) | MRR | 95% CI (MRR) | Median latency |",
        "|---|---|" + "---|" * len(ks) + "---|---|---|---|",
    ]
    for size, by_mode in retrieval.items():
        for mode, r in by_mode.items():
            lo, hi = r["recall_ci"]
            mlo, mhi = r["mrr_ci"]
            lines.append(
                f"| {size} | {MODE_LABELS[mode]} | "
                + " | ".join(pct(r["recall"][k]) for k in ks)
                + f" | {pct(lo)}-{pct(hi)} | {r['mrr']:.3f} | {mlo:.3f}-{mhi:.3f} | {r['latency_ms']:.1f} ms |"
            )

    safe = recommend_threshold(sweep)
    lines += [
        "",
        "## Refusal by similarity threshold (no LLM call)",
        "",
        f"Chunk size {default_size}. A question is refused when its best cosine similarity is below the threshold.",
        f"Configured threshold: **{configured_threshold:.2f}**. Highest threshold that wrongly refuses no "
        f"answerable question: **{safe:.2f}**." if safe is not None else
        f"Configured threshold: **{configured_threshold:.2f}**. Every tested threshold wrongly refuses "
        "some answerable questions.",
        "",
        "The threshold is a cheap first filter, so it should not refuse real questions: a wrong refusal "
        "here hides an answer the LLM never sees, while an off-topic question that slips through can "
        "still be refused by the LLM.",
        "",
        "| Threshold | Unanswerable correctly refused | Answerable wrongly refused |",
        "|---|---|---|",
    ]
    for row in sweep:
        marker = " (configured)" if abs(row["threshold"] - configured_threshold) < 1e-9 else ""
        lines.append(f"| {row['threshold']:.2f}{marker} | {pct(row['correct_refusal'])} | {pct(row['false_refusal'])} |")

    if llm:
        lines += [
            "",
            "## Full pipeline with LLM (threshold + model refusal)",
            "",
            f"Questions evaluated: {llm['n_ok']} of {llm['n_planned']} planned (errors: {llm['n_errors']}).",
            *([f"**Partial run.** {llm['stopped_reason']}"] if llm.get("stopped_reason") else []),
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Unanswerable questions refused | {pct(llm['unanswerable_refused'])} "
            f"({llm['refused_by_threshold']} by threshold, {llm['refused_by_llm']} by the LLM) |",
            f"| Answerable questions wrongly refused | {pct(llm['answerable_wrongly_refused'])} |",
            f"| Answers citing an expected page | {pct(llm['answers_citing_expected_page'])} |",
            f"| Answers with an invented citation | {pct(llm['answers_with_invalid_citations'])} |",
        ]

    # Error analysis: questions the default hybrid configuration missed.
    if default_size in retrieval and "hybrid" in retrieval[default_size]:
        text_by_id = {q.id: q.question for q in questions}
        misses = [row for row in retrieval[default_size]["hybrid"]["questions"]
                  if row["rank"] is None or row["rank"] > 3]
        lines += ["", f"## Error analysis (hybrid, chunk size {default_size}, not in top 3)", ""]
        if not misses:
            lines.append("None - every answerable question had a relevant chunk in the top 3.")
        for row in misses:
            where = f"rank {row['rank']}" if row["rank"] else f"not in top {max_k}"
            lines.append(f"- **{row['id']}** ({where}): {text_by_id[row['id']]}  \n"
                         f"  retrieved: {', '.join(row['retrieved'][:3])}")

    lines += ["", "_Generated by `eval/run_eval.py`._", ""]
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def plot_recall(path: Path, retrieval: dict, k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = list(retrieval)
    modes = list(next(iter(retrieval.values())))
    width = 0.8 / len(modes)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, mode in enumerate(modes):
        xs = [j + (i - (len(modes) - 1) / 2) * width for j in range(len(sizes))]
        values = [100 * retrieval[s][mode]["recall"][k] for s in sizes]
        bars = ax.bar(xs, values, width=width * 0.92, color=MODE_COLORS[mode], label=MODE_LABELS[mode],
                      edgecolor=SURFACE, linewidth=1)
        for bar, value in zip(bars, values):  # value labels: identity is never colour-only
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f}", ha="center",
                    va="bottom", fontsize=8, color=INK_MUTED)
    ax.set_xticks(range(len(sizes)), [f"{s} chars" for s in sizes], color=INK)
    ax.set_ylim(0, 108)
    ax.set_ylabel(f"Recall@{k} (%)", color=INK)
    ax.set_xlabel("Chunk size", color=INK)
    ax.set_title(f"Recall@{k} by chunk size and search mode", color=INK, loc="left", fontsize=12, pad=30)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    # Legend in a row above the plot so it never covers the data.
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(modes), fontsize=9,
              handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_refusal(path: Path, sweep: list[dict], configured: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["threshold"] for r in sweep]
    correct = [100 * r["correct_refusal"] for r in sweep]
    false = [100 * r["false_refusal"] for r in sweep]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(xs, correct, color="#2a78d6", linewidth=2, marker="o", markersize=5,
            label="Unanswerable correctly refused (higher is better)")
    ax.plot(xs, false, color="#eb6834", linewidth=2, marker="o", markersize=5,
            label="Answerable wrongly refused (lower is better)")
    ax.axvline(configured, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(configured + 0.005, 2, f"configured {configured:.2f}", color=INK_MUTED, fontsize=8,
            va="bottom", ha="left")
    ax.set_ylim(-3, 105)
    ax.set_xlabel("min_similarity threshold (best cosine score)", color=INK)
    ax.set_ylabel("Questions (%)", color=INK)
    ax.set_title("Refusal trade-off by similarity threshold", color=INK, loc="left", fontsize=12, pad=30)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9,
              handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NoteMind retrieval and refusals.")
    parser.add_argument("--eval-set", type=Path, default=None, help="Path to eval_set.json")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="Override the PDF folder in the eval set")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 500, 800, 1000])
    parser.add_argument("--modes", nargs="+", default=["faiss", "bm25", "hybrid"],
                        choices=["heap", "faiss", "bm25", "hybrid"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], help="Values of k for Recall@k")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[round(0.05 * i, 2) for i in range(1, 11)])
    parser.add_argument("--with-llm", action="store_true", help="Also run the LLM (one API call per question)")
    parser.add_argument("--llm-delay", type=float, default=4.0, help="Seconds between LLM calls")
    parser.add_argument("--llm-sample", type=int, default=None,
                        help="Evaluate only N questions (balanced answerable/unanswerable) to fit a free-tier quota")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: eval/results)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = load_config()
    setup_logging(config.log_level)

    eval_path = args.eval_set or config.paths.eval_set
    questions, pdf_dir = load_eval_set(eval_path)
    pdf_dir = args.pdf_dir or pdf_dir
    out_dir = args.out or config.paths.eval_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(args.k))
    sizes = sorted(set(args.chunk_sizes))
    default_size = config.chunking.chunk_size if config.chunking.chunk_size in sizes else sizes[len(sizes) // 2]

    n_ans = sum(q.answerable for q in questions)
    print(f"Eval set: {n_ans} answerable + {len(questions) - n_ans} unanswerable questions; PDFs from {pdf_dir}")

    embedder = Embedder(config.embedding)
    retrieval: dict[int, dict] = {}
    default_pipeline: RAGPipeline | None = None
    max_sim_default: dict[str, float] = {}

    for size in sizes:
        overlap = min(150, size // 3)
        pipeline = build_pipeline(config, embedder, pdf_dir, size, overlap, max(ks))
        if size == sizes[0]:
            check_labels(questions, pipeline)
        print(f"Evaluating chunk size {size} ({pipeline.num_chunks} chunks) ...")
        retrieval[size], max_sim = evaluate_retrieval(pipeline, questions, args.modes, ks)
        if size == default_size:
            default_pipeline, max_sim_default = pipeline, max_sim

    sweep = threshold_sweep(questions, max_sim_default, sorted(set(args.thresholds)
                                                                | {config.retrieval.min_similarity}))

    llm_result = None
    if args.with_llm and default_pipeline is not None:
        llm_questions = stratified_sample(questions, args.llm_sample) if args.llm_sample else questions
        print(f"Running the full pipeline with {config.generation.provider}/{config.generation.model} "
              f"on {len(llm_questions)} questions (delay {args.llm_delay}s) ...")
        llm_result = evaluate_with_llm(default_pipeline, llm_questions, args.llm_delay)

    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": config.embedding.model_name,
        "pdfs": [d.source for d in default_pipeline.documents] if default_pipeline else [],
        "n_answerable": n_ans,
        "n_unanswerable": len(questions) - n_ans,
    }
    markdown = write_markdown(out_dir / "results.md", meta, retrieval, ks, sweep,
                              config.retrieval.min_similarity, llm_result, default_size, questions)
    (out_dir / "results.json").write_text(json.dumps({
        "meta": meta,
        "retrieval": {str(size): modes for size, modes in retrieval.items()},
        "max_similarity": max_sim_default,
        "threshold_sweep": sweep,
        "llm": llm_result,
    }, indent=2, default=list), encoding="utf-8")

    try:
        plot_recall(out_dir / "recall_by_chunk_size.png", retrieval, max(ks))
        plot_refusal(out_dir / "refusal_threshold.png", sweep, config.retrieval.min_similarity)
    except ImportError:
        print("matplotlib is not installed (pip install -r requirements-dev.txt); skipping charts.")

    print("\n" + markdown)
    print(f"Saved results to {out_dir}")
    return 0


if __name__ == "__main__":
    from notemind.config import run_cli

    raise SystemExit(run_cli(main))
