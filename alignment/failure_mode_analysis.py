import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import OpenRouterLLMClient, LLMResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
dotenv.load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "failure_mode_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SUBSAMPLE_PATH = PROJECT_ROOT / "alignment" / "data" / "FINAL_subsample_300.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark_final"
OUTPUT_SUFFIX = "_subsample300"

CANONICAL_RUNS = {
    "google": "gemma-3-12b-it_20260728_212136.json",
    "meta-llama": "llama-3.1-8b-instruct_20260728_213400.json",
    "mistralai": "mistral-nemo_20260728_215353.json",
    "openai": "gpt-oss-20b_20260729_082724.json",
    "Qwen": "Qwen2.5-7B-Instruct-Turbo_20260728_212309.json",
}

SHORT_NAMES = {
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "Qwen2.5-7B",
    "google/gemma-3-12b-it": "Gemma-3-12B",
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "mistralai/mistral-nemo": "Mistral-Nemo",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
}

LABEL_TO_INDEX = {
    "SUPPORTED": 0,
    "PARTIALLY_SUPPORTED": 1,
    "UNSUPPORTED": 2,
    "UNCERTAIN": 3,
}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

ERROR_TYPES = {
    "semantic_ambiguity": "The claim is phrased in a way that is inherently ambiguous, making it unclear what specific factual assertion is being made or what evidence would be relevant.",
    "evidence_retrieval_failure": "The retrieved evidence chunks lack the necessary information to properly evaluate the claim — the relevant passage was not retrieved or does not exist in the document.",
    "annotation_error": "The ground-truth label appears incorrect or inconsistent with the available evidence, suggesting a mistake in the original dataset annotation.",
    "complex_inferential_chain": "The claim requires multi-step reasoning across scattered pieces of evidence, and the model failed to connect all the necessary facts correctly.",
    "inference_hallucination": "The model invented facts, made unsupported logical leaps, or cited evidence that does not actually support its conclusion.",
}

JUDGE_MODEL = "google/gemma-3-12b-it"
MAX_CONCURRENCY = 100

PROMPT_TEMPLATE = """You are an expert in fact-checking error analysis. Given a claim, its ground-truth label, the model's prediction, the model's reasoning, and the retrieved evidence, classify the error into exactly ONE of the following types.

Error Types:
{error_type_definitions}

Respond in valid JSON with exactly two fields:
  - "error_type": one of the 5 error type keys above
  - "reasoning": a compact 1-2 sentence explanation of why this error occurred

---
Claim: "{claim}"
Ground-truth label: {true_label}
Model prediction: {predicted_label}
Model reasoning: {reasoning}
{evidence_block}
---
Return only the JSON object, no other text."""


def model_runs() -> list[dict]:
    runs = []
    for org, fname in CANONICAL_RUNS.items():
        org_dir = RESULTS_DIR / f"benchmark_results_{org}"
        result_path = org_dir / fname
        data = json.load(open(result_path, "r", encoding="utf-8"))
        model_name = data.get("meta", {}).get("llm", {}).get("model", "unknown")
        metrics_path = org_dir / f"{model_name.rsplit('/', 1)[-1]}_subsample_300_metrics.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
        runs.append({
            "org": org,
            "model": model_name,
            "short_name": SHORT_NAMES.get(model_name, model_name),
            "result_path": result_path,
            "data": data,
            "metrics": metrics,
        })
    return runs


def load_subsample() -> list[dict]:
    data = json.load(open(SUBSAMPLE_PATH, "r", encoding="utf-8"))
    return data if isinstance(data, list) else data.get("instances", [])


def extract_misclassified(force: bool = False) -> list[dict]:
    subsample = load_subsample()
    subsample_ids = {inst["instance_id"] for inst in subsample}
    subsample_map = {inst["instance_id"]: inst for inst in subsample}

    summaries = []
    for run in model_runs():
        org_dir = RESULTS_DIR / f"benchmark_results_{run['org']}"
        model_slug = run["model"].replace("/", "_")
        out_path = org_dir / f"{model_slug}{OUTPUT_SUFFIX}_misclassified.json"
        if out_path.exists() and not force:
            logger.info(f"Exists, skipping: {out_path.name}")
            summaries.append(json.load(open(out_path, "r", encoding="utf-8")))
            continue

        results = run["data"].get("results", [])
        sub_results = [r for r in results if r.get("instance_id") in subsample_ids]
        misclassified = [r for r in sub_results if r.get("predicted_label") != r.get("true_alignment")]

        for r in misclassified:
            sub = subsample_map.get(r.get("instance_id"), {})
            true_outputs = sub.get("true_outputs", {})
            r["expert_rationale"] = true_outputs.get("expert_rationale")
            r["true_existence"] = true_outputs.get("true_existence")
            r["true_hallucination_category"] = true_outputs.get("true_hallucination_category")
            r["retrieved_evidences"] = sub.get("retrieved_evidences")

        groups = defaultdict(list)
        for r in misclassified:
            groups[r["true_alignment"]].append(r)
        label_distribution = {INDEX_TO_LABEL.get(k, str(k)): len(v) for k, v in sorted(groups.items())}

        output = {
            "model": run["model"],
            "mode": run["data"].get("meta", {}).get("mode"),
            "accuracy_on_sub300": run["metrics"].get("accuracy"),
            "total_misclassified": len(misclassified),
            "label_distribution": label_distribution,
            "samples": {INDEX_TO_LABEL.get(k, str(k)): v for k, v in sorted(groups.items())},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(misclassified)} misclassified -> {out_path.name}")
        summaries.append(output)

    return summaries


def _build_evidence_block(sample: dict) -> str:
    chunks = sample.get("extractive_chunks") or []
    if not chunks:
        return "Retrieved evidence: (none)"
    lines = []
    for i, c in enumerate(chunks[:5]):
        text = c.get("extractive_text", "")
        lines.append(f"Chunk {i + 1}: {text[:500]}")
    return "Retrieved evidence:\n" + "\n\n".join(lines)


def _parse_json_response(raw: str) -> dict | None:
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


def _build_prompt(sample: dict) -> str:
    true_label = sample.get("true_alignment")
    predicted_label = sample.get("predicted_label")
    return PROMPT_TEMPLATE.format(
        error_type_definitions="\n".join(f'  - "{k}": {v}' for k, v in ERROR_TYPES.items()),
        claim=sample.get("claim_text", ""),
        true_label=INDEX_TO_LABEL.get(true_label, str(true_label)),
        predicted_label=INDEX_TO_LABEL.get(predicted_label, str(predicted_label)),
        reasoning=sample.get("reasoning", ""),
        evidence_block=_build_evidence_block(sample),
    )


async def classify_one(client: OpenRouterLLMClient, sem: asyncio.Semaphore, sample: dict) -> dict:
    async with sem:
        try:
            response: LLMResponse = await client.generate(_build_prompt(sample), temperature=0.2)
            result = _parse_json_response(response.content)
            if result is None:
                sample["error_type"] = "parse_failure"
                sample["error_reasoning"] = ""
            else:
                sample["error_type"] = result.get("error_type", "unknown")
                sample["error_reasoning"] = result.get("reasoning", "")
        except Exception as e:
            logger.error(f"Failed to analyze instance {sample.get('instance_id')}: {e}")
            sample["error_type"] = "unknown"
            sample["error_reasoning"] = ""
    return sample


async def classify_failure_modes(force: bool = False, api_key: str | None = None) -> None:
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    client = OpenRouterLLMClient(model=JUDGE_MODEL, temperature=0.2, api_key=api_key)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    for run in model_runs():
        org_dir = RESULTS_DIR / f"benchmark_results_{run['org']}"
        model_slug = run["model"].replace("/", "_")
        mis_path = org_dir / f"{model_slug}{OUTPUT_SUFFIX}_misclassified.json"
        out_path = org_dir / f"{model_slug}{OUTPUT_SUFFIX}_error_analysis.json"

        if not mis_path.exists():
            logger.warning(f"Missing misclassified file, run extract first: {mis_path.name}")
            continue

        mis = json.load(open(mis_path, "r", encoding="utf-8"))
        samples = []
        for group in mis.get("samples", {}).values():
            samples.extend(group)

        if not force and out_path.exists():
            existing = json.load(open(out_path, "r", encoding="utf-8"))
            done_map = {}
            for group in existing.get("samples", {}).values():
                for s in group:
                    done_map[s.get("instance_id")] = s
            todo = [s for s in samples if s.get("instance_id") not in done_map]
            pending = [s for s in samples if s.get("instance_id") in done_map and done_map[s.get("instance_id")].get("error_type") not in ERROR_TYPES]
            if not todo and not pending:
                logger.info(f"All {len(samples)} already analyzed: {out_path.name}")
                continue
            samples = todo + pending
            logger.info(f"Resuming {model_slug}: {len(samples)} remaining of {mis['total_misclassified']}")
        else:
            logger.info(f"Analyzing {len(samples)} misclassified for {model_slug}")

        if not samples:
            continue

        processed = 0
        for i in range(0, len(samples), MAX_CONCURRENCY):
            batch = samples[i:i + MAX_CONCURRENCY]
            await asyncio.gather(*[classify_one(client, sem, s) for s in batch])
            processed += len(batch)
            logger.info(f"  {model_slug}: {processed}/{len(samples)} processed")

        results = samples
        if not force and out_path.exists():
            existing = json.load(open(out_path, "r", encoding="utf-8"))
            merged = []
            for group in existing.get("samples", {}).values():
                merged.extend(group)
            merged_by_id = {s.get("instance_id"): s for s in merged}
            for s in results:
                merged_by_id[s.get("instance_id")] = s
            results = list(merged_by_id.values())

        grouped = defaultdict(list)
        for s in results:
            lbl = INDEX_TO_LABEL.get(s.get("true_alignment"), "UNKNOWN")
            grouped[lbl].append(s)

        output = {
            "meta": {
                "created_at": datetime.now().isoformat(),
                "judge_model": JUDGE_MODEL,
                "model": mis["model"],
                "mode": mis.get("mode"),
                "total_analyzed": len(results),
                "total_misclassified": mis["total_misclassified"],
            },
            "samples": {k: v for k, v in sorted(grouped.items())},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved error analysis -> {out_path.name}")


def load_analysis_files() -> list[dict]:
    analyses = []
    for run in model_runs():
        org_dir = RESULTS_DIR / f"benchmark_results_{run['org']}"
        model_slug = run["model"].replace("/", "_")
        ana_path = org_dir / f"{model_slug}{OUTPUT_SUFFIX}_error_analysis.json"
        if not ana_path.exists():
            logger.warning(f"Missing analysis file: {ana_path.name}")
            continue
        data = json.load(open(ana_path, "r", encoding="utf-8"))
        samples = []
        for group in data.get("samples", {}).values():
            samples.extend(group)
        analyses.append({
            "org": run["org"],
            "model": run["model"],
            "short_name": run["short_name"],
            "metrics": run["metrics"],
            "total_analyzed": data["meta"].get("total_analyzed", len(samples)),
            "total_misclassified": data["meta"].get("total_misclassified"),
            "samples": samples,
        })
    return analyses


def _fmt_pct(count: int, total: int) -> str:
    if total == 0:
        return "-"
    return f"{count / total * 100:.1f}%"


def generate_report() -> Path:
    analyses = load_analysis_files()
    if not analyses:
        raise RuntimeError("No analysis files found; run extract + classify first")

    out_path = RESULTS_DIR / "failure_mode_report.md"
    lines = []
    lines.append("# Failure-Mode Analysis of Misclassified Instances (FINAL_subsample_300)")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Judge model: `{JUDGE_MODEL}` (OpenRouter, temperature 0.2)")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Model | Accuracy (sub-300) | Misclassified | Rate | Analyzed |")
    lines.append("|---|---|---|---|---|")
    for a in analyses:
        acc = a["metrics"].get("accuracy")
        acc_s = f"{acc:.4f}" if acc is not None else "-"
        total = a["metrics"].get("count", 300)
        lines.append(
            f"| {a['short_name']} | {acc_s} | {a['total_misclassified']} | "
            f"{_fmt_pct(a['total_misclassified'], total)} | {a['total_analyzed']} |"
        )
    lines.append("")

    lines.append("## Failure-Mode Definitions")
    lines.append("")
    for k, v in ERROR_TYPES.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Error-Type Distribution per Model")
    lines.append("")
    header = ["Model"] + list(ERROR_TYPES.keys()) + ["parse_failure", "unknown"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    all_counts = Counter()
    all_total = 0
    for a in analyses:
        counts = Counter(s.get("error_type", "unknown") for s in a["samples"])
        all_counts.update(counts)
        all_total += len(a["samples"])
        row = [a["short_name"]]
        for et in header[1:]:
            c = counts.get(et, 0)
            row.append(f"{c} ({_fmt_pct(c, len(a['samples']))})")
        lines.append("| " + " | ".join(row) + " |")
    row = ["**All models**"]
    for et in header[1:]:
        c = all_counts.get(et, 0)
        row.append(f"**{c}** ({_fmt_pct(c, all_total)})")
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-Label Error-Type Breakdown")
    lines.append("")
    for a in analyses:
        lines.append(f"### {a['short_name']}")
        lines.append("")
        lines.append("| True label | " + " | ".join(ERROR_TYPES.keys()) + " | Total |")
        lines.append("|" + "---|" * (len(ERROR_TYPES) + 2))
        for label in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]:
            group = [s for s in a["samples"] if INDEX_TO_LABEL.get(s.get("true_alignment")) == label]
            if not group:
                continue
            counts = Counter(s.get("error_type", "unknown") for s in group)
            row = [label]
            for et in ERROR_TYPES:
                row.append(str(counts.get(et, 0)))
            row.append(str(len(group)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Error Overlap Across Models")
    lines.append("")
    lines.append("Instances misclassified by multiple models (top 15 by agreement count):")
    lines.append("")
    per_inst = defaultdict(list)
    for a in analyses:
        for s in a["samples"]:
            per_inst[s["instance_id"]].append(a["short_name"])
    multi = {iid: models for iid, models in per_inst.items() if len(models) > 1}
    lines.append(f"Instances missed by **1** model: {sum(1 for m in per_inst.values() if len(m) == 1)}, "
                 f"**2**: {sum(1 for m in per_inst.values() if len(m) == 2)}, "
                 f"**3**: {sum(1 for m in per_inst.values() if len(m) == 3)}, "
                 f"**4**: {sum(1 for m in per_inst.values() if len(m) == 4)}, "
                 f"**5**: {sum(1 for m in per_inst.values() if len(m) == 5)}")
    lines.append("")
    lines.append("| Instance ID | Models that failed |")
    lines.append("|---|---|")
    for iid, models in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:15]:
        lines.append(f"| {iid} | {', '.join(models)} |")
    lines.append("")

    lines.append("## Representative Examples")
    lines.append("")
    for a in analyses:
        by_type = defaultdict(list)
        for s in a["samples"]:
            by_type[s.get("error_type", "unknown")].append(s)
        for et in ERROR_TYPES:
            pool = by_type.get(et, [])
            if not pool:
                continue
            lines.append(f"### {a['short_name']} — {et} ({len(pool)})")
            lines.append("")
            for s in pool[:2]:
                true_lbl = INDEX_TO_LABEL.get(s.get("true_alignment"), "?")
                pred_lbl = s.get("predicted_classification", INDEX_TO_LABEL.get(s.get("predicted_label"), "?"))
                claim = s.get("claim_text", "")
                lines.append(f"- **Instance {s.get('instance_id')}**: true=`{true_lbl}` predicted=`{pred_lbl}`")
                lines.append(f"  - Claim: {claim[:300]}")
                lines.append(f"  - Judge rationale: {s.get('error_reasoning', '')}")
            lines.append("")

    lines.append("## Key Takeaways")
    lines.append("")
    dominant = all_counts.most_common()
    if dominant:
        top_type, top_count = dominant[0]
        lines.append(f"- The most frequent failure mode across all models is **{top_type}** "
                     f"({top_count}/{all_total}, {_fmt_pct(top_count, all_total)}).")
    lines.append("- (Add narrative conclusions here after reviewing the tables above.)")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report saved -> {out_path}")
    return out_path


async def main_async(args: argparse.Namespace) -> None:
    if args.step in ("extract", "all"):
        extract_misclassified(force=args.force)
    if args.step in ("classify", "all"):
        await classify_failure_modes(force=args.force, api_key=args.api_key)
    if args.step in ("report", "all"):
        generate_report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure-mode analysis of sub-300 misclassified instances")
    parser.add_argument("step", choices=["extract", "classify", "report", "all"])
    parser.add_argument("--force", action="store_true", help="Re-run even if output files exist")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key (defaults to OPENROUTER_API_KEY env)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
