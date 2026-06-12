"""
run_pipeline.py — end-to-end orchestrator for the unified multi-dataset framework.

Stages (run any subset via --stage):
  load      normalize all datasets -> DocPair parquet
  generate  multi-LLM generation over the pairs
  evaluate  compute every metric per row
  aggregate cross-dataset / cross-model / cross-prompt tables + plot
  ira       stratified 25-sample, export rater template
  prompt    EXP-004: run all prompt variants (one dataset) for the prompt study
  meta      EXP-004b: meta-prompting (Suzgun) + self-refinement (one dataset)

Examples:
  python run_pipeline.py --stage load generate evaluate aggregate
  python run_pipeline.py --stage prompt --dataset code_review
  python run_pipeline.py --stage ira --one-model groq
"""
from __future__ import annotations
import argparse, json
import pandas as pd
import config as C
from loaders import get_loader
from schema import DocPair


def stage_load(datasets):
    all_pairs = []
    for name in datasets:
        spec = C.DATASETS[name]
        loader = get_loader(name, spec["raw"], spec["lang"])
        if not loader.available():
            print(f"  [skip] {name}: raw not found at {spec['raw']}")
            continue
        try:
            pairs = loader.load(limit=C.MAX_PAIRS_PER_DATASET)
        except Exception as e:
            print(f"  [error] {name}: {type(e).__name__}: {e}  (skipped)")
            continue
        print(f"  {name}: {len(pairs)} pairs")
        all_pairs += pairs
    df = pd.DataFrame([p.to_dict() for p in all_pairs])
    df.to_parquet(C.DATA_PROCESSED / "all_pairs.parquet")
    df.to_json(C.DATA_PROCESSED / "all_pairs.jsonl", orient="records", lines=True)
    print(f"Total normalized pairs: {len(df)}")
    return all_pairs


def _load_pairs():
    rows = [json.loads(l) for l in open(C.DATA_PROCESSED / "all_pairs.jsonl", encoding="utf-8")]
    return [DocPair(**{k: r[k] for k in
            ("dataset","task_type","language","code_unit","reference_doc","repo","pair_id")}) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--dataset", default="code_review")
    ap.add_argument("--one-model", default="groq")
    ap.add_argument("--per-dataset", type=int, default=None)  # default: IRA_TOTAL_SAMPLES/datasets (~60)
    ap.add_argument("--results", nargs="+", default=None)   # input jsonl(s) for evaluate
    ap.add_argument("--metrics", default="metrics.csv")     # metrics csv name
    ap.add_argument("--xlsx", default="cross_comparison.xlsx")  # workbook name
    args = ap.parse_args()

    if "load" in args.stage:
        stage_load(args.datasets)

    if "generate" in args.stage:
        from generation.generate import run
        run(_load_pairs())

    if "prompt" in args.stage:                      # EXP-004 prompt study
        from generation.generate import run
        pairs = [p for p in _load_pairs() if p.dataset == args.dataset]
        run(pairs, variants=C.PROMPT_VARIANTS,
            out_path=C.DATA_RESULTS / "prompt_variants_results.jsonl")

    if "meta" in args.stage:                        # EXP-004b meta + self-refine
        from generation.run_meta import run_meta
        pairs = [p for p in _load_pairs() if p.dataset == args.dataset]
        run_meta(pairs, model_key=args.one_model)

    if "evaluate" in args.stage:
        from evaluation.evaluate import evaluate
        out_csv = C.DATA_RESULTS / args.metrics
        if args.results:
            import pandas as pd
            parts = [evaluate(results_path=C.DATA_RESULTS / r,
                              out_csv=C.DATA_RESULTS / f"_tmp_{i}.csv")
                     for i, r in enumerate(args.results)]
            pd.concat(parts, ignore_index=True).to_csv(out_csv, index=False)
            print(f"Combined metrics -> {out_csv}")
        else:
            evaluate(out_csv=out_csv)

    if "aggregate" in args.stage:
        from evaluation.aggregate import aggregate
        print(aggregate(metrics_csv=C.DATA_RESULTS / args.metrics, xlsx_name=args.xlsx))

    if "ira" in args.stage:
        from ira.sample import stratified_sample
        from ira.export_template import export
        stratified_sample(one_model=args.one_model, per_dataset=args.per_dataset)
        export()


if __name__ == "__main__":
    main()
