#!/usr/bin/env python3
"""[exp09-01] UltraFeedback prompts -> disjoint train / calibration / eval splits.

train prompts: used only to build proxy-labelled preference pairs for DPO.
calibration prompts: used only to estimate (A, B) from the reference policy.
eval prompts: held out for realized-gain measurement. No overlap anywhere.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


DEFAULT_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
DEFAULT_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pinned Hugging Face dataset commit.",
    )
    ap.add_argument("--source-split", default="train_prefs")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-cal", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prompt-chars", type=int, default=1500)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    if len(args.revision) != 40:
        ap.error("--revision must be a 40-character commit")

    from datasets import load_dataset
    ds = load_dataset(
        args.dataset,
        split=args.source_split,
        revision=args.revision,
    )

    seen, prompts = set(), []
    for row in ds:
        p = (row.get("prompt") or "").strip()
        if not p or len(p) > args.max_prompt_chars or len(p) < 20:
            continue
        h = hashlib.sha256(p.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        prompts.append({"prompt_id": h[:16], "prompt": p})

    need = args.n_train + args.n_cal + args.n_eval
    if len(prompts) < need:
        raise SystemExit(f"only {len(prompts)} unique prompts < required {need}")
    rng = random.Random(args.seed)
    rng.shuffle(prompts)
    splits = {
        "train": prompts[: args.n_train],
        "calibration": prompts[args.n_train : args.n_train + args.n_cal],
        "eval": prompts[args.n_train + args.n_cal : need],
    }
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    output_paths = [out / f"prompts_{name}.jsonl" for name in splits]
    output_paths.append(out / "prompts_run_spec.json")
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite prompt artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    for name, rows in splits.items():
        with (out / f"prompts_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} prompts")
    ids = [set(r["prompt_id"] for r in rows) for rows in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "split leak!"
    spec = vars(args) | {
        "dataset_revision": args.revision,
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
        "n_unique_source": len(prompts),
        "output_sha256": {
            path.name: sha256_file(path)
            for path in output_paths
            if path.name != "prompts_run_spec.json"
        },
    }
    (out / "prompts_run_spec.json").write_text(
        json.dumps(spec, indent=2),
        encoding="utf-8",
    )
    print("splits disjoint OK")


if __name__ == "__main__":
    main()
