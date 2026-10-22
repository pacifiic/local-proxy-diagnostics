#!/usr/bin/env python3
"""Build every publication display group and merge their manifests."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GROUPS = ["bon", "flex", "ab", "dpo", "fig1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    out = args.outdir.resolve()
    out.relative_to(ROOT)  # Manifest outputs use portable repository-relative paths.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SOURCE_DATE_EPOCH="0", CAMERA_READY_OUT=str(out))
    for group in GROUPS:
        print(f"== {group}", flush=True)
        subprocess.run([sys.executable, str(HERE / f"{group}.py")], cwd=HERE, check=True, env=env)
    merged = {}
    for group in GROUPS:
        records = json.loads((out / f"manifest_{group}.json").read_text())
        clash = set(records) & set(merged)
        if clash:
            raise SystemExit(f"duplicate display names: {sorted(clash)}")
        merged.update(records)
    registered = {path for record in merged.values() for path in record["outputs"]}
    prior_manifest = out / "MANIFEST.json"
    prior_registered = set()
    if prior_manifest.exists():
        prior = json.loads(prior_manifest.read_text())
        prior_registered = {path for record in prior.values() for path in record["outputs"]}
    stale = sorted(path for path in prior_registered - registered if (ROOT / path).exists())
    if stale:
        raise SystemExit("stale outputs from this builder:\n  " + "\n  ".join(stale))
    (out / "MANIFEST.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    lines = ["# Display Manifest", "",
             "| Display | Placement | Description | Inputs | Outputs |", "|---|---|---|---|---|"]
    for name, record in sorted(merged.items()):
        inputs = "<br>".join(f"`{path}`" for path in sorted(record["inputs"]))
        outputs = "<br>".join(f"`{Path(path).name}`" for path in sorted(record["outputs"]))
        lines.append(f"| `{name}` | {record['placement']} | {record['description']} | {inputs} | {outputs} |")
    (out / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"{len(merged)} displays built")


if __name__ == "__main__":
    main()
