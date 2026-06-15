from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dataloader.class_policy import apply_class_policy
from dataloader.config_loader import discover_batches, duplicate_rules, load_config, resolve_path
from dataloader.dataset_write import clean_output, write_dataset_outputs
from dataloader.duplicate_resolve import resolve_duplicates
from dataloader.manifest_write import write_manifests
from dataloader.source_clean import parse_and_clean_pairs
from dataloader.source_scan import scan_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RT-DETRv4 training dataset with source cleanup and class policy.")
    parser.add_argument("--config", default="configs/dataset_build.yml", help="Dataset build config.")
    parser.add_argument("--class-policy", default="configs/class_policy_raw_classes.csv", help="Raw class policy CSV.")
    parser.add_argument("--source", action="append", default=[], help="Override/add raw source directory.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only; do not write generated dataset files.")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of hard-linking where possible.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = resolve_path(args.output_dir or config.get("output_dir", "datasets"))
    class_policy_path = resolve_path(args.class_policy)
    batches = discover_batches(config, output_dir, args.source)
    exclude_dirs = set(config.get("exclude_dirs") or [])

    scan = scan_sources(batches, exclude_dirs)
    parsed = parse_and_clean_pairs(scan.pairs, scan.unlabel_images, scan.issues)
    deduped = resolve_duplicates(parsed.samples, duplicate_rules(config))
    policy = apply_class_policy(deduped.samples, class_policy_path)
    issues = [*parsed.issues, *deduped.issues, *policy.issues]

    pipeline_stats = {
        "mode": "dry-run" if args.dry_run else "apply",
        "config": str(config_path.resolve()),
        "class_policy": str(class_policy_path),
        "output_dir": str(output_dir),
        "raw_images": scan.raw_images,
        "raw_labels": scan.raw_labels,
        "paired_before_clean": len(scan.pairs),
        "source_clean_samples": len(parsed.samples),
        "deduplicated_samples": len(deduped.samples),
        "train_samples": len(policy.samples),
        "hold_samples": len(policy.hold_samples),
        "unknown_class_samples": len(policy.unknown_samples),
        "dropped_empty_samples": len(policy.dropped_samples),
        "repeated_image_label_conflict_groups": len(deduped.conflicts),
        "source_clean_stats": dict(sorted(parsed.stats.items())),
        "deduplicate_stats": dict(sorted(deduped.stats.items())),
        "class_policy_stats": dict(sorted(policy.stats.items())),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not args.dry_run:
        clean_output(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        doublecheck_rows = write_dataset_outputs(
            output_dir=output_dir,
            samples=policy.samples,
            unlabels=parsed.unlabel_images,
            conflicts=deduped.conflicts,
            hold_samples=policy.hold_samples,
            unknown_samples=policy.unknown_samples,
            dropped_samples=policy.dropped_samples,
            config=config,
            copy=args.copy,
        )
        write_manifests(output_dir, policy.samples, parsed.unlabel_images, issues, config, doublecheck_rows, pipeline_stats)
    print(json.dumps(pipeline_stats, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("\nDry run only. Re-run without --dry-run to rebuild outputs.")


if __name__ == "__main__":
    main()
