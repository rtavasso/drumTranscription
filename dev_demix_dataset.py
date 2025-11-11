#!/usr/bin/env python
"""
Batch demix a dataset directory using Demucs.

Features:
- Recursively scans an input directory for audio files.
- Mirrors the directory structure under the output directory.
- Saves stems as <basename>_<stem>.wav to retain clear mapping.
- Writes a JSONL manifest mapping inputs -> outputs.
- Skips files whose stem outputs already exist (optional).

Example:
    python demucs/dev_demix_dataset.py data/babyslakh_16k \
        --out stems --model htdemucs --device cuda --progress
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Dict

from demucs.api import Separator, save_audio


DEFAULT_EXTS = [
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma", ".mp4", ".m4b",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recursively demix all audio files in a dataset directory."
    )
    p.add_argument(
        "root",
        type=Path,
        help="Root directory of the dataset to demix (recursively scanned).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("stems"),
        help="Output directory root where stems will be written (mirrors input structure).",
    )
    p.add_argument(
        "--model",
        default="htdemucs",
        help="Pretrained model name (see demucs README for options).",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Force device. Defaults to CUDA if available, else CPU.",
    )
    p.add_argument(
        "--ext",
        action="append",
        dest="exts",
        default=None,
        help=(
            "Audio file extension to include (case-insensitive, include dot). "
            "Repeatable. Defaults to common audio: " + ", ".join(DEFAULT_EXTS)
        ),
    )
    p.add_argument(
        "--name",
        default=None,
        help=(
            "If set, only process files whose basename matches this exactly (e.g., 'mix.wav')."
        ),
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show Demucs progress bars during separation.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files if all expected stem outputs already exist.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Path for JSONL manifest. Defaults to <out>/manifest.jsonl. "
            "Each line: {input, outputs:{stem:path,...}, samplerate}"
        ),
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of files to process (debugging).",
    )
    return p.parse_args()


def _gather_files(root: Path, exts: Iterable[str], name: str | None, out_root: Path) -> List[Path]:
    root = root.resolve()
    out_root = out_root.resolve()
    exts = {e.lower() for e in exts}
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Do not accidentally reprocess outputs if out is inside root
        try:
            _ = path.resolve().relative_to(out_root)
            # It's inside the output tree — skip
            continue
        except Exception:
            pass
        if name is not None and path.name != name:
            continue
        if path.suffix.lower() not in exts:
            continue
        files.append(path)
    files.sort()
    return files


def _expected_outputs(out_dir: Path, base: str, stems: Iterable[str]) -> List[Path]:
    return [out_dir / f"{base}_{stem}.wav" for stem in stems]


def main() -> None:
    args = _parse_args()
    if not args.root.exists() or not args.root.is_dir():
        raise SystemExit(f"Input root '{args.root}' must exist and be a directory.")

    exts = args.exts if args.exts else DEFAULT_EXTS
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.out / "manifest.jsonl")

    sep_kwargs = {"model": args.model, "progress": args.progress}
    if args.device:
        sep_kwargs["device"] = args.device
    separator = Separator(**sep_kwargs)

    # Informative device print — parameters may still be on CPU until inference starts.
    try:
        target_dev = getattr(separator, "_device", None)
        if target_dev is None:
            target_dev = next(separator.model.parameters()).device
        print(f"Using device: {target_dev}")
    except Exception:
        pass

    # Build file list
    files = _gather_files(args.root, exts, args.name, args.out)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        print("No matching audio files found.")
        return

    stems_names = list(separator.model.sources)
    total = len(files)
    processed = 0
    errors: Dict[str, str] = {}

    # Open manifest for writing (overwrite)
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx, in_path in enumerate(files, 1):
            rel = in_path.resolve().relative_to(args.root.resolve())
            out_dir = (args.out / rel.parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            expected = _expected_outputs(out_dir, in_path.stem, stems_names)

            if args.skip_existing and all(p.exists() for p in expected):
                print(f"[{idx}/{total}] SKIP existing: {rel}")
                # Write manifest line pointing to existing outputs
                mf.write(
                    json.dumps(
                        {
                            "input": str(in_path),
                            "outputs": {
                                stem: str(out_dir / f"{in_path.stem}_{stem}.wav")
                                for stem in stems_names
                            },
                            "samplerate": separator.samplerate,
                            "skipped": True,
                        }
                    )
                    + "\n"
                )
                continue

            print(f"[{idx}/{total}] Demixing: {rel}")
            try:
                separated = separator.separate_audio_file(in_path)[1]
                for stem_name, audio in separated.items():
                    stem_path = out_dir / f"{in_path.stem}_{stem_name}.wav"
                    save_audio(
                        audio,
                        stem_path,
                        samplerate=separator.samplerate,
                        bits_per_sample=16,
                        as_float=False,
                    )
                mf.write(
                    json.dumps(
                        {
                            "input": str(in_path),
                            "outputs": {
                                stem: str(out_dir / f"{in_path.stem}_{stem}.wav")
                                for stem in stems_names
                            },
                            "samplerate": separator.samplerate,
                            "skipped": False,
                        }
                    )
                    + "\n"
                )
                processed += 1
            except KeyboardInterrupt:
                print("Interrupted by user.")
                break
            except Exception as e:  # continue on error
                msg = f"{type(e).__name__}: {e}"
                print(f"  ERROR processing {rel}: {msg}")
                errors[str(rel)] = msg

    print(f"Done. Processed {processed}/{total} files. Manifest: {manifest_path}")
    if errors:
        errors_path = args.out / "errors.json"
        with open(errors_path, "w", encoding="utf-8") as ef:
            json.dump(errors, ef, indent=2)
        print(f"There were {len(errors)} errors. See {errors_path}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Allow piping to tools like `head` without noisy tracebacks
        try:
            sys.stderr.close()
        except Exception:
            pass
        try:
            sys.stdout.close()
        except Exception:
            pass
