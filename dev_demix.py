#!/usr/bin/env python
"""
Tiny helper to demix a single track with Demucs.

Example:
    python demucs/dev_demix.py path/to/song.wav --out stems --model htdemucs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from demucs.api import Separator, save_audio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demix one audio file and dump stems.")
    parser.add_argument("track", type=Path, help="Path to the audio file to demix.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("stems"),
        help="Directory where the separated stems will be written.",
    )
    parser.add_argument(
        "--model",
        default="htdemucs",
        help="Pretrained model name (see demucs/README.md for the list).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Force device. Defaults to CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar while running the separation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.track.exists():
        raise SystemExit(f"Input file '{args.track}' does not exist.")

    sep_kwargs = {"model": args.model, "progress": args.progress}
    if args.device:
        sep_kwargs["device"] = args.device
    separator = Separator(**sep_kwargs)
    # Print the actual compute device Demucs will use. The model parameters are
    # kept on CPU until inference begins, so checking parameter.device is misleading.
    try:
        target_dev = getattr(separator, "_device", None)
        if target_dev is None:
            target_dev = next(separator.model.parameters()).device
        print(f"Using device: {target_dev}")
    except Exception:
        pass
    stems = separator.separate_audio_file(args.track)[1]

    args.out.mkdir(parents=True, exist_ok=True)
    for stem_name, audio in stems.items():
        stem_path = args.out / f"{args.track.stem}_{stem_name}.wav"
        save_audio(
            audio,
            stem_path,
            samplerate=separator.samplerate,
            bits_per_sample=16,
            as_float=False,
        )
        print(f"Saved {stem_path}")


if __name__ == "__main__":
    main()
