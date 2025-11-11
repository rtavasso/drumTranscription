# Dataset Preparation

The repo ships with BabySlakh16k but expects resampled drum stems plus TSV labels
in `onsets-and-frames/data/babyslakh_drums`. Follow the steps below to generate
everything from scratch. For each step we show both **Git Bash (Windows)** and
**Linux** commands—pick whichever matches your host shell.

1. **Generate demixed stems on CUDA**  
   This demixes every track in `data/babyslakh_16k` and writes the drum/bass/other/vocal stems
   into `stems/TrackXXXXX/`. The `MSYS_NO_PATHCONV` prefix keeps Docker from rewriting the volume mount.

   *Git Bash (Windows)*
   ```bash
   MSYS_NO_PATHCONV=1 docker compose run --rm -v "$(pwd -W):/workspace" demucs-gpu \
     python dev_demix_dataset.py data/babyslakh_16k \
     --out stems --device cuda --progress --name mix.wav --skip-existing
   ```

   *Linux*
   ```bash
   docker compose run --rm -v "$(pwd):/workspace" demucs-gpu \
     python dev_demix_dataset.py data/babyslakh_16k \
     --out stems --device cuda --progress --name mix.wav --skip-existing
   ```

2. **Resample drum stems + build TSV labels**  
   The preprocessing script converts each `mix_drums.wav` to mono 16 kHz audio, drops
   the files into `onsets-and-frames/data/babyslakh_drums`, and saves merged drum
   MIDI annotations as `.tsv` files so the Onsets-and-Frames dataloader can ingest them.

   *Git Bash (Windows)*
   ```bash
   docker compose run --rm -v "$(pwd -W):/workspace" demucs-gpu \
     python onsets-and-frames/data/prepare_babyslakh_drums.py
   ```

   *Linux*
   ```bash
   docker compose run --rm -v "$(pwd):/workspace" demucs-gpu \
     python onsets-and-frames/data/prepare_babyslakh_drums.py
   ```

Once both steps complete, launch training inside the Docker environment so PyTorch,
CUDA, and dependencies match the rest of the workflow:

*Git Bash (Windows)*
```bash
WANDB_API_KEY=bfbf4066fa95e427897e143fcf9d3e17fa70e774 docker compose run --rm -v "$(pwd -W):/workspace" demucs-gpu \
  python onsets-and-frames/train.py with \
  train_on=BabySlakh dataset_path=onsets-and-frames/data/babyslakh_drums
```

*Linux*
```bash
WANDB_API_KEY=bfbf4066fa95e427897e143fcf9d3e17fa70e774 docker compose run --rm -v "$(pwd):/workspace" demucs-gpu \
  python onsets-and-frames/train.py with \
  train_on=BabySlakh dataset_path=onsets-and-frames/data/babyslakh_drums
```

## Legacy Python Compatibility

The project still pins `sacred==0.7.4`, which expects `collections.Mapping` and `pprint._safe_repr`.
Modern Python releases (3.10+/3.12+) removed those symbols, so we add tiny shims directly inside
`onsets-and-frames/train.py`. When the script starts it backfills the deprecated `collections`
aliases from `collections.abc` and injects a fallback `_safe_repr` into the standard `pprint`
module. This lets the existing Sacred configuration/printing code run unchanged inside the Docker
image without bumping dependencies or rebuilding anything.
