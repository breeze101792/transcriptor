# transcriptor

Transcribe iPhone Voice Memos (or any audio) **fully locally** — nothing leaves your machine. Works on macOS and Linux.

Pipeline: **Whisper (STT) → local LLM (cleanup) → local LLM (summary)**

## One-time setup

> All examples use `./start.sh`, the bundled wrapper. It installs `uv` if
> missing, runs `uv sync`, and (on Linux only) wires up the bundled cu12
> cuBLAS / cuDNN libraries that `faster-whisper`'s ctranslate2 wheel needs.
> On macOS that section is skipped automatically — Apple Silicon uses Metal
> via the Accelerate framework. macOS users who prefer the bare
> `uv run transcribe.py …` invocation can still use it.

### macOS

```bash
# 1. system deps
brew install ffmpeg ollama

# 2. start Ollama in a terminal and leave it running
ollama serve

# 3. in another terminal, pull the LLM you want (downloads ~1.5 GB once)
ollama pull qwen3.5:2b-mlx

# 4. (only the first time) make start.sh executable
chmod +x start.sh
```

### Linux

The `faster-whisper` / ctranslate2 wheel on PyPI is built against CUDA 12.
`start.sh` handles the cu12 libs for you, but it can't install your system
packages — those need to be in place before the first run.

```bash
# 1. system deps (Arch shown; adapt to your distro)
sudo pacman -S ffmpeg nvidia-utils cuda                    # Arch
#   or, on Ubuntu / Debian:
# sudo apt install ffmpeg && curl -fsSL https://ollama.com/install.sh | sh

# 2. start Ollama in a terminal and leave it running
ollama serve

# 3. in another terminal, pull the LLM you want (downloads ~1.5 GB once)
ollama pull qwen3.5:2b                                    # Linux (no -mlx variant)

# 4. (only the first time) make start.sh executable
chmod +x start.sh
```

`start.sh` will:

- install `uv` to `~/.local/bin` if it's not on `PATH`
- run `uv sync` to create/update the venv
- on Linux: install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` into the
  venv (no-op if already there) and export `LD_LIBRARY_PATH` so the
  cu12-built `ctranslate2` can find them — the system has CUDA 13's
  `libcublas.so.13`, but the wheel wants `libcublas.so.12`
- on macOS: skip the cu12 section entirely
- `exec uv run transcribe.py …` with all forwarded args

## Usage

```bash
# basic
./start.sh recording.m4a

# with options (use qwen3.5:2b-mlx on macOS, qwen3.5:2b on Linux)
./start.sh recording.m4a \
    --output-dir ./output \
    --whisper-model small \
    --llm-model qwen3.5:2b-mlx    # on macOS — use qwen3.5:2b on Linux

# only produce the raw transcript (no LLM)
./start.sh recording.m4a --skip-cleanup

# only transcribe + clean (no summary)
./start.sh recording.m4a --skip-summary

# re-run cleanup + summary on an existing transcript (skips STT entirely)
./start.sh --from-raw ./output/old_run/recording.raw.txt

# re-run only the summary on an existing cleaned transcript (skips STT + cleanup)
./start.sh --from-cleaned ./output/old_run/recording.cleaned.txt
```

## Outputs

For every run, a per-run folder is created and the three files go inside it:

```
./output/
└── mymemo_20260611-103045/
    ├── mymemo.raw.txt
    ├── mymemo.cleaned.txt
    └── mymemo.summary.md
```

| File                      | What it is                              |
|---------------------------|------------------------------------------|
| `<name>.raw.txt`          | Verbatim Whisper output                  |
| `<name>.cleaned.txt`      | Grammar/punctuation/structure fixed by LLM |
| `<name>.summary.md`       | Structured markdown summary              |

The folder name uses the audio file's stem (with spaces and special characters replaced by `_`) plus a timestamp, so re-running on the same file creates a new folder instead of overwriting. Override the root with `--output-dir`; the per-run subfolder is always created.

## Options

```
input (provide exactly one):
  audio_path                path to .m4a (or any ffmpeg-supported audio)
  --from-raw TXT            skip STT; run cleanup+summary on an existing transcript
  --from-cleaned TXT        skip STT and cleanup; run only summary on a cleaned transcript

options:
  -o, --output-dir DIR      output root (default: ./output; a per-run subfolder is created)
  -w, --whisper-model NAME  tiny|base|small|medium|large-v3 (default: small)
  -l, --llm-model NAME      any ollama model tag (default: auto-detected by platform —
                              macOS → qwen3.5:2b-mlx, Linux → qwen3.5:2b)
  --ollama-url URL          Ollama base URL (default: http://127.0.0.1:11434)
  --language CODE           ISO-639-1 like 'en' or 'zh' (default: auto-detect)
  --skip-cleanup            stop after raw transcript
  --skip-summary            stop after cleaned transcript
  --debug                   print every streamed chunk from Ollama (no progress bars)
```

## Model notes

- **Whisper `small`** is the default — accurate enough that the LLM cleanup mostly fixes punctuation and run-ons, not content. Bump to `medium` if you notice lots of word errors. `large-v3` is overkill for personal use.
- **Default LLM is platform-detected**: macOS uses `qwen3.5:2b-mlx` (Apple Silicon-optimized), Linux uses `qwen3.5:2b`. Override with `--llm-model`. Other good options: `llama3.1`, `gemma4:e4b`, `mistral:7b`. Pull first with `ollama pull <name>`.

## Progress

- **Phase 1 (STT)** — shows audio-position progress (`X/Y seconds`) driven by Whisper segment timestamps.
- **Phase 2 (cleanup)** — shows a *predicted* progress bar that fills to 100%. The total is set to the input token count, since cleanup output is roughly the same length as the input.
- **Phase 3 (summary)** — no bar (the output length is unpredictable). Instead prints a live tok/s rate line that overwrites itself every 0.5s, then a final `done: N tok in T s = R tok/s` line.

## Token budgets

Before each LLM phase, the script counts the tokens in the input text and prints a budget line:

```
        phase 2 input (raw transcript): 4521 / 8192 tokens (55%, exact)
        phase 3 input (cleaned transcript): 4480 / 16384 tokens (27%, exact)
```

- `exact` — counted via Ollama's `/api/tokenize` endpoint, using the model's actual tokenizer.
- `approximate` — Ollama is too old (pre-0.31) to expose `/api/tokenize`, so the script falls back to a char-based heuristic (~85% accurate on natural English, worse on code or mixed CJK). Labels the count as `approximate` so you know to take it with a grain of salt.

**Auto-raising the context window.** Default `num_ctx` is 16384 for both phases. If the input exceeds it, the script automatically raises `num_ctx` (rounded up to a power of 2, capped at 32768) and logs the change, e.g.:

```
        auto-raised num_ctx: 16384 -> 32768 (to fit 22100 input tokens)
        phase 2 input (raw transcript): 22100 / 32768 tokens (67%, exact)
```

Above 32768 input tokens the script exits with a clear error suggesting the audio be split into shorter chunks, since most local Ollama models can't run with a context window larger than 32k without OOM-ing.

## Summary structure

The summary phase produces a markdown file with these sections:

- **Session** — date, source file, duration, language. Populated automatically from the run; `--from-cleaned` runs show `"unknown"` for duration and language (no Whisper info available).
- **Summary** — 2–4 sentence overview of the main point.
- **Key Points** — 3–7 bullets covering the main ideas, decisions, or topics.
- **Terms** — glossary of jargon, proper nouns, acronyms, project/product names, and technical concepts the speaker used. Common words are skipped. Presented as a 2-column markdown table (`| Term | Meaning |`) with the standard `| --- | --- |` header separator. 3–10 rows. If no notable terms appear, the section contains only `None` (no table). One sentence per meaning; pipes inside cells are escaped as `\|`.
- **Action Items** — specific tasks, follow-ups, or commitments mentioned.

## Performance & debugging

- The script passes `think=False` to Ollama, disabling chain-of-thought for Qwen 3 and similar reasoning models. Cleanup and summary are mechanical — the model thinking deeply before answering just adds latency with no quality gain.
- Pass `--debug` to skip the progress bars and print every streamed chunk from Ollama as it arrives, plus a final summary of chunks/total-time/tokens-per-second. Useful when:
  - A run feels stuck and you want to see if the model is emitting anything
  - You suspect the model is producing thinking tokens (you'll see them in the chunk stream)
  - You're iterating on prompts and want to spot regressions

## Privacy

- No cloud API calls. All processing is local (Whisper runs in-process; Ollama runs on `localhost:11434`).
- Voice Memo audio is never uploaded anywhere.

## Troubleshooting

- **`Cannot reach Ollama`** — start it: `ollama serve`, or pass `--ollama-url` to point at a different host/port
- **`Model 'X' not found`** — pull it: `ollama pull X`
- **`RuntimeError: Library libcublas.so.12 is not found or cannot be loaded`** (Linux) — the system has CUDA 13 but the cu12 wheel of ctranslate2 needs cu12. Use `./start.sh` (it installs `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` and sets `LD_LIBRARY_PATH`), or set `LD_LIBRARY_PATH` manually to wherever `libcublas.so.12` lives.
- **First Whisper run is slow** — the model (~460 MB for `small`) downloads once and is cached afterwards.
