#!/usr/bin/env python3
"""Transcribe an iPhone Voice Memo locally in 3 phases:

    1. STT         - faster-whisper transcribes the audio file
    2. Cleanup     - local LLM (Ollama) fixes grammar, punctuation, structure
    3. Summary     - same LLM produces a structured markdown summary

Usage:
    uv run transcribe.py path/to/recording.m4a
    uv run transcribe.py memo.m4a --whisper-model medium --llm-model qwen2.5:7b
    uv run transcribe.py --from-raw existing_transcript.txt
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel
from ollama import Client, ResponseError
from tqdm import tqdm

import httpx

# ─── defaults ────────────────────────────────────────────────────────────────
DEFAULT_WHISPER_MODEL = "small"      # tiny|base|small|medium|large-v3
DEFAULT_LLM_MODEL_MAC = "qwen3.5:0.8b-mlx"   # macOS (Apple Silicon)
DEFAULT_LLM_MODEL_LINUX = "qwen3.5:0.8b"     # Linux
DEFAULT_OUTPUT_ROOT = "./output"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

def default_llm_model() -> str:
    """Pick the platform-appropriate default LLM. Override with --llm-model."""
    if sys.platform == "darwin":
        return DEFAULT_LLM_MODEL_MAC
    if sys.platform.startswith("linux"):
        return DEFAULT_LLM_MODEL_LINUX
    # Windows / other: fall back to the macOS tag (works on any platform that can run Ollama)
    return DEFAULT_LLM_MODEL_MAC

# ─── LLM prompts ─────────────────────────────────────────────────────────────
CLEANUP_PROMPT = """You are a transcription editor. Clean up the following raw speech-to-text transcript by:
- Fixing grammar, punctuation, and capitalization
- Breaking run-on sentences into proper sentences
- Removing filler words (um, uh, like, you know) only when they add no meaning
- Preserving the speaker's original voice, vocabulary, and meaning exactly

Do NOT:
- Add new content or change what the speaker said
- Rewrite for style or flow
- Summarize or omit anything
- Add commentary, headers, notes, or surrounding text

Output only the cleaned transcript, nothing else.

Transcript:
{text}
"""

SUMMARY_PROMPT = r"""Summarize the following cleaned transcript in markdown. Use this exact structure:

## Session
- **Date**: {session_date}
- **Source file**: {source_file}
- **Duration**: {duration}
- **Language**: {language}
(If any value is unknown, write "unknown" rather than guessing.)

## Summary
2-4 sentences capturing the main point.

## Key Points
- 3-7 bullets covering the main ideas, decisions, or topics

## Terms
Present as a markdown table with two columns: `Term` and `Meaning`. Include jargon, proper nouns, acronyms, project names, product names, or technical concepts the speaker used. Skip common everyday words. If the transcript uses no notable terms, write `None` on its own line (no table at all).

- Aim for 3-10 rows.
- Use the exact format below, including the header separator row (three or more dashes per column). Do not wrap the table in a code fence.
- Keep the `Meaning` column concise: one short sentence, no line breaks inside a cell. If pipes (`|`) appear inside a term or meaning, escape them as `\|`.
- Example of the right shape:
  | Term   | Meaning                                              |
  | ------ | ---------------------------------------------------- |
  | Falcon | Internal codename for the new dashboard project      |
  | MRR    | Monthly recurring revenue                            |

## Action Items
- Specific tasks, follow-ups, or commitments mentioned (write "None" if absent)

Be faithful to the transcript. Do not invent details. If something is ambiguous, say so briefly.

Transcript:
{text}
"""

# Context windows: default 2048 silently truncates long memos.
CLEANUP_CTX = 16384
SUMMARY_CTX = 16384
MAX_CTX = 32768    # hard cap: beyond this, splitting the input is the only option


# ─── phase 1: speech-to-text ────────────────────────────────────────────────
def transcribe(audio_path: Path, model_name: str, language: str | None) -> tuple[str, dict]:
    """Transcribe audio, returning (text, info_dict) where info_dict has
    'duration' (seconds) and 'language' for use in the summary's session header."""
    print(f"[1/3] Loading Whisper model '{model_name}' (first run downloads it)...")
    model = WhisperModel(model_name, device="auto", compute_type="int8")

    print(f"[1/3] Transcribing {audio_path.name}...")
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,    # skip silence, prevents hallucinating text in pauses
    )

    total = info.duration or 0
    with tqdm(total=total, unit="s", disable=total <= 0,
              desc="    transcribe") as bar:
        parts: list[str] = []
        last_end = 0.0
        for seg in segments:
            parts.append(seg.text.strip())
            # advance the bar by however much audio this segment covers
            advance = max(0.0, min(seg.end - last_end, bar.total - bar.n))
            if advance > 0:
                bar.update(advance)
            last_end = seg.end

    text = " ".join(parts)
    print(
        f"[1/3] Done. Detected language: {info.language} "
        f"(prob {info.language_probability:.2f}), {len(text)} chars."
    )
    info_dict = {"duration": info.duration, "language": info.language}
    return text, info_dict


# ─── phase 2: grammar / structure cleanup ────────────────────────────────────
def _stream_with_bar(client: Client, model: str, prompt: str, num_ctx: int,
                     temperature: float, desc: str, total: int, debug: bool) -> str:
    """Stream with a tqdm progress bar that fills to `total` (predicted output size).

    Used by phase 2 (cleanup) where the output is roughly the same length as the
    input, so we can show a meaningful progress bar.
    """
    chunks: list[str] = []
    start = time.monotonic()
    with tqdm(total=total, unit="tok", desc=desc,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} tok [{rate_fmt}, {elapsed}<{remaining}]",
              disable=debug) as bar:
        stream = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": num_ctx, "temperature": temperature},
            stream=True,
            think=False,    # disable chain-of-thought; cleanup & summary don't need it
        )
        n = 0
        for chunk in stream:
            msg = chunk.get("message") or {}
            token = msg.get("content")
            if token:
                chunks.append(token)
                n += 1
                if debug:
                    print(f"  [{n:04d}] ({len(token):3d} chars) {token!r}", flush=True)
                else:
                    bar.update(1)
    elapsed = time.monotonic() - start
    if debug:
        rate = n / elapsed if elapsed > 0 else 0
        print(f"  ---- {n} chunks in {elapsed:.2f}s = {rate:.1f} tok/s ----", flush=True)
    return "".join(chunks).strip()


def _stream_with_rate(client: Client, model: str, prompt: str, num_ctx: int,
                      temperature: float, desc: str, debug: bool) -> str:
    """Stream and print a live tok/s rate line, no progress bar.

    Used by phase 3 (summary) where the output length is unpredictable, so a
    bar that fills to 100% would be misleading. The rate is printed on the
    same line, overwritten every second, so the user sees activity and speed.
    """
    chunks: list[str] = []
    start = time.monotonic()
    last_print = start
    n = 0
    if debug:
        stream_iter = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": num_ctx, "temperature": temperature},
            stream=True,
            think=False,
        )
        for chunk in stream_iter:
            msg = chunk.get("message") or {}
            token = msg.get("content")
            if token:
                chunks.append(token)
                n += 1
                print(f"  [{n:04d}] ({len(token):3d} chars) {token!r}", flush=True)
    else:
        # Print a stable header line, then overwrite with a live rate line that
        # we re-render in place using \r. End the live line with \n on completion.
        print(f"{desc} streaming...", flush=True)
        stream_iter = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": num_ctx, "temperature": temperature},
            stream=True,
            think=False,
        )
        for chunk in stream_iter:
            msg = chunk.get("message") or {}
            token = msg.get("content")
            if token:
                chunks.append(token)
                n += 1
                now = time.monotonic()
                if now - last_print >= 0.5:
                    elapsed = now - start
                    rate = n / elapsed if elapsed > 0 else 0
                    print(f"\r{desc} {n} tok, {rate:.1f} tok/s, {elapsed:.1f}s elapsed", end="", flush=True)
                    last_print = now
    elapsed = time.monotonic() - start
    rate = n / elapsed if elapsed > 0 else 0
    if not debug:
        # overwrite the rate line with a final summary
        print(f"\r{desc} done: {n} tok in {elapsed:.1f}s = {rate:.1f} tok/s" + " " * 20, flush=True)
    else:
        print(f"  ---- {n} chunks in {elapsed:.2f}s = {rate:.1f} tok/s ----", flush=True)
    return "".join(chunks).strip()


def clean_with_llm(raw_text: str, model: str, client: Client,
                   input_tokens: int = 0, debug: bool = False) -> str:
    """Clean raw transcript with the LLM, auto-sizing num_ctx for the input.

    Progress bar uses `input_tokens` as the predicted total — cleanup output
    is roughly the same length as the input, so the bar fills to ~100%.
    """
    num_ctx = _auto_num_ctx(input_tokens, CLEANUP_CTX)
    if num_ctx > CLEANUP_CTX:
        print(f"        auto-raised num_ctx: {CLEANUP_CTX} -> {num_ctx} (to fit {input_tokens} input tokens)")
    print(f"[2/3] Cleaning up with Ollama model '{model}'...")
    prompt = CLEANUP_PROMPT.format(text=raw_text)
    try:
        return _stream_with_bar(client, model, prompt, num_ctx, 0.2,
                                desc="    cleanup  ", total=input_tokens, debug=debug)
    except ResponseError as e:
        msg = str(e).lower()
        if "not found" in msg and "model" in msg:
            sys.exit(f"Model '{model}' not found in Ollama. Run: ollama pull {model}")
        raise


# ─── phase 3: summary ───────────────────────────────────────────────────────
def _format_duration(seconds: float | None) -> str:
    """Convert seconds to a human-friendly 'X min Y sec' string, or 'unknown'."""
    if not seconds or seconds <= 0:
        return "unknown"
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h} h {m} min {s} sec"
    if m:
        return f"{m} min {s} sec"
    return f"{s} sec"

def summarize_with_llm(cleaned_text: str, model: str, client: Client,
                       session_meta: dict | None = None,
                       input_tokens: int = 0,
                       debug: bool = False) -> str:
    """Summarize cleaned transcript, auto-sizing num_ctx for the input.

    No progress bar — summary output length is unpredictable, so a filling
    bar would be misleading. Instead prints a live tok/s rate line that
    updates every 0.5s and a final summary line.
    """
    num_ctx = _auto_num_ctx(input_tokens, SUMMARY_CTX)
    if num_ctx > SUMMARY_CTX:
        print(f"        auto-raised num_ctx: {SUMMARY_CTX} -> {num_ctx} (to fit {input_tokens} input tokens)")
    print(f"[3/3] Summarizing with '{model}'...")
    meta = session_meta or {}
    prompt = SUMMARY_PROMPT.format(
        text=cleaned_text,
        session_date=meta.get("date", "unknown"),
        source_file=meta.get("source_file", "unknown"),
        duration=_format_duration(meta.get("duration")),
        language=meta.get("language", "unknown"),
    )
    return _stream_with_rate(client, model, prompt, num_ctx, 0.3,
                             desc="    summary  ", debug=debug)


# ─── io helper ──────────────────────────────────────────────────────────────
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")

def sanitize_stem(stem: str) -> str:
    """Turn 'Voice Memo (2024-01-15)' into 'Voice_Memo_2024-01-15'."""
    s = _SAFE_STEM_RE.sub("_", stem).strip("_")
    return s or "transcript"

def make_output_dir(root: Path, audio_stem: str) -> Path:
    """Create a per-run subfolder: <root>/<safe_stem>_<YYYYMMDD-HHMMSS>/."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{sanitize_stem(audio_stem)}_{ts}"

def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"        wrote {path} ({len(content)} chars)")


# ─── token counting ─────────────────────────────────────────────────────────
def _heuristic_tokens(text: str) -> int:
    """Rough token count when Ollama's /api/tokenize isn't available.

    Heuristic: English text averages ~4 chars per token, CJK text averages
    ~1.5 chars per token (since each CJK char is usually its own token).
    We blend based on the CJK fraction of the text. Accuracy is ~85% on
    natural English, worse on code/technical content.
    """
    if not text:
        return 0
    cjk = sum(1 for c in text if "一" <= c <= "鿿" or "぀" <= c <= "ヿ" or "가" <= c <= "힣")
    frac_cjk = cjk / len(text)
    chars_per_token = 4.0 * (1 - frac_cjk) + 1.5 * frac_cjk
    return max(1, round(len(text) / chars_per_token))

def count_tokens(text: str, ollama_url: str, model: str) -> int:
    """Token count for the input text. Tries Ollama's /api/tokenize first
    (exact, model-specific); falls back to a heuristic if that endpoint
    isn't available (older Ollama versions don't expose it).
    """
    try:
        r = httpx.post(
            f"{ollama_url.rstrip('/')}/api/tokenize",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return len(data)
        return len(data.get("tokens", []))
    except Exception:
        # Fall back to heuristic. Silent: the caller labels it "approximate"
        # when heuristic is used.
        return _heuristic_tokens(text)

def token_count_source(ollama_url: str, model: str) -> str:
    """Return 'exact' if /api/tokenize works, 'approximate' otherwise."""
    try:
        r = httpx.post(
            f"{ollama_url.rstrip('/')}/api/tokenize",
            json={"model": model, "prompt": "ping"},
            timeout=10,
        )
        r.raise_for_status()
        return "exact"
    except Exception:
        return "approximate"

def format_budget(label: str, used: int, budget: int, source: str) -> str:
    """Format a 'phase 2 input: 4521 / 8192 tokens (55%)' line."""
    pct = (used / budget) * 100 if budget else 0
    warn = "  <-- EXCEEDS BUDGET, output will be truncated" if used > budget else ""
    return f"        {label}: {used} / {budget} tokens ({pct:.0f}%, {source}){warn}"


def _auto_num_ctx(input_tokens: int, default_ctx: int) -> int:
    """Pick a num_ctx large enough for the input plus expected output.

    Rule of thumb: input + ~2.5x headroom for the response (a cleanup pass
    produces roughly the same length as the input; a summary can be longer
    in token density per character). Round up to a power of two for clean
    KV-cache allocation in Ollama. Cap at MAX_CTX; beyond that, the input
    must be split manually.
    """
    target = max(default_ctx, int(input_tokens * 2.5))
    target = min(target, MAX_CTX)
    # round up to next power of 2
    p = 1
    while p < target:
        p <<= 1
    return p


# ─── cli ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe an iPhone Voice Memo locally (Whisper + Ollama LLM).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("audio_path", nargs="?",
                     help="path to the audio file (e.g. recording.m4a)")
    src.add_argument("--from-raw", dest="from_raw", metavar="TXT",
                     help="skip STT; run cleanup+summary on an existing .txt transcript")
    src.add_argument("--from-cleaned", dest="from_cleaned", metavar="TXT",
                     help="skip STT and cleanup; run only summary on an existing cleaned .txt")
    p.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_ROOT,
                   help=f"root output dir (default: {DEFAULT_OUTPUT_ROOT}; "
                        f"a per-run subfolder is created inside)")
    p.add_argument("-w", "--whisper-model", default=DEFAULT_WHISPER_MODEL,
                   help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})")
    p.add_argument("-l", "--llm-model", default=None,
                   help=f"Ollama model tag (default: auto-detected; "
                        f"macOS -> {DEFAULT_LLM_MODEL_MAC}, "
                        f"Linux -> {DEFAULT_LLM_MODEL_LINUX})")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                   help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    p.add_argument("--language", default=None,
                   help="ISO-639-1 code like 'en' or 'zh' (default: auto-detect)")
    p.add_argument("--skip-cleanup", action="store_true",
                   help="stop after producing the raw transcript")
    p.add_argument("--skip-summary", action="store_true",
                   help="stop after producing the cleaned transcript")
    p.add_argument("--debug", action="store_true",
                   help="disable progress bars and print every streamed chunk from Ollama")
    return p.parse_args()


# ─── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # resolve the LLM default based on platform (--llm-model overrides)
    if args.llm_model is None:
        args.llm_model = default_llm_model()
        print(f"[info] using LLM model: {args.llm_model} (platform default for {sys.platform})")

    # phase 2 + 3 share one Ollama client (created up front so the user sees
    # connection errors before we spend time on phase 1)
    try:
        client = Client(host=args.ollama_url)
    except Exception as e:
        sys.exit(
            f"Cannot reach Ollama at {args.ollama_url}. "
            f"Start it with: ollama serve\n({e})"
        )

    # session metadata for the summary header (date, source, duration, language).
    # populated progressively as we move through the pipeline.
    session_meta: dict = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": "unknown",
        "duration": None,
        "language": "unknown",
    }

    if args.from_cleaned:
        # ── skip STT and cleanup, run only summary on an existing cleaned file ──
        cleaned_path_in = Path(args.from_cleaned).expanduser().resolve()
        if not cleaned_path_in.is_file():
            sys.exit(f"Cleaned transcript file not found: {cleaned_path_in}")
        src_stem = cleaned_path_in.stem
        if src_stem.endswith(".cleaned"):
            src_stem = src_stem[:-8]
        base = src_stem or "transcript"

        out_root = Path(args.output_dir).expanduser().resolve()
        out_dir = make_output_dir(out_root, base)
        summary_path = out_dir / f"{base}.summary.md"
        # copy the input cleaned file into the output folder for traceability
        copied_cleaned = out_dir / f"{base}.cleaned.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cleaned_path_in, copied_cleaned)
        print(f"        copied {cleaned_path_in} -> {copied_cleaned}")

        cleaned = copied_cleaned.read_text(encoding="utf-8")
        session_meta["source_file"] = cleaned_path_in.name
        print(f"[info] using existing cleaned transcript ({len(cleaned)} chars), "
              f"skipping STT (phase 1) and cleanup (phase 2)")
    elif args.from_raw:
        # ── skip STT, run cleanup + summary on an existing transcript ─────────
        raw_path_in = Path(args.from_raw).expanduser().resolve()
        if not raw_path_in.is_file():
            sys.exit(f"Raw transcript file not found: {raw_path_in}")
        # base name: strip a trailing ".raw" if the user passed foo.raw.txt,
        # so output is foo.cleaned.txt / foo.summary.md (not foo.raw.cleaned.txt)
        src_stem = raw_path_in.stem
        if src_stem.endswith(".raw"):
            src_stem = src_stem[:-4]
        base = src_stem or "transcript"

        out_root = Path(args.output_dir).expanduser().resolve()
        out_dir = make_output_dir(out_root, base)
        cleaned_path = out_dir / f"{base}.cleaned.txt"
        summary_path = out_dir / f"{base}.summary.md"
        # copy the input transcript into the output folder for traceability
        copied_raw = out_dir / f"{base}.raw.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_path_in, copied_raw)
        print(f"        copied {raw_path_in} -> {copied_raw}")

        raw = copied_raw.read_text(encoding="utf-8")
        session_meta["source_file"] = raw_path_in.name
        print(f"[info] using existing raw transcript ({len(raw)} chars), "
              f"skipping STT (phase 1)")
    else:
        # ── normal flow: transcribe audio then clean + summarize ─────────────
        audio = Path(args.audio_path).expanduser().resolve()
        if not audio.is_file():
            sys.exit(f"Audio file not found: {audio}")

        out_root = Path(args.output_dir).expanduser().resolve()
        out_dir = make_output_dir(out_root, audio.stem)
        base = audio.stem
        raw_path = out_dir / f"{base}.raw.txt"
        cleaned_path = out_dir / f"{base}.cleaned.txt"
        summary_path = out_dir / f"{base}.summary.md"
        session_meta["source_file"] = audio.name

        # phase 1
        raw, info_dict = transcribe(audio, args.whisper_model, args.language)
        session_meta["duration"] = info_dict.get("duration")
        session_meta["language"] = info_dict.get("language", "unknown")
        write(raw_path, raw)
        if args.skip_cleanup:
            print(f"\nDone. Output files in: {out_dir}")
            return

    if not args.from_cleaned:
        # phase 2
        tok_src = token_count_source(args.ollama_url, args.llm_model)
        n_raw_tok = count_tokens(raw, args.ollama_url, args.llm_model)
        ctx2 = _auto_num_ctx(n_raw_tok, CLEANUP_CTX)
        print(format_budget("phase 2 input (raw transcript)", n_raw_tok, ctx2, tok_src))
        if n_raw_tok > MAX_CTX // 2:
            sys.exit(
                f"\nInput is {n_raw_tok} tokens, exceeding the {MAX_CTX} hard cap.\n"
                f"Split the audio into shorter chunks (e.g. with --skip-summary and a\n"
                f"manual break) or use a model with a larger context window."
            )
        cleaned = clean_with_llm(raw, args.llm_model, client,
                                 input_tokens=n_raw_tok, debug=args.debug)
        write(cleaned_path, cleaned)
        if args.skip_summary:
            print(f"\nDone. Output files in: {out_dir}")
            return
    else:
        # phase 3 only: still detect whether the tokenizer is exact or approximate
        tok_src = token_count_source(args.ollama_url, args.llm_model)

    # phase 3
    n_clean_tok = count_tokens(cleaned, args.ollama_url, args.llm_model)
    ctx3 = _auto_num_ctx(n_clean_tok, SUMMARY_CTX)
    print(format_budget("phase 3 input (cleaned transcript)", n_clean_tok, ctx3, tok_src))
    if n_clean_tok > MAX_CTX // 2:
        sys.exit(
            f"\nCleaned transcript is {n_clean_tok} tokens, exceeding the {MAX_CTX} hard cap.\n"
            f"Try running with --skip-cleanup=False on smaller audio chunks."
        )
    summary = summarize_with_llm(cleaned, args.llm_model, client,
                                 session_meta=session_meta,
                                 input_tokens=n_clean_tok,
                                 debug=args.debug)
    write(summary_path, summary)

    print(f"\nDone. Output files in: {out_dir}")


if __name__ == "__main__":
    main()
