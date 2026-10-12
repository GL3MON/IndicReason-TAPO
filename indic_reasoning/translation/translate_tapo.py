#!/usr/bin/env python3
"""Translate GSM8K TRAIN for TAPO with bounded concurrency and resumable JSONL.

Run with the same flags to resume; successful source IDs are never repeated.
Sarvam prompt: https://huggingface.co/sarvamai/sarvam-translate
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

import aiohttp
from datasets import load_dataset


def has_target_script(text, language):
    ranges = {"ta": ("\u0b80", "\u0bff"), "ml": ("\u0d00", "\u0d7f")}
    bounds = ranges.get(language)
    return bool(text.strip()) and (bounds is None or any(bounds[0] <= c <= bounds[1] for c in text))


def encode_row(row):
    # TAPO uses str.splitlines(), which also splits on Unicode line separators.
    text = json.dumps(row, ensure_ascii=False)
    for separator in ("\u0085", "\u2028", "\u2029"):
        text = text.replace(separator, f"\\u{ord(separator):04x}")
    return (text + "\n").encode("utf-8")


def recover_output(handle, dataset, language):
    """Check every saved alignment; recover only an interrupted trailing write."""
    done = set()
    handle.seek(0)
    while True:
        offset = handle.tell()
        line = handle.readline()
        if not line:
            break
        if not line.endswith(b"\n"):
            handle.truncate(offset)
            break
        row = json.loads(line)
        index = int(row["source_id"].removeprefix("gsm8k_train_"))
        if not 0 <= index < len(dataset):
            raise ValueError("Saved source ID outside GSM8K TRAIN")
        original = dataset[index]
        if (row["source_id"] != f"gsm8k_train_{index}" or index in done
                or row["language"] != language or not row["question"].strip()
                or row["english_question"] != original["question"]
                or row["answer"] != original["answer"].rsplit("####", 1)[-1].strip()):
            raise ValueError(f"Invalid saved alignment: {row['source_id']}")
        done.add(index)
    handle.seek(0, 2)
    return done


async def run(args, dataset, output, done):
    pending = [i for i in range(min(args.limit or len(dataset), len(dataset))) if i not in done]
    queue = asyncio.Queue()
    for index in pending:
        queue.put_nowait(index)
    started = time.monotonic()
    stats = dict(saved=0, failed=0, retries=0, output_tokens=0)
    failure_path = args.output.with_suffix(".errors.jsonl")
    # Every append is flushed immediately; fsync every second and at shutdown.
    last_sync = time.monotonic()
    failures = open(failure_path, "a", encoding="utf-8", buffering=1)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def worker():
            nonlocal last_sync
            while not queue.empty():
                index = queue.get_nowait()
                original = dataset[index]
                error = None
                max_tokens = args.max_tokens
                for attempt in range(args.retries + 1):
                    payload = dict(model=args.model, messages=[
                        {"role": "system", "content": f"Translate the text below to {args.target_language}."},
                        {"role": "user", "content": original["question"]}],
                        temperature=0.01, max_tokens=max_tokens,
                        repetition_penalty=min(args.repetition_penalty + .1 * attempt, 1.3))
                    try:
                        async with session.post(args.url, json=payload) as response:
                            response.raise_for_status()
                            result = await response.json()
                        choice = result["choices"][0]
                        text = choice["message"]["content"].strip()
                        if choice.get("finish_reason") != "stop":
                            max_tokens = min(max_tokens * 2, 2048)
                            raise ValueError("Translation did not finish normally")
                        if not has_target_script(text, args.language):
                            raise ValueError(f"Empty translation or missing {args.target_language} script")
                        row = dict(question=text, english_question=original["question"],
                                   answer=original["answer"].rsplit("####", 1)[-1].strip(),
                                   language=args.language, source_id=f"gsm8k_train_{index}")
                        output.write(encode_row(row))
                        output.flush()
                        if time.monotonic() - last_sync >= 1:
                            os.fsync(output.fileno())
                            last_sync = time.monotonic()
                        done.add(index)
                        stats["saved"] += 1
                        stats["output_tokens"] += result.get("usage", {}).get("completion_tokens", 0)
                        error = None
                        break
                    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
                        error = str(exc)
                        if attempt < args.retries:
                            stats["retries"] += 1
                            await asyncio.sleep(min(2 ** attempt, 30) + random.random())
                if error is not None:
                    stats["failed"] += 1
                    failures.write(json.dumps(dict(source_id=f"gsm8k_train_{index}", error=error)) + "\n")
                queue.task_done()

        async def progress():
            while True:
                await asyncio.sleep(10)
                elapsed = time.monotonic() - started
                rate = stats["saved"] / elapsed
                remaining = len(pending) - stats["saved"] - stats["failed"]
                print(json.dumps(dict(total_saved=len(done), total=len(dataset), **stats,
                                      rows_per_second=round(rate, 2),
                                      tokens_per_second=round(stats["output_tokens"] / elapsed, 1),
                                      eta_seconds=round(remaining / rate) if rate else None)), flush=True)

        reporter = asyncio.create_task(progress())
        try:
            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        finally:
            reporter.cancel()
            output.flush()
            os.fsync(output.fileno())
            failures.close()
    elapsed = time.monotonic() - started
    summary = dict(total_saved=len(done), dataset_rows=len(dataset), **stats,
                   elapsed_seconds=round(elapsed, 2), rows_per_second=round(stats["saved"] / max(elapsed, .001), 2),
                   complete=len(done) == len(dataset))
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if stats["failed"]:
        raise SystemExit("Some rows failed; rerun to retry only missing rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/gsm8k_train_ta.jsonl"))
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="sarvamai/sarvam-translate")
    parser.add_argument("--language", default="ta")
    parser.add_argument("--target-language", default="Tamil")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Pilot prefix size; 0 = full train split")
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or args.limit < 0 or args.retries < 0:
        parser.error("Invalid concurrency, token, limit or retry setting")
    if not 1.0 <= args.repetition_penalty <= 1.3:
        parser.error("Repetition penalty must be between 1.0 and 1.3")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.output) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        source_hash = hashlib.sha256(json.dumps(list(dataset), sort_keys=True).encode()).hexdigest()
        manifest_path = args.output.with_suffix(".manifest.json")
        manifest = dict(dataset="openai/gsm8k", config="main", split="train", rows=len(dataset),
                        source_sha256=source_hash, language=args.language, target_language=args.target_language,
                        model=args.model, temperature=0.01, prompt=f"Translate the text below to {args.target_language}.")
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if any(existing.get(key) != value for key, value in manifest.items()):
                raise ValueError("Output manifest differs; use another output path")
            manifest = existing
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        with open(args.output, "a+b") as output:
            done = recover_output(output, dataset, args.language)
            print(f"GSM8K TRAIN: {len(dataset)} rows, {len(done)} already saved; concurrency={args.concurrency}", flush=True)
            asyncio.run(run(args, dataset, output, done))


if __name__ == "__main__":
    main()
