"""End-to-end verification of the FastAPI streaming endpoint (Part 4).

Sends four queries to ``POST /chat``, parses the SSE stream
incrementally, prints tokens to stdout as they arrive, and writes both
``api_run_<UTC>.txt`` and ``api_run_<UTC>.json`` under ``tests/logs/``
(same artefact style as Parts 2 and 3). Per query we record
**time-to-first-token (TTFT)** alongside total latency: the gap between
them is the user-visible value of streaming.

Usage:
    # Terminal 1: launch the API
    .\\scripts\\run_api.ps1

    # Terminal 2 (after the API logs "Application startup complete"):
    python tests\\verify_api.py
    python tests\\verify_api.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx


DEMO_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "What chunk size does this RAG pipeline use, and what is the overlap?",
        "rag",
    ),
    (
        "What is 17 multiplied by 24?",
        "direct",
    ),
    (
        "Who are you?",
        "direct",
    ),
    (
        "What is the recommended recipe for a chocolate cake?",
        "rag",  # router -> RAG, retriever -> 0 hits, fallback fires.
    ),
)


@dataclass
class QueryResult:
    """Aggregated outcome of one streamed ``/chat`` round-trip."""

    query: str
    expected_route: str
    actual_route: str | None = None
    routing_method: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    answer: str = ""
    total_elapsed_ms: float | None = None
    time_to_first_token_ms: float | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def streamed_text(self) -> str:
        return "".join(self.tokens)


class TeeLog:
    """Tee writer: prints to stdout while buffering for a log file."""

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def write(self, line: str = "") -> None:
        print(line, flush=True)
        self._buffer.append(line)

    def section(self, title: str) -> None:
        self.write("")
        self.write("=" * 70)
        self.write(title)
        self.write("=" * 70)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._buffer) + "\n", encoding="utf-8")


def _iter_sse_events(response: httpx.Response) -> Iterator[dict[str, Any]]:
    """Parse a chunked ``text/event-stream`` body into a stream of events.

    The SSE format is line-oriented; events are separated by blank lines
    and each event has ``event:`` and ``data:`` fields. ``sse-starlette``
    also emits ``: ping`` keepalive comments, which we silently skip.
    """
    event_name: str | None = None
    data_lines: list[str] = []

    for raw_line in response.iter_lines():
        if raw_line == "":
            if event_name is not None and data_lines:
                yield {
                    "event": event_name,
                    "data": json.loads("\n".join(data_lines)),
                }
            event_name = None
            data_lines = []
            continue

        if raw_line.startswith(":"):
            # SSE comment (keepalive) - ignore.
            continue

        if raw_line.startswith("event:"):
            event_name = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            data_lines.append(raw_line[len("data:"):].lstrip())


def _stream_one_query(
    client: httpx.Client,
    log: TeeLog,
    query: str,
    expected_route: str,
) -> QueryResult:
    """Send one ``POST /chat`` and consume the SSE stream live."""
    log.write("")
    log.write("-" * 70)
    log.write(f"QUERY: {query}")
    log.write(f"EXPECTED ROUTE: {expected_route}")
    log.write("-" * 70)

    result = QueryResult(query=query, expected_route=expected_route)

    perf_start = time.perf_counter()

    with client.stream(
        "POST",
        "/chat",
        json={"query": query},
        headers={"Accept": "text/event-stream"},
    ) as response:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            result.error = f"HTTP {response.status_code}: {exc}"
            log.write(f"[FAIL] {result.error}")
            return result

        first_token_seen = False

        for event in _iter_sse_events(response):
            name = event["event"]
            data = event["data"]
            elapsed_ms = round((time.perf_counter() - perf_start) * 1000.0, 2)
            result.events.append(
                {
                    "event": name,
                    "data": data,
                    "wall_elapsed_ms": elapsed_ms,
                }
            )

            if name == "route":
                result.actual_route = data.get("route")
                result.routing_method = data.get("method")
                log.write("")
                log.write(
                    f"  [route]      route={data.get('route')}  "
                    f"method={data.get('method')}  "
                    f"server_elapsed_ms={data.get('elapsed_ms')}"
                )

            elif name == "tool_call":
                result.tool_calls.append(data)
                log.write(
                    f"  [tool_call]  name={data.get('name')}  "
                    f"hits={data.get('hits')}  "
                    f"top_score={data.get('top_score')}  "
                    f"tool_elapsed_ms={data.get('tool_elapsed_ms')}"
                )

            elif name == "token":
                token = data.get("value") or ""
                result.tokens.append(token)
                if not first_token_seen:
                    first_token_seen = True
                    result.time_to_first_token_ms = elapsed_ms
                    log.write("")
                    log.write(f"  [stream]     (first token at {elapsed_ms:.1f} ms)")
                # Stdout-only: the assembled answer is recorded below.
                print(token, end="", flush=True)

            elif name == "done":
                result.answer = data.get("answer") or ""
                result.total_elapsed_ms = data.get("total_elapsed_ms")
                if first_token_seen:
                    print()
                log.write("")
                log.write(
                    f"  [done]       total_elapsed_ms={result.total_elapsed_ms}  "
                    f"answer_chars={len(result.answer)}"
                )

            elif name == "error":
                result.error = f"{data.get('kind')}: {data.get('message')}"
                if first_token_seen:
                    print()
                log.write("")
                log.write(f"  [error]      {result.error}")

    log.write("")
    log.write("ANSWER (assembled):")
    for line in (result.answer or "").splitlines() or [""]:
        log.write(f"  > {line}")

    if result.actual_route is None and result.error is None:
        result.error = "stream ended without a 'done' event"

    log.write("")
    if result.error:
        log.write(f"[FAIL] {result.error}")
    elif result.actual_route != result.expected_route:
        log.write(
            f"[WARN] expected_route={result.expected_route}  "
            f"actual_route={result.actual_route}"
        )
    else:
        log.write(
            f"[OK]   expected_route={result.expected_route}  "
            f"actual_route={result.actual_route}  "
            f"TTFT={result.time_to_first_token_ms:.1f} ms  "
            f"total={result.total_elapsed_ms:.1f} ms"
        )
    return result


def _check_health(client: httpx.Client, log: TeeLog) -> bool:
    """Hit ``GET /health`` and log the response."""
    try:
        response = client.get("/health", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        log.write(f"[FAIL] /health unreachable: {exc}")
        return False

    payload = response.json()
    log.write(f"  status:           {payload.get('status')}")
    log.write(f"  ollama_alive:     {payload.get('ollama_alive')}")
    log.write(f"  retriever_ready:  {payload.get('retriever_ready')}")
    log.write(f"  retriever_chunks: {payload.get('retriever_chunks')}")
    log.write(f"  ollama_model:     {payload.get('ollama_model')}")
    log.write(f"  embed_model:      {payload.get('embed_model')}")
    return payload.get("status") == "ok"


def _save_artifacts(
    log: TeeLog,
    log_dir: Path,
    timestamp: str,
    base_url: str,
    results: list[QueryResult],
) -> tuple[Path, Path]:
    txt_path = log_dir / f"api_run_{timestamp}.txt"
    log.save(txt_path)

    json_path = log_dir / f"api_run_{timestamp}.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "queries": [
            {
                "query": r.query,
                "expected_route": r.expected_route,
                "actual_route": r.actual_route,
                "routing_method": r.routing_method,
                "tool_calls": r.tool_calls,
                "answer": r.answer,
                "answer_chars": len(r.answer),
                "num_tokens": len(r.tokens),
                "time_to_first_token_ms": r.time_to_first_token_ms,
                "total_elapsed_ms": r.total_elapsed_ms,
                "error": r.error,
                "events": r.events,
            }
            for r in results
        ],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Where the FastAPI server is reachable (default: %(default)s).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("tests/logs"),
        help="Directory for run logs (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout seconds (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = TeeLog()

    log.section("API SSE STREAMING VERIFICATION (Part 4)")
    log.write(f"Timestamp:  {datetime.now(timezone.utc).isoformat()}")
    log.write(f"Base URL:   {args.base_url}")
    log.write(f"Timeout:    {args.timeout}s")

    log.section("HEALTH PROBE  (GET /health)")
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        if not _check_health(client, log):
            log.write("")
            log.write("[FAIL] health probe did not return status='ok' - aborting.")
            txt, jsn = _save_artifacts(
                log, args.log_dir, timestamp, args.base_url, results=[]
            )
            print(f"\nFull trace saved to: {txt}")
            print(f"JSON trace saved to: {jsn}")
            return 1

        log.section("DEMO QUERIES  (POST /chat - SSE)")
        results: list[QueryResult] = []
        for query, expected_route in DEMO_QUERIES:
            results.append(_stream_one_query(client, log, query, expected_route))

    log.section("SUMMARY")
    routes_ok = sum(
        1 for r in results if r.error is None and r.actual_route == r.expected_route
    )
    streaming_proof = sum(
        1
        for r in results
        if r.time_to_first_token_ms is not None
        and r.total_elapsed_ms is not None
        and r.time_to_first_token_ms < r.total_elapsed_ms - 100
    )
    log.write(f"Queries executed:                {len(results)} / {len(DEMO_QUERIES)}")
    log.write(f"Routes as expected:              {routes_ok} / {len(results)}")
    log.write(
        f"Streaming verified (TTFT < total): {streaming_proof} / {len(results)}"
    )
    log.write("")
    for r in results:
        ttft = (
            f"{r.time_to_first_token_ms:.0f} ms"
            if r.time_to_first_token_ms is not None
            else "n/a"
        )
        total = (
            f"{r.total_elapsed_ms:.0f} ms"
            if r.total_elapsed_ms is not None
            else "n/a"
        )
        status = "OK" if r.error is None and r.actual_route == r.expected_route else "FAIL"
        log.write(
            f"  [{status:4}] route={r.actual_route or '?':<6}  "
            f"TTFT={ttft:<10}  total={total:<10}  "
            f"tokens={len(r.tokens):<3}  query=\"{r.query[:60]}\""
        )

    log.section("DONE")
    txt_path, json_path = _save_artifacts(
        log, args.log_dir, timestamp, args.base_url, results
    )
    print(f"\nFull trace saved to: {txt_path}")
    print(f"JSON trace saved to: {json_path}")
    return 0 if routes_ok == len(DEMO_QUERIES) else 1


if __name__ == "__main__":
    sys.exit(main())
