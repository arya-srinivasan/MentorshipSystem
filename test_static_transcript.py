"""
test_static_transcript.py

Fallback / sanity-check path: feeds a static transcript (a text file with
one caption line per row, optionally "Speaker: text") through the exact
same chunking + backend pipeline that live_transcription_tool.py uses,
without needing Zoom, Playwright, or RTMS at all.

Usage:
    # 1. Start the backend in another terminal:
    #    python backend.py
    #
    # 2. Then run this:
    python test_static_transcript.py sample_transcript.txt
"""

import asyncio
import os
import sys

import httpx

from live_transcription_tool import RealTimeTranscriptChunker, send_chunk_to_backend

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
SESSION_ID = os.environ.get("TRANSCRIPT_SESSION_ID", "static_test_session")
DELAY_BETWEEN_LINES_S = float(os.environ.get("STATIC_TEST_DELAY", "0"))


def parse_line(line: str) -> tuple:
    """Splits 'Speaker: text' into (speaker, text). Falls back to ('Unknown', line)."""
    line = line.strip()
    if ":" in line:
        speaker, _, text = line.partition(":")
        speaker = speaker.strip()
        text = text.strip()
        if speaker and text:
            return speaker, text
    return "Unknown", line


async def main(transcript_path: str):
    with open(transcript_path, "r") as f:
        lines = [l for l in f.readlines() if l.strip()]

    print(f"Loaded {len(lines)} caption lines from {transcript_path}")
    print(f"Posting chunks to {BACKEND_URL}/transcript/chunk (session_id={SESSION_ID})\n")

    chunker = RealTimeTranscriptChunker(target_word_count=100, overlap_word_count=20)

    async with httpx.AsyncClient() as http_client:
        try:
            r = await http_client.get(f"{BACKEND_URL}/health", timeout=5)
            r.raise_for_status()
            print("Backend health check OK\n")
        except Exception as e:
            print(f"Could not reach backend at {BACKEND_URL}: {e}")
            print("Make sure `python backend.py` is running first.")
            return

        for line in lines:
            speaker, text = parse_line(line)
            chunk = chunker.add_transcript_chunk(speaker, text)
            if chunk:
                await send_chunk_to_backend(chunk, http_client)

            if DELAY_BETWEEN_LINES_S:
                await asyncio.sleep(DELAY_BETWEEN_LINES_S)

        # flush whatever's left at the end
        chunk = chunker.force_flush()
        if chunk:
            await send_chunk_to_backend(chunk, http_client)

    print("\nDone. Check /transcript/chunks or your Pinecone index to confirm storage.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python test_static_transcript.py <transcript.txt>")

    asyncio.run(main(sys.argv[1]))
