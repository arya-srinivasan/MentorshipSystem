"""
live_transcription_tool.py

Joins a Zoom meeting as a headless browser "guest", turns on closed
captions, scrapes the live caption DOM, groups captions into rolling
text chunks, and POSTs each chunk to the backend's
POST /transcript/chunk endpoint (see backend.py), which stores it in
Pinecone and runs the meeting_copilot_agent on it.

Usage:
    export ZOOM_LINK="https://us05web.zoom.us/j/..."
    export BACKEND_URL="http://localhost:8000"   # optional, this is the default
    python live_transcription_tool.py

Notes / known limitations:
- The CSS selectors in SELECTORS are best guesses at Zoom's web-client
  caption DOM. Zoom's class names are partially hashed/obfuscated and
  do change between releases, so if captions aren't being picked up,
  open the Zoom web client in a normal (non-headless) browser, turn on
  captions, and inspect the DOM to update these selectors.
- For local debugging, set headless=False (see run_zoom_bot) so you can
  watch the bot join the meeting.
"""

import asyncio
import os
import re
import time
import uuid
from typing import Awaitable, Callable, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from playwright.async_api import Page, async_playwright

load_dotenv()

ZOOM_LINK = os.environ.get("ZOOM_LINK")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
SESSION_ID = os.environ.get("TRANSCRIPT_SESSION_ID", "live_session")

SELECTORS = {
    "name_input": 'input[placeholder*="name"], input[id*="inputname"]',
    "join_button": 'button[class*="preview-join-button"], button[class*="joinBtn"]',
    "cc_btn": 'button[aria-label*="caption"], button[aria-label="CC"]',
    "caption_item": '[class*="caption-line"], [class*="transcript-item"]',
    "caption_speaker": '[class*="caption-speaker"], [class*="speaker-name"]',
    "caption_text": '[class*="caption-text"], [class*="caption-content"]',
    "ended_overlay": '[class*="meeting-ended"], [class*="leave-meeting"]',
}


# ---------------------------------------------------------------------------
# Chunking - groups individual caption lines into ~target_word_count chunks
# with a small sliding-window overlap, so the LLM gets coherent context
# instead of single short caption fragments.
# ---------------------------------------------------------------------------

class RealTimeTranscriptChunker:
    def __init__(self, target_word_count: int = 100, overlap_word_count: int = 20):
        self.target_word_count = target_word_count
        self.overlap_word_count = overlap_word_count
        # Each entry is (word, speaker, timestamp)
        self.current_chunk_words: List[tuple] = []

    def add_transcript_chunk(self, speaker: str, text: str) -> Optional[Dict]:
        now = time.time()
        for word in text.split():
            self.current_chunk_words.append((word, speaker, now))

        if len(self.current_chunk_words) >= self.target_word_count:
            return self.flush_and_slide()
        return None

    def force_flush(self) -> Optional[Dict]:
        if self.current_chunk_words:
            return self.flush_and_slide()
        return None

    def flush_and_slide(self) -> Dict:
        words = self.current_chunk_words

        full_text = " ".join(item[0] for item in words)
        unique_speakers = list({item[1] for item in words})
        start_time = words[0][2]
        end_time = words[-1][2]

        chunk_payload = {
            "text": full_text,
            "metadata": {
                "speakers": unique_speakers,
                "start_time": start_time,
                "end_time": end_time,
                "word_count": len(words),
            },
        }

        # keep the tail `overlap_word_count` words for context continuity
        self.current_chunk_words = words[-self.overlap_word_count:] if self.overlap_word_count else []
        return chunk_payload


# ---------------------------------------------------------------------------
# Sending chunks to the backend
# ---------------------------------------------------------------------------

async def send_chunk_to_backend(chunk: Dict, http_client: httpx.AsyncClient):
    speakers = chunk["metadata"]["speakers"]
    speaker = speakers[0] if len(speakers) == 1 else ", ".join(speakers)

    payload = {
        "text": chunk["text"],
        "timestamp": str(chunk["metadata"]["end_time"]),
        "speaker": speaker,
        "session_id": SESSION_ID,
    }

    try:
        resp = await http_client.post(f"{BACKEND_URL}/transcript/chunk", json=payload, timeout=30)
        resp.raise_for_status()
        print(f"[sent] {len(chunk['text'].split())} words -> {resp.json()}")
    except Exception as e:
        print(f"[error] failed to send chunk to backend: {e}")


# ---------------------------------------------------------------------------
# Main bot loop
# ---------------------------------------------------------------------------

async def run_zoom_bot(
    meeting_url: str,
    on_caption: Optional[Callable[[str, str], Awaitable[None]]] = None,
    display_name: str = "Transcription Bot",
    poll_interval_ms: int = 1000,
    headless: bool = True,
):
    chunker = RealTimeTranscriptChunker(target_word_count=100, overlap_word_count=20)

    async with httpx.AsyncClient() as http_client:

        async def default_on_caption(speaker: str, text: str):
            chunk = chunker.add_transcript_chunk(speaker, text)
            if chunk:
                await send_chunk_to_backend(chunk, http_client)

        callback = on_caption or default_on_caption

        seen: Dict[str, float] = {}
        interval_s = poll_interval_ms / 1000

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=headless,
                args=[
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream",
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            context = await browser.new_context(
                permissions=["microphone", "camera"],
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            try:
                await _join(page, meeting_url, display_name)
                await _enable_captions(page)

                while True:
                    if await page.query_selector(SELECTORS["ended_overlay"]):
                        print("Meeting has ended. Exiting.")
                        break

                    now = time.time()
                    caption_items = await page.query_selector_all(SELECTORS["caption_item"])

                    for item in caption_items:
                        caption_speaker = await item.query_selector(SELECTORS["caption_speaker"])
                        caption_text = await item.query_selector(SELECTORS["caption_text"])
                        if not caption_text:
                            continue

                        speaker = (await caption_speaker.inner_text()) if caption_speaker else "Unknown"
                        text = (await caption_text.inner_text()).strip()
                        if not text:
                            continue

                        key = f"{speaker}:{text}"
                        if now - seen.get(key, 0) > 5.0:
                            seen[key] = now
                            try:
                                await callback(speaker, text)
                            except Exception as e:
                                print(f"Error in on_caption callback: {e}")

                    # prune old entries so `seen` doesn't grow unbounded
                    seen = {k: v for k, v in seen.items() if now - v < 30.0}

                    # sleep once per poll cycle, not once per caption item
                    await asyncio.sleep(interval_s)

            except Exception as e:
                print(f"Error in run_zoom_bot: {e}")
                raise
            finally:
                chunk = chunker.force_flush()
                if chunk:
                    await send_chunk_to_backend(chunk, http_client)
                await browser.close()
                print("Browser closed, bot exiting.")


async def _join(page: Page, meeting_url: str, display_name: str):
    await page.goto(meeting_url, wait_until="domcontentloaded", timeout=30_000)

    try:
        await page.wait_for_selector(SELECTORS["name_input"], timeout=15_000)
        await page.fill(SELECTORS["name_input"], display_name)
    except Exception:
        print("Failed to find name input field.")
        raise

    try:
        btn = await page.wait_for_selector(SELECTORS["join_button"])
        await btn.click()
    except Exception:
        print("Failed to find or click join button.")
        raise

    await page.wait_for_selector(SELECTORS["cc_btn"], timeout=60_000)
    print("Joined meeting successfully.")


async def _enable_captions(page: Page):
    try:
        btn = await page.wait_for_selector(SELECTORS["cc_btn"], timeout=15_000)
        if btn and await btn.get_attribute("aria-pressed") != "true":
            await btn.click()
            await asyncio.sleep(2)
            print("Closed captions enabled.")
    except Exception:
        print("Failed to find or click closed captions button.")
        raise


def _extract_meeting_id(meeting_url: str) -> Optional[str]:
    match = re.search(r"zoom\.us/(?:j|my)/(\d+)", meeting_url)
    return match.group(1) if match else None


if __name__ == "__main__":
    if not ZOOM_LINK:
        raise SystemExit("Set the ZOOM_LINK environment variable to a Zoom meeting URL.")

    # headless=False while you're getting the selectors right is very helpful
    asyncio.run(run_zoom_bot(ZOOM_LINK, display_name="Transcription Bot", headless=True))
