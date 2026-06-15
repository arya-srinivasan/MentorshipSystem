import asyncio
import os
import sys
import time

import httpx

# add repo root to path so import works when run from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from live_transcription_tool import run_zoom_bot

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
ZOOM_LINK = os.environ.get("ZOOM_LINK")
HEADLESS = os.environ.get("HEADLESS", "true").lower() not in ("false", "0", "no")

if not ZOOM_LINK:
    print("Set the ZOOM_LINK environment variable to a Zoom meeting URL.")
    raise SystemExit(1)


def check_backend_health(url: str, timeout_s: float = 5.0) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=timeout_s)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"Could not reach backend at {url}: {e}")
        return False


async def main():
    print(f"Checking backend at {BACKEND_URL}...", end=" ")
    if not check_backend_health(BACKEND_URL):
        print("\nPlease start the backend first: python backend.py")
        return
    print("OK")

    print("Starting live transcription bot. Press Ctrl+C to quit.")

    try:
        await run_zoom_bot(ZOOM_LINK, headless=HEADLESS)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
    except Exception as e:
        print(f"Bot exited with error: {e}")
    finally:
        print("Live demo finished.")


if __name__ == '__main__':
    asyncio.run(main())
