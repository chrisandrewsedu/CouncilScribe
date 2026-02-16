"""Download meeting videos from URLs and CATS TV archive."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import requests

CATSTV_BASE_URL = "https://catstv.net/government.php"
CATSTV_BLOB_BASE = "https://catstv.blob.core.windows.net/videoarchive"

# Timeout for downloads (connect, read) in seconds
_CONNECT_TIMEOUT = 30
_READ_TIMEOUT = 600  # 10 minutes for large video files


def download_from_url(
    url: str,
    output_path: str | Path,
    progress: bool = True,
) -> Path:
    """Download a video file from a direct URL.

    Supports any direct video URL (mp4, m4v, mkv, etc.) and also CATS TV
    page URLs — if the URL points to a catstv.net page, it extracts the
    video blob URL automatically.

    Args:
        url: Direct video URL or CATS TV page URL.
        output_path: Local path to save the downloaded file.
        progress: If True, print download progress.

    Returns:
        Path to the downloaded file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If it's a CATS TV page URL, resolve to the blob URL
    resolved = _resolve_video_url(url)

    resp = requests.get(resolved, stream=True, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 8192

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if progress and total > 0:
                pct = (downloaded / total) * 100
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                print(f"\r  Downloading: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    if progress:
        print()  # newline after progress

    return output_path


def _resolve_video_url(url: str) -> str:
    """If url is a CATS TV page, extract the blob video URL. Otherwise return as-is."""
    parsed = urlparse(url)

    # Already a direct blob URL
    if "catstv.blob.core.windows.net" in parsed.netloc:
        return url

    # CATS TV page URL — scrape the video filename
    if "catstv.net" in parsed.netloc:
        return _extract_blob_url_from_page(url)

    # Any other direct URL — return as-is
    return url


def _extract_blob_url_from_page(page_url: str) -> str:
    """Scrape a CATS TV page to find the video blob URL.

    The page loads videos via JavaScript with data-m4v attributes or
    inline jPlayer config. We try both approaches.
    """
    from bs4 import BeautifulSoup

    resp = requests.get(page_url, timeout=(_CONNECT_TIMEOUT, 60))
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Try 1: Find jPlayer config in inline script (default video for the page)
    for script in soup.find_all("script"):
        text = script.string or ""
        match = re.search(r'm4v:\s*["\']([^"\']+\.m4v)["\']', text)
        if match:
            m4v = match.group(1)
            if m4v.startswith("http"):
                return m4v
            return f"{CATSTV_BLOB_BASE}/{m4v}"

    # Try 2: Find data-m4v attributes on result links
    link = soup.find("a", attrs={"data-m4v": True})
    if link:
        m4v = link["data-m4v"]
        if m4v.startswith("http"):
            return m4v
        return f"{CATSTV_BLOB_BASE}/{m4v}"

    raise ValueError(
        f"Could not find a video URL on the CATS TV page: {page_url}\n"
        "Try using a direct blob URL instead (https://catstv.blob.core.windows.net/videoarchive/...)."
    )


# ---------------------------------------------------------------------------
# CATS TV Meeting Browser
# ---------------------------------------------------------------------------

def fetch_catstv_meetings(search_url: str | None = None) -> list[dict]:
    """Scrape CATS TV archive and return a list of available meetings.

    Each meeting dict contains:
        - name: Meeting title
        - subtitle: Additional description
        - date: Meeting date string
        - duration: Duration string
        - m4v: Filename on blob storage
        - video_url: Full blob download URL
        - permalink: CATS TV permalink
        - has_agenda: Whether an agenda link exists
        - documents_url: Link to meeting documents

    Args:
        search_url: CATS TV search URL. Defaults to the full government archive.

    Returns:
        List of meeting dicts sorted by date (newest first).
    """
    from bs4 import BeautifulSoup

    if search_url is None:
        search_url = f"{CATSTV_BASE_URL}?issearch=govt"

    resp = requests.get(search_url, timeout=(_CONNECT_TIMEOUT, 60))
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    meetings = []
    for link in soup.find_all("a", attrs={"data-m4v": True}):
        m4v = link.get("data-m4v", "")
        if not m4v:
            continue

        video_url = f"{CATSTV_BLOB_BASE}/{m4v}" if not m4v.startswith("http") else m4v

        meetings.append({
            "name": link.get("data-name", "").strip(),
            "subtitle": link.get("data-subtitle", "").strip(),
            "date": link.get("data-date", "").strip(),
            "duration": link.get("data-duration", "").strip(),
            "m4v": m4v,
            "video_url": video_url,
            "permalink": link.get("data-permalink", "").strip(),
            "has_agenda": link.get("data-hasagenda", "").lower() == "true",
            "documents_url": link.get("data-documentsurl", "").strip(),
        })

    return meetings


def display_catstv_meetings(meetings: list[dict], limit: int = 25) -> None:
    """Print a numbered table of meetings for user selection."""
    shown = meetings[:limit]
    print(f"{'#':>4}  {'Date':<12} {'Duration':<10} Title")
    print(f"{'─'*4}  {'─'*12} {'─'*10} {'─'*50}")
    for i, m in enumerate(shown):
        title = m["name"]
        if m["subtitle"]:
            title += f" — {m['subtitle']}"
        if len(title) > 60:
            title = title[:57] + "..."
        print(f"{i:>4}  {m['date']:<12} {m['duration']:<10} {title}")

    if len(meetings) > limit:
        print(f"\n  ... and {len(meetings) - limit} more. Pass a larger limit to see all.")
