"""
Scrape presidential speech transcripts from the Miller Center.

Targets speeches from 2000-2024 (Bush through Biden/Trump) to build a
domain-specific corpus for Stage 1 pre-fine-tuning. Earlier speeches are
available but less relevant to modern political language patterns.

Output: raw/miller_center_speeches.csv
Columns: president, date, title, url, full_text, paragraphs (JSON list)
"""

import os
import re
import json
import time
import argparse
import csv
from urllib.parse import urljoin
from datetime import datetime

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://millercenter.org"
SPEECH_INDEX = BASE_URL + "/the-presidency/presidential-speeches"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")

# President IDs on Miller Center (for filtering)
PRESIDENTS = {
    "George W. Bush": 8391,
    "Barack Obama": 8392,
    "Donald Trump": 8396,
    "Joseph Biden": 8397,
}

HEADERS = {
    "User-Agent": "NostalgiaClassifier/1.0 (academic research; contact: jacobcrainic2008@gmail.com)"
}

# Polite delay between requests (seconds)
DELAY = 1.5


def get_speech_urls(president_id=None, max_pages=20):
    """Crawl the speech index to collect individual speech URLs."""
    urls = []
    page = 0

    while page < max_pages:
        params = {"page": page}
        if president_id:
            params[f"field_president_target_id[{president_id}]"] = president_id

        resp = requests.get(SPEECH_INDEX, params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"  Page {page}: HTTP {resp.status_code}, stopping")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find speech links -- they follow a consistent URL pattern
        links = soup.find_all("a", href=re.compile(r"/the-presidency/presidential-speeches/"))
        new_urls = []
        for link in links:
            href = link.get("href", "")
            if href and href != "/the-presidency/presidential-speeches":
                full_url = urljoin(BASE_URL, href)
                if full_url not in urls and full_url not in new_urls:
                    new_urls.append(full_url)

        if not new_urls:
            break

        urls.extend(new_urls)
        print(f"  Page {page}: found {len(new_urls)} speeches (total: {len(urls)})")
        page += 1
        time.sleep(DELAY)

    return urls


def extract_speech(url):
    """Extract transcript and metadata from a single speech page."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title is in the main heading
    title_el = soup.find("h2") or soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    # President name from breadcrumb or page metadata
    president = ""
    meta_pres = soup.find("meta", {"name": "dcterms.creator"})
    if meta_pres:
        president = meta_pres.get("content", "")
    else:
        # Try extracting from page title
        page_title = soup.find("title")
        if page_title:
            text = page_title.get_text()
            for pname in PRESIDENTS:
                if pname.split()[-1].lower() in text.lower():
                    president = pname
                    break

    # Date from title (format: "Month DD, YYYY: ...")
    date_str = ""
    date_match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}",
        title
    )
    if date_match:
        date_str = date_match.group()

    # Transcript is in the expandable text section or main content area
    transcript_div = (
        soup.find("div", class_="field-docs-content")
        or soup.find("div", class_="transcript-inner")
        or soup.find("div", id="dp-expandable-text")
        or soup.find("div", class_="view-mode-full")
    )

    if not transcript_div:
        # Fallback: look for the largest text block on the page
        all_divs = soup.find_all("div")
        best = None
        best_len = 0
        for div in all_divs:
            text = div.get_text(strip=True)
            if len(text) > best_len and len(text) > 500:
                best = div
                best_len = len(text)
        transcript_div = best

    if not transcript_div:
        return None

    # Extract paragraphs
    paragraphs = []
    for p in transcript_div.find_all("p"):
        text = p.get_text(strip=True)
        # Skip very short paragraphs (applause markers, etc.)
        if len(text) > 30:
            # Clean up common artifacts
            text = re.sub(r"\[.*?\]", "", text)  # Remove [Applause], [Laughter], etc.
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)

    if len(full_text) < 200:
        return None  # Too short to be a real transcript

    return {
        "president": president,
        "date": date_str,
        "title": title,
        "url": url,
        "full_text": full_text,
        "paragraphs": json.dumps(paragraphs),
        "n_paragraphs": len(paragraphs),
        "n_words": len(full_text.split()),
    }


def scrape_all(start_year=2000, presidents=None):
    """Scrape all speeches, optionally filtered by president."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("MILLER CENTER SPEECH SCRAPER")
    print("=" * 60)

    all_speeches = []

    if presidents:
        for pname in presidents:
            pid = PRESIDENTS.get(pname)
            if not pid:
                print(f"Unknown president: {pname}")
                continue
            print(f"\nScraping speeches for {pname} (ID: {pid})...")
            urls = get_speech_urls(president_id=pid)
            for i, url in enumerate(urls):
                print(f"  [{i+1}/{len(urls)}] {url.split('/')[-1][:50]}...")
                speech = extract_speech(url)
                if speech:
                    all_speeches.append(speech)
                    print(f"    OK: {speech['n_paragraphs']} paragraphs, {speech['n_words']} words")
                else:
                    print(f"    SKIP: no transcript found")
                time.sleep(DELAY)
    else:
        print("\nScraping all speeches (no president filter)...")
        urls = get_speech_urls()
        for i, url in enumerate(urls):
            print(f"  [{i+1}/{len(urls)}] {url.split('/')[-1][:50]}...")
            speech = extract_speech(url)
            if speech:
                # Filter by year if specified
                if start_year and speech["date"]:
                    try:
                        year = int(re.search(r"\d{4}", speech["date"]).group())
                        if year < start_year:
                            print(f"    SKIP: year {year} < {start_year}")
                            continue
                    except (AttributeError, ValueError):
                        pass
                all_speeches.append(speech)
                print(f"    OK: {speech['n_paragraphs']} paragraphs, {speech['n_words']} words")
            else:
                print(f"    SKIP: no transcript found")
            time.sleep(DELAY)

    # Save
    output_path = os.path.join(OUTPUT_DIR, "miller_center_speeches.csv")
    if all_speeches:
        keys = all_speeches[0].keys()
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_speeches)

    print(f"\n{'='*60}")
    print(f"DONE: {len(all_speeches)} speeches saved to {output_path}")
    total_words = sum(s["n_words"] for s in all_speeches)
    total_paras = sum(s["n_paragraphs"] for s in all_speeches)
    print(f"Total: {total_words:,} words across {total_paras:,} paragraphs")
    print(f"{'='*60}")

    return all_speeches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Miller Center speeches")
    parser.add_argument("--start-year", type=int, default=2000,
                        help="Earliest year to include (default: 2000)")
    parser.add_argument("--presidents", nargs="+",
                        choices=list(PRESIDENTS.keys()),
                        help="Filter to specific presidents")
    parser.add_argument("--all", action="store_true",
                        help="Scrape ALL presidents (overrides --start-year filter)")
    args = parser.parse_args()

    if args.all:
        scrape_all(start_year=None, presidents=None)
    else:
        scrape_all(start_year=args.start_year, presidents=args.presidents)
