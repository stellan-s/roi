from pathlib import Path
import time, feedparser, pandas as pd

def fetch_news(feed_urls, cache_dir: str) -> pd.DataFrame:
    import concurrent.futures
    import threading

    def fetch_single_feed(url, timeout_seconds=15):
        """Fetch a single news feed with timeout"""
        result = []
        try:
            print(f"Fetching news from {url}...")
            # Use thread-based timeout instead of signal (which doesn't work reliably on macOS)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(feedparser.parse, url)
                try:
                    feed = future.result(timeout=timeout_seconds)
                    for e in feed.entries:
                        result.append({
                            "source": url,
                            "title": getattr(e, "title", ""),
                            "summary": getattr(e, "summary", ""),
                            "published": getattr(e, "published", ""),
                            "link": getattr(e, "link", "")
                        })
                    print(f"✅ Fetched {len(result)} articles from {url}")
                except concurrent.futures.TimeoutError:
                    print(f"⏰ Timeout ({timeout_seconds}s) för news feed {url}")
                    return []
        except Exception as e:
            print(f"❌ Misslyckades hämta news från {url}: {e}")
            return []
        return result

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    for url in feed_urls:
        feed_rows = fetch_single_feed(url)
        rows.extend(feed_rows)
        time.sleep(0.2)  # artigt
    # Build DataFrame with fixed schema so empty fetches still yield expected columns.
    df = pd.DataFrame(rows, columns=["source", "title", "summary", "published", "link"])

    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        df = df.sort_values("published", ascending=False)

    df.to_parquet(Path(cache_dir)/"news.parquet", index=False)
    return df
