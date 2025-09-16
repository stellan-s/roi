from pathlib import Path
import time, feedparser, pandas as pd

def fetch_news(feed_urls, cache_dir: str) -> pd.DataFrame:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    rows=[]
    for url in feed_urls:
        feed = feedparser.parse(url)
        for e in feed.entries:
            rows.append({
                "source": url,
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "published": getattr(e, "published", ""),
                "link": getattr(e, "link", "")
            })
        time.sleep(0.2)  # artigt
    # Build DataFrame with fixed schema so empty fetches still yield expected columns.
    df = pd.DataFrame(rows, columns=["source", "title", "summary", "published", "link"])

    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        df = df.sort_values("published", ascending=False)

    df.to_parquet(Path(cache_dir)/"news.parquet", index=False)
    return df
