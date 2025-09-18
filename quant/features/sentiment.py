import pandas as pd, re

POS = re.compile(r"\b(strong|record|raises|wins|contract|growth)\b", re.I)
NEG = re.compile(r"\b(profit warning|cuts|loss|weak|layoffs|investigation)\b", re.I)

def naive_sentiment(news_df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    # enkel mapping: matcha bolagsnamn/tickers i title/summary
    out=[]
    for _,r in news_df.iterrows():
        text = f"{r.get('title','')} {r.get('summary','')}"
        score = 0
        if POS.search(text): score += 1
        if NEG.search(text): score -= 1
        # crude mapping: if a ticker string appears in the text
        hits=[t for t in tickers if t.split(".")[0].replace("-","") in text.replace("-","").upper()]
        if not hits: continue
        for t in hits:
            out.append({"ticker": t, "published": r.get("published"), "title": r.get("title",""), "sent_score": score})
    df=pd.DataFrame(out)
    if df.empty: return pd.DataFrame(columns=["ticker","date","sent_score"])
    df["date"]=pd.to_datetime(df["published"]).dt.date
    senti = df.groupby(["date","ticker"])["sent_score"].mean().reset_index()
    return senti
