import html
import re


def preprocess(series):
    texts = (
        series.fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()  # keep as list for downstream preprocessing
    )
    texts = [preprocess_token(t) for t in texts]
    return texts


def preprocess_token(t):
    t = html.unescape(str(t))
    t = t.lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&?#?[a-z0-9]+;s?", " ", t)  # kill &gt; &lt; &amp; etc
    t = re.sub(r"[^a-z0-9\s\-\.\,]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()
