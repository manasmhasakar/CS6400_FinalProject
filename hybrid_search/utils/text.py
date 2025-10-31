import html
import re
from typing import List, Optional, Union

PRICE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def to_text(x: Optional[Union[str, List[str]]]) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        parts = []
        for v in x:
            if v is None:
                continue
            s = str(v)
            if s:
                parts.append(s)
        return " ".join(parts)
    return str(x)


def build_text_field(title: Optional[str], description: Optional[List[str]], features: Optional[List[str]], max_chars: int = 4096) -> str:
    t = []
    if title:
        t.append(strip_html(str(title)))
    desc = strip_html(to_text(description))
    if desc:
        t.append(desc)
    feats = strip_html(to_text(features))
    if feats:
        t.append(feats)
    out = " \n".join([s for s in t if s])
    if len(out) > max_chars:
        out = out[:max_chars]
    return out


def parse_price(val: Optional[object]) -> float:
    if val is None:
        return float("nan")
    s = str(val).strip()
    if not s or s.lower() == "none" or s.lower() == "nan":
        return float("nan")
    # Extract first numeric substring
    m = PRICE_RE.search(s.replace(",", ""))
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except Exception:
        return float("nan")
