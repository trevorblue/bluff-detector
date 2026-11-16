"""Build the Kenyan document collections from public sources (data/corpus/sources.yaml).

Downloads the Constitution of Kenya 2010 (Wikisource, chapter by chapter) and a list of
Kenya-related English Wikipedia articles as plain text, removes reference sections, and writes:
- data/corpus/retrieval_chunks.jsonl  passages of about 300-800 tokens, for M1 RAG and M3
- data/corpus/finetune_docs.jsonl     whole documents, for M2 QLoRA and M3
Each document goes to exactly one collection (`collection:` in sources.yaml), never both. Every
line records its source URL and licence. Downloaded pages are cached in data/raw/corpus/, so a
re-run (e.g. after editing the list) only fetches what is new.

CRISP-DM: data preparation.
Run:    bluff build-corpus --config configs/variants/m1_rag.yaml
Reads:  data/corpus/sources.yaml (and the web, once per page)
Writes: data/corpus/retrieval_chunks.jsonl, data/corpus/finetune_docs.jsonl,
        data/corpus/manifest.json
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from bluff.utils import ensure_dir, write_json, write_jsonl

log = logging.getLogger(__name__)

USER_AGENT = "bluff-detector-corpus-builder/0.1 (academic research project)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
WIKIPEDIA_PAGE = "https://en.wikipedia.org/wiki/"
WIKISOURCE_PAGE = "https://en.wikisource.org/wiki/"
DROP_SECTIONS = {"see also", "references", "external links", "further reading", "notes",
                 "bibliography", "sources", "citations", "footnotes", "gallery"}
CHUNK_WORDS = (230, 450, 600)  # min / target / max words: about 300 / 585 / 780 tokens
PAUSE_S = 1.5  # between requests: Wikipedia rate-limits fast clients


def _get_json(url: str, params: dict[str, Any], tries: int = 6) -> dict[str, Any]:
    """GET with a pause before each request and polite retries when the wiki says "slow down"
    (HTTP 429 / 503), honouring its Retry-After header."""
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": USER_AGENT})
    for attempt in range(tries):
        time.sleep(PAUSE_S)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == tries - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
            log.info("rate-limited; waiting %.0f s", wait)
            time.sleep(min(wait, 120))
    raise RuntimeError("unreachable")


def fetch_wikipedia(title: str) -> tuple[str, str]:
    """Plain text of an English Wikipedia article and its canonical title."""
    data = _get_json(WIKIPEDIA_API, {"action": "query", "prop": "extracts", "explaintext": 1,
                                     "redirects": 1, "format": "json", "formatversion": 2,
                                     "titles": title})
    page = data["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        raise LookupError(f"Wikipedia has no article {title!r}")
    return page["extract"], page["title"]


def fetch_wikisource(page: str) -> str:
    """Rendered text of a Wikisource page (the Constitution is transcluded, so parse it)."""
    data = _get_json(WIKISOURCE_API, {"action": "parse", "page": page, "prop": "text",
                                      "format": "json", "formatversion": 2})
    raw = data["parse"]["text"]
    raw = re.sub(r"<(style|script)\b.*?</\1>", " ", raw, flags=re.S)
    raw = re.sub(r"<(br|/p|/div|/li|/h\d|/tr)\b[^>]*>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw)).replace("​", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    if len(text) < 200:
        raise LookupError(f"Wikisource page {page!r} is empty")
    return text


def clean_wikipedia(text: str) -> str:
    """Drop reference-style sections (See also, References, ...) and empty headings."""
    out, skip = [], False
    for line in text.splitlines():
        m = re.match(r"^(=+)\s*(.*?)\s*=+$", line.strip())
        if m:
            level, heading = len(m.group(1)), m.group(2).lower()
            if level == 2:
                skip = heading in DROP_SECTIONS
            if not skip:
                out.append(f"\n{m.group(2)}\n")
            continue
        if not skip and line.strip():
            out.append(line.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def chunk(text: str, min_w: int = CHUNK_WORDS[0], target_w: int = CHUNK_WORDS[1],
          max_w: int = CHUNK_WORDS[2]) -> list[str]:
    """Group paragraphs into passages of about target_w words, never over max_w (long
    paragraphs are split by sentence), merging a short last piece into the one before."""
    pieces: list[str] = []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        if len(para.split()) <= max_w:
            pieces.append(para)
            continue
        sentences, buf = re.split(r"(?<=[.;:])\s+", para), []
        for s in sentences:
            if buf and len(" ".join(buf + [s]).split()) > target_w:
                pieces.append(" ".join(buf))
                buf = []
            buf.append(s)
        if buf:
            pieces.append(" ".join(buf))
    chunks, cur = [], []
    for piece in pieces:
        if cur and len(" ".join(cur + [piece]).split()) > target_w:
            chunks.append("\n".join(cur))
            cur = []
        cur.append(piece)
    if cur:
        if chunks and len(" ".join(cur).split()) < min_w and \
                len((chunks[-1] + " " + " ".join(cur)).split()) <= max_w:
            chunks[-1] += "\n" + "\n".join(cur)
        else:
            chunks.append("\n".join(cur))
    return chunks


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _cached(cache: Path, key: str, fetch) -> dict[str, str]:
    path = cache / f"{_slug(key)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    doc = fetch()
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return doc


def load_documents(sources: dict[str, Any], cache: Path) -> tuple[list[dict], list[str]]:
    """Every source as {doc_id, title, text, source, url, licence, domain, collection}."""
    docs, failed = [], []
    c = sources.get("constitution")
    if c:
        for part in c["parts"]:
            page = c["prefix"] + part
            try:
                text = _cached(cache, "ws-" + page, lambda p=page: {"text": fetch_wikisource(p)})
            except Exception as e:  # noqa: BLE001 - a missing page is reported, not fatal
                failed.append(f"{page}: {e}")
                continue
            docs.append({
                "doc_id": f"constitution-{_slug(part)}",
                "title": f"Constitution of Kenya 2010, {part}",
                "text": text["text"], "source": "Constitution of Kenya 2010",
                "url": WIKISOURCE_PAGE + urllib.parse.quote(page.replace(" ", "_")),
                "licence": "public legal text (Wikisource)", "domain": c["domain"],
                "collection": c["collection"]})
    for s in sources.get("wikipedia", []):
        try:
            got = _cached(cache, "wp-" + s["title"], lambda t=s["title"]: dict(
                zip(("text", "title"), fetch_wikipedia(t), strict=True)))
        except Exception as e:  # noqa: BLE001
            failed.append(f"Wikipedia {s['title']}: {e}")
            continue
        docs.append({
            "doc_id": f"wp-{_slug(got['title'])}", "title": got["title"],
            "text": clean_wikipedia(got["text"]), "source": "Wikipedia",
            "url": WIKIPEDIA_PAGE + urllib.parse.quote(got["title"].replace(" ", "_")),
            "licence": "CC BY-SA 4.0", "domain": s["domain"], "collection": s["collection"]})
    return docs, failed


def run(cfg: dict[str, Any],
        sources_path: str | Path = "data/corpus/sources.yaml") -> dict[str, Any]:
    d = cfg["data"]
    retrieval_path, finetune_path = Path(d["retrieval_corpus"]), Path(d["finetune_corpus"])
    sources = yaml.safe_load(Path(sources_path).read_text(encoding="utf-8"))
    cache = ensure_dir(Path(cfg["paths"]["data_dir"]) / "raw" / "corpus")
    docs, failed = load_documents(sources, cache)

    seen: dict[str, str] = {}
    for doc in docs:  # the same document must never sit in both collections
        if seen.setdefault(doc["doc_id"], doc["collection"]) != doc["collection"]:
            raise ValueError(f"{doc['doc_id']} is listed in both collections")

    meta_keys = ("doc_id", "title", "source", "url", "licence", "domain")
    chunks = [{"id": f"{doc['doc_id']}-{i:03d}", "text": f"{doc['title']}\n{text}",
               **{k: doc[k] for k in meta_keys}}
              for doc in docs if doc["collection"] == "retrieval"
              for i, text in enumerate(chunk(doc["text"]), start=1)]
    finetune = [{"id": doc["doc_id"], "text": f"{doc['title']}\n\n{doc['text']}",
                 **{k: doc[k] for k in meta_keys}}
                for doc in docs if doc["collection"] == "finetune"]
    write_jsonl(retrieval_path, chunks)
    write_jsonl(finetune_path, finetune)

    words = [len(c["text"].split()) for c in chunks]
    manifest = {
        "documents": {"retrieval": sum(x["collection"] == "retrieval" for x in docs),
                      "finetune": len(finetune)},
        "retrieval_chunks": len(chunks),
        "chunk_words": {"min": min(words, default=0), "max": max(words, default=0),
                        "mean": round(sum(words) / max(1, len(words)))},
        "finetune_words": sum(len(x["text"].split()) for x in finetune),
        "failed": failed,
    }
    write_json(retrieval_path.parent / "manifest.json", manifest)
    log.info("corpus: %d retrieval chunks from %d documents, %d fine-tuning documents, %d failed",
             len(chunks), manifest["documents"]["retrieval"], len(finetune), len(failed))
    return manifest
