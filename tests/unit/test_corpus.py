"""The corpus builder, offline: cleaning, passage sizes, one collection per document."""

import json

import pytest

from bluff import corpus


def test_clean_wikipedia_drops_reference_sections():
    text = ("Kenya is a country.\n\n== History ==\nIndependence in 1963.\n\n"
            "== See also ==\nUganda\n\n== References ==\n[1] A book.\n")
    out = corpus.clean_wikipedia(text)
    assert "Independence in 1963." in out and "History" in out
    assert "Uganda" not in out and "A book" not in out


def test_chunks_stay_within_bounds():
    paras = [" ".join(f"w{i}_{j}" for j in range(120)) + "." for i in range(20)]  # 2,400 words
    chunks = corpus.chunk("\n".join(paras))
    sizes = [len(c.split()) for c in chunks]
    assert sum(sizes) == 2400
    assert max(sizes) <= corpus.CHUNK_WORDS[2]
    assert all(s >= corpus.CHUNK_WORDS[0] for s in sizes)


def test_a_long_paragraph_is_split_by_sentence():
    para = " ".join(f"Sentence {i} has a few words here." for i in range(300))  # 2,100 words
    assert all(len(c.split()) <= corpus.CHUNK_WORDS[2] for c in corpus.chunk(para))


def _cache(tmp_path, title, text):
    (tmp_path / "raw" / "corpus").mkdir(parents=True, exist_ok=True)
    key = corpus._slug("wp-" + title)
    (tmp_path / "raw" / "corpus" / f"{key}.json").write_text(
        json.dumps({"text": text, "title": title}), encoding="utf-8")


def _cfg(tmp_path):
    return {"paths": {"data_dir": str(tmp_path)},
            "data": {"retrieval_corpus": str(tmp_path / "r.jsonl"),
                     "finetune_corpus": str(tmp_path / "f.jsonl")}}


def test_build_from_cache_splits_collections(tmp_path):
    _cache(tmp_path, "Nairobi", "Nairobi is the capital of Kenya. " * 200)
    _cache(tmp_path, "Mombasa", "Mombasa is a coastal city. " * 50)
    sources = tmp_path / "s.yaml"
    sources.write_text("wikipedia:\n"
                       "  - {title: Nairobi, domain: geo, collection: retrieval}\n"
                       "  - {title: Mombasa, domain: geo, collection: finetune}\n",
                       encoding="utf-8")
    m = corpus.run(_cfg(tmp_path), sources)
    assert m["documents"] == {"retrieval": 1, "finetune": 1} and m["failed"] == []
    r = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(c["doc_id"] == "wp-nairobi" and c["licence"] == "CC BY-SA 4.0" for c in r)


def test_same_document_in_both_collections_is_refused(tmp_path):
    _cache(tmp_path, "Nairobi", "Nairobi is the capital of Kenya. " * 50)
    sources = tmp_path / "s.yaml"
    sources.write_text("wikipedia:\n"
                       "  - {title: Nairobi, domain: geo, collection: retrieval}\n"
                       "  - {title: Nairobi, domain: geo, collection: finetune}\n",
                       encoding="utf-8")
    with pytest.raises(ValueError, match="both collections"):
        corpus.run(_cfg(tmp_path), sources)
