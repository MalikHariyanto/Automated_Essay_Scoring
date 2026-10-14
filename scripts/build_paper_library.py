#!/usr/bin/env python3
"""Build the offline paper guide from public, paraphrased learning notes.

No uploaded document or PDF text is needed to rebuild this guide.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/paper_library_data.json"
TEMPLATE = ROOT / "scripts/templates/paper_library.html"
OUTPUT = ROOT / "docs/paper_library_interaktif.html"


def validate(data):
    papers = data["papers"]
    expected = {f"{prefix}{i:02d}" for prefix in ("p", "m") for i in range(1, 22)}
    assert {p["id"] for p in papers} == expected and len(papers) == 42, "Source inventory differs"
    assert data["collection"]["total_sources"] == len(papers)
    topics = {t["id"] for t in data["topics"]}
    required = ("title", "authors", "venue", "publication_type", "version_note", "index",
                "topic", "subtopic", "summary", "process", "formulas", "terms", "findings",
                "limitations", "aes_relevance", "reading_path", "sources", "coverage")
    for p in papers:
        assert all(k in p for k in required), p["id"]
        assert p["topic"] in topics and p["authors"] and p["title"], p["id"]
        assert p["source_kind"] in ("individual", "merged")
        assert p["index"]["status"] in ("unverified", "not_applicable"), "Unsupported index claim"
        assert p["terms"] and p["findings"] and p["reading_path"] and p["coverage"]["scope"], p["id"]
        assert p["process"] or p["id"] == "m19", "Missing available process"
        lo, hi = (p["source_page_range"][k] for k in ("start", "end"))
        for group, key in (("process", "page"), ("formulas", "page"), ("terms", "page"),
                           ("findings", "page"), ("model_roles", "evidence_page"),
                           ("reading_path", "pages"), ("sources", "pages")):
            for item in p[group]:
                page = str(item[key])
                # Ignore table/section numbering after the PDF page citation.
                page = re.split(r"Tabel|Table|§", page)[0]
                refs = [int(n) for n in re.findall(r"\d+", page)]
                assert refs and all(lo <= n <= hi for n in refs), (p["id"], group, item[key], lo, hi)
                if group == "process":
                    assert all(item.get(k) for k in ("title", "input", "action", "output", "note"))
                if group == "formulas":
                    assert all(item.get(k) for k in ("label", "expression", "explanation", "variables"))
    return papers


def build(output=OUTPUT):
    raw = DATA.read_text(encoding="utf-8")
    data = json.loads(raw)
    papers = validate(data)
    artwork = {}
    for variable, filename in (("comic-backdrop", "paper_splash.svg"),
                               ("hero-skyline", "paper_city.svg")):
        svg = (ROOT / "scripts/assets" / filename).read_bytes()
        artwork[variable] = "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")
    colour_css = ":root{" + ";".join(f'--{key}:url("{value}")' for key, value in artwork.items()) + "}\n"
    colour_css += (ROOT / "scripts/templates/paper_library_full_colour.css").read_text(encoding="utf-8")
    values = {
        "__FULL_COLOUR_CSS__": colour_css,
        "__PAPER_LIBRARY_DATA__": json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"),
        "__KATEX_JS__": (ROOT / "scripts/vendor/paper_katex.js").read_text().replace("</script", "<\\/script"),
        "__KATEX_CSS__": (ROOT / "scripts/vendor/paper_katex.css").read_text(),
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    for marker, value in values.items():
        assert html.count(marker) == 1, f"Missing or repeated marker: {marker}"
        html = html.replace(marker, value)
    assert not re.search(r"__(?:PAPER_LIBRARY_DATA|KATEX_JS|KATEX_CSS|FULL_COLOUR_CSS)__", html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return {"sources": len(papers), "process_steps": sum(len(p["process"]) for p in papers),
            "term_contexts": sum(len(p["terms"]) for p in papers), "output": str(output.relative_to(ROOT)),
            "data_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "html_sha256": hashlib.sha256(html.encode()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
