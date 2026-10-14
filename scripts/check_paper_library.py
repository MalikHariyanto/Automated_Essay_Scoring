#!/usr/bin/env python3
"""Exercise the published guide in Chromium; never executes research models."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/paper_library_interaktif.html"


def check(url):
    errors, external = [], []
    outcomes = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 1050}, reduced_motion="reduce")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external.append(request.url) if not request.url.startswith(url.rsplit("/", 2)[0]) and not request.url.startswith("data:") else None)
        page.goto(url)
        page.wait_for_function("window.PaperLibrary && document.querySelectorAll('.paper-card').length===42")
        assert page.evaluate("PaperLibrary.getState().paused"), "Reduced motion ignored"
        assert page.locator(".hero-mascot").evaluate("e=>getComputedStyle(e).animationName") == "none"
        outcomes.append("reduced-motion and 42-source inventory")
        ids = page.evaluate("PaperLibrary.data.papers.map(p=>p.id)")
        tabs = ["identity", "core", "process", "formulas", "terms", "findings", "reading"]
        panes = 0
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1050 if width > 500 else 844})
            for pid in ids:
                page.evaluate("id=>PaperLibrary.openPaper(id)", pid)
                for tab in tabs:
                    page.evaluate("tab=>PaperLibrary.setTab(tab)", tab)
                    assert page.locator("#readerContent").inner_text().strip(), (pid, tab)
                    assert page.locator("#readerContent").evaluate("e=>!e.innerText.includes('[object Object]')"), (pid, tab)
                    assert page.locator("#readerDialog").evaluate("e=>e.open"), pid
                    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1"), (width, pid, tab)
                    assert page.locator("#readerScroll").evaluate("e=>e.scrollWidth<=e.clientWidth+1"), (width, pid, tab, "reader horizontal overflow")
                    if tab == "formulas":
                        assert page.locator(".katex-error").count() == 0, pid
                    panes += 1
            page.evaluate("PaperLibrary.closeReader()")
        outcomes.append(f"{panes} reading panes: desktop/mobile, all 42 sources × 7 tabs")
        page.set_viewport_size({"width": 1440, "height": 1050})
        page.locator("[data-topic=embedding]").click()
        assert page.locator(".paper-card").count() == 9
        page.locator("#bgeFilter").click()
        bge_cards = page.locator(".paper-card").count()
        assert 0 < bge_cards < 9
        page.evaluate("PaperLibrary.resetFilters()")
        page.locator("#paperSearch").fill("Structure-Aware")
        assert page.locator(".paper-card").count() == 1
        page.locator("#paperSearch").fill("zzzz-no-matching-source")
        assert page.locator(".paper-card").count() == 0
        page.evaluate("PaperLibrary.resetFilters()")
        page.locator("#sourceFilter").select_option("other")
        assert 0 < page.locator(".paper-card").count() < 42
        page.evaluate("PaperLibrary.resetFilters()")
        page.locator("[data-subtopic]").nth(1).click()
        assert page.locator(".paper-card").count() > 0
        page.evaluate("PaperLibrary.resetFilters()")
        outcomes.append("topic, subtopic, BGE, source-type, text-search and empty-state filters")
        # Reproduce the long-sidebar bug using real wheel input, then select the last item.
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1050 if width > 500 else 844})
            page.evaluate("PaperLibrary.resetFilters()")
            page.locator(".catalog-body").evaluate("e=>e.scrollIntoView({block:'start'})")
            box = page.locator("#subtopicList").bounding_box()
            main_scroll = page.evaluate("scrollY")
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.wheel(0, 10000)
            page.wait_for_timeout(200)
            assert page.evaluate("scrollY") == main_scroll, (width, "filter moved main page")
            list_scroll = page.locator("#subtopicList").evaluate("e=>e.scrollTop")
            assert list_scroll > 0, (width, "filter failed to scroll")
            page.mouse.wheel(0, 10000)
            page.wait_for_timeout(150)
            assert page.evaluate("scrollY") == main_scroll, (width, "filter scroll chained at bottom")
            page.locator("#subtopicList [data-subtopic]").last.click()
            assert page.locator("#subtopicList").evaluate("e=>e.scrollTop") == list_scroll, "Selection lost list position"
            assert page.locator(".paper-card").count() == 1
            if width > 500:
                button = page.locator(".open-paper").bounding_box()
                assert button["y"] >= 0 and button["y"] + button["height"] <= 1050, "Single result needs page hunting"
            page.locator(".open-paper").click()
            assert page.locator("#readerDialog").evaluate("e=>e.open")
            page.evaluate("PaperLibrary.closeReader()")
        page.set_viewport_size({"width": 1440, "height": 1050})
        page.evaluate("PaperLibrary.resetFilters()")
        outcomes.append("desktop/mobile independent filter wheel scrolling, no scroll chaining, preserved last selection and immediate paper opening")
        page.locator("[data-compare=p14]").check()
        page.locator("[data-compare=p13]").check()
        page.locator("#openCompare").click()
        assert page.locator("#compareDialog").evaluate("e=>e.open")
        assert page.locator("#compareContent thead th").count() == 3
        page.keyboard.press("Escape")
        page.locator("#clearCompare").click()
        outcomes.append("two-source comparison and Escape dismissal")
        page.evaluate("PaperLibrary.openPaper('p14','core')")
        page.locator("#readerContent .term-link").first.click()
        assert page.locator("#termDialog").evaluate("e=>e.open")
        assert page.locator("#termContent .term-meaning").inner_text()
        page.keyboard.press("Escape")
        assert page.locator("#readerDialog").evaluate("e=>e.open")
        page.locator("#tab-core").focus()
        page.keyboard.press("ArrowRight")
        assert page.evaluate("PaperLibrary.getState().activeTab") == "process"
        assert page.evaluate("document.activeElement.id") == "tab-process"
        original_scroll = page.evaluate("scrollY")
        page.locator("#readerScroll").evaluate("e=>e.scrollTop=e.scrollHeight")
        assert page.locator("#readerScroll").evaluate("e=>e.scrollTop>0")
        assert page.evaluate("scrollY") == original_scroll, "Reader moved the main page"
        page.locator("#markRead").click()
        page.reload()
        page.evaluate("PaperLibrary.openPaper('p14')")
        assert page.locator("#markRead").get_attribute("aria-pressed") == "true"
        page.evaluate("PaperLibrary.closeReader()")
        page.locator("#openGlossary").click()
        page.locator("#glossarySearch").fill("Cosine")
        assert page.locator("#glossaryResults .term-card").count() > 0
        page.keyboard.press("Escape")
        outcomes.append("context glossary, keyboard tabs, nested dialogs, bounded scrolling and saved reading progress")
        page.emulate_media(reduced_motion="no-preference")
        page.evaluate("PaperLibrary.openPaper('p14','process'); PaperLibrary.setMotion(false)")
        assert page.locator("#readerContent .spider-friend").count() >= 5
        animations = page.locator(".spider-friend").evaluate_all("nodes=>nodes.map(e=>getComputedStyle(e).animationName)")
        assert len(set(animations) - {"none"}) >= 4, "Missing distinct animated poses"
        assert page.locator(".spider-friend").evaluate_all("nodes=>nodes.every(e=>getComputedStyle(e).pointerEvents==='none' && e.closest('[aria-hidden=true]'))")
        page.locator("#readerMotion").click()
        assert page.locator(".spider-friend").evaluate_all("nodes=>nodes.every(e=>getComputedStyle(e).animationPlayState==='paused')")
        page.wait_for_timeout(100)
        stopped = page.locator(".spider-friend").evaluate_all("nodes=>nodes.map(e=>getComputedStyle(e).transform)")
        page.wait_for_timeout(250)
        assert stopped == page.locator(".spider-friend").evaluate_all("nodes=>nodes.map(e=>getComputedStyle(e).transform)"), "Paused pose still moves"
        page.locator("#readerMotion").click()
        assert page.locator(".spider-friend").evaluate_all("nodes=>nodes.every(e=>getComputedStyle(e).animationPlayState==='running')")
        page.emulate_media(reduced_motion="reduce")
        assert page.locator(".spider-friend").evaluate_all("nodes=>nodes.every(e=>getComputedStyle(e).animationName==='none')")
        outcomes.append("four distinct companion animations, multiple reader/notes poses, global pause/resume, static reduced-motion and noninteractive decorations")
        assert not errors, errors
        assert not external, external
        outcomes.append("no browser exceptions, no external requests, offline equations")
        browser_version = browser.version
        browser.close()
    return {"scope": "UI/browser verification only; not model training or research accuracy",
            "passed": outcomes, "failed": [], "skipped": [], "xfailed": [],
            "reading_panes_checked": panes, "python": platform.python_version(),
            "playwright": importlib.metadata.version("playwright"), "chromium": browser_version,
            "html_sha256": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
            "head_before_publication": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/docs/paper_library_interaktif.html")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/paper_verification/browser_checks.json")
    args = parser.parse_args()
    result = check(args.url)
    result.update(command="python scripts/check_paper_library.py", cwd=str(ROOT), exit_code=0)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
