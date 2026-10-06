"""Checks on the GlobeTrot landing site: generated pages are current and every local link resolves.

Run with:  python -m unittest test_site -v
"""
from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import threading
import unittest
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

SITE = Path(__file__).parent / "super-travel"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"site_{name}", SITE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build")
serve = _load("serve")


class _Page(HTMLParser):
    """Collects the ids on a page and every link it points at."""

    def __init__(self, text: str):
        super().__init__()
        self.ids, self.refs, self.h1 = set(), [], 0
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "h1":
            self.h1 += 1
        for name in ("href", "src"):
            if tag in ("a", "link", "script") and attrs.get(name):
                self.refs.append(attrs[name])


def site_pages():
    return [SITE / "index.html", *sorted((SITE / "journeys").rglob("index.html"))]


def parse(path: Path) -> _Page:
    return _Page(path.read_text(encoding="utf-8"))


class SiteTests(unittest.TestCase):
    def test_generated_pages_are_current(self):
        for path, text in build.render_all().items():
            self.assertEqual(path.read_text(encoding="utf-8"), text, f"{path.relative_to(SITE)} is stale: run python super-travel/build.py")

    def test_every_journey_has_a_page_a_listing_card_and_a_home_card(self):
        journeys = json.loads((SITE / "journeys.json").read_text(encoding="utf-8"))["journeys"]
        home = (SITE / "index.html").read_text(encoding="utf-8")
        listing = (SITE / "journeys" / "index.html").read_text(encoding="utf-8")
        for j in journeys:
            page = SITE / "journeys" / j["slug"] / "index.html"
            self.assertTrue(page.exists(), j["slug"])
            self.assertIn(f'<a class="case__view" href="journeys/{j["slug"]}/"', home)
            self.assertIn(f'<a class="case__view" href="{j["slug"]}/"', listing)
            self.assertEqual(len(re.findall(r'class="dayrow ', page.read_text(encoding="utf-8"))), j["days"], j["slug"])
            self.assertEqual(parse(page).h1, 1, f"{j['slug']} needs exactly one h1")

    def test_view_journey_button_is_the_cards_only_link(self):
        for name in ("index.html", "journeys/index.html"):
            cards = re.findall(r'<article class="case.*?</article>', (SITE / name).read_text(encoding="utf-8"), re.S)
            self.assertEqual(len(cards), 6, name)
            for card in cards:
                self.assertEqual(card.count("<a "), 1, f"{name}: the View Journey button is the only link")
                self.assertNotIn('href="#', card)
                link = re.search(r'<a class="case__view" href="[^"]+"[^>]*>(.*?)</a>', card, re.S)
                self.assertIsNotNone(link, name)
                self.assertEqual(re.sub(r"<[^>]+>", " ", link.group(1)).split(), ["View", "Journey"])
                self.assertLess(card.index("case__view"), card.index("case__meta"), f"{name}: the button sits on the photo")
                title = re.search(r'<h\d class="case__title">(.*?)</h\d>', card, re.S).group(1)
                self.assertNotIn("<a", title, f"{name}: the title is plain text, the link is the button")

    def test_home_page_has_no_view_all_link(self):
        self.assertNotIn("work__more", (SITE / "index.html").read_text(encoding="utf-8"))

    def test_pages_link_the_current_asset_version(self):
        version = build.asset_version()
        for page in site_pages():
            refs = [r for r in parse(page).refs if "styles.css" in r or "main.js" in r]
            self.assertEqual(len(refs), 2, page.name)
            for ref in refs:
                self.assertTrue(ref.endswith(f"?v={version}"), f"{page.relative_to(SITE)}: {ref} is not versioned {version}")

    def test_local_links_resolve(self):
        cache = {}
        for page in site_pages():
            for ref in parse(page).refs:
                if ref == "#" or re.match(r"(https?:|mailto:|data:)", ref):
                    continue
                parts = urlsplit(ref)
                path_part, fragment = parts.path, parts.fragment
                target = (page.parent / path_part).resolve() if path_part else page
                if target.is_dir():
                    target = target / "index.html"
                self.assertTrue(target.exists(), f"{page.relative_to(SITE)} links to {ref}, which does not exist")
                if fragment:
                    ids = cache.setdefault(target, parse(target).ids)
                    self.assertIn(fragment, ids, f"{page.relative_to(SITE)} links to {ref}, but there is no #{fragment}")


class ServeTests(unittest.TestCase):
    def test_dev_server_makes_browsers_revalidate(self):
        server = serve.make_server(0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for path in ("/styles.css", "/journeys/goa/"):
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}{path}", timeout=5) as response:
                    self.assertEqual(response.status, 200, path)
                    self.assertEqual(response.headers["Cache-Control"], "no-cache", path)
        finally:
            server.shutdown()
            server.server_close()


class BuildValidationTests(unittest.TestCase):
    def load_with(self, mutate):
        data = json.loads((SITE / "journeys.json").read_text(encoding="utf-8"))
        mutate(data["journeys"][0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journeys.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with mock.patch.object(build, "DATA", path):
                return build.load()

    def test_valid_data_loads(self):
        self.assertTrue(self.load_with(lambda j: None)["journeys"])

    def test_budget_breakdown_must_add_up(self):
        with self.assertRaisesRegex(SystemExit, "adds up to"):
            self.load_with(lambda j: j["budget_breakdown"][0].update(amount=1))

    def test_day_count_must_match_the_itinerary(self):
        with self.assertRaisesRegex(SystemExit, "lists"):
            self.load_with(lambda j: j.update(days=j["days"] + 1))

    def test_unknown_setting_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "setting"):
            self.load_with(lambda j: j["itinerary"][0].update(setting="Underwater"))


if __name__ == "__main__":
    unittest.main()
