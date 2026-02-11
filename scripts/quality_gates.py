import os
import sys
import argparse
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "site"


def links_in_markdown(md: str) -> List[Tuple[str, str]]:
    """Return list of (label, url) markdown links: [label](url)."""
    if not md:
        return []
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", md):
        label = (m.group(1) or "").strip()
        url = (m.group(2) or "").strip()
        if url:
            out.append((label, url))
    return out


def _word_count(text: str) -> int:
    """Rough word count for perf budgets."""
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def performance_budget(site_cfg: Dict, pages: List[Path]) -> None:
    """Enforce page-count + total-word-count budgets.

    Budgets come from either:
      - env PERF_MAX_PAGES / PERF_MAX_WORDS (if set)
      - site_cfg['gates']['max_pages'] / ['max_words'] (defaults set by bootstrap)

    We treat each content/pages/<slug>/index.md as a "page".
    """
    gates = site_cfg.get("gates", {}) if isinstance(site_cfg, dict) else {}

    env_max_pages = os.getenv("PERF_MAX_PAGES")
    env_max_words = os.getenv("PERF_MAX_WORDS")
    try:
        max_pages = int(env_max_pages) if env_max_pages else int(gates.get("max_pages", 1000))
    except Exception:
        max_pages = 1000
    try:
        max_words = int(env_max_words) if env_max_words else int(gates.get("max_words", 600000))
    except Exception:
        max_words = 600000

    page_count = len(pages)
    if page_count > max_pages:
        raise SystemExit(f"Performance budget: too many pages ({page_count}) > max_pages ({max_pages}).")

    total_words = 0
    for p in pages:
        try:
            md = p.read_text(encoding="utf-8")
        except Exception:
            md = p.read_text(errors="ignore")
        body = md
        # Drop front matter quickly
        if body.startswith("---"):
            parts = body.split("---", 2)
            if len(parts) == 3:
                body = parts[2]
        total_words += _word_count(body)

        # Small early-exit for speed
        if total_words > max_words:
            break

    if total_words > max_words:
        raise SystemExit(f"Performance budget: too many words ({total_words}) > max_words ({max_words}).")


def _apply_site_root_early():
    """Allow running core scripts from inside thin-repo.

    Priority:
      1) --site-root <path>
      2) --site-slug <slug>  -> chdir sites/<slug>
      3) env SITE_SLUG (if set) -> chdir sites/<SITE_SLUG>
      4) env BOOTSTRAP_NICHE / NICHE (slugified) -> chdir sites/<slug>
    """
    if "--site-root" in sys.argv:
        i = sys.argv.index("--site-root")
        if i + 1 < len(sys.argv) and sys.argv[i + 1]:
            os.chdir(sys.argv[i + 1])
        return

    slug = None
    if "--site-slug" in sys.argv:
        i = sys.argv.index("--site-slug")
        if i + 1 < len(sys.argv):
            slug = (sys.argv[i + 1] or "").strip()

    slug = slug or os.getenv("SITE_SLUG", "").strip()
    if not slug:
        slug = _slugify(os.getenv("BOOTSTRAP_NICHE") or os.getenv("NICHE") or "")

    if slug:
        target = Path("sites") / slug
        target.mkdir(parents=True, exist_ok=True)
        os.chdir(target)


_apply_site_root_early()

SITE_CONFIG_PATH = os.getenv("SITE_CONFIG_PATH", "data/site.yaml")
CONTENT_ROOT = Path(os.getenv("CONTENT_ROOT", "content/pages"))
DELETE_ON_FAIL = os.getenv("DELETE_ON_FAIL", "1").strip() == "1"

# ---------------------------
# Helpers
# ---------------------------

def load_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def read_frontmatter(md_text: str) -> Tuple[Dict, str]:
    """Return (frontmatter_dict, body_text_without_frontmatter)."""
    if not md_text.startswith("---"):
        return {}, md_text
    parts = md_text.split("\n---\n", 2)
    if len(parts) < 3:
        return {}, md_text
    fm_raw = parts[1]
    body = parts[2]
    try:
        fm = yaml.safe_load(fm_raw) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    return fm, body

def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))

def extract_markdown_links(md: str) -> List[Tuple[str, str]]:
    # [text](url)
    return re.findall(r"\[([^\]]+)\]\(([^)]+)\)", md)

def split_paragraphs(md: str) -> List[str]:
    # Split on blank lines, keep simple
    parts = re.split(r"\n{2,}", md.strip())
    return [p.strip() for p in parts if p.strip()]

def sentence_count(paragraph: str) -> int:
    # ignore headings, lists, code fences
    if paragraph.startswith("#"):
        return 0
    if re.match(r"^(\-|\*|\d+\.)\s+", paragraph):
        return 0
    if paragraph.startswith("```"):
        return 0
    # crude sentence count
    return len(re.findall(r"[.!?](?:\s|$)", paragraph))

def has_only_h2_h3(md: str) -> bool:
    # Disallow H1 and H4+
    if re.search(r"^#\s+", md, flags=re.M):
        return False
    if re.search(r"^####\s+", md, flags=re.M):
        return False
    return True

def extract_h2_sequence(md: str) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(r"^##\s+(.+?)\s*$", md, flags=re.M)]

def section_text(md: str, h2: str) -> str:
    # Return text under the H2 heading until next H2
    pat = re.compile(rf"^##\s+{re.escape(h2)}\s*$", re.M)
    m = pat.search(md)
    if not m:
        return ""
    start = m.end()
    rest = md[start:]
    m2 = re.search(r"^##\s+", rest, flags=re.M)
    return (rest[:m2.start()] if m2 else rest).strip()

def contains_any(text: str, patterns: List[str]) -> bool:
    for p in patterns:
        if re.search(p, text, flags=re.I|re.M):
            return True
    return False

# ---------------------------
# Rules (default fallbacks)
# ---------------------------

DEFAULT_FORBIDDEN = [
    r"\bdiagnos(e|is)\b",
    r"\bprescrib(e|ed|ing)\b",
    r"\bsue\b",
    r"\btreat(ment|ments|ing)?\b",
    r"\bcure(s|d)?\b",
    r"\btherapy\b",
    r"\btherapist\b",
    r"\blawyer\b",
    r"\baccountant\b",
]

DEFAULT_NO_DATES = [
    r"\b(19|20)\d{2}\b",  # years
    r"\brecent(ly)?\b",
    r"\bcurrently\b",
    r"\bthis\s+year\b",
    r"\blast\s+year\b",
    r"\btoday\b",
    r"\bnow\b",
]

DEFAULT_NO_PRICES = [
    r"[$€£¥]\s?\d",
    r"\b\d+(?:\.\d+)?\s?(?:usd|aud|cad|eur|gbp|dollars|bucks)\b",
    r"\bprice\b",
    r"\bcost\b",
]


DEFAULT_NO_STATS = [
    r"\bstudies\s+show\b",
    r"\bresearch\s+shows\b",
    r"\baccording\s+to\b",
    r"\bsurvey\b",
    r"\bstatistic(s)?\b",
    r"\b\d{1,3}%\b",
    r"\b\d+(?:\.\d+)?\s?(?:percent|per\s*cent)\b",
    r"\b\d+(?:,\d{3})+\b",
]
DEFAULT_NO_GUARANTEES = [
    r"\bguarantee(d|s)?\b",
    r"\b100%\b",
    r"\bwill\s+definitely\b",
    r"\balways\b",
    r"\bnever\b",
]

DEFAULT_NO_FIRST_PERSON = [
    r"\b(i|i'm|i’ve|i've|my|mine|me|we|we're|we’ve|we've|our|ours|us)\b",
]

DEFAULT_NO_CALLS_TO_ACTION = [
    r"\bclick\s+here\b",
    r"\bsign\s+up\b",
    r"\bsubscribe\b",
    r"\bbuy\b",
    r"\bpurchase\b",
    r"\bdownload\b",
    r"\bjoin\b",
    r"\btry\s+this\b",
    r"\byou\s+should\b",
    r"\bmake\s+sure\s+to\b",
    r"\bconsider\s+doing\b",
]

DEFAULT_NO_AFFILIATE = [
    r"\baffiliate\b",
    r"\bsponsored\b",
    r"\breview\b",
    r"\bcoupon\b",
    r"\bdiscount\b",
]

DEFAULT_SUPERLATIVES = [
    r"\bbest\b",
    r"\bworst\b",
    r"\bbetter\s+than\b",
    r"\bmore\s+than\b\s+(?:any|everyone)\b",
    r"\btop\s+\d+\b",
]

# ---------------------------
# Validation
# ---------------------------

def validate_page(md_path: Path, cfg: dict) -> Tuple[bool, List[str], int, int]:
    """
    Returns (ok, failures, passed_rules, total_rules_scored)
    Only "scored" rules contribute to compliance percentage.
    """
    failures: List[str] = []
    scored_total = 0
    scored_pass = 0

    gates = (cfg.get("gates") or {}) if isinstance(cfg, dict) else {}
    generation = (cfg.get("generation") or {}) if isinstance(cfg, dict) else {}
    internal = (cfg.get("internal_linking") or {}) if isinstance(cfg, dict) else {}

    required_outline = generation.get("outline_h2") or []
    wc_cfg = generation.get("wordcount") or {}
    wc_min = int(wc_cfg.get("min", gates.get("wordcount_min", 800)))
    wc_max = int(wc_cfg.get("max", gates.get("wordcount_max", 2000)))
    ideal_min = int(wc_cfg.get("ideal_min", 1000))
    ideal_max = int(wc_cfg.get("ideal_max", 1400))

    max_sent = int(gates.get("max_sentences_per_paragraph", generation.get("style_rules", {}).get("max_sentences_per_paragraph", 3)))
    min_links = int(internal.get("min_links", gates.get("min_internal_links", 3)))
    forbid_external = bool(internal.get("forbid_external", gates.get("forbid_external_links", True)))

    # Hugo section/home index pages (_index.md) are often intentionally short.
    # Treat them as a special case so the generator doesn't fail immediately on
    # a placeholder/intro home page.
    is_index_page = md_path.name == "_index.md"
    if is_index_page:
        required_outline = []
        wc_min = 0
        wc_max = max(wc_max, 2000)
        ideal_min = 0
        ideal_max = max(ideal_max, 500)
        min_links = 0
        forbid_external = False
        max_sent = max(max_sent, 6)

    raw = md_path.read_text(encoding="utf-8")
    fm, body = read_frontmatter(raw)

    # 1) Frontmatter keys
    fm_required = ["title", "slug", "description", "date", "hub", "page_type", "summary"]
    if is_index_page:
        fm_required = ["title", "description"]

    for k in fm_required:
        scored_total += 1
        if fm.get(k) is None or str(fm.get(k)).strip() == "":
            failures.append(f"Missing frontmatter key: {k}")
        else:
            scored_pass += 1

    # 2) Headings restrictions (skip for index pages)
    if not is_index_page:
        scored_total += 1
        if not has_only_h2_h3(body):
            failures.append("Headings must be H2/H3 only (no H1 or H4+).")
        else:
            scored_pass += 1

    # 3) Outline (exact H2 set and order)
    if required_outline:
        scored_total += 1
        got = extract_h2_sequence(body)
        if got != required_outline:
            failures.append(f"H2 outline mismatch. Expected exactly: {required_outline}. Got: {got}.")
        else:
            scored_pass += 1

    # 4) Wordcount
    wc = word_count(body)
    scored_total += 1
    if wc < wc_min or wc > wc_max:
        failures.append(f"Wordcount out of bounds: {wc} (min {wc_min}, max {wc_max}).")
    else:
        scored_pass += 1

    # 5) Paragraph sentence limit
    scored_total += 1
    bad_paras = 0
    for p in split_paragraphs(body):
        sc = sentence_count(p)
        if sc > max_sent:
            bad_paras += 1
    if bad_paras > 0:
        failures.append(f"Too many long paragraphs: {bad_paras} paragraphs exceed {max_sent} sentences.")
    else:
        scored_pass += 1

    # 6) Internal links
    links = extract_markdown_links(body)
    internal_links = [u for _, u in links if u.startswith("/")]
    external_links = [u for _, u in links if re.match(r"^(https?:)?//", u) or u.startswith("www.")]
    scored_total += 1
    if len(internal_links) < min_links:
        failures.append(f"Too few internal links: {len(internal_links)} (min {min_links}).")
    else:
        scored_pass += 1

    scored_total += 1
    if any("click here" in (t or "").lower() for t, _ in links):
        failures.append('Link text "click here" is forbidden.')
    else:
        scored_pass += 1

    scored_total += 1
    if forbid_external and external_links:
        failures.append(f"External links forbidden (found {len(external_links)}).")
    else:
        scored_pass += 1


    # Related topics section should carry the internal links (makes linking consistent)
    def extract_section(md: str, h2_title: str) -> str:
        # Find "## <title>" section and return its contents until next "## "
        pat = re.compile(rf"^##\s+{re.escape(h2_title)}\s*$", re.M)
        m = pat.search(md)
        if not m:
            return ""
        start = m.end()
        # Next H2
        m2 = re.search(r"^\s*##\s+", md[start:], flags=re.M)
        end = start + m2.start() if m2 else len(md)
        return md[start:end].strip()

    related_section = extract_section(body, "Related topics and deeper reading")
    related_links = []
    if related_section:
        for t, u in links_in_markdown(related_section):
            if (u or "").startswith("/"):
                related_links.append((t, u))

    scored_total += 1
    if len(related_links) < min_links:
        failures.append(f'Related topics section must include at least {min_links} internal links (found {len(related_links)}).')
    else:
        scored_pass += 1

    # 7) Hard prohibitions in body + frontmatter
    full_text = (yaml.safe_dump(fm, sort_keys=False) + "\n" + body)

    def score_rule(ok: bool, msg: str):
        nonlocal scored_total, scored_pass
        scored_total += 1
        if ok:
            scored_pass += 1
        else:
            failures.append(msg)

    score_rule(not contains_any(full_text, DEFAULT_FORBIDDEN), "Forbidden medical/legal term hit.")
    score_rule(not contains_any(full_text, DEFAULT_NO_DATES), "Date/recency language is forbidden.")
    score_rule(not contains_any(full_text, DEFAULT_NO_PRICES), "Price/cost language is forbidden.")
    score_rule(not contains_any(full_text, DEFAULT_NO_STATS), "Statistics/numbered claims are forbidden.")
    score_rule(not contains_any(full_text, DEFAULT_NO_GUARANTEES), "Guarantee/promise language is forbidden.")
    score_rule(not contains_any(full_text, DEFAULT_NO_FIRST_PERSON), "First-person language is forbidden.")
    score_rule(not contains_any(full_text, DEFAULT_NO_CALLS_TO_ACTION), "Calls-to-action / directive phrasing is forbidden.")
    score_rule(not contains_any(full_text, DEFAULT_NO_AFFILIATE), "Affiliate/review language is forbidden.")
    score_rule(not contains_any(full_text, DEFAULT_SUPERLATIVES), "Superlative/superiority language is forbidden (stay neutral).")

    # 8) Structural content presence within sections
    # Require meaningful text in the key sections
    required_sections = ["Intro", "Definitions and key terms", "How it typically works", "Clarifying examples", "Neutral summary"]
    for sec in required_sections:
        scored_total += 1
        txt = section_text(body, sec)
        if word_count(txt) < 40:
            failures.append(f'Section "{sec}" is too thin (<40 words).')
        else:
            scored_pass += 1

    # 9) FAQs count (look for ### Q: lines or bold questions)
    scored_total += 1
    faq_txt = section_text(body, "FAQs")
    # Count question-like lines
    q_count = len(re.findall(r"^###\s+.+", faq_txt, flags=re.M)) + len(re.findall(r"^\*\*Q[:\s].+\*\*", faq_txt, flags=re.M))
    if q_count < int(gates.get("faq_min", 4)):
        failures.append(f"Too few FAQs: {q_count} (min {int(gates.get('faq_min', 4))}).")
    else:
        scored_pass += 1

    ok = len(failures) == 0
    return ok, failures, scored_pass, scored_total

def main() -> int:
    cfg = load_yaml(SITE_CONFIG_PATH)

    pages = sorted(CONTENT_ROOT.glob("*/index.md"))
    performance_budget(site_cfg, pages)
    if not pages:
        print("No pages found to validate.")
        return 0

    total_scored = 0
    total_passed = 0
    failures_total = 0

    for md in pages:
        ok, fails, passed, scored = validate_page(md, cfg)
        total_scored += scored
        total_passed += passed

        if not ok:
            failures_total += len(fails)
            slug = md.parent.name
            for f in fails:
                print(f"[FAIL] {slug}: {f}")
            if DELETE_ON_FAIL:
                try:
                    # delete entire page folder
                    for p in md.parent.glob("**/*"):
                        p.unlink(missing_ok=True)
                    md.parent.rmdir()
                    print(f"[DEL]  {slug}: removed page folder")
                except Exception:
                    pass

    compliance = 0.0 if total_scored == 0 else (total_passed / total_scored) * 100.0
    print(f"\nCompliance score: {compliance:.1f}% ({total_passed}/{total_scored} checks passed)")
    if failures_total:
        print(f"Total failures: {failures_total}")
        return 1

    print("All pages passed quality gates.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
