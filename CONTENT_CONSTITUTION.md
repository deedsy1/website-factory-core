# Content Constitution (Enforced Forever)

This factory generates evergreen, neutral informational pages designed to be useful in any country and still true in 10+ years.

These rules are enforced by:
- `data/site.yaml` (source of truth)
- `scripts/generate_pages.py` (prompt + contract)
- `scripts/quality_gates.py` (hard validation + deletion on fail)

## Hard Prohibitions (Never Allowed)

- Dates or time-sensitive framing (years, “recent”, “currently”, “this year”, “today”, “now”)
- Prices, costs, or financial claims
- Medical, legal, or financial advice
- Guarantees or promises (“always”, “never”, “100%”, “will definitely”, “guarantee”)
- First-person language (“I”, “we”, “our”, “my”)
- Calls to action / directive phrasing (“you should”, “try this”, “make sure to”, “sign up”, “buy”, “download”)
- Affiliate/product review language (affiliate, sponsored, review, coupon, discount)
- Superlatives / superiority claims without neutrality (“best”, “worst”, “better than”)

## Required Style Rules

- Neutral, encyclopedic tone (no hype, no fear-based framing)
- Beginner-friendly: explain concepts before comparisons
- Short paragraphs: max 2–3 sentences (lists and headings excluded)
- Headings: H2 / H3 only (no H1, no H4+)
- No external links

## Structural Requirements (H2 Outline)

Every page must use the exact H2 outline and order defined in `data/site.yaml`:

- Intro
- Definitions and key terms
- Why this topic exists
- How people usually experience this
- How it typically works
- When this topic tends to come up
- Clarifying examples
- Common misconceptions
- Why this topic gets misunderstood online
- Related situations that feel similar
- Related topics and deeper reading
- Neutral summary
- FAQs

## Internal Linking Rules

- Minimum 3 internal links per page
- Links must be contextually relevant
- No “click here”
- Use relative URLs only (e.g. `/pages/<slug>/`)

## Length & Depth

- Minimum: 800 words
- Target: 1,000–1,400 words
- Maximum: 2,000 words
- Thin content is forbidden (key sections must have substance)

## Evergreen Filters

A page must pass all:
- Would this still be true in 10 years?
- Does this avoid trends, brands, and pricing?
- Could this be read in any country without local-law assumptions?
