"""Title-text facet mining for CC-PACE data prep (profiles + coarse categories).

The same-candidate task files carry rich item TITLES but no structured
category/brand/price metadata (the prepared item_records.jsonl is empty-shell).
This module mines coarse facets from titles with small keyword lexicons so that

  1. the user PROFILE SLOTS (schema.py: top_categories, liked_brands, concerns,
     routine_step, ingredient_prefs, price_band) can be aggregated from the
     user's TRAIN history titles only (no future leakage), and
  2. each candidate gets a coarse ``category`` facet + train popularity, which
     feed the residualizer's ``facet_bucket`` / ``log_pop`` nuisance coordinates.

Lexicons are deliberately simple, deterministic, and domain-keyed: this is data
prep, not part of the method. Domains without a curated lexicon fall back to
brand/top-token mining only (profile slots stay sparse, which render_profile
handles by skipping empty slots).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# --- beauty lexicons (keyword -> facet). Keywords are matched case-insensitively
# as substrings of the title; multiword keys are matched as phrases. ---

BEAUTY_CATEGORY_LEXICON: dict[str, str] = {
    "serum": "skincare", "moisturizer": "skincare", "cleanser": "skincare",
    "face wash": "skincare", "toner": "skincare", "face cream": "skincare",
    "eye cream": "skincare", "facial": "skincare", "skin": "skincare",
    "sunscreen": "skincare", "spf": "skincare", "face mask": "skincare",
    "retinol": "skincare", "collagen": "skincare", "anti aging": "skincare",
    "anti-aging": "skincare", "wrinkle": "skincare", "pore": "skincare",
    "shampoo": "hair care", "conditioner": "hair care", "hair": "hair care",
    "scalp": "hair care", "keratin": "hair care", "curl": "hair care",
    "lipstick": "makeup", "lip gloss": "makeup", "mascara": "makeup",
    "eyeliner": "makeup", "foundation": "makeup", "concealer": "makeup",
    "eyeshadow": "makeup", "palette": "makeup", "blush": "makeup",
    "makeup": "makeup", "brow": "makeup", "lash": "makeup",
    "nail": "nails", "polish": "nails", "manicure": "nails", "pedicure": "nails",
    "perfume": "fragrance", "cologne": "fragrance", "eau de": "fragrance",
    "fragrance": "fragrance", "body spray": "fragrance",
    "brush": "tools & accessories", "sponge": "tools & accessories",
    "mirror": "tools & accessories", "tweezer": "tools & accessories",
    "roller": "tools & accessories", "applicator": "tools & accessories",
    "curler": "tools & accessories", "trimmer": "tools & accessories",
    "body wash": "bath & body", "body lotion": "bath & body",
    "soap": "bath & body", "scrub": "bath & body", "bath": "bath & body",
    "deodorant": "bath & body", "hand cream": "bath & body",
    "beard": "men's grooming", "razor": "men's grooming",
    "shave": "men's grooming", "aftershave": "men's grooming",
}

BEAUTY_ROUTINE_LEXICON: dict[str, str] = {
    "cleanser": "cleanse", "face wash": "cleanse", "makeup remover": "cleanse",
    "micellar": "cleanse",
    "scrub": "exfoliate", "peel": "exfoliate", "exfoliat": "exfoliate",
    "serum": "treat", "treatment": "treat", "retinol": "treat",
    "essence": "treat", "ampoule": "treat", "spot": "treat",
    "moisturizer": "moisturize", "face cream": "moisturize",
    "lotion": "moisturize", "hydrating": "moisturize", "eye cream": "moisturize",
    "sunscreen": "protect", "spf": "protect",
    "shampoo": "hair wash", "conditioner": "hair condition",
    "hair mask": "hair treat", "styling": "hair style", "hairspray": "hair style",
    "mousse": "hair style", "pomade": "hair style",
}

BEAUTY_CONCERN_LEXICON: dict[str, str] = {
    "anti aging": "anti-aging", "anti-aging": "anti-aging",
    "wrinkle": "anti-aging", "firming": "anti-aging", "fine lines": "anti-aging",
    "collagen production": "anti-aging",
    "hydrating": "hydration", "dry skin": "hydration", "moisture": "hydration",
    "hyaluronic": "hydration", "dehydrated": "hydration",
    "acne": "acne", "blemish": "acne", "salicylic": "acne", "pimple": "acne",
    "pore": "pores", "blackhead": "pores",
    "brightening": "brightening", "dark spot": "brightening",
    "whitening": "brightening", "radiance": "brightening", "dull": "brightening",
    "sensitive": "sensitivity", "gentle": "sensitivity",
    "hypoallergenic": "sensitivity", "fragrance-free": "sensitivity",
    "frizz": "frizz control", "smoothing": "frizz control",
    "volume": "volume", "volumizing": "volume", "thickening": "volume",
    "hair growth": "hair growth", "regrowth": "hair growth", "hair loss": "hair growth",
    "dandruff": "dandruff", "color treated": "color protection",
    "color-treated": "color protection", "uv protection": "sun protection",
}

BEAUTY_INGREDIENT_LEXICON: dict[str, str] = {
    "retinol": "retinol", "hyaluronic acid": "hyaluronic acid",
    "hyaluronic": "hyaluronic acid", "vitamin c": "vitamin c",
    "vitamin e": "vitamin e", "collagen": "collagen",
    "niacinamide": "niacinamide", "salicylic acid": "salicylic acid",
    "glycolic": "glycolic acid", "argan": "argan oil",
    "coconut oil": "coconut oil", "tea tree": "tea tree",
    "aloe": "aloe vera", "charcoal": "charcoal", "keratin": "keratin",
    "biotin": "biotin", "caffeine": "caffeine", "jojoba": "jojoba oil",
    "rosehip": "rosehip oil", "shea butter": "shea butter",
    "snail": "snail mucin", "ceramide": "ceramides", "peptide": "peptides",
    "squalane": "squalane", "tourmaline": "tourmaline", "castor oil": "castor oil",
}

_DOMAIN_LEXICONS: dict[str, dict[str, dict[str, str]]] = {
    "beauty": {
        "category": BEAUTY_CATEGORY_LEXICON,
        "routine": BEAUTY_ROUTINE_LEXICON,
        "concern": BEAUTY_CONCERN_LEXICON,
        "ingredient": BEAUTY_INGREDIENT_LEXICON,
    },
}

# Tokens that disqualify a leading-token run from being read as a brand name.
_BRAND_STOPWORDS = {
    "the", "a", "an", "new", "pure", "organic", "natural", "premium", "best",
    "professional", "mini", "set", "pack", "for", "with", "and", "anti",
    "hair", "face", "skin", "body", "men", "women", "kids",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'&-]*")


def _match_lexicon(text_lower: str, lexicon: dict[str, str]) -> list[str]:
    """Return facet values whose keyword appears in the text (dedup, keyword order)."""
    hits: list[str] = []
    for kw, facet in lexicon.items():
        if kw in text_lower and facet not in hits:
            hits.append(facet)
    return hits


def coarse_category(title: str, domain: str = "beauty") -> str:
    """Single coarse category facet for one item title ('' when nothing matches)."""
    lex = _DOMAIN_LEXICONS.get(domain, {}).get("category")
    if not lex:
        return ""
    hits = _match_lexicon(str(title).lower(), lex)
    return hits[0] if hits else ""


def _is_product_word(tok_lower: str) -> bool:
    """True when the token matches any lexicon keyword (so it's a product word, not a brand)."""
    for lex in (
        BEAUTY_CATEGORY_LEXICON, BEAUTY_ROUTINE_LEXICON,
        BEAUTY_CONCERN_LEXICON, BEAUTY_INGREDIENT_LEXICON,
    ):
        for kw in lex:
            if tok_lower == kw or tok_lower in kw.split():
                return True
    return False


def _brandish_strong(tok: str) -> bool:
    """ALL-CAPS (len>=2) or internal capitals: 'OZNaturals', \"L'ANGE\", 'HAIR'."""
    return (len(tok) >= 2 and tok.isupper()) or any(ch.isupper() for ch in tok[1:])


def extract_brand(title: str) -> str:
    """Heuristic brand from a title's leading tokens.

    'OZNaturals Retinol Serum' -> 'OZNaturals'; "L'ANGE HAIR Argan-Infused ..."
    -> "L'ANGE HAIR"; 'Pure Hyaluronic Acid Serum' -> '' (ordinary words).
    First token qualifies if it is strongly name-like (internal caps / all-caps)
    or a Title-case word that is neither a stopword nor a product word.
    Continuation tokens are kept only while strongly name-like.
    """
    tokens = _WORD_RE.findall(str(title))
    if not tokens:
        return ""
    first = tokens[0]
    low = first.lower()
    if low in _BRAND_STOPWORDS or _is_product_word(low):
        return ""
    if not (_brandish_strong(first) or (first[0].isupper() and len(first) >= 4)):
        return ""
    brand = [first]
    # continuation tokens only while ALL-CAPS ("L'ANGE HAIR" yes; "Argan-Infused" no)
    for tok in tokens[1:3]:
        if len(tok) >= 2 and tok.isupper():
            brand.append(tok)
        else:
            break
    return " ".join(brand)


def _top_with_counts(counter: Counter, k: int) -> list[str]:
    return [f"{name} ({cnt})" if cnt > 1 else name for name, cnt in counter.most_common(k) if name]


def profile_from_history(
    history_titles: list[str],
    *,
    domain: str = "beauty",
    max_per_slot: int = 3,
) -> dict[str, Any]:
    """Aggregate one user's TRAIN-history titles into CC-PACE profile slots.

    Only history (pre-cutoff) text goes in -- never candidate or target fields.
    Slots with no evidence are omitted; schema.render_profile skips empty slots.
    """
    lexicons = _DOMAIN_LEXICONS.get(domain, {})
    cats: Counter = Counter()
    routines: Counter = Counter()
    concerns: Counter = Counter()
    ingredients: Counter = Counter()
    brands: Counter = Counter()

    for title in history_titles:
        low = str(title).lower()
        for facet in _match_lexicon(low, lexicons.get("category", {})):
            cats[facet] += 1
        for facet in _match_lexicon(low, lexicons.get("routine", {})):
            routines[facet] += 1
        for facet in _match_lexicon(low, lexicons.get("concern", {})):
            concerns[facet] += 1
        for facet in _match_lexicon(low, lexicons.get("ingredient", {})):
            ingredients[facet] += 1
        brand = extract_brand(title)
        if brand:
            brands[brand.lower()] += 1

    profile: dict[str, Any] = {}
    if cats:
        profile["top_categories"] = _top_with_counts(cats, max_per_slot)
    # a brand is 'liked' if it recurs; otherwise report nothing rather than noise
    liked = Counter({b: c for b, c in brands.items() if c >= 2})
    if liked:
        profile["liked_brands"] = _top_with_counts(liked, max_per_slot)
    if concerns:
        profile["concerns"] = _top_with_counts(concerns, max_per_slot)
    if routines:
        profile["routine_step"] = _top_with_counts(routines, max_per_slot)
    if ingredients:
        profile["ingredient_prefs"] = _top_with_counts(ingredients, max_per_slot)
    # price_band: the task files carry no price metadata; omitted (renderer skips it)
    return profile
