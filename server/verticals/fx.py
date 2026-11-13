"""Currency card: ECB reference rates via Frankfurter (no key); crypto spot via Coinbase's public price endpoint (no key).

ECB rates are published once per working day (~16:00 CET): the card says so ("ECB reference rate of 2026-11-12"),
it never claims a live market price. Crypto spot prices are live."""
from __future__ import annotations

import re

from . import Card, TTLCache, get_json, now_iso

_cache = TTLCache(900, 256)          # ECB: daily rates; 15 min is plenty
_crypto = TTLCache(30, 64)           # spot: 30 s

# spoken currency word (singular stem) -> ISO code; modifiers said before it pick among the dollars / pounds
WORD = {"dollar": "USD", "buck": "USD", "usd": "USD", "euro": "EUR", "eur": "EUR", "pound": "GBP", "sterling": "GBP",
        "gbp": "GBP", "quid": "GBP", "yen": "JPY", "jpy": "JPY", "yuan": "CNY", "rmb": "CNY", "renminbi": "CNY",
        "rupee": "INR", "inr": "INR", "peso": "MXN", "franc": "CHF", "chf": "CHF", "won": "KRW", "rand": "ZAR",
        "real": "BRL", "reai": "BRL", "lira": "TRY", "krona": "SEK", "kronor": "SEK", "krone": "NOK", "kroner": "NOK",
        "zloty": "PLN", "baht": "THB", "ringgit": "MYR", "shekel": "ILS", "cad": "CAD", "aud": "AUD",
        "bitcoin": "BTC", "btc": "BTC", "ether": "ETH", "eth": "ETH", "ethereum": "ETH", "dirham": "AED",
        "riyal": "SAR", "dong": "VND", "ruble": "RUB", "rouble": "RUB"}
MOD = {("canadian", "dollar"): "CAD", ("australian", "dollar"): "AUD", ("hong", "dollar"): "HKD",
       ("kong", "dollar"): "HKD", ("singapore", "dollar"): "SGD", ("new", "dollar"): "NZD",
       ("zealand", "dollar"): "NZD", ("mexican", "peso"): "MXN", ("philippine", "peso"): "PHP",
       ("indian", "rupee"): "INR", ("pakistani", "rupee"): "PKR", ("swiss", "franc"): "CHF",
       ("south", "won"): "KRW", ("korean", "won"): "KRW", ("swedish", "krona"): "SEK", ("danish", "krone"): "DKK",
       ("norwegian", "krone"): "NOK", ("turkish", "lira"): "TRY", ("brazilian", "real"): "BRL",
       ("chinese", "yuan"): "CNY", ("japanese", "yen"): "JPY", ("british", "pound"): "GBP",
       ("american", "dollar"): "USD", ("us", "dollar"): "USD", ("u.s", "dollar"): "USD"}
CRYPTO = {"BTC", "ETH"}
NAMES = {"USD": "US dollars", "EUR": "euros", "GBP": "British pounds", "JPY": "yen", "CNY": "yuan", "INR": "rupees",
         "MXN": "Mexican pesos", "CHF": "Swiss francs", "KRW": "won", "ZAR": "rand", "BRL": "reais", "TRY": "lira",
         "SEK": "kronor", "NOK": "kroner", "DKK": "kroner", "PLN": "zloty", "THB": "baht", "MYR": "ringgit",
         "ILS": "shekels", "CAD": "Canadian dollars", "AUD": "Australian dollars", "HKD": "Hong Kong dollars",
         "SGD": "Singapore dollars", "NZD": "New Zealand dollars", "PHP": "Philippine pesos", "BTC": "bitcoin",
         "ETH": "ether", "AED": "dirhams", "SAR": "riyals", "VND": "dong", "RUB": "rubles", "PKR": "rupees"}


def _stem(w: str) -> str:
    w = w.lower().strip(".,")
    if w in ("reais",):
        return "real"
    if w.endswith("ies"):
        return w[:-3] + "y"
    return w[:-1] if w.endswith("s") and w not in ("us", "chf", "gbp") and len(w) > 3 else w


def code(unit_word: str, context: str = "") -> str | None:
    """'dollars' (+ the words said before it, e.g. 'canadian') -> 'CAD'."""
    s = _stem(unit_word)
    iso = WORD.get(s)
    if iso is None:
        return None
    prev = [_stem(w) for w in re.findall(r"[a-z.]+", context.lower())][-3:]
    for p in prev:
        if (p, s) in MOD:
            return MOD[(p, s)]
    return iso


def is_currency(word: str) -> bool:
    return _stem(word) in WORD


async def rate(src: str, dst: str) -> tuple[float, str, str, str]:
    """(units of dst per 1 src, as_of, source, url)."""
    if src in CRYPTO or dst in CRYPTO:
        return await _crypto_rate(src, dst)
    key = (src, dst)
    hit = _cache.get(key)
    if hit:
        return hit
    data = await get_json("https://api.frankfurter.dev/v1/latest", {"base": src, "symbols": dst})
    r = float(data["rates"][dst])
    out = (r, data.get("date", ""), "ECB reference rate via Frankfurter", "https://frankfurter.dev/")
    _cache.put(key, out)
    return out


async def _spot(sym: str, fiat: str) -> float:
    hit = _crypto.get((sym, fiat))
    if hit:
        return hit
    data = await get_json(f"https://api.coinbase.com/v2/prices/{sym}-{fiat}/spot")
    v = float(data["data"]["amount"])
    _crypto.put((sym, fiat), v)
    return v


async def _crypto_rate(src: str, dst: str):
    url = "https://www.coinbase.com/price"
    if src in CRYPTO and dst not in CRYPTO:
        return await _spot(src, dst), now_iso(), "Coinbase spot price", url
    if dst in CRYPTO and src not in CRYPTO:
        return 1.0 / await _spot(dst, src), now_iso(), "Coinbase spot price", url
    a, b = await _spot(src, "USD"), await _spot(dst, "USD")
    return a / b, now_iso(), "Coinbase spot price", url


def _fmt(v: float, iso: str) -> str:
    if iso in CRYPTO:
        return f"{v:,.6f}".rstrip("0").rstrip(".")
    if abs(v - round(v)) < 0.005 or abs(v) >= 10000:
        return f"{round(v):,}"
    return f"{v:,.2f}" if abs(v) >= 1 else f"{v:,.4f}"


def _rate(v: float) -> str:
    return f"{v:,.4f}".rstrip("0").rstrip(".") if v < 100 else f"{v:,.2f}"


async def card(amount: float | None, src: str, dst: str, said: str = "") -> Card:
    r, as_of, source, url = await rate(src, dst)
    amt = 1.0 if amount is None else amount
    val = amt * r
    rate_txt = f"1 {src} = {_rate(r)} {dst}"
    if amount is None or amount == 1 and src in CRYPTO:
        spoken = f"One {NAMES.get(src, src)} is {_fmt(val, dst)} {NAMES.get(dst, dst)} ({rate_txt})"
    else:
        spoken = f"{_fmt(amt, src)} {NAMES.get(src, src)} is about {_fmt(val, dst)} {NAMES.get(dst, dst)}, at {rate_txt}"
    when = f"{source.split(' via')[0].replace(' reference rate', '')} rate of {as_of}" if src not in CRYPTO and \
        dst not in CRYPTO else f"spot price, {as_of[11:16]}"
    spoken += f" ({when})." if "ECB" in source else "."
    return Card(vertical="fx", title=f"{src} → {dst}", spoken=spoken,
                detail=f"Rate: {rate_txt}; inverse 1 {dst} = {_rate(1 / r)} {src}.", source=source, url=url,
                as_of=as_of, args={"amount": amt, "src": src, "dst": dst},
                data={"amount": amt, "src": src, "dst": dst, "rate": r, "value": val, "said": said})


async def warm():
    try:
        await rate("USD", "EUR")
    except Exception:
        pass
