"""Unit conversion card: pure Python, no network. US customary kitchen volumes (cup = 236.6 ml), avoirdupois weights.

parse() reads "how many ounces are in one cup", "26 miles in km", "convert 50 kg to pounds", "how many teaspoons are in
a quarter cup" (the LAST amount + unit said wins, so "a tablespoon, actually make that a quarter cup" converts the cup).
"ounces" against a volume means fluid ounces."""
from __future__ import annotations

import re

from . import Card

# unit -> (dimension, factor to the base unit: ml | g | m | m2 | km/h)
U = {}


def _add(dim, factor, *names):
    for n in names:
        U[n] = (dim, factor)


_add("vol", 4.92892, "teaspoon", "teaspoons", "tsp")
_add("vol", 14.7868, "tablespoon", "tablespoons", "tbsp")
_add("vol", 29.5735, "fluid ounce", "fluid ounces", "fl oz")
_add("vol", 236.588, "cup", "cups")
_add("vol", 473.176, "pint", "pints")
_add("vol", 946.353, "quart", "quarts")
_add("vol", 3785.41, "gallon", "gallons")
_add("vol", 1000.0, "liter", "liters", "litre", "litres", "l")
_add("vol", 1.0, "milliliter", "milliliters", "millilitre", "millilitres", "ml")
_add("mass", 28.3495, "ounce", "ounces", "oz")
_add("mass", 453.592, "pound", "pounds", "lb", "lbs")
_add("mass", 1000.0, "kilogram", "kilograms", "kilo", "kilos", "kg", "kgs")
_add("mass", 1.0, "gram", "grams", "g")
_add("mass", 6350.29, "stone", "stones")
_add("mass", 1e6, "tonne", "tonnes", "metric ton", "metric tons")
_add("len", 1609.344, "mile", "miles", "mi")
_add("len", 1000.0, "kilometer", "kilometers", "kilometre", "kilometres", "km", "kms")
_add("len", 1.0, "meter", "meters", "metre", "metres", "m")
_add("len", 0.01, "centimeter", "centimeters", "centimetre", "centimetres", "cm")
_add("len", 0.001, "millimeter", "millimeters", "millimetre", "millimetres", "mm")
_add("len", 0.3048, "foot", "feet", "ft")
_add("len", 0.0254, "inch", "inches", "in")
_add("len", 0.9144, "yard", "yards", "yd")
_add("area", 4046.86, "acre", "acres")
_add("area", 10000.0, "hectare", "hectares")
_add("area", 0.092903, "square foot", "square feet")
_add("area", 1.0, "square meter", "square meters", "square metre", "square metres")
_add("speed", 1.609344, "mph", "miles per hour")
_add("speed", 1.0, "kph", "km/h", "kilometers per hour", "kilometres per hour")
for n in ("celsius", "fahrenheit", "kelvin", "degrees celsius", "degrees fahrenheit"):
    U[n] = ("temp", 1.0)

SINGULAR = {"teaspoons": "teaspoon", "tablespoons": "tablespoon", "cups": "cup", "pints": "pint", "quarts": "quart",
            "gallons": "gallon", "liters": "liter", "litres": "litre", "ounces": "ounce", "pounds": "pound",
            "kilograms": "kilogram", "grams": "gram", "miles": "mile", "kilometers": "kilometer", "meters": "meter",
            "feet": "foot", "inches": "inch", "yards": "yard", "acres": "acre", "hectares": "hectare",
            "fluid ounces": "fluid ounce", "centimeters": "centimeter", "millimeters": "millimeter",
            "milliliters": "milliliter", "stones": "stone", "tonnes": "tonne", "kilos": "kilo"}
NUMW = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                   "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
NUMW.update({w: 20 + 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())})
FRAC = {"half": 0.5, "quarter": 0.25, "third": 1 / 3, "halves": 0.5, "quarters": 0.25, "thirds": 1 / 3}
MULT = {"hundred": 100, "thousand": 1000, "million": 1e6}
# no "in": it is a preposition far more often than a unit ("how many ounces are IN one cup")
_NAMES = sorted((n for n in U if n not in ("in", "m", "g", "l", "mi")), key=len, reverse=True)
_UNIT_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in _NAMES) + r")\b")


def _amount(words: list[str]) -> float | None:
    """Trailing amount words before a unit: 'a quarter', 'one', '26.2', 'two and a half', 'three quarters', 'half a'."""
    cur, num, art, frac, add = 0.0, False, False, None, False
    for w in words:
        w = w.replace(",", "")
        if w in ("of", "and"):
            continue
        if re.fullmatch(r"\d+(\.\d+)?", w):
            cur, num = cur + float(w), True
        elif re.fullmatch(r"\d+/\d+", w):
            x, y = w.split("/")
            cur, num = cur + float(x) / float(y), True
        elif w in NUMW:
            cur, num = cur + NUMW[w], True
        elif w in MULT:
            cur, num = (cur or 1) * MULT[w], True
        elif w in FRAC:
            frac = FRAC[w]
        elif w in ("a", "an"):
            if num and frac is None:
                add = True              # "two and a half" (not "three quarters of a")
            art = True
        else:
            cur, num, art, frac, add = 0.0, False, False, None, False   # only the trailing run counts
    if frac is not None:
        if num and add:
            return cur + frac
        return cur * frac if num else frac
    if num:
        return cur
    return 1.0 if art else None


def parse(text: str) -> dict | None:
    """{"amount", "src", "dst", "dim"} or None."""
    t = re.sub(r"[^a-z0-9./ ]+", " ", text.lower().replace("table spoon", "tablespoon").replace("tea spoon", "teaspoon"))
    t = re.sub(r"\s+", " ", t).strip()
    ms = list(_UNIT_RE.finditer(t))
    if len(ms) < 2:
        return None
    hm = re.search(r"\bhow (?:many|much)\s+(" + "|".join(re.escape(n) for n in _NAMES) + r")\b", t)
    dst = hm.group(1) if hm else None
    if dst is None:
        tm = re.search(r"\b(?:in|to|into|as)\s+(?:the\s+)?(" + "|".join(re.escape(n) for n in _NAMES) + r")\b\s*$", t)
        if not tm:
            tm = re.search(r"\b(?:in|to|into)\s+(" + "|".join(re.escape(n) for n in _NAMES) + r")\b", t)
        dst = tm.group(1) if tm else None
    if dst is None:
        return None
    src, amount = None, None
    for m in reversed(ms):
        if m.group(1) == dst or (hm and m.start() == hm.start(1)):
            continue
        before = t[:m.start()].split()[-5:]
        a = _amount(before)
        if a is not None:
            src, amount = m.group(1), a
            break
    if src is None:
        return None
    ds, dd = U.get(src), U.get(dst)
    if not ds or not dd:
        return None
    if ds[0] == "vol" and dd[0] == "mass" and dst.startswith(("ounce", "oz")):
        dst = "fluid ounces"                       # "how many ounces in a cup": fluid ounces
        dd = U[dst]
    if dd[0] == "vol" and ds[0] == "mass" and src.startswith(("ounce", "oz")):
        src = "fluid ounces"
        ds = U[src]
    if ds[0] != dd[0]:
        return None
    return {"amount": amount, "src": src, "dst": dst, "dim": ds[0]}


def _temp(v: float, src: str, dst: str) -> float:
    s, d = src.split()[-1], dst.split()[-1]
    c = v if s == "celsius" else (v - 32) * 5 / 9 if s == "fahrenheit" else v - 273.15
    return c if d == "celsius" else c * 9 / 5 + 32 if d == "fahrenheit" else c + 273.15


def convert(amount: float, src: str, dst: str) -> float:
    if U[src][0] == "temp":
        return _temp(amount, src, dst)
    return amount * U[src][1] / U[dst][1]


def _num(v: float) -> str:
    if abs(v - round(v)) < 0.005:
        return f"{round(v):,}"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _name(unit: str, v: float) -> str:
    one = SINGULAR.get(unit, unit)
    many = next((p for p, s in SINGULAR.items() if s == one), unit)
    return one if abs(v - 1) < 1e-9 else many


_FRAC_SAID = {0.25: "a quarter", 0.5: "half a", 0.75: "three quarters of a"}


def _qty(a: float, unit: str) -> str:
    for f, said in _FRAC_SAID.items():
        if abs(a - f) < 1e-6:
            return f"{said} {SINGULAR.get(unit, unit)}"
    return f"{_num(a)} {_name(unit, a)}"


def card(p: dict) -> Card:
    v = convert(p["amount"], p["src"], p["dst"])
    a = p["amount"]
    q = _qty(a, p["src"])
    spoken = f"{q[0].upper() + q[1:]} is {_num(v)} {_name(p['dst'], v)}."
    if p["dim"] == "vol" and any(x in (p["src"], p["dst"]) for x in ("cup", "cups", "pint", "pints", "quart", "quarts",
                                                                       "gallon", "gallons", "fluid ounces")):
        spoken = spoken[:-1] + " (US measures)."
    return Card(vertical="units", title=f"{p['src']} → {p['dst']}", spoken=spoken,
                detail=f"Exact factor: 1 {SINGULAR.get(p['src'], p['src'])} = "
                       f"{_num(convert(1, p['src'], p['dst']))} {p['dst']}.",
                source="unit conversion (local)", args={"amount": round(a, 4), "src": p["src"], "dst": p["dst"]},
                data={**p, "value": v}, live=False)
