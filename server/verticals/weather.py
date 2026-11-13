"""Weather card: Open-Meteo forecast (no key), MET Norway locationforecast as the fallback (no key, UA required).

Days: "today" / "tonight" / "now" -> 0, "tomorrow" -> 1, "day after tomorrow" -> 2, a weekday name -> its next
occurrence, "this weekend" -> Saturday + Sunday, "this week" / "next few days" -> 5 days. More than 7 days out is not a
forecast question (the router sends it to the web)."""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from zoneinfo import ZoneInfo

from . import Card, TTLCache, get_json, host_down, now_iso
from .geo import Place

_cache = TTLCache(600, 512)          # forecasts: 10 min

WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow",
       73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers", 81: "rain showers", 82: "heavy showers",
       85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms", 96: "thunderstorms with hail",
       99: "thunderstorms with hail"}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_day(text: str) -> tuple | None:
    """Words -> ("now"|"day"|"days", offsets...) or None when no forecast window is said (= today/now).
    Returns ("far",) for a window beyond the 7-day forecast ("next month", "in March")."""
    t = " " + re.sub(r"[^a-z' ]+", " ", text.lower()) + " "
    if re.search(r" (next month|next year|in (january|february|march|april|may|june|july|august|september|october|"
                 r"november|december)|last (week|month|year)|yesterday|last night) ", t):
        return ("far",)
    if " day after tomorrow " in t:
        return ("day", 2)
    if " tomorrow" in t:
        return ("day", 1)
    if re.search(r" (this )?weekend ", t):
        return ("weekend",)
    if re.search(r" (this week|next few days|coming days|next days|week|rest of the week) ", t):
        return ("days", 5)
    for i, d in enumerate(DAYS):
        if f" {d} " in t or f" {d}'s " in t:
            return ("weekday", i)
    if re.search(r" (tonight|this evening) ", t):
        return ("day", 0)
    if re.search(r" (right now|now|currently|at the moment|outside) ", t):
        return ("now",)
    return None


def offsets(when: tuple | None, today: dt.date) -> list[int]:
    if when is None or when[0] == "now":
        return [0]
    k = when[0]
    if k == "day":
        return [when[1]]
    if k == "days":
        return list(range(when[1]))
    if k == "weekday":
        return [(when[1] - today.weekday()) % 7]
    if k == "weekend":
        wd = today.weekday()
        if wd == 6:
            return [0]
        if wd == 5:
            return [0, 1]
        return [5 - wd, 6 - wd]
    return [0]


def when_key(when: tuple | None) -> str:
    return "now" if when is None else ":".join(str(x) for x in when)


def _c2f(c: float) -> int:
    return round(c * 9 / 5 + 32)


def _temp(c: float | None, imperial: bool) -> str:
    if c is None:
        return "?"
    return f"{_c2f(c)}°F ({round(c)}°C)" if imperial else f"{round(c)}°C ({_c2f(c)}°F)"


def _short(c: float | None, imperial: bool) -> str:
    if c is None:
        return "?"
    return f"{_c2f(c)}°F" if imperial else f"{round(c)}°C"


async def _open_meteo(p: Place) -> dict:
    data = await get_json("https://api.open-meteo.com/v1/forecast", {
        "latitude": p.lat, "longitude": p.lon, "timezone": p.tz or "auto", "forecast_days": 7,
        "current": "temperature_2m,weather_code,wind_speed_10m,precipitation",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                 "precipitation_sum,sunrise,sunset"})
    d = data["daily"]
    days = []
    for i, day in enumerate(d["time"]):
        days.append({"date": day, "cond": WMO.get(d["weather_code"][i], "mixed"), "hi": d["temperature_2m_max"][i],
                     "lo": d["temperature_2m_min"][i], "pop": d.get("precipitation_probability_max", [None] * 7)[i],
                     "precip_mm": d.get("precipitation_sum", [None] * 7)[i]})
    cur = data.get("current") or {}
    return {"days": days, "now": {"t": cur.get("temperature_2m"), "cond": WMO.get(cur.get("weather_code"), ""),
                                  "wind_kmh": cur.get("wind_speed_10m")},
            "source": "Open-Meteo", "url": "https://open-meteo.com/", "as_of": cur.get("time") or now_iso()}


_SYM = re.compile(r"_(day|night|polartwilight)$")
_SYM_WORDS = [("clearsky", "clear"), ("fair", "mostly clear"), ("partlycloudy", "partly cloudy"),
              ("andthunder", " and thunder"), ("showers", " showers"), ("light", "light "), ("heavy", "heavy ")]


def sym_text(sym: str) -> str:
    """MET Norway symbol code -> words: "lightrainshowers_day" -> "light rain showers"."""
    t = _SYM.sub("", sym or "")
    for a, b in _SYM_WORDS:
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


async def _met_no(p: Place, tz: dt.tzinfo) -> dict:
    from .. import config
    data = await get_json("https://api.met.no/weatherapi/locationforecast/2.0/compact",
                          {"lat": round(p.lat, 3), "lon": round(p.lon, 3)}, timeout=config.ROUTER_SLOW_TIMEOUT_S)
    ts = data["properties"]["timeseries"]
    by: dict = {}
    for x in ts:
        t = dt.datetime.fromisoformat(x["time"].replace("Z", "+00:00")).astimezone(tz)
        day = t.date().isoformat()
        b = by.setdefault(day, {"temps": [], "syms": [], "precip": 0.0, "wet_h": 0, "h": 0})
        b["temps"].append(x["data"]["instant"]["details"].get("air_temperature"))
        n1 = x["data"].get("next_1_hours") or x["data"].get("next_6_hours") or {}
        sym = (n1.get("summary") or {}).get("symbol_code")
        if sym and 7 <= t.hour <= 20:
            b["syms"].append(_SYM.sub("", sym))
        pr = (n1.get("details") or {}).get("precipitation_amount")
        if pr is not None and "next_1_hours" in x["data"]:
            b["precip"] += pr
            b["h"] += 1
            b["wet_h"] += pr >= 0.1
    days = []
    for day in sorted(by)[:7]:
        b = by[day]
        temps = [t for t in b["temps"] if t is not None]
        sym = Counter(b["syms"]).most_common(1)[0][0] if b["syms"] else ""
        # MET Norway's compact forecast has no precipitation PROBABILITY: report the amount, never a made-up %
        days.append({"date": day, "cond": sym_text(sym) or "mixed", "hi": max(temps) if temps else None,
                     "lo": min(temps) if temps else None, "pop": None, "precip_mm": round(b["precip"], 1)})
    first = ts[0]["data"]
    sym0 = ((first.get("next_1_hours") or {}).get("summary") or {}).get("symbol_code") or ""
    return {"days": days, "now": {"t": first["instant"]["details"].get("air_temperature"),
                                  "cond": sym_text(sym0),
                                  "wind_kmh": round((first["instant"]["details"].get("wind_speed") or 0) * 3.6)},
            "source": "MET Norway", "url": "https://api.met.no/", "as_of": data["properties"]["meta"]["updated_at"]}


def _tz(p: Place) -> dt.tzinfo:
    try:
        return ZoneInfo(p.tz) if p.tz else dt.timezone(dt.timedelta(hours=round(p.lon / 15)))
    except Exception:
        return dt.timezone.utc


async def forecast(p: Place) -> dict:
    key = (round(p.lat, 2), round(p.lon, 2))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    tz = _tz(p)
    data = None
    if not host_down("api.open-meteo.com"):
        try:
            data = await _open_meteo(p)
        except Exception:
            data = None
    if data is None:
        data = await _met_no(p, tz)        # raises if both are down: the router falls back to the web
    _cache.put(key, data)
    return data


def _rain(d: dict) -> str:
    if d.get("pop") is not None:
        return f", {d['pop']}% chance of rain" if d["pop"] >= 20 else ""
    mm = d.get("precip_mm") or 0.0
    return f", {mm:g} mm of rain" if mm >= 0.2 else ""


def _day_name(date: str, off: int) -> str:
    if off == 0:
        return "Today"
    if off == 1:
        return "Tomorrow"
    return dt.date.fromisoformat(date).strftime("%A")


async def card(p: Place, when: tuple | None, rain_q: bool = False) -> Card | None:
    data = await forecast(p)
    tz = _tz(p)
    today = dt.datetime.now(tz).date()
    offs = [o for o in offsets(when, today) if 0 <= o < 7]
    days = {d["date"]: d for d in data["days"]}
    picked = []
    for o in offs:
        date = (today + dt.timedelta(days=o)).isoformat()
        if date in days:
            picked.append((o, days[date]))
    if not picked:
        return None
    imp = p.imperial
    where = p.label
    parts = []
    for o, d in picked:
        rain = _rain(d)
        parts.append(f"{_day_name(d['date'], o)} {d['cond']}, high {_short(d['hi'], imp)}, low {_short(d['lo'], imp)}"
                     f"{rain}")
    now = data.get("now") or {}
    if (when is None or when[0] == "now") and now.get("t") is not None:
        d = picked[0][1]
        spoken = (f"Right now in {where} it's {_temp(now['t'], imp)} and {now.get('cond') or d['cond']}; "
                  f"today's high {_short(d['hi'], imp)}, low {_short(d['lo'], imp)}{_rain(d)}.")
    elif rain_q and len(picked) == 1:
        o, d = picked[0]
        pop, mm = d.get("pop"), d.get("precip_mm") or 0.0
        if pop is not None:
            verdict = "Likely" if pop >= 60 else ("Possibly" if pop >= 30 else "Unlikely")
            chance = f"{pop}% chance of rain"
        else:
            verdict = "Likely" if mm >= 1.0 else ("Possibly" if mm >= 0.2 else "Unlikely")
            chance = f"{mm:g} mm of rain forecast" if mm >= 0.1 else "no rain forecast"
        spoken = (f"{verdict}: {chance} {_day_name(d['date'], o).lower()} in {where}, {d['cond']}, "
                  f"high {_short(d['hi'], imp)}, low {_short(d['lo'], imp)}.")
    else:
        spoken = f"{where}: " + "; ".join(parts) + "."
    detail = " | ".join(f"{d['date']}: {d['cond']}, {_temp(d['hi'], imp)} / {_temp(d['lo'], imp)}"
                        + (f", rain {d['pop']}%" if d.get("pop") is not None else f", {d.get('precip_mm') or 0:g} mm")
                        for d in data["days"][:7])
    return Card(vertical="weather", title=f"Weather · {where}", spoken=spoken, detail=f"7-day forecast: {detail}",
                source=data["source"], url=data["url"], as_of=data["as_of"],
                args={"place": p.said.lower(), "when": when_key(when)},
                data={"place": where, "imperial": imp, "now": now,
                      "days": [{**d, "label": _day_name(d["date"], o)} for o, d in picked]})


async def warm():
    """TLS to both forecast hosts (the fallback is warm too)."""
    p = Place("Chicago", "Illinois", "United States", "US", 41.85, -87.65, "America/Chicago")
    try:
        await forecast(p)
    except Exception:
        pass
    try:
        await _met_no(p, _tz(p))
    except Exception:
        pass
    _cache.clear()
