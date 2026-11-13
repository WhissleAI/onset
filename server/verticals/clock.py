"""Time / date and sunrise / sunset cards: local zoneinfo + the NOAA solar equations (no network beyond geocoding)."""
from __future__ import annotations

import datetime as dt
import math
from zoneinfo import ZoneInfo

from . import Card, now_iso
from .geo import Place


def _tz(p: Place | None):
    if p is None:
        return dt.datetime.now().astimezone().tzinfo
    try:
        return ZoneInfo(p.tz) if p.tz else dt.timezone(dt.timedelta(hours=round(p.lon / 15)))
    except Exception:
        return dt.timezone.utc


def _clock(t: dt.datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def time_card(p: Place | None, what: str = "time") -> Card:
    tz = _tz(p)
    t = dt.datetime.now(tz)
    where = p.label if p else "here"
    off = t.utcoffset() or dt.timedelta()
    h, m = divmod(int(off.total_seconds()) // 60, 60)
    utc = f"UTC{'+' if off >= dt.timedelta() else '-'}{abs(h)}" + (f":{abs(m):02d}" if m else "")
    zname = getattr(tz, "key", None) or t.tzname() or utc
    if what == "date":
        spoken = f"It's {t.strftime('%A, %B')} {t.day}, {t.year}" + (f" in {where}." if p else ".")
    else:
        spoken = f"It's {_clock(t)} on {t.strftime('%A')} in {where} ({t.tzname() or utc})." if p else \
            f"It's {_clock(t)} on {t.strftime('%A, %B')} {t.day}."
    return Card(vertical="time", title=f"Local time · {where}", spoken=spoken,
                detail=f"Time zone {zname}, {utc}. Date {t.date().isoformat()}.", source="zoneinfo (local clock)",
                as_of=now_iso(), args={"place": (p.said.lower() if p else ""), "what": what},
                data={"place": where, "iso": t.isoformat(), "tz": zname, "utc": utc}, live=False)


def _sun(date: dt.date, lat: float, lon: float, rising: bool) -> dt.datetime | None:
    """NOAA sunrise equation (zenith 90.833°): UTC datetime, or None (polar day / night)."""
    n = date.timetuple().tm_yday
    lng_h = lon / 15
    t = n + ((6 if rising else 18) - lng_h) / 24
    m = 0.9856 * t - 3.289
    L = (m + 1.916 * math.sin(math.radians(m)) + 0.020 * math.sin(math.radians(2 * m)) + 282.634) % 360
    ra = math.degrees(math.atan(0.91764 * math.tan(math.radians(L)))) % 360
    ra = (ra + (math.floor(L / 90) * 90 - math.floor(ra / 90) * 90)) / 15
    sin_dec = 0.39782 * math.sin(math.radians(L))
    cos_dec = math.cos(math.asin(sin_dec))
    cos_h = (math.cos(math.radians(90.833)) - sin_dec * math.sin(math.radians(lat))) / (cos_dec * math.cos(math.radians(lat)))
    if cos_h > 1 or cos_h < -1:
        return None
    h = (360 - math.degrees(math.acos(cos_h))) if rising else math.degrees(math.acos(cos_h))
    T = h / 15 + ra - 0.06571 * t - 6.622
    ut = (T - lng_h) % 24
    return dt.datetime.combine(date, dt.time(0), dt.timezone.utc) + dt.timedelta(hours=ut)


def sun_card(p: Place, which: str, day_offset: int = 0) -> Card:
    tz = _tz(p)
    date = dt.datetime.now(tz).date() + dt.timedelta(days=day_offset)
    rise, set_ = _sun(date, p.lat, p.lon, True), _sun(date, p.lat, p.lon, False)
    day = "today" if day_offset == 0 else "tomorrow" if day_offset == 1 else date.strftime("%A")
    fmt = lambda x: _clock(x.astimezone(tz)) if x else "none (polar day/night)"   # noqa: E731
    if rise is not None:
        rise_l = rise.astimezone(tz)
        if rise_l.date() != date:                  # the UTC hour wrapped: same local date, one day over
            rise = rise + dt.timedelta(days=(date - rise_l.date()).days)
    if set_ is not None:
        set_l = set_.astimezone(tz)
        if set_l.date() != date:
            set_ = set_ + dt.timedelta(days=(date - set_l.date()).days)
    main = set_ if which == "sunset" else rise
    other = rise if which == "sunset" else set_
    spoken = (f"{'Sunset' if which == 'sunset' else 'Sunrise'} in {p.label} {day} is at {fmt(main)} "
              f"({'sunrise' if which == 'sunset' else 'sunset'} {fmt(other)}).")
    return Card(vertical="sun", title=f"{which.title()} · {p.label}", spoken=spoken,
                detail=f"Computed with the NOAA solar equations for {p.lat:.2f}, {p.lon:.2f} ({getattr(tz, 'key', '')}); "
                       f"accurate to about a minute.", source="NOAA solar equations (local)", as_of=now_iso(),
                args={"place": p.said.lower(), "which": which, "day": day_offset},
                data={"place": p.label, "sunrise": fmt(rise), "sunset": fmt(set_), "date": date.isoformat()},
                live=False)
