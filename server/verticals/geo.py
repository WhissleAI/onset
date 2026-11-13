"""Place name -> (name, admin, country, lat, lon, timezone). Open-Meteo geocoding (no key), Nominatim fallback.
Cached for a day: places don't move."""
from __future__ import annotations

from dataclasses import dataclass

from . import TTLCache, get_json, hedged

_cache = TTLCache(24 * 3600, 2048)
# ~140 big cities (name, admin1, country, cc, lat, lon, tz) taken from Open-Meteo geocoding, so the commonest weather /
# time questions need no geocoding call at all
try:
    import json as _json
    from pathlib import Path as _Path
    CITIES: dict = _json.loads((_Path(__file__).with_name("cities.json")).read_text())
except (OSError, ValueError):
    CITIES = {}


@dataclass
class Place:
    name: str
    admin: str
    country: str
    country_code: str
    lat: float
    lon: float
    tz: str
    said: str = ""            # the words the speaker used

    @property
    def label(self) -> str:
        bits = [self.name]
        if self.admin and self.admin != self.name and self.country_code in ("US", "CA", "AU"):
            bits.append(self.admin)
        elif self.country and self.country != self.name:
            bits.append(self.country)
        return ", ".join(bits)

    @property
    def imperial(self) -> bool:
        return self.country_code in ("US", "LR", "MM")


# spoken / tagged forms that the geocoder doesn't resolve as said
ALIAS = {"nyc": "new york", "new york city": "new york", "sf": "san francisco", "la": "los angeles", "dc": "washington",
         "washington dc": "washington", "vegas": "las vegas", "the bay area": "san francisco", "philly": "philadelphia"}


async def geocode(name: str) -> Place | None:
    q = " ".join(name.lower().split())
    q = ALIAS.get(q, q)
    if not q:
        return None
    hit = _cache.get(q)
    if hit is not None:
        return hit or None
    c = CITIES.get(q)
    if c:                                     # big cities: offline gazetteer (0 ms, works with no network)
        return Place(name=c[0], admin=c[1], country=c[2], country_code=c[3], lat=c[4], lon=c[5], tz=c[6], said=name)
    async def om():
        data = await get_json("https://geocoding-api.open-meteo.com/v1/search",
                              {"name": q, "count": 5, "language": "en", "format": "json"})
        res = data.get("results") or []
        if not res:
            return False                      # an answer: no such place (False, not None: the hedge keeps it)
        # biggest place among the top hits ("paris" -> Paris FR, not Paris TX)
        r = max(res[:5], key=lambda x: (x.get("population") or 0) * (2 if res.index(x) == 0 else 1))
        return Place(name=r.get("name", q.title()), admin=r.get("admin1", ""), country=r.get("country", ""),
                     country_code=r.get("country_code", ""), lat=r["latitude"], lon=r["longitude"],
                     tz=r.get("timezone") or "UTC", said=name)

    async def nom():
        data = await get_json("https://nominatim.openstreetmap.org/search",
                              {"q": q, "format": "jsonv2", "limit": 1, "addressdetails": 1},
                              headers={"Accept-Language": "en"})
        if not data:
            return False
        r = data[0]
        addr = r.get("address") or {}
        return Place(name=(r.get("name") if (r.get("name") or "").isascii() else q.title()), admin=addr.get("state", ""),
                     country=addr.get("country", ""), country_code=(addr.get("country_code") or "").upper(),
                     lat=float(r["lat"]), lon=float(r["lon"]), tz="", said=name)

    try:
        place = await hedged(om, nom, 0.3)
    except Exception:
        place = None
    _cache.put(q, place or False, ttl=None if place else 300)
    return place or None
