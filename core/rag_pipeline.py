"""
core/rag_pipeline.py
Two-stage NOTAM → 4D JSON constraint pipeline.
Architecture mirrors the notebook's Cell 11 implementation.
Stage 1: intent classification (NOTAM class R/P/TFR/D/W)
Stage 2: format-specific coordinate extraction
"""

import re
import time
import numpy as np

# ── VOR navaid reference coordinates (Bay Area) ──────────────────────────────
VOR_COORDS = {
    "SFO": (37.619, -122.375),
    "OAK": (37.721, -122.221),
    "SJC": (37.363, -121.929),
    "SAU": (37.909, -122.522),
    "ENI": (37.424, -121.928),
    "OSI": (34.668, -118.839),
    "SGD": (37.750, -122.200),
}

NOTAM_CLASS_LABELS = {
    "R":   "Restricted Area",
    "P":   "Prohibited Area",
    "TFR": "Temporary Flight Restriction",
    "D":   "Danger Area",
    "W":   "Weather Advisory (SIGMET/METAR)",
}

NOTAM_CLASS_COLORS = {
    "R":   "#FF9800",
    "P":   "#F44336",
    "TFR": "#F44336",
    "D":   "#FF9800",
    "W":   "#2196F3",
}


# ── Stage 1: Intent classifier ───────────────────────────────────────────────
def classify_notam(text: str) -> str:
    """
    Classify NOTAM into one of 5 classes: R / P / TFR / D / W
    Mirrors the two-stage architecture from the project's RAG pipeline.
    """
    t = text.upper()
    if any(kw in t for kw in ["TFR", "TEMPORARY FLIGHT RESTRICTION", "VIP", "SECURITY OPS",
                                "SEARCH AND RESCUE", "FIREFIGHTING", "DISASTER RELIEF",
                                "AERIAL DEMONSTRATION", "EMERGENCY TFR"]):
        return "TFR"
    if any(kw in t for kw in ["PROHIBITED", "/P-", "P-AREA", "NO AIRCRAFT PERMITTED",
                                "VIOLATIONS SUBJECT"]):
        return "P"
    if any(kw in t for kw in ["RESTRICTED", "/R-", "R-AREA", "MILITARY OPS",
                                "CONTACT", "CLEARANCE REQUIRED", "R-2508"]):
        return "R"
    if any(kw in t for kw in ["DANGER", "/D-", "D-AREA", "MILITARY EXERCISE",
                                "HAZARDOUS OPS"]):
        return "D"
    if any(kw in t for kw in ["METAR", "SIGMET", "TURBULENCE", "WIND", "ICING",
                                "SEVERE", "WEATHER", "PIREP"]):
        return "W"
    return "R"  # default to restricted


# ── Stage 2: Format-specific extractors ─────────────────────────────────────
def _extract_lat_lon(text: str):
    """Try multiple coordinate formats to extract lat/lon."""
    patterns = [
        # 37.2N 122.1W  or  37.20N 122.10W
        r"(\d{2,3}\.\d+)\s*N\s+(\d{2,3}\.\d+)\s*W",
        # 37.2N, 122.1W
        r"(\d{2,3}\.\d+)\s*N,?\s*(\d{2,3}\.\d+)\s*W",
        # ICAO compressed: 3742N12217W
        r"(\d{2})(\d{2})N(\d{3})(\d{2})W",
        # Decimal with minus: 37.62 -122.38
        r"(3[678]\.\d{2,4})\s+(-12[123]\.\d{2,4})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            g = m.groups()
            if len(g) == 4 and len(g[0]) == 2:
                # ICAO format
                lat = float(g[0]) + float(g[1]) / 60.0
                lon = -(float(g[2]) + float(g[3]) / 60.0)
            elif len(g) == 2 and g[1].startswith("-"):
                lat = float(g[0])
                lon = float(g[1])
            else:
                lat = float(g[0])
                lon = -float(g[1])
            if 36.0 <= lat <= 39.0 and -124.0 <= lon <= -120.0:
                return round(lat, 4), round(lon, 4)
    return None, None


def _extract_altitude(text: str):
    """Extract floor and ceiling altitudes (ft)."""
    floor, ceiling = 0, 18000
    t = text.upper()

    if re.search(r"\bSFC\b|\bSURFACE\b", t):
        floor = 0
    floor_m = re.search(r"(\d{3,5})\s*FT?\s*(?:MSL|AGL|TO|FLOOR)", t)
    if floor_m:
        floor = int(floor_m.group(1))

    fl_m = re.search(r"FL\s*(\d{2,3})", t)
    if fl_m:
        ceiling = int(fl_m.group(1)) * 100

    ceil_m = re.search(r"TO\s+(?:FL\s*)?(\d{4,6})\s*FT", t)
    if ceil_m:
        ceiling = int(ceil_m.group(1))

    return max(0, floor), min(60000, max(ceiling, floor + 100))


def _extract_radius(text: str) -> int:
    m = re.search(r"(\d{1,3})\s*NM", text.upper())
    return int(m.group(1)) if m else 10


def _extract_time(text: str):
    times = re.findall(r"\b(\d{4})Z?\b", text)
    if len(times) >= 2:
        return times[0] + "Z", times[1] + "Z"
    return "0000Z", "2359Z"


def _extract_danger_area(text: str):
    """Specialist extractor for D-class (bearing/distance from VOR)."""
    t = text.upper()
    vor_m = re.search(r"\b(SFO|OAK|SJC|SAU|ENI|OSI|SGD)\b\s*VOR", t)
    bear_m = re.search(r"(\d{1,3})\s*(?:DEG|°)\s*/?\s*(\d{1,3})\s*(?:NM|DME)", t)
    if vor_m and bear_m:
        vor_name = vor_m.group(1)
        bearing = float(bear_m.group(1))
        dist_nm = float(bear_m.group(2))
        vor_lat, vor_lon = VOR_COORDS.get(vor_name, (37.619, -122.375))
        dist_deg = dist_nm / 60.0
        lat = vor_lat + dist_deg * np.cos(np.radians(bearing))
        lon = vor_lon + dist_deg * np.sin(np.radians(bearing))
        return round(lat, 4), round(lon, 4)
    return _extract_lat_lon(text)


# ── Full pipeline ────────────────────────────────────────────────────────────
def parse_notam_to_json(notam_text: str) -> dict:
    """
    Full two-stage RAG pipeline.
    Returns a dict with 4D spatial constraint + metadata.
    """
    t0 = time.perf_counter()

    # Stage 1
    notam_class = classify_notam(notam_text)

    # Stage 2 (format-specific)
    if notam_class == "D":
        lat, lon = _extract_danger_area(notam_text)
    else:
        lat, lon = _extract_lat_lon(notam_text)

    floor, ceiling = _extract_altitude(notam_text)
    radius = _extract_radius(notam_text)
    t_start, t_end = _extract_time(notam_text)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Schema validation
    valid = (lat is not None and lon is not None
             and 0 <= floor < ceiling <= 60000)

    return {
        "notam_class":         notam_class,
        "class_label":         NOTAM_CLASS_LABELS[notam_class],
        "latitude_center":     lat,
        "longitude_center":    lon,
        "radius_nm":           radius,
        "altitude_floor_ft":   floor,
        "altitude_ceiling_ft": ceiling,
        "time_start_utc":      t_start,
        "time_end_utc":        t_end,
        "severity":            "HIGH" if notam_class in ("P", "TFR") else "MEDIUM",
        "_valid":              valid,
        "_latency_ms":         latency_ms,
    }


# ── Pre-loaded NOTAM examples ────────────────────────────────────────────────
EXAMPLE_NOTAMS = {
    "🔴 Emergency TFR (PDF Test Case)":
        "Emergency TFR active 37.2N 122.1W radius 10 NM SFC to FL180 immediate effect",

    "🟠 R-Class Military Restricted Area":
        "!SFO 1234/26 NOTAMN Q) ZOA/QRTCA/IV/BO/W/000/400/3742N12217W010 "
        "A) KSFO B) 2604140000 C) 2604142359 E) R-2508 AIRSPACE ACTIVE SFC TO FL400 "
        "DUE TO MILITARY OPERATIONS. CONTACT 123.45 FOR CLEARANCE.",

    "🔴 P-Class Prohibited Area":
        "!SFO NOTAM P-2515: PROHIBITED AREA ACTIVE SFC TO FL250. "
        "COORDINATES: 37.55N 122.30W RADIUS 15NM. NO AIRCRAFT PERMITTED. "
        "SECURITY OPERATIONS IN EFFECT.",

    "🟠 D-Class Danger Area (VOR Reference)":
        "DANGER AREA D-2540: DEFINED AS 045DEG/20NM FROM OAK VOR. "
        "RADIUS 10NM. ALTITUDE 5000FT TO FL180. ACTIVE 0800-1600Z. "
        "MILITARY EXERCISE IN PROGRESS.",

    "🔵 SIGMET Weather Advisory":
        "SIGMET UNIFORM 5: SEVERE TURBULENCE. 37.1N-38.2N 122.0W-123.0W. "
        "FL060 TO FL180. OBSERVED AND FORECAST. VALID 0600-1200Z.",
}
