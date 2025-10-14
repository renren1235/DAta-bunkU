import math
from typing import List, Dict, Any, Tuple


def calculate_sigma(thickness_mm: float, resistance_ohm: float, electrode_diameter_mm: float) -> float:
    """Calculate conductivity sigma in S/cm from thickness (mm), resistance (ohm), electrode diameter (mm).

    sigma = t_cm / (R * A_cm2)
    where A_cm2 = pi * (d_cm / 2)^2
    """
    if resistance_ohm <= 0:
        raise ValueError("Resistance must be > 0")
    if thickness_mm <= 0:
        raise ValueError("Thickness must be > 0")
    if electrode_diameter_mm <= 0:
        raise ValueError("Electrode diameter must be > 0")

    t_cm = thickness_mm / 10.0
    d_cm = electrode_diameter_mm / 10.0
    area_cm2 = math.pi * (d_cm / 2.0) ** 2
    sigma = t_cm / (resistance_ohm * area_cm2)
    return sigma


def make_arrhenius_points(rows: List[Dict[str, Any]]) -> List[List[float]]:
    """Given rows with keys: T_c (Celsius), T_k, sigma_S_cm -> returns list of [1000/T, ln(sigma*T)] sorted ascending by 1000/T"""
    pts = []
    for r in rows:
        T_k = r.get("T_k") or (r.get("T_c") + 273.15)
        sigma = r.get("sigma_S_cm")
        if sigma is None or sigma <= 0:
            continue
        x = 1000.0 / T_k
        y = math.log(sigma * T_k)
        pts.append([x, y])
    pts.sort(key=lambda p: p[0])
    return pts


ELEMENTS = ["Ba", "Zr", "Ce", "Y", "Yb", "Zn", "Ni", "Co", "Fe"]


def normalize_composition(comp: Dict[str, float]) -> Dict[str, float]:
    """Normalize composition so that Ba == 1.0 if Ba present; otherwise normalize so sum == 1.0.

    Input: {'Ba':1.0, 'Zr':0.4, ...}
    Output: {'Ba':1.0, 'Zr':0.4, ...} (floats)
    """
    if not comp:
        return {}
    comp_num: Dict[str, float] = {}
    for k, v in comp.items():
        try:
            comp_num[k] = float(v)
        except Exception:
            continue
    ba_val = comp_num.get('Ba')
    if ba_val is not None and ba_val != 0:
        return {k: (v / ba_val) for k, v in comp_num.items()}
    total = sum(comp_num.values())
    if total == 0:
        return comp_num
    return {k: (v / total) for k, v in comp_num.items()}


def element_numeric_fields(comp_norm: Dict[str, float]) -> Dict[str, float]:
    """Return a dict of numeric element fields with lowercase keys for MI-ready columns.

    Missing elements are returned as None.
    """
    out: Dict[str, float] = {}
    for e in ELEMENTS:
        val = comp_norm.get(e)
        out[e.lower()] = float(val) if val is not None else None
    return out


def comp_to_feature_vector(comp_norm: Dict[str, float]) -> Tuple[List[str], List[float]]:
    """Return ordered element names and vector (NaN->None) suitable for ML features."""
    names = [e.lower() for e in ELEMENTS]
    vec = [comp_norm.get(e) if comp_norm.get(e) is not None else None for e in ELEMENTS]
    return names, vec

