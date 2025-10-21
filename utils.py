import math
from typing import List, Dict, Any, Tuple, Optional


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


def compute_measured_density(pellet_mass_g: float, pellet_thickness_mm: float, pellet_diameter_mm: float) -> Optional[float]:
    """Compute measured density (g/cm^3) from pellet mass (g), thickness (mm), diameter (mm).

    Returns None if inputs are invalid or non-positive.
    """
    try:
        pmass = float(pellet_mass_g)
        th = float(pellet_thickness_mm)
        dia = float(pellet_diameter_mm)
        if pmass <= 0 or th <= 0 or dia <= 0:
            return None
        th_cm = th / 10.0
        d_cm = dia / 10.0
        vol_cm3 = math.pi * (d_cm / 2.0) ** 2 * th_cm
        if vol_cm3 <= 0:
            return None
        return pmass / vol_cm3
    except Exception:
        return None


def compute_relative_density(measured_g_cm3: Optional[float], theoretical_g_cm3: Optional[float]) -> Optional[float]:
    """Compute relative density (%) given measured and theoretical densities.

    Returns None if inputs are invalid or theoretical <= 0.
    """
    try:
        if measured_g_cm3 is None or theoretical_g_cm3 is None:
            return None
        m = float(measured_g_cm3)
        t = float(theoretical_g_cm3)
        if t <= 0:
            return None
        return (m / t) * 100.0
    except Exception:
        return None


def compute_theoretical_density(composition: Dict[str, float], unit_cell_vol_ang3: float, z_per_cell: float = 1.0) -> Optional[float]:
    """Compute theoretical density (g/cm^3) from composition, unit cell volume (Å^3), and Z per cell.

    ρ = (Σ n_i M_i) * Z / (N_A * V_cell),  V_cell[cm^3] = unit_cell_vol_ang3 * 1e-24
    Returns None if inputs are invalid.
    """
    try:
        if not composition or not unit_cell_vol_ang3:
            return None
        atomic_weights = {
            'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
            'C': 12.011, 'N': 14.007, 'O': 16.00, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
            'S': 32.06, 'Cl': 35.45, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956,
            'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845,
            'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723,
            'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
            'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906,
            'Mo': 95.95, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42,
            'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71, 'Sb': 121.76,
            'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33,
            'La': 138.91, 'Ce': 140.116, 'Pr': 140.91, 'Nd': 144.24, 'Pm': 145.0,
            'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
            'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97,
            'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23,
            'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38,
            'Pb': 207.2, 'Bi': 208.98
        }
        mass_per_formula = 0.0
        for el, coeff in composition.items():
            if el is None or coeff is None:
                continue
            el_str = str(el)
            # normalize symbol capitalization
            el_sym = el_str.capitalize() if len(el_str) == 1 else (el_str[0].upper() + el_str[1:].lower())
            aw = atomic_weights.get(el_sym) or atomic_weights.get(el_str.upper())
            if aw is None:
                continue
            mass_per_formula += aw * float(coeff)
        v_cm3 = float(unit_cell_vol_ang3) * 1e-24
        if v_cm3 <= 0:
            return None
        NA = 6.02214076e23
        return (mass_per_formula * float(z_per_cell)) / (NA * v_cm3)
    except Exception:
        return None


def compute_theoretical_density_debug(composition: Dict[str, float], unit_cell_vol_ang3: float, z_per_cell: float = 1.0) -> Dict[str, Any]:
    """Return intermediate values for theoretical density calculation for debugging/inspection."""
    dbg: Dict[str, Any] = {
        'composition': composition,
        'mass_per_formula': None,
        'v_cm3': None,
        'NA': 6.02214076e23,
        'base_per_formula': None,
        'final_density': None,
    }
    rho = compute_theoretical_density(composition, unit_cell_vol_ang3, z_per_cell)
    try:
        # recompute pieces for debug info
        if not composition or not unit_cell_vol_ang3:
            return dbg
        atomic_weights = {
            'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
            'C': 12.011, 'N': 14.007, 'O': 16.00, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
            'S': 32.06, 'Cl': 35.45, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956,
            'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845,
            'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723,
            'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
            'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906,
            'Mo': 95.95, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42,
            'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71, 'Sb': 121.76,
            'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33,
            'La': 138.91, 'Ce': 140.116, 'Pr': 140.91, 'Nd': 144.24, 'Pm': 145.0,
            'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
            'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97,
            'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23,
            'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38,
            'Pb': 207.2, 'Bi': 208.98
        }
        mass_per_formula = 0.0
        for el, coeff in composition.items():
            if el is None or coeff is None:
                continue
            el_str = str(el)
            el_sym = el_str.capitalize() if len(el_str) == 1 else (el_str[0].upper() + el_str[1:].lower())
            aw = atomic_weights.get(el_sym) or atomic_weights.get(el_str.upper())
            if aw is None:
                continue
            mass_per_formula += aw * float(coeff)
        v_cm3 = float(unit_cell_vol_ang3) * 1e-24
        dbg['mass_per_formula'] = mass_per_formula
        dbg['v_cm3'] = v_cm3
        if v_cm3 > 0:
            dbg['base_per_formula'] = mass_per_formula / (dbg['NA'] * v_cm3)
            dbg['final_density'] = rho
    except Exception:
        pass
    return dbg

