import sys
import os
import pytest

# Ensure project root is on sys.path so tests can import local modules when workspace path contains spaces
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import calculate_sigma, make_arrhenius_points
from utils import normalize_composition, element_numeric_fields


def test_calculate_sigma_basic():
    # thickness 1 mm, electrode dia 10 mm, area = pi*(0.5 cm)^2 = ~0.7854 cm2
    # R = 1 ohm => sigma = 0.1 / (1 * 0.7854) ~ 0.1273 S/cm
    sigma = calculate_sigma(1.0, 1.0, 10.0)
    assert sigma > 0
    assert pytest.approx(sigma, rel=1e-3) == 0.127323954


def test_make_arrhenius_points():
    rows = [
        {"T_c": 600.0, "T_k": 873.15, "sigma_S_cm": 1e-2},
        {"T_c": 500.0, "T_k": 773.15, "sigma_S_cm": 2e-3},
    ]
    pts = make_arrhenius_points(rows)
    assert len(pts) == 2
    # ascending 1000/T: 1000/873.15 < 1000/773.15
    assert pts[0][0] < pts[1][0]


def test_normalize_composition_and_numeric_fields():
    comp = {"Ba": 1.0, "Zr": 0.4, "Ce": 0.4, "Y": 0.2}
    norm = normalize_composition(comp)
    assert norm["Ba"] == 1.0
    assert norm["Zr"] == 0.4
    num = element_numeric_fields(norm)
    assert num["ba"] == 1.0
    assert num["zr"] == 0.4
