import math
import os
import sys
import pytest

# Ensure project root is on sys.path so tests can import local modules when workspace path contains spaces
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import compute_measured_density, compute_relative_density
from utils import normalize_composition


def test_compute_measured_density_basic():
    # pellet: mass=0.5 g, thickness=1.0 mm, diameter=10.0 mm
    # volume = pi*(0.5 cm)^2*0.1 cm = ~0.07854 cm3
    # density ~ 0.5 / 0.07854 = 6.3662 g/cm3
    d = compute_measured_density(0.5, 1.0, 10.0)
    assert d is not None
    assert pytest.approx(d, rel=1e-4) == 6.3661977


def test_compute_measured_density_invalid():
    assert compute_measured_density(0.5, 0.0, 10.0) is None
    assert compute_measured_density(0.5, 1.0, 0.0) is None
    assert compute_measured_density(0.0, 1.0, 10.0) is None


def test_compute_relative_density():
    # measured 6.0, theoretical 6.25 => 96%
    rel = compute_relative_density(6.0, 6.25)
    assert pytest.approx(rel, rel=1e-6) == 96.0

    # invalid inputs
    assert compute_relative_density(None, 6.0) is None
    assert compute_relative_density(6.0, None) is None
    assert compute_relative_density(6.0, 0.0) is None
