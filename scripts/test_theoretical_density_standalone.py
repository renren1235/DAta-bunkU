# Quick standalone test for compute_theoretical_density
import math

def compute_theoretical_density(composition: dict, unit_cell_vol_ang3: float, z_per_cell: float = 1.0) -> float:
    if not composition or not unit_cell_vol_ang3:
        return None
    atomic_weights = {
        'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
        'C': 12.011, 'N':14.007, 'O':16.00, 'F':18.998, 'Ne':20.180,
        'Na':22.990, 'Mg':24.305, 'Al':26.982, 'Si':28.085, 'P':30.974,
        'S':32.06, 'Cl':35.45, 'K':39.098, 'Ca':40.078, 'Sc':44.956,
        'Ti':47.867, 'V':50.942, 'Cr':51.996, 'Mn':54.938, 'Fe':55.845,
        'Co':58.933, 'Ni':58.693, 'Cu':63.546, 'Zn':65.38, 'Ga':69.723,
        'Ge':72.630, 'As':74.922, 'Se':78.971, 'Br':79.904, 'Kr':83.798,
        'Rb':85.468, 'Sr':87.62, 'Y':88.906, 'Zr':91.224, 'Nb':92.906,
        'Mo':95.95, 'Tc':98.0, 'Ru':101.07, 'Rh':102.91, 'Pd':106.42,
        'Ag':107.87, 'Cd':112.41, 'In':114.82, 'Sn':118.71, 'Sb':121.76,
        'Te':127.60, 'I':126.90, 'Xe':131.29, 'Cs':132.91, 'Ba':137.33,
        'La':138.91, 'Ce':140.116, 'Pr':140.91, 'Nd':144.24, 'Pm':145.0,
        'Sm':150.36, 'Eu':151.96, 'Gd':157.25, 'Tb':158.93, 'Dy':162.50,
        'Ho':164.93, 'Er':167.26, 'Tm':168.93, 'Yb':173.05, 'Lu':174.97,
        'Hf':178.49, 'Ta':180.95, 'W':183.84, 'Re':186.21, 'Os':190.23,
        'Ir':192.22, 'Pt':195.08, 'Au':196.97, 'Hg':200.59, 'Tl':204.38,
        'Pb':207.2, 'Bi':208.98
    }
    mass_per_formula = 0.0
    for el, coeff in composition.items():
        if not el or coeff is None:
            continue
        el_sym = el.capitalize() if len(el) == 1 else (el[0].upper() + el[1:].lower())
        aw = atomic_weights.get(el_sym) or atomic_weights.get(el.upper())
        if aw is None:
            continue
        mass_per_formula += aw * float(coeff)
    v_cm3 = unit_cell_vol_ang3 * 1e-24
    NA = 6.02214076e23
    if v_cm3 <= 0:
        return None
    rho = mass_per_formula / (NA * v_cm3)
    return rho


def compute_theoretical_density_debug(composition: dict, unit_cell_vol_ang3: float, z_per_cell: float = 1.0) -> dict:
    debug = {'composition': composition, 'mass_per_formula': None, 'v_cm3': None, 'NA':6.02214076e23, 'base_theo': None}
    if not composition or not unit_cell_vol_ang3:
        return debug
    atomic_weights = {
        'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
        'C': 12.011, 'N':14.007, 'O':16.00, 'F':18.998, 'Ne':20.180,
        'Na':22.990, 'Mg':24.305, 'Al':26.982, 'Si':28.085, 'P':30.974,
        'S':32.06, 'Cl':35.45, 'K':39.098, 'Ca':40.078, 'Sc':44.956,
        'Ti':47.867, 'V':50.942, 'Cr':51.996, 'Mn':54.938, 'Fe':55.845,
        'Co':58.933, 'Ni':58.693, 'Cu':63.546, 'Zn':65.38, 'Ga':69.723,
        'Ge':72.630, 'As':74.922, 'Se':78.971, 'Br':79.904, 'Kr':83.798,
        'Rb':85.468, 'Sr':87.62, 'Y':88.906, 'Zr':91.224, 'Nb':92.906,
        'Mo':95.95, 'Tc':98.0, 'Ru':101.07, 'Rh':102.91, 'Pd':106.42,
        'Ag':107.87, 'Cd':112.41, 'In':114.82, 'Sn':118.71, 'Sb':121.76,
        'Te':127.60, 'I':126.90, 'Xe':131.29, 'Cs':132.91, 'Ba':137.33,
        'La':138.91, 'Ce':140.116, 'Pr':140.91, 'Nd':144.24, 'Pm':145.0,
        'Sm':150.36, 'Eu':151.96, 'Gd':157.25, 'Tb':158.93, 'Dy':162.50,
        'Ho':164.93, 'Er':167.26, 'Tm':168.93, 'Yb':173.05, 'Lu':174.97,
        'Hf':178.49, 'Ta':180.95, 'W':183.84, 'Re':186.21, 'Os':190.23,
        'Ir':192.22, 'Pt':195.08, 'Au':196.97, 'Hg':200.59, 'Tl':204.38,
        'Pb':207.2, 'Bi':208.98
    }
    mass_per_formula = 0.0
    for el, coeff in composition.items():
        if not el or coeff is None:
            continue
        el_sym = el.capitalize() if len(el) == 1 else (el[0].upper() + el[1:].lower())
        aw = atomic_weights.get(el_sym) or atomic_weights.get(el.upper())
        if aw is None:
            continue
        mass_per_formula += aw * float(coeff)
    v_cm3 = unit_cell_vol_ang3 * 1e-24
    debug['mass_per_formula'] = mass_per_formula
    debug['v_cm3'] = v_cm3
    if v_cm3 and v_cm3 > 0:
        debug['base_theo'] = mass_per_formula / (debug['NA'] * v_cm3)
    return debug


if __name__ == '__main__':
    comp = {'Ba':1.0, 'Zr':0.4, 'Ce':0.4, 'Y':0.2, 'O':3.0}
    ucv = 240.0
    z = 1.0
    print('composition:', comp)
    print('unit_cell_vol:', ucv)
    print(f'theoretical density (Z={z}):', compute_theoretical_density(comp, ucv, z))
    print('debug:', compute_theoretical_density_debug(comp, ucv, z))
