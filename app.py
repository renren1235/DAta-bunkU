import os

# --- Unified field list for Add/Update Sample ---
SAMPLE_FORM_FIELDS = [
	# 基本情報
	'sample_no', 'composition', 'comp_normalized', 'element_numeric',
	# 構造・格子
	'crystal_system', 'a', 'a_err', 'b', 'b_err', 'c', 'c_err',
	'unit_cell_volume', 'unit_cell_volume_err', 'z_per_cell',
	# 密度
	'theoretical_density_g_cm3', 'measured_density_g_cm3', 'relative_density_pct',
	# ペレット・電極
	'pellet_mass_g', 'pellet_thickness_mm', 'pellet_diameter_mm',
	'thickness_mm', 'electrode_diameter_mm',
	# 合成・測定条件
	'synthesis_method', 'calcination_temp_c', 'calcination_time_h', 'atmosphere',
	# 抵抗
	'resistances',
	# その他
	'created_at', 'meta'
]
import json
import uuid
import sqlite3
import math
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List
import streamlit as st
import pandas as pd
from utils import calculate_sigma, make_arrhenius_points, normalize_composition, element_numeric_fields
from utils import compute_measured_density, compute_relative_density
from utils import compute_theoretical_density, compute_theoretical_density_debug

# Paths
DB_PATH = os.environ.get('SAMPLES_DB_PATH', 'samples.db')
PRESET_PATH = os.environ.get('PRESETS_PATH', 'presets.json')


def get_db_connection():
	"""Return a sqlite3 connection configured for better concurrency (WAL)."""
	conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
	try:
		# enable WAL mode for better concurrency (readers won't block writers as much)
		cur = conn.cursor()
		cur.execute("PRAGMA journal_mode=WAL;")
		# reduce sync overhead for speed; improves throughput but relaxes durability slightly
		cur.execute("PRAGMA synchronous=NORMAL;")
		conn.commit()
	except Exception:
		# ignore PRAGMA errors and continue with default connection
		pass
	return conn


def init_db():
	"""Initialize the local SQLite database with required tables."""
	conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
	cur = conn.cursor()
	cur.execute(
		"""
		CREATE TABLE IF NOT EXISTS samples (
			id TEXT PRIMARY KEY,
			sample_no TEXT,
			created_at TEXT,
			data_json TEXT
		)
		"""
	)
	conn.commit()
	conn.close()


def get_storage_mode() -> str:
	"""Return current storage mode from session (local or supabase)."""
	try:
		mode = st.session_state.get('storage_mode') or 'local'
		return mode
	except Exception:
		return 'local'


def save_sample_local(sample: dict):
	conn = get_db_connection()
	cur = conn.cursor()
	sid = sample.get("id") or str(uuid.uuid4())
	# keep existing created_at when editing; set if new
	created_at = sample.get("created_at") or datetime.utcnow().isoformat()
	# also reflect these into the JSON blob for consistency
	sample["id"] = sid
	sample["created_at"] = created_at
	cur.execute("REPLACE INTO samples (id, sample_no, created_at, data_json) VALUES (?, ?, ?, ?)",
				(sid, sample.get("sample_no"), created_at, json.dumps(sample, ensure_ascii=False)))
	conn.commit()
	conn.close()
	return sid


def load_all_samples_local() -> List[Dict[str, Any]]:
	conn = get_db_connection()
	cur = conn.cursor()
	cur.execute("SELECT id, sample_no, created_at, data_json FROM samples ORDER BY created_at DESC")
	rows = cur.fetchall()
	conn.close()
	samples: List[Dict[str, Any]] = []
	for r in rows:
		try:
			data = json.loads(r[3])
		except Exception:
			data = {}
		samples.append({"id": r[0], "sample_no": r[1], "created_at": r[2], **data})
	return samples


def delete_sample_local(sample_id: str):
	conn = get_db_connection()
	cur = conn.cursor()
	cur.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
	conn.commit()
	conn.close()
	return True


# Supabase stubs — will raise to trigger fallback if selected without setup
def supabase_save_sample(sample: dict):
	raise RuntimeError("Supabase not configured")


def supabase_load_all_samples() -> List[Dict[str, Any]]:
	raise RuntimeError("Supabase not configured")


def supabase_delete_sample(sample_id: str):
	raise RuntimeError("Supabase not configured")


def save_sample(sample: dict):
	# Attempt to save to the selected storage. If Supabase is chosen but fails, fall back to local.
	if get_storage_mode() == 'supabase':
		try:
			return supabase_save_sample(sample)
		except Exception as e:
			try:
				st.warning(f"Supabase 保存に失敗したためローカル DB に保存します: {e}")
			except Exception:
				pass
			return save_sample_local(sample)
	else:
		return save_sample_local(sample)


def load_all_samples() -> List[Dict[str, Any]]:
	if get_storage_mode() == 'supabase':
		try:
			return supabase_load_all_samples()
		except Exception as e:
			try:
				st.warning(f"Supabase 取得に失敗しました。ローカル DB を使用します: {e}")
			except Exception:
				pass
			return load_all_samples_local()
	else:
		return load_all_samples_local()


def compute_theoretical_density(composition: dict, unit_cell_vol_ang3: float, z_per_cell: float = 1.0) -> float:
	"""Compute theoretical density (g/cm3) given composition dict and unit cell volume in Å^3.
	If z_per_cell is provided, it is used as the number of formula units per crystallographic unit cell.
	Returns None if cannot compute.
	"""
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
	# convert Å^3 to cm^3 (1 Å^3 = 1e-24 cm^3)
	v_cm3 = unit_cell_vol_ang3 * 1e-24
	NA = 6.02214076e23
	if v_cm3 <= 0:
		return None
	# mass_per_formula is in g/mol; divide by NA to get g per formula unit,
	# multiply by z_per_cell to get mass per unit cell (g), then divide by unit cell volume (cm^3)
	rho = (mass_per_formula * float(z_per_cell)) / (NA * v_cm3)
	return rho



def compute_theoretical_density_debug(composition: dict, unit_cell_vol_ang3: float, z_per_cell: float = 1.0) -> dict:
	"""Compute same values as compute_theoretical_density but return intermediate values for debugging.
	Returns a dict with keys: mass_per_formula (g/mol), v_cm3, NA, base_per_formula (g/cm3 for Z=1),
	and final_density (g/cm3) which includes z_per_cell.
	"""
	debug = {
		'composition': composition,
		'mass_per_formula': None,
		'v_cm3': None,
		'NA': 6.02214076e23,
		'base_per_formula': None,
		'final_density': None,
	}
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
		# base per formula unit (Z = 1)
		debug['base_per_formula'] = mass_per_formula / (debug['NA'] * v_cm3)
		# final density including Z formula units per cell
		debug['final_density'] = (mass_per_formula * float(z_per_cell)) / (debug['NA'] * v_cm3)
	return debug


def safe_float(val, default=0.0):
	"""Return float(val) or default if val is None or cannot be converted."""
	try:
		if val is None:
			return float(default)
		return float(val)
	except Exception:
		return float(default)


def float_or_none(val):
	"""Convert to float if possible; return None if val is None/empty/invalid."""
	try:
		if val is None:
			return None
		if isinstance(val, str) and val.strip() == "":
			return None
		return float(val)
	except Exception:
		return None


def delete_sample(sample_id: str):
	if get_storage_mode() == 'supabase':
		return supabase_delete_sample(sample_id)
	else:
		return delete_sample_local(sample_id)


def render_delete_confirmation():
	"""If st.session_state['confirm_delete'] is set, render a confirmation UI and handle actions."""
	cid = st.session_state.get('confirm_delete')
	if not cid:
		return
	st.warning(f"以下のサンプルを本当に削除しますか？: {cid}")
	c1, c2 = st.columns([1,1])
	with c1:
		if st.button("はい、削除する", key=f"confirm_yes_{cid}"):
			try:
				# support list or single id
				if isinstance(cid, list):
					for sid in list(cid):
						delete_sample(sid)
				else:
					delete_sample(cid)
				st.success("削除しました")
			except Exception as e:
				st.error(f"削除に失敗しました: {e}")
			finally:
				st.session_state.pop('confirm_delete', None)
				st.rerun()
	with c2:
		if st.button("キャンセル", key=f"confirm_no_{cid}"):
			st.session_state.pop('confirm_delete', None)
			st.rerun()


def load_presets() -> Dict[str, Dict[str, float]]:
	try:
		with open(PRESET_PATH, 'r', encoding='utf-8') as f:
			return json.load(f)
	except Exception:
		# default presets
		defaults = {
			"Ba1Zr0.4Ce0.4Y0.2O3": {"Ba": 1.0, "Zr": 0.4, "Ce": 0.4, "Y": 0.2, "O": 3.0},
			"Ba1Zr0.4Ce0.4Y0.1Yb0.1O3": {"Ba": 1.0, "Zr": 0.4, "Ce": 0.4, "Y": 0.1, "Yb": 0.1, "O": 3.0},
			"Ba1Zr0.1Ce0.7Y0.2O3": {"Ba": 1.0, "Zr": 0.1, "Ce": 0.7, "Y": 0.2, "O": 3.0},
			"Ba1Zr0.1Ce0.7Y0.1Yb0.1O3": {"Ba": 1.0, "Zr": 0.1, "Ce": 0.7, "Y": 0.1, "Yb": 0.1, "O": 3.0},
			"Ba1Ce0.8Y0.1Yb0.1O3": {"Ba": 1.0, "Ce": 0.8, "Y": 0.1, "Yb": 0.1, "O": 3.0},
			"Ba1Zr0.37Ce0.4Y0.2Zn0.03O3": {"Ba": 1.0, "Zr": 0.37, "Ce": 0.4, "Y": 0.2, "Zn": 0.03, "O": 3.0},
			"Ba1Zr0.07Ce0.7Y0.2Zn0.03O3": {"Ba": 1.0, "Zr": 0.07, "Ce": 0.7, "Y": 0.2, "Zn": 0.03, "O": 3.0},
		}
		return defaults


def save_presets(presets: Dict[str, Dict[str, float]]):
	with open(PRESET_PATH, 'w', encoding='utf-8') as f:
		json.dump(presets, f, ensure_ascii=False, indent=2)


def excel_bytes_from_df(df: pd.DataFrame, include_header: bool = True) -> bytes:
	buffer = BytesIO()
	with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
		# pandas to_excel always writes header if columns are present; to emulate exclude header,
		# write with header if include_header True, otherwise write without header and shift rows.
		if include_header:
			df.to_excel(writer, index=False, sheet_name='arrhenius')
		else:
			# write without header by setting header=False, but to_excel still expects columns; this will write values starting at row 0
			df.to_excel(writer, index=False, header=False, sheet_name='arrhenius')
	return buffer.getvalue()


st.set_page_config(page_title="DAta bankU", layout="wide")
init_db()

# Small visual polish: chemistry-like accent and cards
st.markdown(
	"""
	<style>
	/* Header style */
	.db-header {font-family: 'Helvetica Neue', Arial, sans-serif; background: linear-gradient(90deg,#0f172a,#0b1220); color: #ffffff; padding: 14px 18px; border-radius: 8px; box-shadow: 0 4px 12px rgba(2,6,23,0.6); text-align:center}
	.db-sub {color: #ffffff; font-size:14px; text-align:center}
	.stButton>button {background-color:#0ea5a5; color:white}
	.card {background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:10px; border-radius:8px}
	</style>
	<div class='db-header'><h1 style='margin:0'>DAta bankU</h1><div class='db-sub'>Arrhenius & Conductivity — Materials database</div></div>
	""",
	unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("Storage / Settings")
st.sidebar.markdown("Data stored locally in `samples.db`. Use Supabase for cloud sync (see README).")
# storage mode selector (local or supabase)
if 'storage_mode' not in st.session_state:
	st.session_state['storage_mode'] = 'local'
st.session_state['storage_mode'] = st.sidebar.selectbox("Storage mode", options=['local', 'supabase'], index=0)

# Load presets
presets = load_presets()
if 'presets' not in st.session_state:
	st.session_state['presets'] = presets

# New sample button (outside form)
col_new1, col_new2 = st.columns([1, 5])
with col_new1:
	if st.button("➕ 新規作成", key='new_sample_btn', help="フォームをクリアして新しいサンプルを作成"):
		# Clear editing state and reset form fields to defaults
		st.session_state['editing_sample'] = None
		st.session_state.pop('editing_loaded_id', None)
		st.session_state['sample_no'] = ''
		st.session_state['composition_text'] = ''
		st.session_state['synthesis_method'] = ''
		st.session_state['calcination_temp_c'] = 1200.0
		st.session_state['calcination_time_h'] = 10.0
		st.session_state['crystal_system'] = ''
		st.session_state['unit_cell_vol'] = ''
		st.session_state['unit_cell_vol_err'] = ''
		st.session_state['pellet_mass_g'] = 0.0
		st.session_state['pellet_thickness_mm'] = 1.0
		st.session_state['pellet_diameter_mm'] = 10.0
		st.session_state['thickness_mm'] = 1.0
		st.session_state['electrode_diameter_mm'] = 5.0
		st.session_state['theoretical_density_input'] = 0.0
		st.session_state['relative_density_pct'] = 95.0
		st.session_state['computed_measured_density'] = None
		st.session_state['computed_theoretical_density'] = None
		st.session_state['computed_relative_density'] = None
		st.session_state['measured_density_display_value'] = 0.0
		st.session_state['theoretical_density_display_value'] = 0.0
		st.session_state['relative_density_display_value'] = 0.0
		st.success("フォームをクリアしました。新規サンプルを作成できます。")
		st.rerun()
with col_new2:
	st.markdown("")

with st.form("sample_form", clear_on_submit=False):
	st.subheader("Add / Update Sample")
	# Prefill fields from session_state when editing
	# Use None (not empty dict) to represent no edit in progress
	if 'editing_sample' not in st.session_state:
		st.session_state['editing_sample'] = None

	# editing may be None when no sample is being edited; use empty dict to avoid AttributeError on .get()
	editing = st.session_state.get('editing_sample') or {}

	# Initialize per-sample/session keys only if missing. Do NOT aggressively clear values on every render
	# This preserves user-entered form data when selecting/saving presets.
	if 'composition_text' not in st.session_state:
		st.session_state['composition_text'] = ''
	if 'unit_cell_vol' not in st.session_state:
		st.session_state['unit_cell_vol'] = ''
	if 'unit_cell_vol_err' not in st.session_state:
		st.session_state['unit_cell_vol_err'] = ''
	if 'pellet_mass_g' not in st.session_state:
		st.session_state['pellet_mass_g'] = 0.0
	if 'pellet_thickness_mm' not in st.session_state:
		st.session_state['pellet_thickness_mm'] = 1.0
	if 'pellet_diameter_mm' not in st.session_state:
		st.session_state['pellet_diameter_mm'] = 10.0
	if 'computed_measured_density' not in st.session_state:
		st.session_state['computed_measured_density'] = None
	if 'computed_theoretical_density' not in st.session_state:
		st.session_state['computed_theoretical_density'] = None
	if 'computed_relative_density' not in st.session_state:
		st.session_state['computed_relative_density'] = None
	if 'theoretical_density_input' not in st.session_state:
		st.session_state['theoretical_density_input'] = 0.0
	if 'relative_density_pct' not in st.session_state:
		st.session_state['relative_density_pct'] = 95.0

	# If we are editing an existing sample, populate session_state with its computed values once
	if editing:
		loaded_id = st.session_state.get('editing_loaded_id')
		if editing.get('id') and editing.get('id') != loaded_id:
			st.session_state['editing_loaded_id'] = editing.get('id')
			# populate form/session fields from the editing sample (only overwrite when loading a new edit)
			st.session_state['composition_text'] = json.dumps(editing.get('composition', {}), ensure_ascii=False) if editing.get('composition') is not None else st.session_state.get('composition_text', '')
			st.session_state['unit_cell_vol'] = str(editing.get('unit_cell_volume')) if editing.get('unit_cell_volume') is not None else st.session_state.get('unit_cell_vol', '')
			st.session_state['unit_cell_vol_err'] = str(editing.get('unit_cell_volume_err')) if editing.get('unit_cell_volume_err') is not None else st.session_state.get('unit_cell_vol_err', '')
			st.session_state['pellet_mass_g'] = editing.get('pellet_mass_g') or st.session_state.get('pellet_mass_g', 0.0)
			st.session_state['pellet_thickness_mm'] = editing.get('pellet_thickness_mm') or editing.get('thickness_mm') or st.session_state.get('pellet_thickness_mm', 1.0)
			st.session_state['pellet_diameter_mm'] = editing.get('pellet_diameter_mm') or editing.get('electrode_diameter_mm') or st.session_state.get('pellet_diameter_mm', 10.0)
			st.session_state['z_per_cell'] = editing.get('z_per_cell') or st.session_state.get('z_per_cell') or 1.0
			# ensure sample_no is available in session_state so the text_input reflects it
			st.session_state['sample_no'] = editing.get('sample_no', '')
			# Prefill synthesis/calcination fields for editing
			st.session_state['synthesis_method'] = editing.get('synthesis_method') or st.session_state.get('synthesis_method', '')
			st.session_state['calcination_temp_c'] = editing.get('calcination_temp_c') if editing.get('calcination_temp_c') is not None else st.session_state.get('calcination_temp_c', 1200)
			st.session_state['calcination_time_h'] = editing.get('calcination_time_h') if editing.get('calcination_time_h') is not None else st.session_state.get('calcination_time_h', 10.0)
			st.session_state['computed_measured_density'] = editing.get('measured_density_g_cm3') or st.session_state.get('computed_measured_density')
			st.session_state['computed_theoretical_density'] = editing.get('theoretical_density_g_cm3') or st.session_state.get('computed_theoretical_density')
			st.session_state['computed_relative_density'] = editing.get('relative_density_pct') or st.session_state.get('computed_relative_density')
			st.session_state['theoretical_density_input'] = editing.get('theoretical_density_g_cm3') or st.session_state.get('theoretical_density_input', 0.0)
			st.session_state['relative_density_pct'] = editing.get('relative_density_pct') or st.session_state.get('relative_density_pct', 0.0)

	# sample_no is session_state-backed so edits update reliably
	sample_no = st.text_input("試料No.", value=st.session_state.get('sample_no', ''), key='sample_no')

	col1, col2 = st.columns(2)
	with col1:
		# Use session_state-backed defaults so editing loads values reliably
		synthesis_method = st.text_input("合成方法", value=st.session_state.get('synthesis_method', ''), key='synthesis_method')
		calcination_temp_c = st.number_input("焼成温度 (℃)", value=safe_float(st.session_state.get('calcination_temp_c', 1200)), key='calcination_temp_c')
		calcination_time_h = st.number_input("焼成時間 (h)", value=safe_float(st.session_state.get('calcination_time_h', 10.0)), format="%.2f", key='calcination_time_h')
		
		# crystal system: allow free text input with selectbox for common choices
		crystal_presets = ["(手動入力)", "立方晶 (Cubic)", "正方晶 (Tetragonal)", "六方晶 (Hexagonal)", "直方晶 (Orthorhombic)", "単斜晶 (Monoclinic)", "三斜晶 (Triclinic)", "菱面体 (Rhombohedral)"]
		
		# Determine current crystal system value and selectbox index
		current_crystal = st.session_state.get('crystal_system', editing.get('crystal_system', ''))
		if current_crystal in crystal_presets:
			default_idx = crystal_presets.index(current_crystal)
		elif current_crystal:
			default_idx = 0  # manual input
		else:
			default_idx = 0
		
		crystal_select = st.selectbox("結晶系", options=crystal_presets, index=default_idx, key='crystal_system_select')
		
		# If manual input is selected, show text input field
		if crystal_select == "(手動入力)":
			crystal_system = st.text_input("結晶系 (手動入力)", value=current_crystal if current_crystal not in crystal_presets[1:] else "", key='crystal_system_manual')
		else:
			crystal_system = crystal_select
		# lattice parameters are shown by default
		st.markdown("---")
		col_la, col_lb, col_lc = st.columns(3)
		with col_la:
			a_val = st.text_input("a (Å)", value=(str(editing.get('a')) if editing.get('a') is not None else ""), key='a_val')
			a_err = st.text_input("a エラー (Å)", value=(str(editing.get('a_err')) if editing.get('a_err') is not None else ""), key='a_err')
		with col_lb:
			b_val = st.text_input("b (Å)", value=(str(editing.get('b')) if editing.get('b') is not None else ""), key='b_val')
			b_err = st.text_input("b エラー (Å)", value=(str(editing.get('b_err')) if editing.get('b_err') is not None else ""), key='b_err')
		with col_lc:
			c_val = st.text_input("c (Å)", value=(str(editing.get('c')) if editing.get('c') is not None else ""), key='c_val')
			c_err = st.text_input("c エラー (Å)", value=(str(editing.get('c_err')) if editing.get('c_err') is not None else ""), key='c_err')
		vc1, vc2 = st.columns(2)
		with vc1:
			unit_cell_vol = st.text_input("格子体積 (Å^3)", value=(str(editing.get('unit_cell_volume')) if editing.get('unit_cell_volume') is not None else ""), key='unit_cell_vol')
		with vc2:
			unit_cell_vol_err = st.text_input("格子体積エラー (Å^3)", value=(str(editing.get('unit_cell_volume_err')) if editing.get('unit_cell_volume_err') is not None else ""), key='unit_cell_vol_err')
		# bind relative density to session_state so external compute button can update it
		relative_density_pct = st.number_input("相対密度 (%)", value=safe_float(st.session_state.get('relative_density_display_value', st.session_state.get('relative_density_pct', editing.get('relative_density_pct', 95.0)))), key='relative_density_pct')
		atmosphere = st.text_input("測定雰囲気", value=editing.get('atmosphere', "Air"))

	with col2:
		# give explicit keys so values are available outside the form for immediate compute
		thickness_mm = st.number_input("試料厚さ (mm)", value=float(editing.get('thickness_mm', 1.0)), format="%.4f", key='thickness_mm')
		electrode_diameter_mm = st.number_input("電極直径 (mm)", value=float(editing.get('electrode_diameter_mm', 5.0)), format="%.4f", key='electrode_diameter_mm')
        
		# pellet mass for density calculation (g)
		pellet_mass_g = st.number_input("ペレット質量 (g)", value=float(editing.get('pellet_mass_g', 0.0)), format="%.6f", key='pellet_mass_g')

		# separate pellet geometry for density calculation (thickness and diameter) — distinct from electrode geometry
		pellet_thickness_mm = st.number_input("ペレット厚さ (mm)", value=float(editing.get('pellet_thickness_mm', editing.get('thickness_mm', 1.0))), format="%.4f", key='pellet_thickness_mm')
		pellet_diameter_mm = st.number_input("ペレット直径 (mm)", value=float(editing.get('pellet_diameter_mm', editing.get('electrode_diameter_mm', 10.0))), format="%.4f", key='pellet_diameter_mm')

		# manual theoretical density input (user requested separate input)
		theoretical_density_input = st.number_input("理論密度 (g/cm^3) — 別入力 (任意)", value=safe_float(st.session_state.get('theoretical_density_display_value', editing.get('theoretical_density_g_cm3', 0.0) or 0.0)), format="%.6f", key='theoretical_density_input')

		# show measured density in-form so users can see/save it after computed results (value driven by display key)
		measured_density_form = st.number_input("実測密度 (g/cm^3) — フォーム表示", value=safe_float(st.session_state.get('measured_density_display_value', editing.get('measured_density_g_cm3', 0.0) or 0.0)), format="%.6f", key='measured_density_form')

		# Z: formula units per unit cell (for theoretical density calculation)
		z_per_cell = st.number_input("Z (単位格子あたりの式数)", value=float(editing.get('z_per_cell', 1.0)), format="%.3f", key='z_per_cell')

		# show computed/measured density (readable field) — bound to session_state so compute updates reflect
		st.markdown("密度の計算結果はフォーム下部の 'Density Readouts' パネルに表示されます。")

		# compute density button moved below the form (see outside form) to avoid Streamlit form callback restrictions

		# Preset selector and save/delete UI
		st.markdown("---")
		st.markdown("### 組成プリセット")
		preset_names = ["(none)"] + list(st.session_state['presets'].keys())

		# Ensure we have a session_state key to hold the composition text
		if 'composition_text' not in st.session_state:
			# initialize from editing or empty
			st.session_state['composition_text'] = json.dumps(editing.get('composition', {}), ensure_ascii=False) if editing else ""

		chosen = st.selectbox("プリセットを選ぶ", preset_names, index=0, key='preset_select')
		# NOTE: callbacks are not allowed inside forms. Instead update session_state directly here.
		val = st.session_state.get('preset_select')
		if val and val != '(none)':
			st.session_state['composition_text'] = json.dumps(st.session_state['presets'].get(val, {}), ensure_ascii=False)
		else:
			# If no preset chosen and we are editing an existing sample, ensure the edit composition is loaded
			if editing and (not st.session_state.get('composition_text')):
				st.session_state['composition_text'] = json.dumps(editing.get('composition', {}), ensure_ascii=False)
		comp_text_area = st.empty()

		# allow user to save current JSON composition as a named preset
		new_preset_name = st.text_input("プリセット名を保存 (任意)")
		# Use form_submit_button for actions inside a form
		save_preset_btn = st.form_submit_button("プリセットを保存")
		delete_preset_btn = st.form_submit_button("選択プリセットを削除")

	st.markdown("---")
	st.markdown("導電率用: 抵抗値を温度ごとに入力してください (Ω)。空欄は未測定扱い。")
	default_temps = [600, 550, 500, 450, 400]
	resistance_inputs = {}
	# determine extra temps from editing sample resistances (if any)
	editing_res = editing.get('resistances', {}) if editing else {}
	extra_temps = []
	if editing_res:
		import re
		for k in editing_res.keys():
			m = re.search(r"(\d{3})", str(k))
			if m:
				val = int(m.group(1))
				if val not in default_temps:
					extra_temps.append(val)
	# build final temperature list preserving order: defaults then extras
	all_temps = default_temps + sorted(set(extra_temps))
	cols = st.columns(len(all_temps))
	for i, T in enumerate(all_temps):
		pref = ""
		if editing_res:
			try:
				val_obj = editing_res.get(str(T)) or editing_res.get(int(T)) or editing_res.get(f"R_{T}") or {}
				if isinstance(val_obj, dict):
					v = val_obj.get('resistance_ohm')
					if v is not None:
						pref = str(v)
			except Exception:
				pref = ""
		val = cols[i].text_input(f"R @ {T}°C (Ω)", value=pref, key=f"r_{T}")
		resistance_inputs[str(T)] = val

	# Use session_state-backed composition text so preset selection can update it immediately
	comp_text = comp_text_area.text_area("組成 JSON (例: {\"Ba\":0.5, \"Zr\":0.5})", value=st.session_state.get('composition_text', ''), height=120, key='composition_text')

	# Two buttons: one for density calculation only, one for save
	st.markdown("---")
	btn_col1, btn_col2 = st.columns(2)
	with btn_col1:
		compute_only = st.form_submit_button("🧮 密度を計算", help="密度を計算して表示します（保存しません）")
	with btn_col2:
		submitted = st.form_submit_button("💾 計算して保存", help="密度を計算してサンプルを保存します")

# Handle compute_only button (density calculation without saving)
if compute_only:
	# Parse composition first
	try:
		composition = json.loads(comp_text) if comp_text and comp_text.strip() else {}
		# use form variables directly (they are available after form submit)
		th = safe_float(pellet_thickness_mm or 0.0)
		dia = safe_float(pellet_diameter_mm or 0.0)
		pmass = safe_float(pellet_mass_g or 0.0)
		manual_theo = safe_float(theoretical_density_input or 0.0)
		z = safe_float(z_per_cell or 1.0)
		ucv_val = safe_float(unit_cell_vol or 0.0)
		# validations
		errs = []
		if pmass <= 0:
			errs.append('ペレット質量が 0 または未設定です（g）。')
		if th <= 0:
			errs.append('ペレット厚さが 0 または未設定です（mm）。')
		if dia <= 0:
			errs.append('ペレット直径が 0 または未設定です（mm）。')
		# compute measured
		meas = None
		if not errs:
			try:
				meas = compute_measured_density(pmass, th, dia)
				if meas is None:
					errs.append('実測密度の計算に必要な入力が不足しています。')
			except Exception as e:
				errs.append(f'実測密度計算で例外: {e}')
		else:
			# show immediate errors about measured input
			for e in errs:
				st.error(e)
		# compute theoretical: prefer manual, else from composition + unit_cell_vol using Z
		theo = None
		comp_dbg = composition
		# if manual theoretical given, use it
		if manual_theo and manual_theo > 0:
			theo = float(manual_theo)
		else:
			if ucv_val <= 0:
				st.warning('格子体積 (unit_cell_vol) が 0 または未設定です。理論密度を計算できません。')
			elif not comp_dbg:
				st.warning('組成が空です。プリセットか有効な JSON を入力してください。')
			else:
				base_theo = None
				try:
					base_theo = compute_theoretical_density(comp_dbg, float(ucv_val), float(z))
				except Exception as e:
					st.error(f'理論密度計算で例外が発生しました: {e}')
				if base_theo is not None:
					theo = base_theo
		# compute relative if possible
		rel = compute_relative_density(meas, theo)
		# write back to session_state and also update the visible numeric input for relative_density_pct
		st.session_state['computed_measured_density'] = meas
		st.session_state['computed_theoretical_density'] = theo
		st.session_state['theoretical_density_g_cm3'] = theo
		st.session_state['computed_relative_density'] = rel
		# also set display keys so form number_inputs reflect computed values immediately
		st.session_state['measured_density_display_value'] = meas
		st.session_state['theoretical_density_display_value'] = theo
		st.session_state['relative_density_display_value'] = rel
		# show detailed debug info in an expander for inspection
		with st.expander('密度計算デバッグ (中間値)', expanded=True):
			try:
				dbg = compute_theoretical_density_debug(comp_dbg, float(ucv_val), float(z)) if ucv_val else {'composition': comp_dbg}
				st.write({'pellet_mass_g': pmass, 'pellet_thickness_mm': th, 'pellet_diameter_mm': dia, 'measured_density_calc': meas, 'manual_theo_input': manual_theo, 'z_per_cell': z, 'unit_cell_vol': ucv_val, 'parsed_composition': comp_dbg, 'theoretical_debug': dbg})
			except Exception as e:
				st.write('デバッグ情報の取得に失敗しました:', e)
		# update the numeric input field value visible in the form (relative_density_pct has no explicit key yet)
		try:
			st.session_state['relative_density_pct'] = float(rel) if rel is not None else st.session_state.get('relative_density_pct', 0.0)
		except Exception:
			pass
		# final user message
		if meas is None:
			st.error('実測密度が計算できませんでした。ペレットの質量/厚さ/直径が正しく入力されているか確認してください。')
		if theo is None:
			st.error('理論密度が計算できませんでした。組成と格子体積を確認してください（または手動で理論密度を入力してください）。')
		if rel is not None:
			st.success(f"計算結果 — 実測密度: {meas:.6f} g/cm^3, 理論密度: {theo:.6f} g/cm^3, 相対密度: {rel:.3f} %")
		else:
			st.info('相対密度は計算されませんでした（実測密度か理論密度が不足しています）。')
		# rerun so the form widget bound to 'relative_density_pct' refreshes and shows updated value
		st.rerun()
	except Exception as e:
		st.error(f'密度計算エラー: {e}')

# auto-reflect of computed values is active; display keys are set by compute handlers

# Density readouts panel (outside form so they update immediately after compute)
st.markdown("---")
st.subheader("Density Readouts")
col_dr1, col_dr2, col_dr3 = st.columns(3)
with col_dr1:
	md = st.session_state.get('computed_measured_density')
	st.metric(label='実測密度 (g/cm^3)', value=f"{safe_float(md):.6f}" if md is not None else "N/A")
with col_dr2:
	td = st.session_state.get('computed_theoretical_density')
	st.metric(label='理論密度 (g/cm^3)', value=f"{safe_float(td):.6f}" if td is not None else "N/A")
with col_dr3:
	rd = st.session_state.get('computed_relative_density')
	st.metric(label='相対密度 (%)', value=f"{safe_float(rd):.3f}" if rd is not None else "N/A")

with st.expander('Density Debug (詳細中間値)', expanded=False):
	try:
		ucv_dbg = float(st.session_state.get('unit_cell_vol') or 0)
		comp_dbg = {}
		try:
			comp_dbg = json.loads(st.session_state.get('composition_text') or '{}')
		except Exception:
			comp_dbg = {}
		z_dbg = float(st.session_state.get('z_per_cell') or 1.0)
		dbg = compute_theoretical_density_debug(comp_dbg, ucv_dbg, z_dbg) if ucv_dbg else {'composition': comp_dbg}
		measured_vol = None
		pmass = st.session_state.get('pellet_mass_g')
		pth = st.session_state.get('pellet_thickness_mm')
		pdia = st.session_state.get('pellet_diameter_mm')
		try:
			if pmass and pth and pdia:
				th_cm = float(pth) / 10.0
				d_cm = float(pdia) / 10.0
				measured_vol = math.pi * (d_cm / 2.0) ** 2 * th_cm
		except Exception:
			measured_vol = None
		# Display comprehensive debug info
		st.json({
			'composition': dbg.get('composition'),
			'mass_per_formula': dbg.get('mass_per_formula'),
			'v_cm3': dbg.get('v_cm3'),
			'NA': dbg.get('NA'),
			'z_per_cell': z_dbg,
			'base_per_formula': dbg.get('base_per_formula'),
			'final_density': dbg.get('final_density'),
			'measured_volume_cm3': measured_vol,
			'measured_mass_g': pmass
		})
	except Exception as e:
		st.write('デバッグ表示エラー:', e)

	# Handle preset save/delete
	if save_preset_btn:
		try:
			parsed = json.loads(comp_text)
			# if oxygen not provided, assume O=3.0 for perovskite-like compositions (Ba-based)
			if parsed and 'O' not in parsed and ('Ba' in parsed or 'A' in parsed):
				try:
					parsed['O'] = 3.0
				except Exception:
					pass
			if not new_preset_name.strip():
				st.warning("プリセット名を入力してください。")
			else:
				st.session_state['presets'][new_preset_name.strip()] = parsed
				save_presets(st.session_state['presets'])
				st.success(f"プリセット '{new_preset_name}' を保存しました")
		except Exception as e:
			st.error(f"JSON のパースに失敗しました: {e}")

	if delete_preset_btn:
		if chosen and chosen != "(none)":
			st.session_state['presets'].pop(chosen, None)
			save_presets(st.session_state['presets'])
			st.success(f"プリセット '{chosen}' を削除しました")
		else:
			st.warning("削除するプリセットを選んでください。")

	if submitted:
		# parse composition
		try:
			composition = json.loads(comp_text) if comp_text.strip() else {}
		except Exception:
			st.error("組成 JSON のパースに失敗しました。正しい JSON を入力してください。")
			composition = {}

		# build sample dict and normalize composition for MI-ready features
		comp_norm = normalize_composition(composition)
		elem_nums = element_numeric_fields(comp_norm)

		# assemble the sample dict (clean indentation and avoid duplicate keys)
		sample = {
			"sample_no": sample_no,
			"composition": composition,
			"comp_normalized": comp_norm,
			"element_numeric": elem_nums,
			"crystal_system": crystal_system,
			"a": float_or_none(a_val),
			"a_err": float_or_none(a_err),
			"b": float_or_none(b_val),
			"b_err": float_or_none(b_err),
			"c": float_or_none(c_val),
			"c_err": float_or_none(c_err),
			"unit_cell_volume": float_or_none(unit_cell_vol),
			"unit_cell_volume_err": float_or_none(unit_cell_vol_err),
			"synthesis_method": synthesis_method,
			"calcination_temp_c": float(calcination_temp_c),
			"calcination_time_h": float(calcination_time_h),
			"relative_density_pct": float(relative_density_pct),
			"atmosphere": atmosphere,
			"thickness_mm": float(thickness_mm),
			"electrode_diameter_mm": float(electrode_diameter_mm),
			"resistances": {},
			"meta": {"saved_at": datetime.utcnow().isoformat()},
		}

		# compute densities using current form values (avoid relying on session_state at submit time)
		theoretical_density = None
		try:
			manual_theo = safe_float(theoretical_density_input or 0.0)
			if manual_theo and manual_theo > 0:
				theoretical_density = manual_theo
			else:
				ucv = safe_float(unit_cell_vol or 0.0)
				if ucv and ucv > 0:
					z_val = safe_float(z_per_cell or 1.0)
					base = compute_theoretical_density(sample.get('composition', {}), float(ucv), float(z_val))
					if base is not None:
						theoretical_density = base
		except Exception:
			theoretical_density = None

		measured_density = None
		try:
			pmass = safe_float(pellet_mass_g or 0.0)
			pth = safe_float(pellet_thickness_mm or 0.0)
			pdia = safe_float(pellet_diameter_mm or 0.0)
			measured_density = compute_measured_density(pmass, pth, pdia)
		except Exception:
			measured_density = None

		# assign computed values into the sample dict (so they are saved per-sample)
		# ensure computed values are stored in the sample (or removed if None)
		if theoretical_density is not None:
			sample['theoretical_density_g_cm3'] = theoretical_density
		else:
			sample.pop('theoretical_density_g_cm3', None)
		if measured_density is not None:
			sample['measured_density_g_cm3'] = measured_density
		else:
			sample.pop('measured_density_g_cm3', None)
		# If we can compute both theoretical and measured densities, overwrite relative_density_pct
		rel_pct = compute_relative_density(measured_density, theoretical_density)
		if rel_pct is not None:
			sample['relative_density_pct'] = rel_pct

		# attach resistances and numeric element fields
		for T, v in resistance_inputs.items():
			if v and str(v).strip():
				try:
					sample['resistances'][T] = {'resistance_ohm': float(v)}
				except Exception:
					st.warning(f"R @ {T}°C の値が数値ではありません: {v}")

		try:
			for k, v in elem_nums.items():
				sample[k] = v
		except Exception:
			pass

		# persist pellet & geometry values (use form variables directly as they are already captured at submit time)
		try:
			sample['pellet_mass_g'] = float(pellet_mass_g)
			sample['thickness_mm'] = float(thickness_mm)
			sample['electrode_diameter_mm'] = float(electrode_diameter_mm)
			sample['pellet_thickness_mm'] = float(pellet_thickness_mm)
			sample['pellet_diameter_mm'] = float(pellet_diameter_mm)
			sample['z_per_cell'] = float(z_per_cell)
		except Exception:
			pass
		# --- measured_density_g_cm3: always reflect form value ---
		try:
			sample['measured_density_g_cm3'] = float(measured_density_form)
		except Exception:
			sample['measured_density_g_cm3'] = measured_density_form

		# preserve id if editing an existing sample
		if st.session_state.get('editing_sample') and st.session_state['editing_sample'].get('id'):
			sample['id'] = st.session_state['editing_sample']['id']
			# also preserve original created_at if present
			try:
				orig_created = st.session_state['editing_sample'].get('created_at')
				if orig_created:
					sample['created_at'] = orig_created
			except Exception:
				pass

		sid = save_sample(sample)
		st.success(f"保存しました: id={sid}")
		# clear editing state and reset some form-backed session_state keys to defaults
		st.session_state['editing_sample'] = None
		st.session_state.pop('editing_loaded_id', None)
		# reset common form session keys to sensible defaults so next new form is clean
		# helper to set session_state keys safely (some contexts can raise StreamlitAPIException)
		def _safe_set_session(key, val):
			try:
				st.session_state[key] = val
			except Exception as _ex:
				# log to UI (non-fatal) so deploy logs capture the issue without crashing the app
				try:
					st.warning(f"session_state set failed for {key}: {_ex}")
				except Exception:
					# last resort: print to stdout (Streamlit will capture logs)
					print(f"session_state set failed for {key}: {_ex}")

		for k, v in {
			'composition_text': '',
			'unit_cell_vol': '',
			'unit_cell_vol_err': '',
			'pellet_mass_g': 0.0,
			'pellet_thickness_mm': 1.0,
			'pellet_diameter_mm': 10.0,
			'z_per_cell': 1.0,
			'theoretical_density_input': 0.0,
			'measured_density_display_value': None,
			'theoretical_density_display_value': None,
			'relative_density_display_value': None,
			'relative_density_pct': 95.0,
			'synthesis_method': '',
			'calcination_temp_c': 1200,
			'calcination_time_h': 10.0,
			'atmosphere': 'Air'
		}.items():
			_safe_set_session(k, v)
		# refresh the app so the new sample appears in the list immediately
		st.rerun()

# Samples list / search
st.subheader("Samples List")
samples = load_all_samples()
if not samples:
	st.info("サンプルがまだありません。フォームから追加してください。")

# Build full dataframe of samples
full_rows = []
for s in samples:
	row = {
		"id": s.get("id"),
		"sample_no": s.get("sample_no", ""),
		"created_at": s.get("created_at", ""),
		"composition": json.dumps(s.get("composition", {}), ensure_ascii=False),
		"comp_normalized": json.dumps(s.get("comp_normalized", {}), ensure_ascii=False),
		"element_numeric": json.dumps(s.get("element_numeric", {}), ensure_ascii=False),
		"crystal_system": s.get("crystal_system"),
		"a": s.get("a"),
		"a_err": s.get("a_err"),
		"b": s.get("b"),
		"b_err": s.get("b_err"),
		"c": s.get("c"),
		"c_err": s.get("c_err"),
		"unit_cell_volume": s.get("unit_cell_volume"),
		"unit_cell_volume_err": s.get("unit_cell_volume_err"),
		"thickness_mm": s.get("thickness_mm"),
		"electrode_diameter_mm": s.get("electrode_diameter_mm"),
		"pellet_thickness_mm": s.get("pellet_thickness_mm"),
		"pellet_diameter_mm": s.get("pellet_diameter_mm"),
		"pellet_mass_g": s.get("pellet_mass_g"),
		"z_per_cell": s.get("z_per_cell"),
		"measured_density_g_cm3": s.get("measured_density_g_cm3"),
		"theoretical_density_g_cm3": s.get("theoretical_density_g_cm3"),
		"resistances": json.dumps(s.get("resistances", {}), ensure_ascii=False),
	}
	# add dynamic R_<T> columns
	try:
		import re
		for k, v in (s.get('resistances') or {}).items():
			# extract numeric temperature from key if needed
			m = re.search(r"(\d{3})", str(k))
			if m:
				Tkey = m.group(1)
			else:
				Tkey = str(k)
			val = None
			if isinstance(v, dict) and 'resistance_ohm' in v:
				val = v.get('resistance_ohm')
			else:
				try:
					val = float(v)
				except Exception:
					val = None
			row[f"R_{Tkey}"] = val
	except Exception:
		pass

	# other metadata
	row["synthesis_method"] = s.get("synthesis_method")
	row["calcination_temp_c"] = s.get("calcination_temp_c")
	row["calcination_time_h"] = s.get("calcination_time_h")
	row["relative_density_pct"] = s.get("relative_density_pct")
	row["atmosphere"] = s.get("atmosphere")
	# include element numeric top-level if present
	for e in ['ba','zr','ce','y','yb','zn','ni','co','fe']:
		row[e] = s.get(e)
	full_rows.append(row)

df = pd.DataFrame(full_rows)

# Field selection for retrieval/export
available_fields = list(df.columns) if not df.empty else ["id","sample_no","created_at","composition"]
# Default to all available fields
default_fields = available_fields.copy()
chosen_fields = st.multiselect("取り出したいフィールドを選択", options=available_fields, default=default_fields)


# Bulk-select and delete flow
selected_for_delete = st.multiselect("削除するサンプルを選択", options=(df['id'].tolist() if not df.empty else []))
if st.button("選択削除"):
	if not selected_for_delete:
		st.warning("削除するサンプルを選んでください。")
	else:
		# show confirmation UI
		st.session_state['confirm_delete'] = selected_for_delete

# render confirmation UI if any
render_delete_confirmation()

colA, colB = st.columns([3, 1])
with colA:
	q = st.text_input("検索 (試料No や組成)  — フィルタ" )
	if q:
		df = df[df.apply(lambda r: q.lower() in str(r.sample_no).lower() or q.lower() in str(r.composition).lower(), axis=1)]
	st.dataframe(df, height=300)

	# Inline per-row edit/delete controls
	st.markdown("---")
	st.markdown("### サンプル一覧（編集）")
	if not df.empty:
		# show a compact list with edit/delete buttons per row
		for _, row in df.iterrows():
			rcols = st.columns([3, 2, 3, 1, 1])
			try:
				title = row.get('sample_no') or row.get('id')
			except Exception:
				title = ''
			rcols[0].markdown(f"**{title}**")
			rcols[1].write(str(row.get('created_at', '')))
			# show short composition
			comp_short = (row.get('composition')[:140] + '...') if isinstance(row.get('composition'), str) and len(row.get('composition')) > 140 else row.get('composition')
			rcols[2].write(str(comp_short))
			edit_key = f"inline_edit_{row.get('id')}"
			del_key = f"inline_del_{row.get('id')}"
			if rcols[3].button("編集", key=edit_key):
				# find full sample object and load into form for editing
				s = next((x for x in samples if x.get('id') == row.get('id')), None)
				if s:
					# ensure form reloads fresh values
					st.session_state.pop('editing_loaded_id', None)
					st.session_state['editing_sample'] = s
					st.rerun()
			if rcols[4].button("削除", key=del_key):
				# request deletion via pending flow
				st.session_state['pending_delete'] = [row.get('id')]
				st.rerun()

	# --- File import section ---
	st.markdown("---")
	st.header("ファイルからインポート")
	uploaded = st.file_uploader("CSV/TSV/XLSX/JSON を選択", type=["csv","tsv","xlsx","json"], accept_multiple_files=False)
	if uploaded:
		# try to parse
		try:
			fname = uploaded.name
			if fname.endswith('.csv') or fname.endswith('.tsv'):
				sep = ',' if fname.endswith('.csv') else '\t'
				df_up = pd.read_csv(uploaded, sep=sep)
			elif fname.endswith('.xlsx'):
				df_up = pd.read_excel(uploaded)
			elif fname.endswith('.json'):
				df_up = pd.read_json(uploaded)
			else:
				st.error("サポートされていないファイル形式です")
				df_up = None
		except Exception as e:
			st.error(f"ファイルの読み込みに失敗しました: {e}")
			df_up = None

		if df_up is not None:
			st.subheader("プレビュー（最初の10行）")
			st.dataframe(df_up.head(10))

			# auto-detect columns
			cols = df_up.columns.tolist()
			st.markdown("自動検出されたカラム: " + ", ".join(cols))

			# Auto-map toggle: when enabled, we skip manual mapping UI and import all detected columns automatically
			auto_map_all = st.checkbox("自動マッピングで全カラムを取り込む（推奨）", value=True)

			# Manual mapping UI (optional)
			if not auto_map_all:
				st.markdown("**マッピング（適切なカラムを選んでください）**")
				map_sample_no = st.selectbox("試料No カラム", options=["(なし)"] + cols, index=0)
				map_composition = st.selectbox("組成（JSONまたは個別元素カラム）", options=["(なし)"] + cols, index=0)
				map_thickness = st.selectbox("厚さ (mm)", options=["(なし)"] + cols, index=0)
				map_electrode = st.selectbox("電極直径 (mm)", options=["(なし)"] + cols, index=0)
				st.markdown("抵抗（R_600 など温度付きカラム）がある場合は以下で選択してください（複数選択可）")
				resistance_cols = st.multiselect("抵抗カラム", options=cols)
			else:
				# placeholders for manual vars so later code can reference them safely
				map_sample_no = '(なし)'
				map_composition = '(なし)'
				map_thickness = '(なし)'
				map_electrode = '(なし)'
				resistance_cols = []

			if st.button("インポート実行"):
				imported = 0
				# flag to show one-time warning if O was auto-added for Ba/A-site perovskite rule
				o_auto_added_any = False
				errors = []
				# quick debug: show number of rows and mapping choices
				try:
					mode = '自動マッピング' if auto_map_all else '手動マッピング'
					st.info(f"インポート開始: {len(df_up)} 行（モード: {mode}）")
				except Exception:
					pass

				import re
				norm_cols = {c.lower().replace(' ','').replace('_',''): c for c in df_up.columns}
				def find_col(candidates):
					for cand in candidates:
						key = cand.lower().replace(' ','').replace('_','')
						if key in norm_cols:
							return norm_cols[key]
					return None

				# --- 強化: 主要フィールドの自動マッピング候補リスト ---
				field_map_candidates = {
					'crystal_system': ['crystalsystem','crystal_system','結晶系'],
					'a': ['a','lattice_a','a(å)','a(ang)','格子定数a'],
					'a_err': ['a_err','aerror','a±','a誤差'],
					'b': ['b','lattice_b','b(å)','b(ang)','格子定数b'],
					'b_err': ['b_err','berror','b±','b誤差'],
					'c': ['c','lattice_c','c(å)','c(ang)','格子定数c'],
					'c_err': ['c_err','cerror','c±','c誤差'],
					'unit_cell_volume': ['unit_cell_volume','ucv','v(å^3)','cellvolume','格子体積'],
					'unit_cell_volume_err': ['unit_cell_volume_err','ucv_err','v_err','cellvolume_err','格子体積誤差'],
					'theoretical_density_g_cm3': ['theoretical_density_g_cm3','theoreticaldensity','理論密度'],
					'measured_density_g_cm3': ['measured_density_g_cm3','measureddensity','実測密度'],
					'relative_density_pct': ['relative_density_pct','relativedensity','相対密度'],
					'atmosphere': ['atmosphere','環境','雰囲気'],
					'calcination_temp_c': ['calcination_temp_c','calcinationtemp','焼成温度'],
					'calcination_time_h': ['calcination_time_h','calcinationtime','焼成時間'],
					'pellet_mass_g': ['pellet_mass_g','massg','mass','質量','ペレット質量'],
					'pellet_thickness_mm': ['pellet_thickness_mm','thickness_mm','厚さ','ペレット厚さ','ペレット厚み'],
					'pellet_diameter_mm': ['pellet_diameter_mm','electrode_diameter_mm','ペレット直径','ペレット径','電極直径'],
					'z_per_cell': ['z_per_cell','zpercell','z','式数'],
					# エレメント
					'ba': ['ba','Ba','Ba2+'],
					'zr': ['zr','Zr'],
					'ce': ['ce','Ce'],
					'y': ['y','Y'],
					'yb': ['yb','Yb'],
					'zn': ['zn','Zn'],
					'ni': ['ni','Ni'],
					'co': ['co','Co'],
					'fe': ['fe','Fe'],
				}
				for idx, row in df_up.iterrows():
					try:
						sample = {}
						# sample_no
						if not auto_map_all and map_sample_no != '(なし)':
							sample['sample_no'] = str(row[map_sample_no])
						else:
							# try common names: sample_no, sample, 試料No, 試料番号
							c = find_col(['sample_no','sample','sampleno','試料no','試料番号','サンプルno'])
							sample['sample_no'] = str(row[c]) if c else f"import_{idx}"
						# composition
						comp = {}
						if (not auto_map_all and map_composition != '(なし)'):
							val = row[map_composition]
							# try JSON parse
							# robust parsing: accept dict, JSON string, key:val pairs, or condensed element+number strings
							def parse_comp_value(v):
								if v is None:
									return {}
								if isinstance(v, dict):
									return v
								if isinstance(v, (int, float)):
									return {}
								if isinstance(v, str):
									s = v.strip()
									# try JSON
									try:
										if s.startswith('{') or s.startswith('['):
											return json.loads(s)
									except Exception:
										pass
									# try key:value pairs like 'Ba:1, Zr:0.4' or 'Ba=1 Zr=0.4'
									import re
									pairs = re.findall(r'([A-Za-z]{1,2})\s*[:=]\s*([0-9]+\.?[0-9]*)', s)
									if pairs:
										return {k: float(v) for k, v in pairs}
									# try condensed like Ba1Zr0.4Ce0.4Y0.2
									pairs2 = re.findall(r'([A-Z][a-z]?)([0-9]*\.?[0-9]+)', s)
									if pairs2:
										return {k: float(v) for k, v in pairs2}
									# try symbol with integer default (e.g., 'Ba Zr Ce')
									symbols = re.findall(r'([A-Z][a-z]?)', s)
									if symbols:
										return {k: 1.0 for k in symbols}
								return {}
							comp = parse_comp_value(val)
						else:
							# auto: prefer a composition/組成 column; else gather element columns case-insensitively
							cand = find_col(['composition','組成'])
							if cand:
								val = row[cand]
								def parse_comp_value(v):
									if v is None:
										return {}
									if isinstance(v, dict):
										return v
									if isinstance(v, (int, float)):
										return {}
									if isinstance(v, str):
										s = v.strip()
										# try JSON
										try:
											if s.startswith('{') or s.startswith('['):
												return json.loads(s)
										except Exception:
											pass
										# try key:value pairs
										pairs = re.findall(r'([A-Za-z]{1,2})\s*[:=]\s*([0-9]+\.?[0-9]*)', s)
										if pairs:
											return {k: float(v) for k, v in pairs}
										# try condensed like Ba1Zr0.4...
										pairs2 = re.findall(r'([A-Z][a-z]?)([0-9]*\.?[0-9]+)', s)
										if pairs2:
											return {k: float(v) for k, v in pairs2}
										# symbols space-separated
										symbols = re.findall(r'([A-Z][a-z]?)', s)
										if symbols:
											return {k: 1.0 for k in symbols}
									return {}
								comp = parse_comp_value(val)
							else:
								cols_lower = {c.lower(): c for c in df_up.columns}
								for e in ['Ba','Zr','Ce','Y','Yb','Zn','Ni','Co','Fe','O']:
									colname = cols_lower.get(e.lower())
									if colname:
										try:
											v = row[colname]
											if pd.isna(v):
												continue
											comp[e] = float(v)
										except Exception:
											pass
						# Perovskite rule: if Ba or A present and O missing, add O=3.0 to allow correct theoretical density calc
						try:
							if isinstance(comp, dict) and ('O' not in comp) and (('Ba' in comp) or ('A' in comp)):
								comp['O'] = 3.0
								o_auto_added_any = True
						except Exception:
							pass
						sample['composition'] = comp
						sample['comp_normalized'] = normalize_composition(comp)
						sample['element_numeric'] = element_numeric_fields(sample['comp_normalized'])


						# --- 強化: 主要フィールドの自動マッピング ---
						try:
							for field, candidates in field_map_candidates.items():
								col = find_col(candidates)
								if col is not None and not pd.isna(row[col]):
									try:
										sample[field] = float(row[col]) if isinstance(row[col], (int, float, str)) and str(row[col]).replace('.','',1).isdigit() else row[col]
									except Exception:
										sample[field] = row[col]
						except Exception:
							pass

						# resistances: accept explicit selection (manual mode), any R_### columns, numeric-temp column names, or a JSON 'resistances' cell
						resistances = {}
						# start with user-selected columns (manual) else none
						res_cols = set(resistance_cols or [])
						# auto-detect R_ prefixed or numeric temp headers
						for c in df_up.columns:
							if re.search(r"(^R_\d{3}\b)|(^\d{3}$)|\bR_?\d{3}\b", str(c)):
								res_cols.add(c)
						# if there's a 'resistances' column which may contain JSON, prefer to parse it
						if 'resistances' in df_up.columns:
							try:
								cell = row['resistances']
								if cell and not (pd.isna(cell)):
									# if it's already a dict
									if isinstance(cell, dict):
										for k, v in cell.items():
											try:
												if isinstance(v, dict) and 'resistance_ohm' in v:
													resistances[str(k)] = {'resistance_ohm': float(v['resistance_ohm'])}
												else:
													resistances[str(k)] = {'resistance_ohm': float(v)}
											except Exception:
												continue
									# if it's a JSON string
									elif isinstance(cell, str):
										try:
											j = json.loads(cell)
											if isinstance(j, dict):
												for k, v in j.items():
														try:
															if isinstance(v, dict) and 'resistance_ohm' in v:
																resistances[str(k)] = {'resistance_ohm': float(v['resistance_ohm'])}
															else:
																resistances[str(k)] = {'resistance_ohm': float(v)}
														except Exception:
															continue
										except Exception:
											pass
							except Exception:
								pass
						# parse other detected columns
						for rc in sorted(res_cols):
							try:
								val = row[rc]
								if pd.isna(val) or val in (None, ''):
									continue
								m = re.search(r"(\d{3})", str(rc))
								if m:
									T = m.group(1)
								else:
									T = str(rc)
								try:
									resistances[str(T)] = {'resistance_ohm': float(val)}
								except Exception:
									# if val is a dict-like string, try to parse numeric inside
									try:
										vj = json.loads(val) if isinstance(val, str) else None
										if isinstance(vj, dict) and 'resistance_ohm' in vj:
											resistances[str(T)] = {'resistance_ohm': float(vj['resistance_ohm'])}
									except Exception:
										pass
							except Exception:
								continue
						sample['resistances'] = resistances


						# auto-map synonyms if explicit columns absent
						try:
							# pellet_mass_g
							if 'pellet_mass_g' not in sample:
								cols_norm = {c.lower().replace(' ','').replace('_',''): c for c in df_up.columns}
								for cand in ['pelletmassg','massg','mass','質量','ペレット質量']:
									if cand in cols_norm:
										cname = cols_norm[cand]
										try:
											sample['pellet_mass_g'] = float(row[cname])
										except Exception:
											sample['pellet_mass_g'] = row.get(cname)
										break
							# z_per_cell
							if 'z_per_cell' not in sample:
								for cand in ['zpercell','z','式数']:
									if cand in cols_norm:
										cname = cols_norm[cand]
										try:
											sample['z_per_cell'] = float(row[cname])
										except Exception:
											sample['z_per_cell'] = row.get(cname)
										break
						except Exception:
							pass

						# z per cell
						if 'z_per_cell' in df_up.columns:
							try:
								sample['z_per_cell'] = float(row['z_per_cell'])
							except Exception:
								sample['z_per_cell'] = row.get('z_per_cell')

						# element columns: support both capitalized symbols and lowercase names
						for el in ['Ba','Zr','Ce','Y','Yb','Zn','Ni','Co','Fe','O','Ba2+','A']:
							if el in df_up.columns:
								try:
									sample[el] = float(row[el])
								except Exception:
									sample[el] = row.get(el)
						# also lowercase keys
						cols_lower = {c.lower(): c for c in df_up.columns}
						for el in ['ba','zr','ce','y','yb','zn','ni','co','fe','o']:
							if el in cols_lower:
								c = cols_lower[el]
								try:
									sample[el] = float(row[c])
								except Exception:
									sample[el] = row.get(c)

						# created_at override if provided (accept string values)
						if 'created_at' in df_up.columns:
							try:
								sample['created_at'] = str(row['created_at'])
							except Exception:
								pass

						# fallback: ensure synthesis_method exists
						sample['synthesis_method'] = sample.get('synthesis_method') or (str(row.get('synthesis_method','')) if 'synthesis_method' in df_up.columns else '')
						# ensure metadata and created_at
						sample['meta'] = sample.get('meta', {})
						sample['meta']['imported_from'] = fname
						if 'created_at' not in sample or not sample.get('created_at'):
							sample['created_at'] = datetime.utcnow().isoformat()
						
						# Auto-calculate densities if fields present (but not already in row)
						try:
							# Theoretical density: if composition + unit_cell_volume + z exist and not already imported
							if 'theoretical_density_g_cm3' not in sample or sample.get('theoretical_density_g_cm3') is None:
								comp_auto = sample.get('composition') or {}
								ucv_auto = sample.get('unit_cell_volume')
								z_auto = sample.get('z_per_cell') or 1.0
								if comp_auto and ucv_auto:
									try:
										theo_calc = compute_theoretical_density(comp_auto, float(ucv_auto), float(z_auto))
										if theo_calc is not None:
											sample['theoretical_density_g_cm3'] = theo_calc
									except Exception:
										pass
							
							# Measured density: if pellet mass + thickness + diameter exist and not already imported
							if 'measured_density_g_cm3' not in sample or sample.get('measured_density_g_cm3') is None:
								pmass_auto = sample.get('pellet_mass_g')
								pth_auto = sample.get('pellet_thickness_mm') or sample.get('thickness_mm')
								pdia_auto = sample.get('pellet_diameter_mm') or sample.get('electrode_diameter_mm')
								if pmass_auto and pth_auto and pdia_auto:
									try:
										th_cm_auto = float(pth_auto) / 10.0
										d_cm_auto = float(pdia_auto) / 10.0
										vol_cm3_auto = math.pi * (d_cm_auto / 2.0) ** 2 * th_cm_auto
										if vol_cm3_auto > 0:
											meas_calc = float(pmass_auto) / vol_cm3_auto
											sample['measured_density_g_cm3'] = meas_calc
									except Exception:
										pass
							
							# Relative density: if both theoretical and measured exist and not already imported
							if 'relative_density_pct' not in sample or sample.get('relative_density_pct') is None:
								theo_f = sample.get('theoretical_density_g_cm3')
								meas_f = sample.get('measured_density_g_cm3')
								if theo_f and meas_f and float(theo_f) > 0:
									try:
										rel_calc = (float(meas_f) / float(theo_f)) * 100.0
										sample['relative_density_pct'] = rel_calc
									except Exception:
										pass
						except Exception:
							pass
						
						# save with fallback: prefer configured storage, but if it fails, save local
						saved = False
						try:
							# try configured storage
							save_sample(sample)
							saved = True
						except Exception as e:
							# try local fallback and surface the original error
							try:
								save_sample_local(sample)
								saved = True
								st.warning(f"保存方法に問題があったためローカルに保存しました: {e}")
							except Exception as e2:
								errors.append(f"行 {idx}: supabase error: {e}; local fallback error: {e2}")
						# increment only when saved
						if saved:
							imported += 1
					except Exception as e:
						errors.append(f"行 {idx}: {e}")
				st.success(f"インポート完了: {imported} 件")
				if errors:
					st.error(f"一部エラーが発生しました: {errors[:5]}")
				if o_auto_added_any:
					st.warning("注意: Ba または A が検出され、酸素が欠落していたため自動的に O=3.0 を追加しました。必要に応じて編集してください。")
				st.rerun()
with colB:
	# allow multiple selection of samples; show friendly labels (sample_no + short id)
	option_labels = []
	display_to_id = {}
	for idx, r in df.iterrows() if not df.empty else []:
		sid = r.get('id')
		label = f"{r.get('sample_no','')} — {str(sid)[:8]}"
		# ensure uniqueness
		if label in display_to_id:
			label = f"{r.get('sample_no','')} — {str(sid)}"
		display_to_id[label] = sid
		option_labels.append(label)

	selected_displays = st.multiselect("サンプル選択", options=option_labels)
	selected_ids = [display_to_id[d] for d in selected_displays]

	# Button: recompute densities for selected samples and save
	if st.button("選択サンプルの密度を再計算して保存"):
		if not selected_ids:
			st.warning("再計算するサンプルを選択してください。")
		else:
			updated = 0
			for sid in selected_ids:
				s = next((x for x in samples if x.get('id') == sid), None)
				if not s:
					continue
				# theoretical density from composition + unit cell vol and Z
				comp = s.get('composition') or {}
				ucv = s.get('unit_cell_volume')
				try:
					ucv_val = float(ucv) if ucv is not None else None
				except Exception:
					ucv_val = None
				base = None
				if comp and ucv_val:
					z = safe_float(s.get('z_per_cell') or 1.0)
					base = compute_theoretical_density(comp, ucv_val, float(z))
					theo = base if base is not None else None
				# measured density from pellet mass/thickness/diameter
				pmass = safe_float(s.get('pellet_mass_g') or 0.0)
				pth = safe_float(s.get('pellet_thickness_mm') or s.get('thickness_mm') or 0.0)
				pdia = safe_float(s.get('pellet_diameter_mm') or s.get('electrode_diameter_mm') or 0.0)
				meas = None
				try:
					if pmass and pth and pdia:
						th_cm = pth / 10.0
						d_cm = pdia / 10.0
						vol_cm3 = math.pi * (d_cm / 2.0) ** 2 * th_cm
						if vol_cm3 > 0:
							meas = pmass / vol_cm3
				except Exception:
					meas = None
				rel = (meas / theo) * 100.0 if (meas and theo and theo > 0) else None
				# store into sample dict
				if theo is not None:
					s['theoretical_density_g_cm3'] = theo
				else:
					s.pop('theoretical_density_g_cm3', None)
				if meas is not None:
					s['measured_density_g_cm3'] = meas
				else:
					s.pop('measured_density_g_cm3', None)
				if rel is not None:
					s['relative_density_pct'] = rel
				else:
					s.pop('relative_density_pct', None)
				# save
				save_sample(s)
				updated += 1
			st.success(f"再計算して保存しました: {updated} 件更新")
			st.rerun()

	# Export selected/all with chosen fields
	export_target = st.selectbox("エクスポート対象", options=["選択サンプル", "全サンプル"], index=0)
	if st.button("エクスポート（CSV）"):
		if export_target == "選択サンプル":
			ids = selected_ids if selected_ids else []
		else:
			ids = df['id'].tolist() if not df.empty else []
		rows_out = [r for r in full_rows if r.get('id') in ids] if ids else []
		if not rows_out and export_target == "選択サンプル":
			st.warning("エクスポートするサンプルが選択されていません。")
		elif not rows_out and export_target == "全サンプル":
			st.warning("エクスポート対象のデータがありません。")
		else:
				df_out = pd.DataFrame(rows_out)
				# defensive: ensure sample_no column present and always included in exported columns
				if 'sample_no' not in df_out.columns:
					# rows_out preserves original order; fill sample_no from rows_out dicts
					df_out.insert(0, 'sample_no', [r.get('sample_no', '') for r in rows_out])
				if chosen_fields:
					# ensure sample_no is included even if user didn't select it
					final_fields = list(chosen_fields)
					if 'sample_no' not in final_fields:
						final_fields.insert(0, 'sample_no')
					df_out = df_out[final_fields]
				csv = df_out.to_csv(index=False)
				st.download_button("ダウンロード CSV", csv, file_name="export_selected.csv", mime="text/csv")

	if st.button("エクスポート（XLSX）"):
		if export_target == "選択サンプル":
			ids = selected_ids if selected_ids else []
		else:
			ids = df['id'].tolist() if not df.empty else []
		rows_out = [r for r in full_rows if r.get('id') in ids] if ids else []
		if not rows_out and export_target == "選択サンプル":
			st.warning("エクスポートするサンプルが選択されていません。")
		elif not rows_out and export_target == "全サンプル":
			st.warning("エクスポート対象のデータがありません。")
		else:
				df_out = pd.DataFrame(rows_out)
				# defensive: ensure sample_no column present and always included in exported columns
				if 'sample_no' not in df_out.columns:
					df_out.insert(0, 'sample_no', [r.get('sample_no', '') for r in rows_out])
				if chosen_fields:
					final_fields = list(chosen_fields)
					if 'sample_no' not in final_fields:
						final_fields.insert(0, 'sample_no')
					df_out = df_out[final_fields]
				xlsx_bytes = excel_bytes_from_df(df_out)
				st.download_button("ダウンロード XLSX", xlsx_bytes, file_name="export_selected.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

	# show details for each selected sample with edit/delete controls
	if selected_ids:
		st.markdown("### 選択サンプル")
		for sid in selected_ids:
			s = next((x for x in samples if x.get("id") == sid), None)
			if not s:
				continue
			with st.expander(f"{s.get('sample_no','')} — {sid[:8]}"):
				st.json(s)
				cdel, cedit = st.columns([1,1])
				with cdel:
					if st.button("削除", key=f"del_{sid}"):
						delete_sample(sid)
						st.rerun()
				with cedit:
					if st.button("編集", key=f"edit_{sid}"):
						# ensure form reloads fresh values for this edit
						st.session_state.pop('editing_loaded_id', None)
						st.session_state['editing_sample'] = s
						st.rerun()

# Arrhenius plot and exports
st.subheader("Arrhenius Plot / Export")
# Allow plotting multiple selected samples together
if not selected_ids:
	st.info("左のパネルからサンプルを選択してください。複数選択可。")
else:
	# collect arrhenius points for each selected sample
	skipped = []
	combined_rows = []
	for sid in selected_ids:
		s = next((x for x in samples if x.get('id') == sid), None)
		if not s:
			continue
		# try primary geometry then fallback to pellet geometry
		thickness_mm = s.get('thickness_mm') or s.get('pellet_thickness_mm')
		electrode_diameter_mm = s.get('electrode_diameter_mm') or s.get('pellet_diameter_mm')
		resistances = s.get('resistances', {}) or {}
		rows = []
		reason = []
		if not thickness_mm or not electrode_diameter_mm:
			reason.append('geometry')
		if not resistances:
			reason.append('no R')
		for T_str, rinfo in resistances.items():
			try:
				import re
				# temperature key parsing (supports '600', 600, 'R_600', 'T600')
				if isinstance(T_str, (int, float)):
					T_c = float(T_str)
				else:
					m = re.search(r"(\d+(?:\.\d+)?)", str(T_str))
					if not m:
						raise ValueError(f"温度キーが数値に変換できません: {T_str}")
					T_c = float(m.group(1))
				# resistance value parsing (dict with 'resistance_ohm' or raw number)
				R = None
				if isinstance(rinfo, dict):
					R = rinfo.get('resistance_ohm') or rinfo.get('R') or rinfo.get('value')
				else:
					R = rinfo
				R = float(R)
				# geometry safeties
				th = float(thickness_mm)
				dia = float(electrode_diameter_mm)
				if th <= 0 or dia <= 0 or R <= 0:
					raise ValueError("幾何か抵抗値が無効です")
				sigma = calculate_sigma(th, R, dia)
				T_k = T_c + 273.15
				rows.append({'sample_id': sid, 'sample_no': s.get('sample_no'), 'T_c': T_c, 'T_k': T_k, 'sigma_S_cm': sigma})
			except Exception:
				continue
		if rows:
			# convert to arrhenius points for this sample
			arr_pts = make_arrhenius_points(rows)
			for p in arr_pts:
				combined_rows.append({'sample_id': sid, 'sample_no': s.get('sample_no'), '1000/T': p[0], 'ln(sigma*T)': p[1]})
		else:
			nm = s.get('sample_no') or (s.get('id') or '')[:8]
			skipped.append(f"{nm}({'+'.join(reason) if reason else 'calc failed'})")

	if not combined_rows:
		msg = "選択サンプルに導電率データがありません。フォームで抵抗値と幾何を入力してください。"
		if skipped:
			msg += f"（スキップ: {', '.join(skipped[:5])} など）"
		st.info(msg)
	else:
		arr_df = pd.DataFrame(combined_rows)
		st.dataframe(arr_df)
		# plot (one series per sample)
		try:
			import numpy as np
			import plotly.express as px
			import plotly.graph_objects as go
			fig = px.scatter(arr_df, x='1000/T', y='ln(sigma*T)', color='sample_no', symbol='sample_no', title='Arrhenius (combined)')
			# per-sample linear fit and Ea estimation
			sample_groups = arr_df.groupby('sample_no')
			fit_texts = []
			for name, grp in sample_groups:
				if len(grp) >= 2:
					x = grp['1000/T'].to_numpy()
					y = grp['ln(sigma*T)'].to_numpy()
					m, b = np.polyfit(x, y, 1)
					# add fit line trace
					x_lin = np.linspace(x.min(), x.max(), 100)
					y_lin = m * x_lin + b
					fig.add_trace(go.Scatter(x=x_lin, y=y_lin, mode='lines', name=f'{name} fit', line={'dash':'dash'}))
					# compute Ea (eV) using slope when x is 1000/T
					kB = 8.617333e-5
					Ea_eV = -m * kB * 1000.0
					fit_texts.append(f"{name}: Ea={Ea_eV:.4f} eV")
			if fit_texts:
				st.markdown("**活性化エネルギー推定:** " + ", ".join(fit_texts))
			st.plotly_chart(fig, use_container_width=True)
		except Exception:
			st.write('プロット用ライブラリ(plotly,numpy)が必要です。インストールしてください。')

		# export combined CSV/TXT/XLSX
		# export headerless two-column by default (1000/T, ln(sigma*T)) per-sample grouped
		include_header = st.checkbox('エクスポートにヘッダ行を含める（CSV/TXT）', value=False)
		# CSV (all rows, include sample_no for traceability)
		csv = arr_df.to_csv(index=False, header=include_header)
		st.download_button('ダウンロード CSV (Arrhenius 結果)', csv, file_name='arrhenius_combined.csv', mime='text/csv')
		# For plain 2-column headerless export required: produce a concatenation per sample with no header
		def two_col_blob(df_in, header=False):
			buf_lines = []
			if header:
				buf_lines.append('1000/T,ln(sigma*T)')
			for _, r in df_in[['1000/T','ln(sigma*T)']].iterrows():
				buf_lines.append(f"{r['1000/T']},{r['ln(sigma*T)']}")
			return '\n'.join(buf_lines)

		txt_blob = two_col_blob(arr_df, header=include_header)
		st.download_button('ダウンロード TXT (2-col, headerless)', txt_blob, file_name='arrhenius_combined.txt', mime='text/plain')
		try:
			xlsx_bytes = excel_bytes_from_df(arr_df, include_header=include_header)
			st.download_button('ダウンロード XLSX (Arrhenius)', xlsx_bytes, file_name='arrhenius_combined.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
		except Exception:
			st.info('Excel 出力には openpyxl が必要です。')

		# --- 新機能: 導電率(S/cm) の別枠エクスポート ---
		st.markdown('---')
		st.markdown('### 導電率 (S/cm) の個別エクスポート')
		# Build conductivity table per selected sample (two-column per sample: T, sigma)
		cond_rows = []
		for sid in selected_ids:
			s = next((x for x in samples if x.get('id') == sid), None)
			if not s:
				continue
			# fall back to pellet geometry if main geometry is missing
			thickness_mm = s.get('thickness_mm') or s.get('pellet_thickness_mm')
			electrode_diameter_mm = s.get('electrode_diameter_mm') or s.get('pellet_diameter_mm')
			resistances = s.get('resistances', {})
			for T_str, rinfo in resistances.items():
				try:
					T_c = float(T_str)
					R = float(rinfo.get('resistance_ohm')) if isinstance(rinfo, dict) else float(rinfo)
					sigma = None
					try:
						sigma = calculate_sigma(thickness_mm, R, electrode_diameter_mm)
					except Exception:
						sigma = None
					if sigma is not None:
						cond_rows.append({'sample_id': sid, 'sample_no': s.get('sample_no'), 'T_c': T_c, 'sigma_S_cm': sigma})
				except Exception:
					continue

		if cond_rows:
			cond_df = pd.DataFrame(cond_rows)
			# Provide per-sample separate downloads: CSV with header and plain 2-col TXT (temperature, conductivity S/cm)
			cond_csv = cond_df.to_csv(index=False, header=include_header)
			st.download_button('ダウンロード CSV (導電率 全選択サンプル)', cond_csv, file_name='conductivity_selected.csv', mime='text/csv')
			# two-col blob for conductivity (T, sigma)
			def cond_two_col_blob(df_in, header=False):
				lines = []
				if header:
					lines.append('T_C,sigma_S_cm')
				for _, r in df_in[['T_c','sigma_S_cm']].iterrows():
					lines.append(f"{r['T_c']},{r['sigma_S_cm']}")
				return '\n'.join(lines)
			cond_txt = cond_two_col_blob(cond_df, header=include_header)
			st.download_button('ダウンロード TXT (2-col: T, sigma_S_cm)', cond_txt, file_name='conductivity_selected.txt', mime='text/plain')
		else:
			st.info('選択したサンプルに導電率データが見つかりません。')

# Bulk export all samples as CSV
st.markdown("---")
if st.button("全サンプルを CSV にエクスポート"):
	# reuse df (full_rows DataFrame) which already contains all mapped fields and dynamic R_T columns
	df_all = df.copy() if 'df' in locals() else pd.DataFrame([])
	if df_all.empty:
		# build from load_all_samples as fallback
		all_samples = load_all_samples()
		rows = []
		for s in all_samples:
			base = {
				"id": s.get("id"),
				"sample_no": s.get("sample_no", ""),
				"created_at": s.get("created_at", ""),
				"composition": json.dumps(s.get("composition", {}), ensure_ascii=False),
				"comp_normalized": json.dumps(s.get("comp_normalized", {}), ensure_ascii=False),
				"element_numeric": json.dumps(s.get("element_numeric", {}), ensure_ascii=False),
				"crystal_system": s.get("crystal_system"),
				"a": s.get("a"),
				"a_err": s.get("a_err"),
				"b": s.get("b"),
				"b_err": s.get("b_err"),
				"c": s.get("c"),
				"c_err": s.get("c_err"),
				"unit_cell_volume": s.get("unit_cell_volume"),
				"unit_cell_volume_err": s.get("unit_cell_volume_err"),
				"thickness_mm": s.get("thickness_mm"),
				"electrode_diameter_mm": s.get("electrode_diameter_mm"),
				"pellet_thickness_mm": s.get("pellet_thickness_mm"),
				"pellet_diameter_mm": s.get("pellet_diameter_mm"),
				"pellet_mass_g": s.get("pellet_mass_g"),
				"z_per_cell": s.get("z_per_cell"),
				"measured_density_g_cm3": s.get("measured_density_g_cm3"),
				"theoretical_density_g_cm3": s.get("theoretical_density_g_cm3"),
				"relative_density_pct": s.get("relative_density_pct"),
				"synthesis_method": s.get("synthesis_method"),
				"calcination_temp_c": s.get("calcination_temp_c"),
				"calcination_time_h": s.get("calcination_time_h"),
				"atmosphere": s.get("atmosphere"),
			}
			# expand resistances to dynamic columns
			try:
				import re
				for k, v in (s.get('resistances') or {}).items():
					m = re.search(r"(\d{3})", str(k))
					Tkey = m.group(1) if m else str(k)
					val = v.get('resistance_ohm') if isinstance(v, dict) else v
					base[f"R_{Tkey}"] = val
			except Exception:
				pass
			# include element numeric fields if present at top-level
			for e in ['ba','zr','ce','y','yb','zn','ni','co','fe']:
				base[e] = s.get(e)
			rows.append(base)
		df_all = pd.DataFrame(rows)
	csv = df_all.to_csv(index=False)
	st.download_button("ダウンロード all_samples.csv", csv, file_name="all_samples.csv", mime="text/csv")

st.markdown("---")
st.caption("このアプリはローカル SQLite を使用します。Supabase でクラウド共有する場合は README の手順を参照してください。")

