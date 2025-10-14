# Materials DataBank (Streamlit prototype)

This is a minimal Streamlit prototype to manage materials samples, compute conductivity from resistance and thickness, generate Arrhenius data (1000/T vs ln(σT)), and export CSV/XLSX.

Requirements
- Python 3.10+
- Install dependencies (see `requirements.txt`)

Quick start (macOS / zsh):

```bash
cd /Users/ren/Desktop/\ DAta\ bankU
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Notes
- By default the app uses a local SQLite DB file `samples.db` in the same folder.
- For cloud sync and group sharing, use Supabase and replace the SQLite helpers with Supabase client calls. A SQL schema is provided in `supabase_schema.sql` (not included yet).

Supabase: create a table `samples` with a JSONB column and use Supabase client to upsert/query.
