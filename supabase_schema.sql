-- Supabase/Postgres schema for samples table

CREATE TABLE IF NOT EXISTS samples (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_no text,
  data jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Index on sample_no for fast lookup
CREATE INDEX IF NOT EXISTS idx_samples_sample_no ON samples(sample_no);

-- Example: store composition in data->'composition'
-- data: {
--  "composition": {"Ba": 0.5, "Zr":0.5},
--  "thickness_mm": 1.0,
--  "electrode_diameter_mm": 5.0,
--  "resistances": {"600": {"resistance_ohm": 10}, ...}
-- }
