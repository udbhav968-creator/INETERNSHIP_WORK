The files in this directory are NOT used by the engine, the database, model training, or any student-facing paper.

neet_unverified_augmented.json
  815 questions (question_id prefix NEET_AUG_, paper_source "NEET NCERT Benchmark") that were part of the
  question bank but have no verifiable official exam provenance. All 815 have correct_option = "a", so a
  student who always picks A would score 100% on them. They were quarantined on the
  fix/data-integrity-honest-metrics branch.

To restore any of these items they need: a documented source, an independently checked answer key, and a
passing `python -m model.data_integrity_gate` run.
