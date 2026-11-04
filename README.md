# Klassed NEET Adaptive Engine — Repository Documentation & Enterprise v2 ML Technical Report

**Author:** Udbhav Yadav (ML Engineer Intern)  
**Institution:** Bennett University  
**Repository:** udbhav968-creator/INETERNSHIP_WORK (Klassed NEET Adaptive Engine)  

---

## 1. Executive Summary & Enterprise v2 Vision

Building upon our successful v1 MVP—which combines a Random Forest question difficulty classifier achieving **99.60% accuracy** (`model/train_difficulty_model.py`, `model/question_difficulty_metrics.json`) with deterministic rolling-window mastery state machines (`backend/app/state_machine.py`) and dual-signal pace tracking (`backend/app/time_tracking.py`)—this document provides a complete architectural walkthrough of all files in the repository alongside comprehensive engineering recommendations for scaling the Klassed NEET Adaptive Engine into an enterprise-grade v2 AI ecosystem.

While v1 successfully proved out closed-loop adaptive testing across **2,394 verified NEET questions** and **63 canonical NCERT chapters** (`backend/klassed.db`, `data/tagged_questions/neet_questions_clean.json`), the v2 architecture transitions the ML layer (`model/` and `data/training/`) from static heuristics to **probabilistic sequence modeling (Bayesian Knowledge Tracing & SAINT+ Transformers)**, **psychometric Item Response Theory (IRT 3PL)**, **Contextual Multi-Armed Bandits (LinUCB)**, and **RAG-Powered Socratic Remediation**.

---

## 2. Complete Inventory of Existing Repository Files

### A. Machine Learning & Data Validation Layer (`model/`)

| File Path | Role in Current Architecture (v1) | Key Inputs, Outputs & Metrics |
| :--- | :--- | :--- |
| **model/train_difficulty_model.py** | Trains the supervised Random Forest difficulty classifier using one-hot encoded categorical metadata and numeric time/mark features with balanced class weighting. | Reads `data/training/question_features.csv` (2,471 rows; 1,976 train / 495 test) and outputs `model/question_difficulty_model.joblib` and `model/question_difficulty_metrics.json` at **99.60% test accuracy**. |
| **model/question_difficulty_metrics.json** | Stores evaluation telemetry for the trained Random Forest classifier across `basic`, `medium`, and `advanced` classes. | Records 1,976 training rows, 495 testing rows, **0.9960 accuracy**, and the 8 core feature columns (`subject`, `chapter`, `topic_tag`, `question_type`, `ideal_time_sec`, `has_diagram`, `marks`, `negative_marks`). |
| **model/question_difficulty_model.joblib** | Serialized scikit-learn pipeline combining `ColumnTransformer` (`OneHotEncoder` + numeric passthrough) and a 300-tree `RandomForestClassifier`. | Production artifact used to verify and predict question difficulty tiers for newly ingested NEET items. |
| **model/build_training_dataset.py** | Extracts structured tabular features from the cleaned NEET question bank and audits missing values, difficulty distributions, and topic counts. | Reads `data/tagged_questions/neet_questions_clean.json` and generates `data/training/question_features.csv` (2,471 verified question feature rows). |
| **model/build_attempt_training_dataset.py** | Joins live student attempt records from the SQLite database with question metadata and computes per-attempt `time_ratio` (`time_taken_seconds / ideal_time_sec`). | Reads `backend/klassed.db` (`test_attempts` table) and `data/tagged_questions/neet_questions_clean.json`, exporting `data/training/student_attempts.csv` for sequence model research. |
| **model/importance_calculator.py** | Core mathematical engine computing fixed syllabus topic weights using exponential recency decay ($0.9^{\text{current\_year} - \text{year}}$) normalized to sum to 100. | Pure library module owned by Model Support so topic importance weights remain deterministic and reproducible. |
| **model/build_topic_importance.py** | Parses historical question frequencies per chapter across 13 years of NEET papers (2013–2026), filtering out generic tags and invoking `importance_calculator.py`. | Reads tagged question banks and writes `model/topic_importance.json` across 65 canonical NCERT chapters. |
| **model/topic_importance.json** | Precomputed normalized importance scores across all NCERT chapters in Physics, Chemistry, and Biology. | Consumed by `backend/app/paper_generator.py` to allocate fixed per-topic question quotas on every adaptive paper. |
| **model/validate_question_bank.py** | Automated quality assurance gate scanning raw question submissions for templated boilerplate explanations, identical reviewer notes, duplicate MCQ options, fake variants, and non-standard difficulty labels. | Rejects corrupted batches before database ingestion and outputs structured JSON audit verdicts. |
| **model/filter_clean_questions.py** | Splits raw tagged question banks into a clean production subset and a quarantined subset requiring remediation (catching duplicate options, generic chapters, and OCR glued-word errors). | Reads `neet_all_papers_2013_2026_flawless.json` and outputs `.clean.json` (safe to load) and `.quarantine.json` (flagged for remediation). |
| **model/validate_weightage_file.py** | Audits multi-year syllabus weightage files for internal consistency, verifying that stated 13-year totals match yearly sums, marks equal four times question count, and no topic exhibits single-year taxonomy bugs. | Validates `data/taxonomy/NEET_Unified_Weightage_And_Marks_2013_2026.json` across 25 units and 13 exam years. |
| **model/state_machine/ & model/notebooks/** | Dedicated Model Support workspaces reserved for prototyping v2 probabilistic knowledge tracing (BKT / SAINT+) and psychometric calibration experiments. | Houses experimental replacements for the hardcoded v1 rolling-window state machine as specified in `docs/spec.md` Section 11. |

---

### B. Curated Datasets, Taxonomy & Training Assets (`data/`)

| File Path | Dataset Contents & Structure | Role in Adaptive Engine |
| :--- | :--- | :--- |
| **data/tagged_questions/neet_questions_clean.json** | Sanitized, verified NEET UG question bank (2013–2026) containing question text, four options, correct option key, chapter, topic tag, skill descriptor, difficulty level, ideal solve time, and explanation. | Primary source of truth loaded into `backend/klassed.db` (2,394 active questions across 63 chapters) and used to build ML training sets. |
| **data/tagged_questions/neet_all_papers_2013_2026_flawless.json** | Complete raw 13-year NEET UG question bank prior to quarantine filtering, accompanied by `.clean.json` and `.quarantine.json` splits generated by `model/filter_clean_questions.py`. | Preserves full provenance of the 2013–2026 question corpus alongside quarantined items isolated for editorial review. |
| **data/taxonomy/NEET_Unified_Weightage_And_Marks_2013_2026.json** | Comprehensive 13-year NCERT weightage analysis covering **25 units, 64 chapters, 197 canonical topics, 2,561 questions, and 10,244 total marks** under the $+4 / -1$ marking scheme. | Establishes locked canonical NCERT chapter naming (`2.1_CANONICAL_NCERT_LOCKED`) and historical year-by-year exam distributions. |
| **data/training/question_features.csv** | Tabular feature matrix of **2,471 verified NEET questions** generated by `model/build_training_dataset.py`. | Training and evaluation dataset for `model/train_difficulty_model.py`. |
| **data/training/student_attempts.csv** | Longitudinal student attempt dataset joining `test_attempts` from SQLite with question metadata and `time_ratio`, generated by `model/build_attempt_training_dataset.py`. | Foundation dataset for training v2 sequence models (Bayesian Knowledge Tracing, IRT 3PL, and contextual bandits). |
| **data/config/time_benchmarks.json** | Configurable expected solve-time benchmarks per difficulty tier: **Basic = 60 seconds**, **Medium = 90 seconds**, **Advanced = 120 seconds**. | Read by `backend/app/time_tracking.py` to compute `time_ratio` and assign `"fast"` vs `"slow"` pace flags without hardcoding thresholds. |

---

### C. Backend Adaptive Engine (`backend/`) & Frontend CBT Interface (`frontend/` & `docs/`)

| Component / File Path | Architectural Responsibility |
| :--- | :--- |
| **docs/spec.md** | Single source of truth defining the v1 rule-based philosophy, taxonomy schema, topic importance formula, database schema, 3-paper diagnostic phase, rolling-window state machine, independent pace tracking, adaptive paper generation, and the v2 Bayesian Knowledge Tracing roadmap. |
| **backend/app/main.py** | FastAPI application exposing `/auth/signup`, `/auth/login`, `/students/{id}/onboarding`, `/submit_attempt`, `/students/{id}/state`, `/students/{id}/next_paper`, `/students/{id}/papers/{paper_number}/review`, and `/students/{id}/analytics`. |
| **backend/app/state_machine.py** | Deterministic per-topic mastery state machine (`process_answer`): promotes `basic` to `medium` on 2 of last 3 correct; promotes `medium` to `advanced` on 3 consecutive correct; demotes `medium` to `basic` on 2 of last 3 wrong; demotes `advanced` to `medium` on 2 consecutive wrong. |
| **backend/app/time_tracking.py** | Independent pace tracker (`update_pace`): computes running average `time_ratio` on correct answers only against `data/config/time_benchmarks.json`, flagging `avg_time_ratio <= 1.2` as `"fast"` and `> 1.2` as `"slow"`. |
| **backend/app/paper_generator.py** | Adaptive 45-question paper generator (`generate_paper`): allocates fixed question counts per topic from `model/topic_importance.json`, samples from the student's current mastery tier in `student_topic_state`, and injects 1–2 supplementary timed drill questions when `pace_flag == "slow"`. |
| **backend/app/analytics.py** | Computes real, database-derived progress statistics (`build_analytics`) from `test_attempts` joined with `questions` and `topics`, returning overall accuracy, topics practiced, subject breakdowns, and recent attempts. |
| **backend/app/auth.py, db_models.py, database.py, schemas.py, load_real_data.py, seed_test_data.py, paths.py** | Complete persistence and identity stack: `bcrypt` password hashing, deterministic student IDs, SQLAlchemy ORM models (`Student`, `Topic`, `Question`, `StudentTopicState`, `TestAttempt`), SQLite database (`backend/klassed.db`), Pydantic schemas, and validated question-bank loader. |
| **frontend/src/ (App.tsx, ExamScreen.tsx, Dashboard.tsx, AuthPage.tsx, OnboardingPage.tsx, AppShell.tsx, api.ts)** | React 18 + TypeScript + Vite + Tailwind CSS single-page application providing real student signup/login, Class 11/12/Dropper onboarding, topic mastery & pace dashboard, full-paper CBT exam interface with per-question timer tracking, sequential attempt submission, and post-exam answer review. |

---

## 3. Recommended New Models & Algorithms (Why & What We Are Fixing in the ML Layer)

### 1. Bayesian Knowledge Tracing (BKT) & SAINT+ Transformers
* **Where It Connects in Repo:** Upgrades `model/state_machine/`, `model/build_attempt_training_dataset.py`, and `backend/app/state_machine.py`.
* **Why:** v1 uses rigid 3-question rolling windows (`2 out of 3` in `state_machine.py`) that treat all binary outcomes equally and fail to capture continuous mastery probability, guess rates, slip rates, or temporal forgetting across sessions.
* **What We Are Fixing:** Replaces hard window rules with probabilistic Hidden Markov Models (BKT) and separated self-attention sequence models (SAINT+) trained on `data/training/student_attempts.csv` to continuously compute posterior mastery probability $P(L_t)$ per topic after every attempt.

### 2. Item Response Theory (IRT — 3PL Model)
* **Where It Connects in Repo:** Upgrades `model/train_difficulty_model.py`, `data/training/question_features.csv`, and `data/tagged_questions/neet_questions_clean.json`.
* **Why:** v1 relies on static categorical difficulty labels (`basic`, `medium`, `advanced`) predicted from metadata by `model/question_difficulty_model.joblib` without accounting for live student cohort pass rates or 4-option MCQ guessing baselines.
* **What We Are Fixing:** Dynamically calibrates continuous item difficulty ($b_i$), item discrimination ($a_i$), and pseudo-guessing ($c_i$) alongside student latent ability ($\theta$) using global cohort telemetry logged in `student_attempts.csv`.

### 3. Contextual Multi-Armed Bandits (LinUCB)
* **Where It Connects in Repo:** Upgrades `backend/app/paper_generator.py` and `model/topic_importance.json`.
* **Why:** In v1, `paper_generator.py` uses uniform random sampling (`random.sample`) within a topic's difficulty pool and lacks an exploration-exploitation trade-off for probing adjacent difficulty tiers.
* **What We Are Fixing:** Implements Linear Upper Confidence Bound (LinUCB) contextual bandits that combine student mastery $P(L_t)$, latent ability $\theta$, and `time_ratio` to route active questions within the student's **Zone of Proximal Development (ZPD)** while automating the retention probe questions noted in `paper_generator.py`.

### 4. RAG-Powered Socratic Remediation
* **Where It Connects in Repo:** Upgrades static `explanation` and `wrong_option_tags` fields in `data/tagged_questions/neet_questions_clean.json` and `backend/app/db_models.py`.
* **Why:** v1 relies on static JSON answer keys and explanations, and `wrong_option_tags` in `load_real_data.py` is currently unpopulated (`None`), leaving mistake-fingerprinting on hold.
* **What We Are Fixing:** Integrates vector search over NCERT Physics, Chemistry, and Biology textbooks via `sentence-transformers` and FAISS to diagnose specific distractor misconceptions and generate dynamic, NCERT-grounded Socratic hints on failure without prematurely leaking the answer.

---

## 4. Recommended New Datasets (Why & What We Are Fixing in `data/`)

| Recommended Dataset | Target Repo Pipeline | Why v1 Needs This | What We Are Fixing |
| :--- | :--- | :--- | :--- |
| **Granular Clickstream Telemetry** | `data/training/student_attempts.csv` & `model/build_attempt_training_dataset.py` | v1 only tracks final binary correctness (`is_correct`) and total elapsed seconds (`time_taken_seconds`). | Captures option-switching logs, erasure timestamps, initial hesitation latency, and dwell time from `ExamScreen.tsx` to distinguish confident mastery from lucky guesses. |
| **Knowledge-Graph Prerequisite DAGs** | `data/taxonomy/NEET_Unified_Weightage_And_Marks_2013_2026.json` | v1 treats all 63 chapters in `model/topic_importance.json` as independent silos. | Maps directed prerequisite dependency trees between chapters and micro-topics (e.g., *Kinematics* $\to$ *Laws of Motion* $\to$ *Work, Energy & Power* $\to$ *Rotational Motion*) so failures in advanced chapters trigger prerequisite diagnostics. |
| **Multimodal NCERT Diagram Datasets** | `data/tagged_questions/neet_questions_clean.json` & `data/training/question_features.csv` | While `question_features.csv` tracks a boolean `has_diagram` column, `image_asset` is empty in v1 and questions are purely text-based. | Incorporates annotated anatomical figures, circuit diagrams, and chemical structures to track visual comprehension and diagram-specific solve times. |
| **Public Pre-Training Corpora (EdNet / ASSISTments)** | `data/training/student_attempts.csv` & `model/state_machine/` | `student_attempts.csv` suffers from cold-start data sparseness when onboarding new students prior to completing diagnostic papers. | Pre-trains BKT and SAINT+ sequence priors using millions of external educational interaction rows so new users receive calibrated routing from their very first paper. |

---

## 5. Component-Specific Improvements via New Models & Datasets

* **Improving Knowledge Tracking via BKT / SAINT+ (`model/state_machine/` & `backend/app/state_machine.py`):** Replaces binary rolling-window jumps (`last_3_results` and `streak`) with a continuous probability curve $P(L_t) \in (0, 1)$, eliminating abrupt state oscillation caused by single careless errors.
* **Improving Psychometric Calibration via IRT (`model/train_difficulty_model.py` & `data/training/question_features.csv`):** Augments the 99.60% Random Forest metadata classifier by dynamically updating item difficulty ($b_i$) and discrimination ($a_i$) parameters based on live population failure rates.
* **Improving Question Selection via LinUCB Bandits (`backend/app/paper_generator.py`):** Replaces uniform `random.sample` within a tier with Upper Confidence Bound scoring that balances exploitation of weak chapters with exploration of borderline and retention-probe topics.
* **Improving Remediation via RAG (`data/tagged_questions/neet_questions_clean.json`):** Activates the misconception-fingerprinting layer described in `docs/spec.md` Section 10 by generating dynamic, NCERT-grounded conceptual explanations tailored to the exact wrong option selected.
* **Improving Cold-Start via EdNet / ASSISTments (`data/training/student_attempts.csv`):** Initializes student ability priors ($\theta_0$ and $P(L_0)$) from pre-trained cohort embeddings and onboarding metadata (`class_level` and `target_neet_year`), solving cold-start data sparsity.

---

## 6. Separated Risk & Vulnerability Audits (Models vs. Datasets)

### A. Model & Algorithmic Risks & Mitigations

| Identified Model Risk | Affected Repo Component | Impact Analysis | Engineering Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Overfitting on Static Features** | `model/train_difficulty_model.py` & `model/question_difficulty_model.joblib` | Unconstrained Random Forest trees can memorize historical chapter/topic-tag combinations and `ideal_time_sec` values in `question_features.csv`. | Regularize tree depth (`max_depth`), enforce minimum leaf sample sizes, evaluate across 5-fold stratified cross-validation, and incorporate adversarial validation splits. |
| **State Oscillation in Rolling Windows** | `backend/app/state_machine.py` | Rigid thresholds (`2 out of 3` correct/wrong) cause rapid promotion/demotion ping-ponging between `basic` and `medium` tiers. | Transition to smoothed Bayesian confidence intervals ($95\%$ credible intervals) with hysteresis deadbands before shifting difficulty tiers. |
| **Cold-Start Bias in Contextual Bandits** | `backend/app/paper_generator.py` | Initial exploration phases in LinUCB can serve suboptimal or overly difficult questions when student context vectors are uninitialized. | Warm-start bandit ridge regression priors using global cohort item difficulty distributions from IRT and Random Forest baseline predictions. |
| **Hallucination in RAG Remediation** | Post-exam review & remediation layer | Generative hints could state inaccurate formulas or non-NCERT terminology. | Enforce strict cosine similarity thresholds ($\ge 0.82$) over verified NCERT textbook chunks and ground prompts in verified question explanations. |

### B. Dataset Risks & Mitigations

| Identified Dataset Risk | Affected Repo Component | Impact Analysis | Engineering Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Label Noise in Historical JSONs** | `data/tagged_questions/neet_all_papers_2013_2026_flawless.json` | Boilerplate explanations, duplicate MCQ options, or generic `"General Biology"` chapters corrupt downstream training and paper generation. | Enforce strict automated validation gates (`model/validate_question_bank.py`) and split corrupted items via `model/filter_clean_questions.py` into `.quarantine.json`. |
| **Class Imbalance Across Difficulty Tiers** | `data/training/question_features.csv` | Overwhelming majority of historical NEET questions reside in the `medium` tier (approx. $72\%$), with fewer `basic` and `advanced` items. | Apply `class_weight="balanced"` in `model/train_difficulty_model.py`, stratified train/test splitting (`stratify=y`), and calibrated minority oversampling. |
| **Data Sparseness for Rare Topics** | `data/taxonomy/NEET_Unified_Weightage_And_Marks_2013_2026.json` & `model/topic_importance.json` | Low-frequency chapters (e.g., *s-Block Elements*, *Environmental Chemistry*) have sparse historical occurrences across the 13-year window. | Audit single-year anomalies via `model/validate_weightage_file.py` and apply Hierarchical Bayesian empirical shrinkage pooling across parent unit nodes. |

---

## 7. Latency & System Design (Recommended v2 Architecture)

To achieve **sub-15ms inference latency** at scale under heavy concurrent loads (50,000+ active NEET aspirants taking simultaneous mock papers), the v2 system design decouples state management and ML model inference from transactional database writes (`backend/klassed.db`):

| Architectural Tier | Target Latency | Technology Stack | Operational Role |
| :--- | :--- | :--- | :--- |
| **1. Low-Latency Feature Store** | `< 2 ms` | Redis / Feast In-Memory Store | Caches rolling student telemetry (`last_3_results`, `avg_time_ratio`, BKT posteriors $P(L_t)$, and IRT ability $\theta$) in-memory, cutting state retrieval latency from ~25ms disk reads to under 2ms. |
| **2. Containerized Model Serving** | `< 10 ms` | NVIDIA Triton / ONNX Runtime | Migrates heavy sequence modeling (SAINT+ / BKT / IRT 3PL) and Random Forest difficulty scoring (`question_difficulty_model.joblib`) out of synchronous Python loops into C++ ONNX runtimes. |
| **3. Asynchronous Vector Retrieval** | `< 40 ms` | FAISS In-Memory HNSW Index | Decouples RAG Socratic hint generation using pre-computed `sentence-transformers` vector embeddings of NCERT chapters and verified explanations stored in local memory indices. |

---

## 8. Recommended Enhanced Machine Learning Workflow

| Stage | Workflow Phase | Component & Pipeline Execution |
| :--- | :--- | :--- |
| **Stage 1** | **Multimodal & Clickstream Telemetry Ingestion** | Captures option-switching logs, hesitation latency, time-taken ratios, diagram interactions, and EdNet pre-training priors via `model/build_attempt_training_dataset.py` and `model/build_training_dataset.py`. |
| **Stage 2** | **Centralized Feature Store & Quality Gates** | Passes all question and weightage data through `model/validate_question_bank.py`, `model/filter_clean_questions.py`, and `model/validate_weightage_file.py`, serving verified features into Feast / Redis. |
| **Stage 3** | **Probabilistic Modeling & Bandit Calibration** | Parallel execution of (a) Probabilistic Knowledge Tracing (`PyKT` / BKT / SAINT+ / IRT 3PL) and (b) Contextual Multi-Armed Bandits (LinUCB / PPO RL) alongside `model/train_difficulty_model.py`. |
| **Stage 4** | **ONNX Runtime / Triton Model Server** | Compiles trained scikit-learn (`question_difficulty_model.joblib`) and sequence models into ONNX computation graphs delivering `< 15ms` state and difficulty inference. |
| **Stage 5** | **Automated Continuous Training (CT) & Drift Monitoring** | Tracks Population Stability Index (PSI) and accuracy/pace drift via Evidently AI, triggering automated re-calibration of `topic_importance.json` and model weights when drift thresholds are exceeded. |

---

## 9. Recommended Enterprise MLOps Architecture

| MLOps Layer | Governance & Automation Mechanism |
| :--- | :--- |
| **1. Automated CI/CD & GitHub Actions** | Automated execution of `pytest` suites (`backend/tests/`), JSON schema validation (`model/validate_question_bank.py` & `model/validate_weightage_file.py`), and linting on every pull request. |
| **2. Model Registry & Artifact Tracking** | MLflow and Weights & Biases version control tracking `model/question_difficulty_model.joblib`, `model/question_difficulty_metrics.json`, `model/topic_importance.json`, and BKT/IRT parameter checkpoints. |
| **3. Containerized Triton / ONNX Serving** | Dockerized FastAPI and Triton inference microservices with horizontal autoscaling, health probes, and sub-15ms p99 latency guarantees. |
| **4. Continuous Monitoring & Drift Governance** | Evidently AI telemetry monitors comparing live student attempt distributions (`student_attempts.csv`) against baseline training distributions (`question_features.csv`) with automated retrain triggers. |

---

## 10. End-to-End v2 Lifecycle Flow

1. **Student Interaction (`frontend/src/features/adaptive-paper/ExamScreen.tsx`):** Student solves an adaptive paper while the client records answer selection, per-question solve duration, option-switching counts, and hesitation timestamps.
2. **FastAPI Microservice Gateway (`backend/app/main.py` & `backend/app/auth.py`):** Authenticates the session, validates the attempt payload (`schemas.AttemptIn`), and routes telemetry into the inference pipeline.
3. **Redis / Feast Feature Store:** Retrieves the student's current per-topic mastery vector, `avg_time_ratio`, and prior latent ability $\theta$ in under 2ms.
4. **Triton / ONNX Model Server (`< 15ms` Inference):**
   * **BKT / SAINT+ Engine:** Updates the student's continuous mastery probability $P(L_t)$ and credible interval for the attempted topic.
   * **IRT 3PL Engine:** Recalibrates item difficulty $b_i$ and updates the student's latent ability $\theta$.
   * **LinUCB Bandit Router:** Evaluates candidate questions across fixed topic weights (`model/topic_importance.json`) and selects the optimal next question within the student's Zone of Proximal Development.
5. **Database Persistence (`backend/app/database.py` & `db_models.py`):** Asynchronously commits the `TestAttempt` and updated `StudentTopicState` rows to PostgreSQL / SQLAlchemy.
6. **RAG Socratic Remediation Engine:** When a student reviews an incorrect attempt (`paper_review`), retrieves relevant NCERT passages via FAISS to surface targeted conceptual hints and misconception diagnostics.
7. **Evidently AI Drift Monitor & MLflow CI/CD:** Continuously logs prediction residuals and latency metrics for model governance.

---

## 11. Final Implementation Plan & Enterprise Technical ML Pitch

### Phased Implementation Plan
* **Phase 1 — Stabilization, Quality Gates & Security:** Enforce canonical string normalization on answer comparisons, expand `model/validate_question_bank.py` and `model/filter_clean_questions.py` across all new question batches, add comprehensive `pytest` coverage in `backend/tests/`, and deploy JWT Bearer authentication and transactional database locks.
* **Phase 2 — Feature Store & Telemetry Expansion:** Instrument `ExamScreen.tsx` and `model/build_attempt_training_dataset.py` to capture option-switching and hesitation telemetry in `data/training/student_attempts.csv`, construct the NCERT Knowledge-Graph prerequisite DAG, and deploy a Redis/Feast feature store.
* **Phase 3 — Probabilistic Modeling, IRT 3PL & Bandit Routing:** Upgrade from rolling-window state machines (`backend/app/state_machine.py`) to BKT/SAINT+ sequence models in `model/state_machine/`, calibrate all 2,394+ questions via IRT 3PL alongside our **99.60% accuracy** Random Forest model (`model/train_difficulty_model.py`), and integrate LinUCB bandit routing into `backend/app/paper_generator.py`.
* **Phase 4 — RAG Remediation & Triton Serving:** Stand up FAISS vector search over NCERT textbooks to populate `wrong_option_tags` and Socratic hints dynamically, and containerize model inference via ONNX Runtime / Triton with Evidently AI drift governance.

### Enterprise Technical ML Pitch
> *"With our v2 roadmap building directly upon our 99.60% accuracy Random Forest baseline and 13-year canonical NEET dataset to introduce Transformer-based knowledge tracing (SAINT+), 3PL Item Response Theory, LinUCB contextual bandits, and NCERT-grounded RAG Socratic remediation, the Klassed NEET Adaptive Engine delivers true mathematical personalization at sub-15ms scale—ensuring every medical aspirant operates continuously within their optimal Zone of Proximal Development."*
