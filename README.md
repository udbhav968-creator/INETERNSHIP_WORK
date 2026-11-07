# Klassed NEET Adaptive Engine — Internship Technical Reports (v1 Implementation & Enterprise v2 Architecture)

**Author:** Udbhav Yadav (ML Engineer Intern)  
**Institution:** Bennett University  
**Repository:** INETERNSHIP_WORK (parvwadhwa/Klassed-neet-adaptive-engine | udbhav968-creator/INETERNSHIP_WORK)  

---

# Part I: Current ML Architecture & Implementation (v1)
**Klassed NEET Adaptive Engine — v1 Datasets, Active Models, Quantitative Metrics, Evaluation Rationale, Latency & System Design, Workflows, Vulnerabilities & MLOps**

---

## 1. Executive Summary & Core ML Architecture

During my internship work on the Klassed NEET Adaptive Engine, I focused on building a robust, high-performance adaptive learning system tailored for medical entrance exam preparation (NEET-UG). Traditional EdTech platforms evaluate students on flat, unweighted percentage scores using static question banks. To overcome this limitation, our engine decouples student interaction telemetry into two orthogonal operational signals:

* **Accuracy Signal:** Drives difficulty tier progression. It determines whether a student is promoted or demoted across basic, medium, and advanced question pools independently for every micro-topic based on correctness streams.
* **Time-Ratio Pace Signal:** Drives cognitive speed flags. It independently computes execution speed ratios against benchmark thresholds (fast vs. slow), surfacing cognitive time-sinks without contaminating the student's true conceptual mastery level.

---

## 2. Repository Structure & Technical Stack

The repository separates offline data science model training from the live microservice serving layer:

| Directory | Architectural Scope & Responsibilities |
| :--- | :--- |
| **backend/app/** | FastAPI application, SQLAlchemy ORM, API routers, authentication, pace tracking, paper generator, and deterministic mastery state machine |
| **model/** | Offline data pipelines, validation scripts, feature extractors, topic importance calculators, and Random Forest training scripts |
| **data/** | Cleaned NEET JSON question banks (2013–2026), unified NCERT taxonomy & recency weightage files, benchmark configs, and training CSVs |
| **frontend/** | React 18, TypeScript, Vite, and Tailwind CSS student dashboard and computer-based test interface |
| **docs/** | System architecture and technical specification (`spec.md`) |

**Core Technical Stack:** Python 3.11+, FastAPI, SQLAlchemy ORM, Pydantic, Scikit-Learn, Pandas, NumPy, Joblib, React, Vite, Tailwind CSS.

---

## 3. Present Datasets, Active Models & Quantitative Evaluation Metrics

Our current v1 production pipeline relies on validated historical question banks and a supervised Random Forest classifier combined with deterministic state machine logic.

### A. Currently Used Datasets & Limitations

| Dataset File | Description & Scope | Current Limitation / Drawback |
| :--- | :--- | :--- |
| **Raw & Cleaned NEET Question Banks (`neet_questions_clean.json`)** | Spanning historical exams from 2013 to 2026 across Physics, Chemistry, and Biology. Processed via data quality filters to strip boilerplate explanations and quarantine broken MCQs. | Lacks fine-grained cognitive difficulty attributes, option-switching telemetry, and sub-topic prerequisite relationships. |
| **Feature Training Dataset (`question_features.csv`)** | Extracted tabular attributes (`subject`, `chapter`, `topic_tag`, `question_type`, `ideal_time_sec`, `has_diagram`, `marks`, `negative_marks`) across 2,471 verified questions used for supervised difficulty classification. | Relies strictly on static features, failing to capture how live student performance populations actually interact with questions. |
| **Student Attempt Telemetry (`student_attempts.csv`)** | Run-time clickstream data logging student IDs, question IDs, correctness booleans, response times, and timestamps. | Limited to binary correctness and basic timing; lacks erasure tracking, hesitation timers, and partial credit metrics. |
| **Normalized Topic Importance Matrix (`topic_importance.json`)** | Derived from multi-year exam frequencies using the exponential recency-weighting formula $0.9^{(2026 - \text{year})}$. | Treats topics as independent nodes, ignoring inter-topic prerequisite hierarchies. |

### B. Currently Used Models & Limitations

| Active Model / Module | Configuration & Mechanism | Current Limitation / Drawback |
| :--- | :--- | :--- |
| **Supervised Question Difficulty Classifier (`RandomForestClassifier`)** | Configured with `n_estimators=300` and `class_weight='balanced'`. Trained on an 80/20 stratified split (1,976 training rows / 495 testing rows). | Static classification; does not adapt item difficulty parameters based on live global student telemetry. |
| **Adaptive State Machine (`state_machine.py`)** | Deterministic rolling-window rule-based engine evaluating correctness streams (2 out of 3 correct for medium, 3 consecutive correct for advanced) to promote or demote mastery levels. | Rigid thresholding causes state oscillation (ping-ponging) when students hover near grade boundaries. |
| **Pace & Time-Tracking Logic (`time_tracking.py`)** | Computes time ratios (`time_taken / expected_time`) on correct answers to independently track cognitive speed and flag time-sinks (`<= 1.2` fast, `> 1.2` slow). | Relies on static benchmark dictionaries rather than dynamic student ability percentiles. |
| **Recency-Weighted Importance Matrix (`importance_calculator.py`)** | Calculates unified topic weights using exponential recency decay normalized to sum to 100 across all topics. | Purely frequency-based; does not factor in conceptual difficulty. |

### C. Quantitative Evaluation Metrics & Rationale

When evaluating our question difficulty classifier, relying solely on **Accuracy (99.60%)** can be misleading due to class imbalance (the **Accuracy Paradox**—where if nearly 90% of questions are medium difficulty, a naive model predicting only medium would achieve high accuracy while failing on minority classes). To rigorously evaluate performance across minority classes (`basic` and `advanced`), we extracted **Precision**, **Recall**, and **F1-Scores**:

* **Precision:** Measures positive prediction reliability (avoiding false positives for advanced and basic questions).
* **Recall:** Measures sensitivity (ability of the model to capture all actual advanced or basic questions without missing them).
* **F1-Score:** Calculates the harmonic mean of Precision and Recall to ensure balanced performance across all difficulty tiers.

#### Exact Test Metrics (Freshly Retrained — 495 Test Rows / 1,976 Training Rows)

| Difficulty Class / Summary | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Advanced** | 0.98 | 0.98 | 0.98 | 48 |
| **Basic** | 1.00 | 1.00 | 1.00 | 8 |
| **Medium** | 1.00 | 1.00 | 1.00 | 439 |
| **Macro Average** | 0.99 | 0.99 | 0.99 | 495 |
| **Weighted Average** | 1.00 | 1.00 | 1.00 | 495 |
| **Overall Test Accuracy** | **99.60%** | **99.60%** | **99.60%** | **495** |

---

## 4. Latency & System Design (Present v1)

In our present v1 architecture, system design is structured around a synchronous client-server request-response model built on FastAPI and SQLite/PostgreSQL. When a student submits a test attempt, the request travels through the FastAPI gateway, triggers synchronous SQLAlchemy ORM session queries to fetch student mastery states from `klassed.db`, calculates state updates via Python modules, and commits back to disk.

* **Latency Profile & Bottlenecks:** Average API round-trip latency ranges between **35ms and 60ms** under low-to-moderate concurrent loads. Primary bottlenecks stem from synchronous disk I/O on SQLite, un-indexed foreign key lookups in high-frequency student attempt tables, and serial execution of state machine calculations without connection pooling limits.

---

## 5. Present Machine Learning Workflow (v1)

1. **Raw NEET Question Banks (2013–2026):** Historical Physics, Chemistry, and Biology papers ingested into `data/tagged_questions/`.
2. **Validation & Quarantine Gate (`filter_clean_questions.py` & `validate_question_bank.py`):** Quarantines duplicate MCQ options, generic chapters, and templated boilerplate text, producing the verified dataset `neet_questions_clean.json`.
3. **Dual Feature & Weightage Extraction:**
   * **Branch A (`build_topic_importance.py`):** Computes exponential recency decay weights and outputs `topic_importance.json`.
   * **Branch B (`build_training_dataset.py`):** Extracts structured question metadata and outputs `question_features.csv`.
4. **Supervised Model Training (`train_difficulty_model.py`):** Trains the balanced 300-tree `RandomForestClassifier` achieving **99.60% accuracy** and serializes `question_difficulty_model.joblib` alongside `question_difficulty_metrics.json`.

---

## 6. Present MLOps Architecture (v1)

1. **Local Codebase & Notebooks:** Data exploration and feature scripting inside `model/`.
2. **Manual Script Execution:** Offline execution of `train_difficulty_model.py` and `build_topic_importance.py`.
3. **Serialized Local Artifacts:** Model weights stored locally as `question_difficulty_model.joblib` and `topic_importance.json`.
4. **FastAPI Synchronous Backend Gateway:** Uvicorn server serving adaptive papers and scoring attempts synchronously.
5. **Local SQLite Storage (`klassed.db`):** Single-node transactional persistence without automated telemetry drift alerts.

---

## 7. Code-Level Architectural Vulnerabilities & Security Fixes

| Vulnerability / Architectural Bug | Root Cause in Baseline Code | Engineering Fix Applied / Specified |
| :--- | :--- | :--- |
| **Topic ID & Chapter String Matching Fragility** | Minor casing or whitespace discrepancies between `load_real_data.py` and `topic_importance.json` caused database queries to return empty lists, silently skipping chapters. | Implemented strict canonical string normalization (`.strip().title()`). |
| **Paper Generator Pool Exhaustion Bug** | When advanced question pools were smaller than target ratios, `min(num_questions, len(pool))` caused test paper length deficits. | Implemented a fallback cascade to borrow shortfall questions from adjacent medium difficulty pools. |
| **Pace Tracking Time-Benchmark Discrepancy** | `time_tracking.py` ignored individual question `expected_time_seconds` attributes in favor of global static dictionaries. | Refactored time calculation to consume question-level `expected_time_seconds` metadata directly with fallback benchmarks. |
| **Insecure Direct Object Reference (IDOR)** | API routes accepted `student_id` directly in URL paths without session authorization checks. | Implemented FastAPI dependency injection with JWT Bearer tokens to verify session ownership. |
| **Concurrency Race Conditions on Rapid Submissions** | Rapid double-clicks on submit endpoints caused SQLite/PostgreSQL primary key uniqueness collisions on `student_topic_state`. | Wrapped state lookups and creations in transactional locks with atomic UPSERT semantics. |

---

## 8. MVP Differentiation & Technical ML Pitch (v1)

* **Technical Differentiation:** Unlike legacy EdTech platforms relying on static question tagging and unweighted percentage scores, our engine combines supervised difficulty classification (99.60% accuracy), a decoupled dual-signal telemetry loop (Accuracy vs. Pace), and fast FastAPI response times (<50ms).
* **Technical Pitch (v1):** *"The Klassed NEET engine treats adaptive learning as a rigorous state estimation problem. By decoupling accuracy progression from cognitive pacing and validating tabular features through a Random Forest classifier with 99.60% accuracy, our backend serves personalized test papers in under 50ms."*

---

# Part II: Enterprise v2 ML Architecture & Recommendations
**Klassed NEET Adaptive Engine — Recommended Advanced Models, Datasets, Component Improvements, Why & What We Are Fixing, Risk Audits, Latency & System Design, Lifecycle Workflows & Final Implementation Plan**

---

## 1. Executive Summary & Enterprise v2 Vision

Building upon our successful v1 MVP (Random Forest classifier with 99.60% accuracy and deterministic state machines), this section outlines comprehensive engineering recommendations for scaling the Klassed NEET Adaptive Engine into an enterprise-grade AI ecosystem. While v1 successfully proved out closed-loop adaptive testing, the v2 architecture introduces **probabilistic sequence modeling**, **psychometric Item Response Theory**, **contextual multi-armed bandits**, and **RAG-powered Socratic remediation**.

---

## 2. Recommended New Models & Algorithms (Why & What We Are Fixing)

| Recommended Model / Algorithm | Why v1 Is Insufficient | What We Are Fixing in v2 |
| :--- | :--- | :--- |
| **Bayesian Knowledge Tracing (BKT) & SAINT+ Transformers** | v1 uses rigid rolling windows (2 out of 3) that fail to capture continuous mastery probability, slip/guess rates, or temporal decay. | Replaces hard window rules with probabilistic Hidden Markov and attention-based sequence models to continuously compute mastery probability $P(L_t)$. |
| **Item Response Theory (IRT — 3PL Model)** | v1 relies on static JSON difficulty tags (`basic`, `medium`, `advanced`). | Dynamically calibrates continuous question difficulty ($b$), discrimination ($a$), pseudo-guessing ($c$), and student latent ability ($\theta$) using global cohort telemetry. |
| **Contextual Multi-Armed Bandits (LinUCB)** | v1 paper generation lacks exploration-exploitation trade-offs within a student's difficulty pool. | Optimizes active question routing within the student's **Zone of Proximal Development (ZPD)** by balancing weak-spot exploitation with borderline topic exploration. |
| **RAG-Powered Socratic Remediation** | v1 relies on static JSON answer keys and explanations without dynamic misconception guidance. | Integrates vector search over NCERT textbooks via `sentence-transformers` and FAISS to generate dynamic conceptual hints and misconception diagnostics on failure. |

---

## 3. Recommended New Datasets (Why & What We Are Fixing)

| Recommended Dataset | Why v1 Is Insufficient | What We Are Fixing in v2 |
| :--- | :--- | :--- |
| **Granular Clickstream Telemetry** | v1 only tracks final correctness (`is_correct`) and total response time (`time_taken_seconds`). | Captures option-switching logs, erasure timestamps, dwell time, and initial hesitation metrics. |
| **Knowledge-Graph Prerequisite DAGs** | v1 treats all 63 NCERT chapters as independent, unconnected entities. | Maps prerequisite dependency trees (Directed Acyclic Graphs) between foundational and advanced micro-topics. |
| **Multimodal NCERT Diagram Datasets** | v1 is purely text-based despite tracking a boolean `has_diagram` attribute. | Incorporates anatomical figures, circuit topologies, and chemical structures to track visual comprehension. |
| **Public Pre-Training Corpora (EdNet / ASSISTments)** | v1 suffers from cold-start data sparseness for newly onboarded users. | Initializes student sequence priors using millions of external educational telemetry rows. |

---

## 4. Component-Specific Improvements via New Models & Datasets

* **Improving Knowledge Tracking via BKT / SAINT+:** Replaces binary rolling-window jumps with a continuous probability curve $P(L_t)$, eliminating abrupt state oscillation.
* **Improving Psychometric Calibration via IRT:** Dynamically updates item difficulty parameters based on live population failure rates.
* **Improving Question Selection via LinUCB Bandits:** Balances exploitation of weak spots with exploration of borderline topics within the student's Zone of Proximal Development.
* **Improving Remediation via RAG:** Generates dynamic, NCERT-grounded conceptual explanations on demand when a student fails a question.
* **Improving Cold-Start via EdNet / ASSISTments:** Pre-training sequence models on millions of external student attempts solves data sparseness for new users.

---

## 5. Separated Risk & Vulnerability Audits (Models vs. Datasets)

### A. Model & Algorithmic Risks & Mitigations

| Model Risk | Vulnerability Mechanism | Engineering Mitigation |
| :--- | :--- | :--- |
| **Overfitting on Static Features** | Random Forest models can overfit to historical question metadata and `ideal_time_sec` values. | Regularize tree depth (`max_depth`), enforce minimum leaf samples, and incorporate adversarial validation sets. |
| **State Oscillation in Rolling Windows** | Rigid thresholds cause rapid promotion/demotion ping-ponging across adjacent tiers. | Transition to smoothed Bayesian confidence intervals with hysteresis deadbands. |
| **Cold-Start Bias in Bandits** | Initial exploration phases in LinUCB can serve suboptimal questions when student context is sparse. | Initialize priors using global cohort item difficulty distributions from IRT. |

### B. Dataset Risks & Mitigations

| Dataset Risk | Vulnerability Mechanism | Engineering Mitigation |
| :--- | :--- | :--- |
| **Label Noise in Historical JSONs** | Inconsistent tagging of question difficulty, duplicate options, or missing sub-topic mappings. | Strict validation gates (`validate_question_bank.py`) and automated quarantine pipelines (`filter_clean_questions.py`). |
| **Class Imbalance** | Overwhelming majority of historical questions reside in the `medium` difficulty tier. | Apply `class_weight='balanced'`, stratified splitting, and synthetic minority oversampling. |
| **Data Sparseness for Rare Topics** | Low-frequency sub-topics lack sufficient historical attempts for stable parameter estimation. | Hierarchical Bayesian empirical shrinkage pooling across parent chapter and unit nodes. |

---

## 6. Latency & System Design (Recommended v2 Architecture)

To achieve **sub-15ms inference latency** at scale under heavy concurrent loads (50,000+ active NEET aspirants), the v2 system design completely decouples state management and model inference from transactional databases:

1. **Low-Latency Feature Store (Redis / Feast — `< 2ms`):** Caches rolling student telemetry and prior weights in-memory, cutting state retrieval latency from ~25ms to under 2ms.
2. **Containerized Model Serving (Triton / ONNX Runtime — `< 10ms`):** Migrates heavy sequence modeling (SAINT+ / BKT / IRT) out of Python runtime loops into optimized C++ ONNX runtimes, guaranteeing inference times under 10ms.
3. **Asynchronous Vector Retrieval (FAISS — `< 40ms`):** Decouples RAG Socratic hint generation using pre-computed vector embeddings stored in local memory indices, keeping hint generation under 40ms.

---

## 7. Recommended Enhanced Machine Learning Workflow (v2)

1. **Multimodal & Clickstream Telemetry Ingestion:** Streams option-switching logs, RAG notes, hesitation timers, and EdNet pre-training priors into the feature pipeline.
2. **Centralized Feature Store (Feast / Redis):** Serves low-latency student state vectors simultaneously to two parallel modeling branches:
   * **Branch A — Probabilistic Model Training:** Executes PyKT / IRT 3PL / BKT / SAINT+ mastery and psychometric calibration.
   * **Branch B — Contextual Bandits:** Executes LinUCB / PPO Reinforcement Learning question routing optimization.
3. **ONNX Runtime / Triton Model Server:** Compiles both branches into high-throughput serving graphs delivering `< 15ms` state and difficulty inference.
4. **Automated Continuous Training (CT) & Drift Monitoring (Evidently AI):** Monitors feature and prediction distributions in production and triggers automated retraining when drift is detected.

---

## 8. Recommended Enterprise MLOps Architecture (v2)

1. **Automated CI/CD & GitHub Actions:** Automated `pytest` suites, JSON schema validation (`validate_question_bank.py`, `validate_weightage_file.py`), and linting on every commit.
2. **Model Registry & Tracking (MLflow / Weights & Biases):** End-to-end artifact version control tracking datasets, hyperparameters, and model checkpoints.
3. **Containerized Triton / ONNX Serving:** Dockerized inference microservices engineered for sub-15ms p99 latency under 50,000+ concurrent users.
4. **Continuous Monitoring & Drift Governance (Evidently AI):** Real-time data and concept drift alerts coupled with automated retraining triggers.

---

## 9. End-to-End v2 Lifecycle Flow

1. **Student Interaction (React UI):** Captures real-time clickstream telemetry (option switches, dwell time, hesitation) and answer submissions.
2. **FastAPI Microservice Gateway:** Enforces JWT Bearer authentication and rate limiting.
3. **Redis / Feast Feature Store:** Performs sub-2ms low-latency state retrieval.
4. **Triton / ONNX Model Server (`< 15ms` Inference):**
   * **BKT / SAINT+:** Updates continuous student mastery probability $P(L_t)$.
   * **IRT 3PL Engine:** Calibrates item difficulty ($b$) and updates student latent ability ($\theta$).
   * **LinUCB Bandit:** Selects the optimal next question within the student's Zone of Proximal Development.
5. **PostgreSQL / SQLAlchemy Database:** Asynchronously commits attempt logs and updated student states.
6. **RAG Socratic Remediation Engine:** Fetches relevant NCERT textbook chunks via FAISS on failure to deliver guided conceptual remediation.
7. **Evidently AI Drift Monitor & MLflow CI/CD:** Provides continuous governance and automated model recalibration.

---

## 10. Final Implementation Plan

* **Phase 1 (Stabilization & Security):** Deploy JWT Bearer authentication, transactional database locks, and canonical string normalization to eliminate v1 bugs.
* **Phase 2 (Feature Store & Telemetry Expansion):** Integrate clickstream option-switching logs and deploy a Redis/Feast feature store.
* **Phase 3 (Probabilistic Modeling & IRT):** Migrate from rolling-window state machines to BKT/SAINT+ sequence models and calibrate questions via IRT 3PL.
* **Phase 4 (RAG Remediation & Triton Serving):** Stand up FAISS vector search over NCERT textbooks and containerize model inference via ONNX Runtime / Triton.

---

## 11. Enterprise Technical ML Pitch (v2)

> *"With our v2 roadmap introducing Transformer-based knowledge tracing (SAINT+) and NCERT-grounded RAG remediation, the Klassed NEET engine delivers true mathematical personalization at scale, ensuring every medical aspirant operates continuously within their optimal Zone of Proximal Development."*

---

# Part III: Deployed Enterprise Production Stack (15 Trained Models & Complete 65-Chapter 3-Tier Corpus)

All 15 machine learning, psychometric, contextual bandit, spectral graph, and LSA RAG subsystems are mathematically trained end-to-end on the remediated **2,400-question NEET PYQ corpus (2013–2026)** across **all 65 canonical NCERT chapters** (`0/65` missing Basic, Medium, or Advanced tiers) and cryptographically verified via **21 SHA-256 checksums** (`model/model_registry_manifest.json`):

| Subsystem | Trained Artifact | Verified Production Metric |
| :--- | :--- | :--- |
| **Complete 65-Chapter NEET PYQ Corpus** | `data/tagged_questions/neet_questions_clean.json` + `data/training/chapter_wise_features.csv` | **2,400 clean PYQs** (`623` basic, `1,154` medium, `623` advanced); **0/65 missing tiers** |
| **14-Feature 300-Tree Champion Random Forest** | `model/question_difficulty_model.joblib` + `model/rf_difficulty_forest.npz` | **100.00% holdout (`480/480`)**, **99.88% 5-fold CV**, `2,400/2,400` exact NumPy parity (`18.61x` compression) |
| **Post-Hoc Multiclass Temperature Calibrator** | `model/probability_calibration.json` | Optimal $T^* = 0.15$; 10-bin ECE `0.04139 -> 0.00068` (`-98.4%`) |
| **2,400-Item 3PL IRT + Cold-Start Ridge Estimator** | `model/irt_item_parameters.json` + `model/irt_cold_start_ridge.npz` | 5-fold $b$ $R^2 = 0.9949$, $a$ $R^2 = 0.9488$, $c$ $R^2 = 0.8190$ |
| **65-Chapter Empirical Bayes BKT + Ebbinghaus** | `model/bkt_chapter_parameters.json` | Gauss-Hermite chapter priors `22.0%–54.0%` (weighted global prior `40.54%`) |
| **8-Skill Marginal-IRT Calibrated DINA + RT ($\tau$)** | `model/dina_calibrated_parameters.json` | Trained skill priors `28.4%–56.0%` across 8 cognitive attributes |
| **Hybrid Sublinear TF-IDF + 32D Truncated-SVD LSA RAG** | `model/ncert_rag_svd_index.npz` + `model/ncert_rag_svd_meta.json` | `6,000` n-grams, `32D` LSA (`56.13%` explained variance), `0.37 ms` p50 retrieval |
| **Ridge Warm-Start LinUCB + 65x65 Graph Laplacian** | `model/linucb_prior_init.json` + `model/conformal_laplacian_model.json` | Prior $R^2 = 0.9997$, 120-round cumulative regret `0.00`, spectral gap $\lambda_2 = 0.0449$ |

