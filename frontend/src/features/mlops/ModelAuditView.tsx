import { useEffect, useState } from "react";
import { api, ApiError } from "../../lib/api";
import type { ColdStartItemPrediction, MlopsAuditOverview } from "../../types";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { Badge, DifficultyBadge } from "../../components/ui/Badge";

interface PresetQuestion {
  label: string;
  subject: string;
  chapter: string;
  has_diagram: boolean;
  question_text: string;
  options: [string, string, string, string];
}

const CALIBRATOR_PRESETS: PresetQuestion[] = [
  {
    label: "Physics · Multi-Step Numerical",
    subject: "Physics",
    chapter: "Electrostatics",
    has_diagram: false,
    question_text:
      "Two point charges +4.0 uC and -9.0 uC are separated in vacuum by 0.30 m. Calculate the magnitude of electric field intensity E = k*q/r^2 at the midpoint between the charges (take 1/(4*pi*epsilon_0) = 9 x 10^9 N m^2 C^-2).",
    options: [
      "5.2 x 10^6 N/C",
      "5.2 x 10^3 N/C",
      "2.0 x 10^6 N/C",
      "3.6 x 10^5 N/C",
    ],
  },
  {
    label: "Chemistry · Assertion–Reason",
    subject: "Chemistry",
    chapter: "Chemical & Ionic Equilibrium",
    has_diagram: false,
    question_text:
      "Assertion (A): Addition of solid NH4Cl to an aqueous solution of NH4OH decreases the pH and suppresses degree of ionization. Reason (R): Which statement is NOT incorrect regarding common-ion equilibrium shift in weak electrolyte buffers?",
    options: [
      "Both A and R are true and R is the correct explanation of A",
      "Both A and R are true but R is not the correct explanation of A",
      "A is true but R is false",
      "Both A and R are false",
    ],
  },
  {
    label: "Biology · NCERT Exception",
    subject: "Biology",
    chapter: "Plant Kingdom",
    has_diagram: false,
    question_text:
      "Which of the following is an exception showing diplontic life cycle among algae in NCERT Plant Kingdom classification?",
    options: ["Fucus", "Spirogyra", "Volvox", "Chlamydomonas"],
  },
];

const MODEL_LABELS: Record<string, string> = {
  random_forest_difficulty_classifier: "300-Tree Balanced Random Forest Classifier",
  rf_300_tree_compact_serverless_runtime: "Compact NumPy 300-Tree Serverless Forest",
  bkt_bayesian_knowledge_tracing: "Bayesian Knowledge Tracing + SAINT+ + Ebbinghaus",
  irt_3pl_psychometric_engine: "3PL Item Response Theory (2,471 Calibrated Items)",
  mirt_3d_skill_profiler: "3D Multidimensional IRT (Conceptual / Formula / Numerical)",
  ncert_prerequisite_dag_nodes: "65-Chapter NCERT Prerequisite DAG & Shrinkage",
  linucb_contextual_bandit: "Disjoint LinUCB Contextual Multi-Armed Bandit",
  rag_ncert_socratic_engine: "NCERT Hybrid TF-IDF Vector RAG Socratic Engine",
  dina_8_skill_q_matrix: "8-Skill DINA Cognitive Q-Matrix + van der Linden RT",
  monte_carlo_conformal_neet_simulator: "500-Trial Monte Carlo Conformal & Graph Laplacian",
};

const SLA_LABELS: Record<string, string> = {
  irt_3pl_map_update: "3PL IRT Newton-Raphson MAP Update",
  bkt_saint_ebbinghaus_update: "BKT + SAINT+ + Ebbinghaus Posterior",
  linucb_context_scoring: "LinUCB Ridge Bandit Context Scoring",
  rag_socratic_remediation: "NCERT TF-IDF Vector RAG Remediation",
  in_memory_feature_store: "Low-Latency In-Memory Feature Store",
};

export function ModelAuditView() {
  const [audit, setAudit] = useState<MlopsAuditOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [subject, setSubject] = useState<string>(CALIBRATOR_PRESETS[0].subject);
  const [chapter, setChapter] = useState<string>(CALIBRATOR_PRESETS[0].chapter);
  const [hasDiagram, setHasDiagram] = useState<boolean>(CALIBRATOR_PRESETS[0].has_diagram);
  const [questionText, setQuestionText] = useState<string>(CALIBRATOR_PRESETS[0].question_text);
  const [options, setOptions] = useState<[string, string, string, string]>(CALIBRATOR_PRESETS[0].options);

  const [calibrating, setCalibrating] = useState(false);
  const [prediction, setPrediction] = useState<ColdStartItemPrediction | null>(null);
  const [predError, setPredError] = useState<string | null>(null);

  async function fetchAudit(isRefresh = false) {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const data = await api.getMlopsAuditOverview();
      setAudit(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load MLOps audit telemetry.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  async function runCalibration(
    overrideSubject?: string,
    overrideChapter?: string,
    overrideDiagram?: boolean,
    overrideText?: string,
    overrideOptions?: [string, string, string, string]
  ) {
    const subj = overrideSubject ?? subject;
    const chap = overrideChapter ?? chapter;
    const diag = overrideDiagram ?? hasDiagram;
    const qText = (overrideText ?? questionText).trim();
    const opts = overrideOptions ?? options;

    if (!qText) {
      setPredError("Enter question text before running calibration.");
      return;
    }

    setCalibrating(true);
    setPredError(null);
    try {
      const res = await api.predictColdStartItem({
        question_text: qText,
        options: opts.filter((o) => o.trim().length > 0),
        has_diagram: diag,
        subject: subj,
        chapter: chap,
        topic_tag: chap,
      });
      setPrediction(res);
    } catch (err) {
      setPredError(err instanceof ApiError ? err.message : "Calibration request failed.");
    } finally {
      setCalibrating(false);
    }
  }

  useEffect(() => {
    fetchAudit(false);
    runCalibration(
      CALIBRATOR_PRESETS[0].subject,
      CALIBRATOR_PRESETS[0].chapter,
      CALIBRATOR_PRESETS[0].has_diagram,
      CALIBRATOR_PRESETS[0].question_text,
      CALIBRATOR_PRESETS[0].options
    );
  }, []);

  function applyPreset(p: PresetQuestion) {
    setSubject(p.subject);
    setChapter(p.chapter);
    setHasDiagram(p.has_diagram);
    setQuestionText(p.question_text);
    setOptions([...p.options] as [string, string, string, string]);
    runCalibration(p.subject, p.chapter, p.has_diagram, p.question_text, p.options);
  }

  function handleSubjectChange(nextSubj: string) {
    setSubject(nextSubj);
    const list = audit?.chapters_by_subject?.[nextSubj] ?? [];
    if (list.length > 0) {
      setChapter(list[0]);
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col gap-6">
        <Card className="h-32 animate-pulse bg-midnight-900/5" />
        <Card className="h-72 animate-pulse bg-midnight-900/5" />
        <Card className="h-64 animate-pulse bg-midnight-900/5" />
      </div>
    );
  }

  if (error || !audit) {
    return (
      <Card className="border-signal-incorrect/20 bg-signal-incorrect/5">
        <p className="font-medium text-signal-incorrect">Could not load Model & MLOps Audit.</p>
        <p className="mt-1 text-sm text-midnight-700/70">{error}</p>
        <Button className="mt-4" size="sm" onClick={() => fetchAudit(false)}>
          Retry
        </Button>
      </Card>
    );
  }

  const rfEval = audit.validation_report.random_forest_holdout_evaluation;
  const cvEval = audit.validation_report.stratified_5fold_cross_validation;
  const irtSummary = audit.validation_report.irt_3pl_calibration_summary;
  const ablation = audit.harness_benchmark.linucb_vs_random_ablation;
  const personas = audit.harness_benchmark.persona_simulations;
  const slas = audit.harness_benchmark.latency_sla_benchmarks ?? {};
  const drift = audit.health_and_drift.drift_and_calibration;
  const fstore = audit.health_and_drift.feature_store;
  const corpus = audit.corpus_integrity;
  const subjectChapters = audit.chapters_by_subject[subject] ?? [chapter];

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-accent">
            Enterprise v2 · Full-Stack ML Governance & Verification
          </p>
          <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight text-midnight-950">
            Model & MLOps Audit
          </h1>
          <p className="mt-1 text-sm text-midnight-700/70">
            Live production telemetry, 5-fold cross-validation metrics, multi-persona psychometric benchmarks, and unseen NEET item calibration.
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => fetchAudit(true)}
          disabled={refreshing}
        >
          {refreshing ? "Refreshing…" : "Refresh Live Telemetry"}
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="p-5">
          <p className="text-xs font-medium text-midnight-700/60">RF Holdout Accuracy</p>
          <p className="mt-1 font-display text-2xl font-semibold text-midnight-950">
            {rfEval ? `${(rfEval.holdout_accuracy * 100).toFixed(2)}%` : "—"}
          </p>
          <p className="mt-1 text-xs text-midnight-700/60">
            {rfEval ? `${rfEval.correct_predictions} / ${rfEval.testing_rows} test items` : "—"} · Macro F1{" "}
            {rfEval ? rfEval.macro_f1.toFixed(4) : "—"}
          </p>
        </Card>

        <Card className="p-5">
          <p className="text-xs font-medium text-midnight-700/60">Verified NEET PYQ Corpus</p>
          <p className="mt-1 font-display text-2xl font-semibold text-midnight-950">
            {corpus.verified_questions_in_db.toLocaleString()} PYQs
          </p>
          <p className="mt-1 text-xs text-midnight-700/60">
            {corpus.canonical_chapters_in_db} NCERT chapters · {corpus.dag_edges_count} DAG edges ·{" "}
            {corpus.dag_acyclic ? "Acyclic" : "Cycle Detected"}
          </p>
        </Card>

        <Card className="p-5">
          <p className="text-xs font-medium text-midnight-700/60">120-Round Bandit Regret</p>
          <p className="mt-1 font-display text-2xl font-semibold text-midnight-950">
            {ablation ? ablation.linucb_cumulative_regret.toFixed(2) : "—"} vs{" "}
            {ablation ? ablation.random_cumulative_regret.toFixed(1) : "—"}
          </p>
          <p className="mt-1 text-xs text-midnight-700/60">
            LinUCB ZPD {ablation ? `${(ablation.linucb_mean_zpd_alignment * 100).toFixed(1)}%` : "—"} vs Random{" "}
            {ablation ? `${(ablation.random_mean_zpd_alignment * 100).toFixed(1)}%` : "—"}
          </p>
        </Card>

        <Card className="p-5">
          <p className="text-xs font-medium text-midnight-700/60">Drift & Feature Store SLA</p>
          <p className="mt-1 font-display text-2xl font-semibold text-midnight-950">
            {drift.drift_status ?? "HEALTHY"}
          </p>
          <p className="mt-1 text-xs text-midnight-700/60">
            PSI {(drift.effective_psi ?? 0).toFixed(4)} · p50 {(fstore.p50_ms ?? 0).toFixed(3)} ms ·{" "}
            {drift.sample_size ?? 0} live samples
          </p>
        </Card>
      </div>

      <Card className="p-6">
        <div className="flex flex-col justify-between gap-3 border-b border-midnight-900/[0.06] pb-4 sm:flex-row sm:items-center">
          <div>
            <h2 className="font-display text-lg font-semibold text-midnight-950">
              Live Unseen NEET Question Calibrator (300-Tree RF + Cold-Start 3PL IRT + DINA Q-Matrix)
            </h2>
            <p className="mt-0.5 text-xs text-midnight-700/65">
              Executes live feature extraction, 300-tree Random Forest probability inference, 3PL IRT parameter estimation (a, b, c), and 8-skill DINA Q-vector tagging on any unseen question.
            </p>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {CALIBRATOR_PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => applyPreset(p)}
              className="rounded-lg border border-midnight-900/10 bg-midnight-900/[0.03] px-3 py-1.5 text-xs font-medium text-midnight-800 transition-colors hover:border-midnight-900/25 hover:bg-midnight-900/[0.06]"
            >
              Load Preset: {p.label}
            </button>
          ))}
        </div>

        <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="flex flex-col gap-3 lg:col-span-6">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-midnight-700/70">Subject</label>
                <select
                  value={subject}
                  onChange={(e) => handleSubjectChange(e.target.value)}
                  className="w-full rounded-xl border border-midnight-900/15 bg-white px-3 py-2 text-xs font-medium text-midnight-900 outline-none focus:border-accent"
                >
                  <option value="Physics">Physics</option>
                  <option value="Chemistry">Chemistry</option>
                  <option value="Biology">Biology</option>
                </select>
              </div>

              <div className="sm:col-span-2">
                <label className="mb-1 block text-xs font-medium text-midnight-700/70">NCERT Chapter</label>
                <select
                  value={chapter}
                  onChange={(e) => setChapter(e.target.value)}
                  className="w-full rounded-xl border border-midnight-900/15 bg-white px-3 py-2 text-xs font-medium text-midnight-900 outline-none focus:border-accent"
                >
                  {subjectChapters.map((ch) => (
                    <option key={ch} value={ch}>
                      {ch}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium text-midnight-700/70">
                Unseen Question Stem
              </label>
              <textarea
                rows={4}
                value={questionText}
                onChange={(e) => setQuestionText(e.target.value)}
                className="w-full rounded-xl border border-midnight-900/15 bg-white p-3 text-xs leading-relaxed text-midnight-900 outline-none focus:border-accent"
                placeholder="Paste any unseen NEET Physics, Chemistry, or Biology MCQ stem..."
              />
            </div>

            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {(["A", "B", "C", "D"] as const).map((letter, idx) => (
                <div key={letter}>
                  <label className="mb-1 block text-[11px] font-medium text-midnight-700/60">
                    Option {letter}
                  </label>
                  <input
                    type="text"
                    value={options[idx]}
                    onChange={(e) => {
                      const copy = [...options] as [string, string, string, string];
                      copy[idx] = e.target.value;
                      setOptions(copy);
                    }}
                    className="w-full rounded-lg border border-midnight-900/15 bg-white px-2.5 py-1.5 text-xs text-midnight-900 outline-none focus:border-accent"
                  />
                </div>
              ))}
            </div>

            <div className="mt-1 flex items-center justify-between gap-3">
              <label className="inline-flex items-center gap-2 text-xs font-medium text-midnight-800">
                <input
                  type="checkbox"
                  checked={hasDiagram}
                  onChange={(e) => setHasDiagram(e.target.checked)}
                  className="h-3.5 w-3.5 rounded border-midnight-900/20"
                />
                Contains circuit / molecular / anatomical diagram
              </label>

              <Button size="sm" onClick={() => runCalibration()} disabled={calibrating}>
                {calibrating ? "Calibrating…" : "Calibrate Unseen Item"}
              </Button>
            </div>

            {predError && <p className="text-xs text-signal-incorrect">{predError}</p>}
          </div>

          <div className="flex flex-col gap-4 rounded-xl border border-midnight-900/10 bg-midnight-900/[0.02] p-4 lg:col-span-6">
            {prediction ? (
              <>
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-midnight-900/[0.06] pb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-midnight-700/60">Calibrated Tier:</span>
                    <DifficultyBadge level={prediction.predicted_difficulty} />
                    <Badge className="uppercase">{prediction.mirt_dimension} MIRT</Badge>
                  </div>
                  <span className="font-mono text-[11px] text-midnight-700/60">
                    Complexity {prediction.complexity_score.toFixed(3)} · {prediction.rf_engine}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="rounded-lg border border-midnight-900/[0.06] bg-white p-2.5">
                    <p className="text-[11px] text-midnight-700/60">Discrimination (a)</p>
                    <p className="mt-0.5 font-mono text-base font-semibold text-midnight-950">
                      {prediction.irt_params.a.toFixed(3)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-midnight-900/[0.06] bg-white p-2.5">
                    <p className="text-[11px] text-midnight-700/60">Difficulty (b)</p>
                    <p className="mt-0.5 font-mono text-base font-semibold text-midnight-950">
                      {prediction.irt_params.b >= 0 ? "+" : ""}
                      {prediction.irt_params.b.toFixed(3)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-midnight-900/[0.06] bg-white p-2.5">
                    <p className="text-[11px] text-midnight-700/60">Pseudo-Guessing (c)</p>
                    <p className="mt-0.5 font-mono text-base font-semibold text-midnight-950">
                      {prediction.irt_params.c.toFixed(3)}
                    </p>
                  </div>
                </div>

                {prediction.rf_probabilities && (
                  <div>
                    <p className="mb-1.5 text-xs font-semibold text-midnight-900">
                      300-Tree Random Forest Class Probabilities ({prediction.resolved_metadata.question_type} ·{" "}
                      {prediction.resolved_metadata.ideal_time_sec}s benchmark)
                    </p>
                    <div className="flex flex-col gap-1.5">
                      {(["basic", "medium", "advanced"] as const).map((tierKey) => {
                        const prob = prediction.rf_probabilities?.[tierKey] ?? 0;
                        const pct = Math.round(prob * 1000) / 10;
                        return (
                          <div key={tierKey} className="flex items-center gap-2 text-xs">
                            <span className="w-20 capitalize text-midnight-700">{tierKey}</span>
                            <div className="h-2 flex-1 overflow-hidden rounded-full bg-midnight-900/10">
                              <div
                                className="h-full rounded-full bg-midnight-900"
                                style={{ width: `${Math.min(100, Math.max(1, pct))}%` }}
                              />
                            </div>
                            <span className="w-12 text-right font-mono text-[11px] font-medium text-midnight-900">
                              {pct.toFixed(1)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-2 text-[11px] text-midnight-700/80 sm:grid-cols-4">
                  <div className="rounded-lg bg-white px-2.5 py-1.5">
                    Words: <span className="font-semibold text-midnight-950">{prediction.nlp_features.word_count}</span>
                  </div>
                  <div className="rounded-lg bg-white px-2.5 py-1.5">
                    Formula:{" "}
                    <span className="font-semibold text-midnight-950">
                      {(prediction.nlp_features.formula_density * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="rounded-lg bg-white px-2.5 py-1.5">
                    Numerical:{" "}
                    <span className="font-semibold text-midnight-950">
                      {(prediction.nlp_features.numerical_complexity * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="rounded-lg bg-white px-2.5 py-1.5">
                    Option Sim:{" "}
                    <span className="font-semibold text-midnight-950">
                      {(prediction.nlp_features.distractor_similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div>
                  <p className="mb-1.5 text-xs font-semibold text-midnight-900">
                    Active 8-Skill DINA Cognitive Q-Matrix Vector ({prediction.active_cognitive_skills.length} / 8 required)
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {prediction.active_cognitive_skills.map((sk) => (
                      <span
                        key={sk.skill_id}
                        title={sk.description}
                        className="rounded-md border border-accent/25 bg-accent/10 px-2 py-1 text-[11px] font-medium text-accent-dim"
                      >
                        {sk.name}
                      </span>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <p className="text-xs text-midnight-700/60">Run calibration to inspect model outputs.</p>
            )}
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="p-6">
          <h2 className="font-display text-lg font-semibold text-midnight-950">
            Active 10-Model Production Registry
          </h2>
          <p className="mt-0.5 text-xs text-midnight-700/65">
            Live verification of serialized artifacts, knowledge bases, and psychometric engines loaded in memory.
          </p>
          <div className="mt-4 divide-y divide-midnight-900/[0.06]">
            {Object.entries(audit.health_and_drift.models_loaded).map(([key, val]) => {
              const isActive = typeof val === "number" ? val > 0 : Boolean(val);
              return (
                <div key={key} className="flex items-center justify-between py-2.5 text-xs">
                  <span className="font-medium text-midnight-900">{MODEL_LABELS[key] ?? key}</span>
                  <span
                    className={`rounded-full px-2.5 py-0.5 font-semibold ${
                      isActive
                        ? "bg-signal-correct/10 text-signal-correct"
                        : "bg-signal-incorrect/10 text-signal-incorrect"
                    }`}
                  >
                    {typeof val === "number" ? `${val} NODES` : isActive ? "ACTIVE" : "OFFLINE"}
                  </span>
                </div>
              );
            })}
          </div>
        </Card>

        <Card className="p-6">
          <h2 className="font-display text-lg font-semibold text-midnight-950">
            Component Latency SLAs & Drift Governance
          </h2>
          <p className="mt-0.5 text-xs text-midnight-700/65">
            Measured p50 / p95 / p99 execution latency (ms) and online probability calibration metrics.
          </p>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-midnight-900/10 text-midnight-700/60">
                  <th className="pb-2 font-medium">Subsystem</th>
                  <th className="pb-2 text-right font-medium">p50 (ms)</th>
                  <th className="pb-2 text-right font-medium">p95 (ms)</th>
                  <th className="pb-2 text-right font-medium">p99 (ms)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-midnight-900/[0.06]">
                {Object.entries(slas).map(([k, v]) => {
                  if (typeof v === "boolean" || !v) return null;
                  return (
                    <tr key={k}>
                      <td className="py-2 font-medium text-midnight-900">{SLA_LABELS[k] ?? k}</td>
                      <td className="py-2 text-right font-mono text-midnight-800">{v.p50_ms.toFixed(4)}</td>
                      <td className="py-2 text-right font-mono text-midnight-800">{v.p95_ms.toFixed(4)}</td>
                      <td className="py-2 text-right font-mono text-midnight-800">{v.p99_ms.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2 rounded-xl bg-midnight-900/[0.03] p-3 text-xs sm:grid-cols-4">
            <div>
              <p className="text-[11px] text-midnight-700/60">Online Brier Score</p>
              <p className="mt-0.5 font-mono font-semibold text-midnight-950">
                {(drift.brier_score ?? 0).toFixed(4)}
              </p>
            </div>
            <div>
              <p className="text-[11px] text-midnight-700/60">Expected Calib. Error</p>
              <p className="mt-0.5 font-mono font-semibold text-midnight-950">
                {(drift.expected_calibration_error ?? 0).toFixed(4)}
              </p>
            </div>
            <div>
              <p className="text-[11px] text-midnight-700/60">Cache Hit Rate</p>
              <p className="mt-0.5 font-mono font-semibold text-midnight-950">
                {((fstore.cache_hit_rate ?? 0) * 100).toFixed(1)}% ({fstore.cached_students ?? 0} active)
              </p>
            </div>
            <div>
              <p className="text-[11px] text-midnight-700/60">Corpus Split (P/C/B)</p>
              <p className="mt-0.5 font-mono font-semibold text-midnight-950">
                {corpus.subject_question_counts.Physics ?? 0}/{corpus.subject_question_counts.Chemistry ?? 0}/
                {corpus.subject_question_counts.Biology ?? 0}
              </p>
            </div>
          </div>
        </Card>
      </div>

      {rfEval && cvEval && (
        <Card className="p-6">
          <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
            <div>
              <h2 className="font-display text-lg font-semibold text-midnight-950">
                Supervised Difficulty Classifier: Holdout Confusion Matrix & 5-Fold Stratified CV
              </h2>
              <p className="mt-0.5 text-xs text-midnight-700/65">
                Trained on {rfEval.training_rows.toLocaleString()} stratified NEET PYQs, evaluated on{" "}
                {rfEval.testing_rows} unseen holdout items + 5-fold out-of-fold validation.
              </p>
            </div>
            <Badge>
              5-Fold Mean CV: {(cvEval.mean_cv_accuracy * 100).toFixed(2)}% (±
              {(cvEval.std_cv_accuracy * 100).toFixed(2)}%)
            </Badge>
          </div>

          <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="overflow-x-auto">
              <p className="mb-2 text-xs font-semibold text-midnight-900">Per-Class Holdout Metrics</p>
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-midnight-900/10 text-midnight-700/60">
                    <th className="pb-2 font-medium">Difficulty Tier</th>
                    <th className="pb-2 text-right font-medium">Precision</th>
                    <th className="pb-2 text-right font-medium">Recall</th>
                    <th className="pb-2 text-right font-medium">F1-Score</th>
                    <th className="pb-2 text-right font-medium">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-midnight-900/[0.06]">
                  {Object.entries(rfEval.per_class_metrics).map(([cls, m]) => (
                    <tr key={cls}>
                      <td className="py-2 font-medium capitalize text-midnight-900">{cls}</td>
                      <td className="py-2 text-right font-mono">{(m.precision * 100).toFixed(2)}%</td>
                      <td className="py-2 text-right font-mono">{(m.recall * 100).toFixed(2)}%</td>
                      <td className="py-2 text-right font-mono">{(m.f1_score * 100).toFixed(2)}%</td>
                      <td className="py-2 text-right font-mono">{m.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="overflow-x-auto">
              <p className="mb-2 text-xs font-semibold text-midnight-900">
                Holdout 3x3 Confusion Matrix (Rows = Actual, Cols = Predicted)
              </p>
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-midnight-900/10 text-midnight-700/60">
                    <th className="pb-2 font-medium">Actual \ Pred</th>
                    {rfEval.confusion_matrix_labels.map((lbl) => (
                      <th key={lbl} className="pb-2 text-right font-medium capitalize">
                        {lbl}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-midnight-900/[0.06]">
                  {rfEval.confusion_matrix.map((row, rIdx) => {
                    const rowLabel = rfEval.confusion_matrix_labels[rIdx];
                    return (
                      <tr key={rowLabel}>
                        <td className="py-2 font-medium capitalize text-midnight-900">{rowLabel}</td>
                        {row.map((cell, cIdx) => (
                          <td
                            key={cIdx}
                            className={`py-2 text-right font-mono ${
                              rIdx === cIdx && cell > 0
                                ? "font-semibold text-signal-correct"
                                : cell > 0
                                ? "text-signal-incorrect"
                                : "text-midnight-700/40"
                            }`}
                          >
                            {cell}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              <p className="mt-3 text-[11px] text-midnight-700/65">
                5-Fold Accuracies:{" "}
                <span className="font-mono font-medium text-midnight-900">
                  {cvEval.fold_accuracies.map((a) => `${(a * 100).toFixed(2)}%`).join(" · ")}
                </span>{" "}
                · OOF Multiclass Brier:{" "}
                <span className="font-mono font-medium text-midnight-900">
                  {cvEval.oof_multiclass_brier_score.toFixed(4)}
                </span>
              </p>
            </div>
          </div>

          {irtSummary && (
            <div className="mt-4 flex flex-wrap gap-3 border-t border-midnight-900/[0.06] pt-3 text-xs text-midnight-700/75">
              <span>
                3PL IRT Calibrated Bank:{" "}
                <strong className="text-midnight-950">{irtSummary.calibrated_questions} items</strong>
              </span>
              <span>
                Discrimination a: [{irtSummary.discrimination_a_range[0]}, {irtSummary.discrimination_a_range[1]}]
              </span>
              <span>
                Difficulty b: [{irtSummary.difficulty_b_range[0]}, +{irtSummary.difficulty_b_range[1]}]
              </span>
              <span>
                Guessing c: [{irtSummary.guessing_c_range[0]}, {irtSummary.guessing_c_range[1]}]
              </span>
            </div>
          )}
        </Card>
      )}

      {personas && ablation && (
        <Card className="p-6">
          <h2 className="font-display text-lg font-semibold text-midnight-950">
            Psychometric Persona Harness & LinUCB Contextual Bandit Ablation
          </h2>
          <p className="mt-0.5 text-xs text-midnight-700/65">
            Closed-loop verification across 4 longitudinal student archetypes and 120-round Fisher-Information + ZPD bandit routing.
          </p>

          <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {personas.strong_master && (
              <div className="rounded-xl border border-midnight-900/10 bg-midnight-900/[0.02] p-4">
                <p className="text-xs font-semibold text-midnight-950">Strong Master Persona</p>
                <p className="mt-1 font-mono text-lg font-semibold text-signal-correct">
                  theta +{personas.strong_master.final_theta.toFixed(2)} ({personas.strong_master.estimated_percentile}%ile)
                </p>
                <p className="mt-1 text-[11px] text-midnight-700/70">
                  Promoted to {personas.strong_master.final_level} · BKT {(personas.strong_master.final_mastery_prob * 100).toFixed(0)}%
                </p>
              </div>
            )}

            {personas.foundational_struggler && (
              <div className="rounded-xl border border-midnight-900/10 bg-midnight-900/[0.02] p-4">
                <p className="text-xs font-semibold text-midnight-950">Foundational Struggler</p>
                <p className="mt-1 font-mono text-lg font-semibold text-midnight-950">
                  {personas.foundational_struggler.detected_bottlenecks} Root Bottlenecks
                </p>
                <p className="mt-1 text-[11px] text-midnight-700/70">
                  theta {personas.foundational_struggler.final_theta.toFixed(2)} · Prerequisite DAG remediation triggered
                </p>
              </div>
            )}

            {personas.impulsive_guesser_telemetry_audit && (
              <div className="rounded-xl border border-midnight-900/10 bg-midnight-900/[0.02] p-4">
                <p className="text-xs font-semibold text-midnight-950">Impulsive Guesser Audit</p>
                <p className="mt-1 font-mono text-lg font-semibold text-midnight-950">
                  -{(personas.impulsive_guesser_telemetry_audit.damping_delta * 100).toFixed(1)}% Damped
                </p>
                <p className="mt-1 text-[11px] text-midnight-700/70">
                  Deliberate {(personas.impulsive_guesser_telemetry_audit.deliberate_posterior * 100).toFixed(1)}% vs Rapid-Switch{" "}
                  {(personas.impulsive_guesser_telemetry_audit.impulsive_guess_posterior * 100).toFixed(1)}%
                </p>
              </div>
            )}

            {personas.spaced_repetition_learner && (
              <div className="rounded-xl border border-midnight-900/10 bg-midnight-900/[0.02] p-4">
                <p className="text-xs font-semibold text-midnight-950">168h Ebbinghaus Decay</p>
                <p className="mt-1 font-mono text-lg font-semibold text-midnight-950">
                  {(personas.spaced_repetition_learner.peak_mastery * 100).toFixed(0)}% to{" "}
                  {(personas.spaced_repetition_learner.decayed_mastery_after_168h * 100).toFixed(1)}%
                </p>
                <p className="mt-1 text-[11px] text-midnight-700/70">
                  Retention Urgency {(personas.spaced_repetition_learner.retention_urgency * 100).toFixed(0)}%
                </p>
              </div>
            )}
          </div>

          <div className="mt-4 grid grid-cols-1 gap-3 rounded-xl bg-midnight-950 p-4 text-xs text-white sm:grid-cols-3">
            <div>
              <p className="text-white/60">Mean ZPD Alignment (Target P ≈ 70%)</p>
              <p className="mt-0.5 font-mono text-sm font-semibold">
                LinUCB {(ablation.linucb_mean_zpd_alignment * 100).toFixed(1)}% vs Random{" "}
                {(ablation.random_mean_zpd_alignment * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-white/60">Mean 3PL Fisher Information I(theta)</p>
              <p className="mt-0.5 font-mono text-sm font-semibold">
                LinUCB {ablation.linucb_mean_fisher_info.toFixed(3)} vs Random{" "}
                {ablation.random_mean_fisher_info.toFixed(3)} (
                {(ablation.linucb_mean_fisher_info / Math.max(0.001, ablation.random_mean_fisher_info)).toFixed(1)}x gain)
              </p>
            </div>
            <div>
              <p className="text-white/60">120-Round Cumulative Bandit Regret</p>
              <p className="mt-0.5 font-mono text-sm font-semibold text-emerald-300">
                LinUCB {ablation.linucb_cumulative_regret.toFixed(2)} vs Random{" "}
                {ablation.random_cumulative_regret.toFixed(2)}
              </p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
