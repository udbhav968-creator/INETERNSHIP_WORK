import { useEffect, useState } from "react";
import { api, ApiError } from "../../lib/api";
import {
  OPTION_LETTERS,
  type Analytics,
  type CATAssessment,
  type Difficulty,
  type FatigueAnalysis,
  type PaperHistorySummary,
  type PaperReviewOut,
  type RevisionCalendar,
  type SocraticHintOut,
  type StudentStudyPlan,
  type TopicState,
  type V2StudentInsights,
} from "../../types";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { DifficultyBadge, PaceBadge } from "../../components/ui/Badge";

interface DashboardProps {
  studentId: string;
  onStartPaper: (topicId?: string, totalLength?: number, mode?: "mock_45" | "neet_180" | "diagnostic_15") => void;
}

type LoadState = "loading" | "ready" | "error" | "empty";
type SubjectFilter = "All" | "Physics" | "Chemistry" | "Biology";

function parseExamYear(qid?: string, fallback?: string): string {
  if (!qid) return fallback || "NEET PYQ";
  if (qid.includes("2026")) return "NEET 2026 Re-NEET";
  const m = qid.match(/NEET(\d{4})/i);
  if (m) return `NEET ${m[1]}`;
  return fallback || "NEET PYQ";
}


export function Dashboard({ studentId, onStartPaper }: DashboardProps) {
  const [topics, setTopics] = useState<TopicState[]>([]);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [v2Insights, setV2Insights] = useState<V2StudentInsights | null>(null);
  const [paperHistory, setPaperHistory] = useState<PaperHistorySummary[]>([]);
  const [studyPlan, setStudyPlan] = useState<StudentStudyPlan | null>(null);
  const [catAssessment, setCatAssessment] = useState<CATAssessment | null>(null);
  const [fatigueAnalysis, setFatigueAnalysis] = useState<FatigueAnalysis | null>(null);
  const [revisionCalendar, setRevisionCalendar] = useState<RevisionCalendar | null>(null);
  const [reviewModal, setReviewModal] = useState<PaperReviewOut | null>(null);
  const [loadingReview, setLoadingReview] = useState<number | null>(null);
  const [reviewError, setReviewError] = useState("");
  const [hints, setHints] = useState<Record<string, SocraticHintOut>>({});
  const [loadingHint, setLoadingHint] = useState<Record<string, boolean>>({});
  const [hintErrors, setHintErrors] = useState<Record<string, string>>({});
  const [state, setState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState("");
  const [subjectFilter, setSubjectFilter] = useState<SubjectFilter>("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSubjectQuick, setSelectedSubjectQuick] = useState<SubjectFilter>("All");
  const [selectedChapterQuick, setSelectedChapterQuick] = useState<string>("Mole Concept / Stoichiometry");

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    api
      .getDashboardSnapshot(studentId)
      .then((snapshot) => {
        if (cancelled) return;
        setTopics(snapshot.topics);
        setAnalytics(snapshot.analytics);
        setV2Insights(snapshot.v2_insights);
        setPaperHistory(snapshot.paper_history);
        setStudyPlan(snapshot.study_plan ?? null);
        setCatAssessment(snapshot.cat_assessment ?? null);
        setFatigueAnalysis(snapshot.fatigue_analysis ?? null);
        setRevisionCalendar(snapshot.revision_calendar ?? null);
        setState(snapshot.topics.length === 0 ? "empty" : "ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setErrorMessage(err instanceof ApiError ? err.message : "Something went wrong.");
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [studentId]);

  async function handleOpenReview(paperNumber: number) {
    setLoadingReview(paperNumber);
    setReviewError("");
    try {
      const review = await api.getPaperReview(studentId, paperNumber);
      setReviewModal(review);
    } catch (err) {
      setReviewError(err instanceof ApiError ? err.message : "Could not load paper review.");
    } finally {
      setLoadingReview(null);
    }
  }

  async function handleFetchSocraticHint(questionId: string, selectedOption: string) {
    setLoadingHint((prev) => ({ ...prev, [questionId]: true }));
    setHintErrors((prev) => ({ ...prev, [questionId]: "" }));
    try {
      const hint = await api.getSocraticHint(questionId, selectedOption);
      setHints((prev) => ({ ...prev, [questionId]: hint }));
    } catch (err) {
      setHintErrors((prev) => ({
        ...prev,
        [questionId]: err instanceof ApiError ? err.message : "Could not load Socratic hint.",
      }));
    } finally {
      setLoadingHint((prev) => ({ ...prev, [questionId]: false }));
    }
  }

  const allSyllabusChapters = v2Insights?.recommended_unlock_topics ?? [];
  const filteredChapters = allSyllabusChapters.filter((ch) => {
    const matchesSubject = subjectFilter === "All" || ch.subject === subjectFilter;
    const matchesQuery =
      !searchQuery.trim() ||
      ch.topic_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (ch.cluster ?? "").toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSubject && matchesQuery;
  });

  const subjectCounts = {
    All: allSyllabusChapters.length,
    Physics: allSyllabusChapters.filter((c) => c.subject === "Physics").length,
    Chemistry: allSyllabusChapters.filter((c) => c.subject === "Chemistry").length,
    Biology: allSyllabusChapters.filter((c) => c.subject === "Biology").length,
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-2">
        <h1 className="font-display text-3xl font-semibold tracking-tight text-midnight-950 md:text-4xl">
          Welcome back.
        </h1>
        <p className="text-midnight-700/70">
          Here's where your preparation stands right now across all 65 NCERT chapters, powered by 3PL IRT, Bayesian Knowledge Tracing & LinUCB.
        </p>
      </div>

      <Card className="flex flex-col items-start justify-between gap-4 bg-midnight-950 p-8 text-white md:flex-row md:items-center">
        <div>
          <p className="text-sm font-medium text-white/70">Ready for your next session?</p>
          <p className="mt-1 font-display text-xl font-semibold text-white">
            Your next paper adapts to your Zone of Proximal Development (target P ≈{" "}
            {Math.round((v2Insights?.zpd_target_success_prob ?? 0.7) * 100)}%).
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button
            variant="secondary"
            onClick={() => onStartPaper(undefined, 45, "mock_45")}
            className="shrink-0 font-semibold"
          >
            Start 45-Q Adaptive Mock
          </Button>
          <Button
            onClick={() => onStartPaper(undefined, 180, "neet_180")}
            className="shrink-0 bg-accent font-semibold text-white hover:bg-accent-dim shadow-md"
          >
            Start 180-Q Full NEET Exam
          </Button>
        </div>
      </Card>

      <Card className="border border-midnight-900/10 bg-surface-raised p-5 shadow-sm">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-base font-bold text-accent">
              🎯
            </div>
            <div>
              <p className="text-sm font-semibold text-midnight-950">
                Chapterwise PYQ Practice Drill (All 65 NCERT Chapters)
              </p>
              <p className="text-xs text-midnight-700/70">
                Select any chapter across Biology, Chemistry, or Physics to practice authentic historical NEET PYQs (15 questions).
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={selectedSubjectQuick}
              onChange={(e) => {
                const subj = e.target.value as SubjectFilter;
                setSelectedSubjectQuick(subj);
                const firstForSubj = allSyllabusChapters.find(
                  (c) => subj === "All" || c.subject === subj
                );
                if (firstForSubj) setSelectedChapterQuick(firstForSubj.topic_id);
              }}
              className="rounded-xl border border-midnight-900/15 bg-white px-3 py-2 text-xs font-semibold text-midnight-900 outline-none focus:border-accent"
            >
              <option value="All">All Subjects (65 Chapters)</option>
              <option value="Biology">Biology (20 Chapters)</option>
              <option value="Chemistry">Chemistry (23 Chapters)</option>
              <option value="Physics">Physics (22 Chapters)</option>
            </select>
            <select
              value={selectedChapterQuick}
              onChange={(e) => setSelectedChapterQuick(e.target.value)}
              className="max-w-[200px] truncate rounded-xl border border-midnight-900/15 bg-white px-3 py-2 text-xs font-semibold text-midnight-950 outline-none focus:border-accent sm:max-w-xs"
            >
              {allSyllabusChapters
                .filter((c) => selectedSubjectQuick === "All" || c.subject === selectedSubjectQuick)
                .map((c) => (
                  <option key={c.topic_id} value={c.topic_id}>
                    {c.topic_id} ({c.subject})
                  </option>
                ))}
            </select>
            <Button
              onClick={() => {
                if (selectedChapterQuick) {
                  onStartPaper(selectedChapterQuick, 15, "diagnostic_15");
                }
              }}
              className="shrink-0 bg-midnight-950 text-xs font-semibold text-white hover:bg-midnight-800"
            >
              Start Chapter Drill →
            </Button>
          </div>
        </div>
      </Card>

      {state === "loading" && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Card key={i} className="h-28 animate-pulse bg-midnight-900/5" />
          ))}
        </div>
      )}

      {state === "error" && (
        <Card className="border-signal-incorrect/20 bg-signal-incorrect/5">
          <p className="font-medium text-signal-incorrect">Couldn't load your progress.</p>
          <p className="mt-1 text-sm text-midnight-700/70">{errorMessage}</p>
        </Card>
      )}

      {state === "empty" && (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <SummaryStat label="NCERT Chapters Mapped" value={allSyllabusChapters.length} />
            <SummaryStat
              label="Mean Empirical BKT Prior"
              value={`${Math.round(
                (allSyllabusChapters.length > 0
                  ? allSyllabusChapters.reduce((sum, ch) => sum + ch.current_mastery, 0) /
                    allSyllabusChapters.length
                  : 0.405) * 100
              )}%`}
            />
            <SummaryStat
              label="Baseline NEET Projection"
              value={v2Insights?.predicted_neet_score_720 != null ? `${v2Insights.predicted_neet_score_720} / 720` : "—"}
            />
            <SummaryStat
              label="3PL IRT Prior Percentile"
              value={v2Insights?.estimated_neet_percentile != null ? `${v2Insights.estimated_neet_percentile}%ile` : "—"}
            />
          </div>

          <Card className="text-center">
            <p className="font-medium text-midnight-900">No study history yet — Cold-Start Prior Initialized.</p>
            <p className="mt-1 text-sm text-midnight-700/70">
              Click "Start sample paper" above and your 3PL IRT ability (θ), BKT mastery, and prerequisite diagnostics will calibrate in real time.
            </p>
          </Card>

          {allSyllabusChapters.length > 0 && (
            <div>
              <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
                Top Recommended Starting Chapters (Zone of Proximal Development)
              </h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {allSyllabusChapters.slice(0, 6).map((rec) => (
                  <Card key={rec.topic_id} className="flex flex-col justify-between gap-2 p-4">
                    <div>
                      <span className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                        {rec.subject} {rec.cluster ? `· ${rec.cluster}` : ""}
                      </span>
                      <p className="mt-0.5 font-medium text-midnight-950">{rec.topic_id}</p>
                      <p className="mt-1 text-xs text-midnight-700/70">
                        Readiness: {Math.round(rec.readiness * 100)}% · Weightage: {rec.neet_weightage ?? 1.5}% · ZPD: {rec.zpd_priority_score.toFixed(2)}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => onStartPaper(rec.topic_id)}
                      className="mt-1 self-start text-xs font-semibold text-accent hover:underline"
                    >
                      Practice chapter →
                    </button>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {state === "ready" && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <SummaryStat label="Questions attempted" value={analytics?.total_attempts ?? 0} />
            <SummaryStat
              label="Accuracy"
              value={analytics ? `${Math.round(analytics.accuracy * 100)}%` : "—"}
            />
            <SummaryStat
              label="Predicted NEET Score"
              value={v2Insights?.predicted_neet_score_720 != null ? `${v2Insights.predicted_neet_score_720} / 720` : "—"}
            />
            <SummaryStat
              label="3PL IRT Percentile"
              value={v2Insights?.estimated_neet_percentile != null ? `${v2Insights.estimated_neet_percentile}%ile` : `${topics.length} topics`}
            />
          </div>

          {analytics && analytics.subject_breakdown.length > 0 && (
            <div>
              <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
                Subject Performance Breakdown
              </h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                {analytics.subject_breakdown.map((s) => (
                  <Card key={s.subject} className="flex flex-col gap-1 p-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                      {s.subject}
                    </p>
                    <p className="font-display text-2xl font-semibold text-midnight-950">
                      {Math.round(s.accuracy * 100)}%
                    </p>
                    <p className="text-xs text-midnight-700/70">
                      {s.correct} / {s.attempts} correct
                    </p>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {v2Insights?.mirt_3d_profile && (
            <div>
              <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
                3D Multidimensional IRT (MIRT) Skill Profile
              </h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <Card className="flex flex-col gap-1 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                    Conceptual Recall (θ)
                  </p>
                  <p className="font-display text-2xl font-semibold text-midnight-950">
                    {v2Insights.mirt_3d_profile.theta_conceptual >= 0
                      ? `+${v2Insights.mirt_3d_profile.theta_conceptual.toFixed(2)}`
                      : v2Insights.mirt_3d_profile.theta_conceptual.toFixed(2)}
                  </p>
                  <p className="text-xs text-midnight-700/70">
                    {v2Insights.mirt_3d_profile.attempts_conceptual} conceptual items calibrated
                  </p>
                </Card>
                <Card className="flex flex-col gap-1 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                    Formula Application (θ)
                  </p>
                  <p className="font-display text-2xl font-semibold text-midnight-950">
                    {v2Insights.mirt_3d_profile.theta_formula >= 0
                      ? `+${v2Insights.mirt_3d_profile.theta_formula.toFixed(2)}`
                      : v2Insights.mirt_3d_profile.theta_formula.toFixed(2)}
                  </p>
                  <p className="text-xs text-midnight-700/70">
                    {v2Insights.mirt_3d_profile.attempts_formula} formula items calibrated
                  </p>
                </Card>
                <Card className="flex flex-col gap-1 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                    Numerical Speed & Precision (θ)
                  </p>
                  <p className="font-display text-2xl font-semibold text-midnight-950">
                    {v2Insights.mirt_3d_profile.theta_numerical >= 0
                      ? `+${v2Insights.mirt_3d_profile.theta_numerical.toFixed(2)}`
                      : v2Insights.mirt_3d_profile.theta_numerical.toFixed(2)}
                  </p>
                  <p className="text-xs text-midnight-700/70">
                    {v2Insights.mirt_3d_profile.attempts_numerical} numerical items calibrated
                  </p>
                </Card>
              </div>
            </div>
          )}

          {v2Insights && v2Insights.root_cause_bottlenecks.length > 0 && (
            <Card className="border-amber-500/20 bg-amber-500/5 p-5">
              <h2 className="font-display text-base font-semibold text-midnight-950">
                Prerequisite Knowledge Graph Diagnostics
              </h2>
              <div className="mt-3 flex flex-col gap-2.5">
                {v2Insights.root_cause_bottlenecks.map((b, idx) => (
                  <div
                    key={idx}
                    className="flex flex-col justify-between gap-3 rounded-xl border border-amber-500/20 bg-amber-500/10 p-3.5 sm:flex-row sm:items-center"
                  >
                    <p className="text-sm leading-snug text-midnight-900">
                      <span className="font-semibold">{b.weak_topic}</span> is bottlenecked by foundational chapter{" "}
                      <span className="font-bold underline text-midnight-950">{b.root_cause_prerequisite}</span> —{" "}
                      <span className="text-midnight-800">{b.recommendation}</span>
                    </p>
                    <button
                      type="button"
                      onClick={() => onStartPaper(b.root_cause_prerequisite, 15, "diagnostic_15")}
                      className="shrink-0 self-start rounded-lg bg-midnight-950 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition hover:bg-midnight-800 sm:self-center"
                    >
                      Remediate Prerequisite →
                    </button>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {v2Insights?.spaced_repetition_reviews && v2Insights.spaced_repetition_reviews.length > 0 && (
            <div>
              <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
                Ebbinghaus Spaced Repetition & Memory Retention Queue
              </h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {v2Insights.spaced_repetition_reviews.map((rev) => (
                  <Card key={rev.topic_id} className="flex flex-col justify-between gap-2 p-4">
                    <div>
                      <p className="font-medium text-midnight-950">{rev.topic_id}</p>
                      <p className="mt-1 text-xs text-midnight-700/75">
                        Peak Mastery: <strong>{Math.round(rev.peak_mastery * 100)}%</strong> → Decayed:{" "}
                        <strong>{Math.round(rev.decayed_mastery * 100)}%</strong>
                      </p>
                      <p className="mt-0.5 text-[11px] text-midnight-700/60">
                        Elapsed: {rev.elapsed_hours.toFixed(1)}h · Retention Urgency: {Math.round(rev.retention_urgency * 100)}%
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => onStartPaper(rev.topic_id, 15, "diagnostic_15")}
                      className="self-start text-xs font-semibold text-accent hover:underline"
                    >
                      Review chapter →
                    </button>
                  </Card>
                ))}
              </div>
            </div>
          )}

          <div>
            <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
              Active Chapters Practiced ({topics.length})
            </h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {topics.map((t) => (
                <Card key={t.topic_id} className="flex flex-col gap-2 p-5">
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-midnight-900">{t.topic_id}</p>
                    <DifficultyBadge level={t.level} />
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <PaceBadge pace={t.pace_flag} />
                    <div className="flex items-center gap-3 text-xs font-medium text-midnight-700/70">
                      {typeof t.mastery_prob === "number" && (
                        <span>
                          BKT Mastery: {Math.round(t.mastery_prob * 100)}%
                          {typeof t.confidence_interval === "number"
                            ? ` (±${Math.round(t.confidence_interval * 100)}%)`
                            : ""}
                        </span>
                      )}
                      {typeof t.irt_theta === "number" && (
                        <span>θ: {t.irt_theta >= 0 ? `+${t.irt_theta.toFixed(2)}` : t.irt_theta.toFixed(2)}</span>
                      )}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => onStartPaper(t.topic_id)}
                    className="mt-1 self-start text-xs font-semibold text-accent hover:underline"
                  >
                    Practice chapter →
                  </button>
                </Card>
              ))}
            </div>
          </div>

          {paperHistory.length > 0 && (
            <div>
              <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
                Completed Papers & Review Archive ({paperHistory.length})
              </h2>
              {reviewError && (
                <p className="mb-2 text-xs text-signal-incorrect">{reviewError}</p>
              )}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {paperHistory.map((p) => (
                  <Card key={p.paper_number} className="flex flex-col justify-between gap-3 p-4">
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="font-display font-semibold text-midnight-950">
                          Sample Paper {p.paper_number}
                        </span>
                        <span
                          className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                            p.score_percent >= 70
                              ? "bg-emerald-500/10 text-emerald-800"
                              : p.score_percent >= 40
                              ? "bg-amber-500/10 text-amber-800"
                              : "bg-rose-500/10 text-rose-800"
                          }`}
                        >
                          {p.score_percent}%
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-midnight-700/70">
                        {p.correct_count} / {p.total_questions} correct · {Math.max(1, Math.round(p.total_time_seconds / 60))} min total
                      </p>
                      {p.topics.length > 0 && (
                        <p className="mt-1.5 text-[11px] text-midnight-700/60 truncate">
                          {p.topics.slice(0, 3).join(", ")}
                          {p.topics.length > 3 ? ` +${p.topics.length - 3} more` : ""}
                        </p>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => handleOpenReview(p.paper_number)}
                      disabled={loadingReview === p.paper_number}
                      className="self-start text-xs font-semibold text-accent hover:underline disabled:opacity-50"
                    >
                      {loadingReview === p.paper_number
                        ? "Loading review…"
                        : "Review answers & Socratic hints →"}
                    </button>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {analytics?.recent_attempts && analytics.recent_attempts.length > 0 && (
            <div>
              <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
                Recent Attempt Activity
              </h2>
              <Card className="divide-y divide-midnight-900/[0.06] p-0 overflow-hidden">
                {analytics.recent_attempts.slice(0, 6).map((att, i) => (
                  <div key={i} className="flex items-center justify-between px-5 py-3 text-sm">
                    <div className="flex items-center gap-3">
                      <span
                        className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                          att.is_correct
                            ? "bg-signal-correct/15 text-signal-correct"
                            : "bg-signal-incorrect/15 text-signal-incorrect"
                        }`}
                      >
                        {att.is_correct ? "✓" : "✗"}
                      </span>
                      <div>
                        <div className="flex items-center gap-1.5">
                          <span className="inline-flex items-center rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] font-bold text-emerald-800 border border-emerald-500/20">
                            {parseExamYear(att.question_id)}
                          </span>
                          <p className="font-medium text-midnight-950">{att.topic_id}</p>
                        </div>
                        <p className="text-[11px] font-mono text-midnight-700/50">{att.question_id}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-midnight-700/70">
                      <span className="rounded-md bg-midnight-900/5 px-2 py-0.5 font-mono">
                        {att.time_taken_seconds}s
                      </span>
                      <span className="text-midnight-700/50">
                        {new Date(att.timestamp).toLocaleDateString(undefined, {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                    </div>
                  </div>
                ))}
              </Card>
            </div>
          )}
        </>
      )}

      {v2Insights?.monte_carlo_conformal_simulator && (
        <div className="flex flex-col gap-4">
          <div>
            <h2 className="font-display text-lg font-semibold text-midnight-950">
              Monte Carlo Conformal NEET Simulator & Negative-Marking Optimizer ({v2Insights.monte_carlo_conformal_simulator.n_trials} Trials)
            </h2>
            <p className="text-xs text-midnight-700/70">
              Split conformal prediction bounds across 180 blueprint questions (+4 / −1 marking) with decision-theoretic skip optimization (E[V] = 5P − 1) & Graph Laplacian Heat-Kernel diffusion leverage.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Card className="flex flex-col gap-1 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                90% Conformal Score Interval
              </p>
              <p className="font-display text-2xl font-semibold text-midnight-950">
                {v2Insights.monte_carlo_conformal_simulator.conformal_90_interval[0]} –{" "}
                {v2Insights.monte_carlo_conformal_simulator.conformal_90_interval[1]}
              </p>
              <p className="text-xs text-midnight-700/70">
                50% IQR: {v2Insights.monte_carlo_conformal_simulator.interquartile_50_interval[0]}–
                {v2Insights.monte_carlo_conformal_simulator.interquartile_50_interval[1]} (±
                {v2Insights.monte_carlo_conformal_simulator.score_std_dev} SD)
              </p>
            </Card>

            <Card className="flex flex-col gap-1 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                Optimal Skip Policy Score
              </p>
              <p className="font-display text-2xl font-semibold text-midnight-950">
                {v2Insights.monte_carlo_conformal_simulator.negative_marking_optimizer.optimal_skip_expected_score_720} / 720
              </p>
              <p className="text-xs text-midnight-700/70">
                +{v2Insights.monte_carlo_conformal_simulator.negative_marking_optimizer.expected_marks_gained_by_skipping} marks gained by skipping P &lt;{" "}
                {Math.round(
                  v2Insights.monte_carlo_conformal_simulator.negative_marking_optimizer.optimal_skip_threshold_prob * 100
                )}%
              </p>
            </Card>

            <Card className="flex flex-col gap-1 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                Optimal 180-Q Attempt Split
              </p>
              <p className="font-display text-2xl font-semibold text-midnight-950">
                {v2Insights.monte_carlo_conformal_simulator.negative_marking_optimizer.recommended_attempt_questions_per_180} Attempt /{" "}
                {v2Insights.monte_carlo_conformal_simulator.negative_marking_optimizer.recommended_skip_questions_per_180} Skip
              </p>
              <p className="text-xs text-midnight-700/70">
                Avoids negative-EV distractor traps
              </p>
            </Card>

            <Card className="flex flex-col gap-1 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                Cutoff Probabilities
              </p>
              <p className="font-display text-lg font-semibold text-midnight-950">
                Qualifying (≥165): {v2Insights.monte_carlo_conformal_simulator.cutoff_probabilities.qualifying_cutoff_165_percent}%
              </p>
              <p className="text-xs text-midnight-700/70">
                Govt MBBS (≥550): {v2Insights.monte_carlo_conformal_simulator.cutoff_probabilities.state_govt_mbbs_550_percent}% · AIIMS (≥650):{" "}
                {v2Insights.monte_carlo_conformal_simulator.cutoff_probabilities.top_aiims_650_percent}%
              </p>
            </Card>
          </div>

          {v2Insights.monte_carlo_conformal_simulator.national_air_projection && (
            <Card className="flex flex-col gap-3 p-4 bg-midnight-50/50">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-midnight-100 pb-2.5">
                <div>
                  <h3 className="font-display text-sm font-semibold text-midnight-950">
                    24-Lakh All-India Rank (AIR) & Top Medical College Projections
                  </h3>
                  <p className="text-xs text-midnight-700/70">
                    Calibrated on official NTA score-percentile curves with 90% conformal uncertainty intervals.
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <p className="text-[11px] uppercase tracking-wide text-midnight-700/60 font-medium">90% Conformal AIR Range</p>
                    <p className="font-display text-base font-semibold text-midnight-950">
                      AIR {v2Insights.monte_carlo_conformal_simulator.national_air_projection.conformal_air_interval[0].toLocaleString()} – {v2Insights.monte_carlo_conformal_simulator.national_air_projection.conformal_air_interval[1].toLocaleString()}
                    </p>
                  </div>
                  <div className="text-right border-l border-midnight-200 pl-3">
                    <p className="text-[11px] uppercase tracking-wide text-midnight-700/60 font-medium">Projected Percentile</p>
                    <p className="font-display text-base font-semibold text-midnight-950">
                      {v2Insights.monte_carlo_conformal_simulator.national_air_projection.projected_percentile_median}%ile
                    </p>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
                {v2Insights.monte_carlo_conformal_simulator.national_air_projection.college_admission_probabilities.map((col) => (
                  <div key={col.college_id} className="flex flex-col justify-between rounded-lg border border-midnight-100 bg-white p-2.5 text-center">
                    <p className="text-[11px] font-medium text-midnight-900 leading-tight truncate" title={col.college_tier}>
                      {col.college_tier}
                    </p>
                    <div className="my-1">
                      <span className={`inline-block rounded px-1.5 py-0.5 text-xs font-semibold ${
                        col.status === "SECURE" ? "bg-emerald-100 text-emerald-800" :
                        col.status === "COMPETITIVE" ? "bg-sky-100 text-sky-800" :
                        col.status === "REACH" ? "bg-amber-100 text-amber-800" :
                        "bg-midnight-100 text-midnight-600"
                      }`}>
                        {col.admission_probability_percent}%
                      </span>
                    </div>
                    <p className="text-[10px] text-midnight-500">
                      Cutoff: {col.target_score_720}+
                    </p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {v2Insights.monte_carlo_conformal_simulator.high_yield_leverage_chapters.length > 0 && (
            <div>
              <h3 className="mb-2.5 font-display text-sm font-semibold text-midnight-950">
                Graph Laplacian Heat-Kernel Leverage: Highest Expected +Marks Chapters
              </h3>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {v2Insights.monte_carlo_conformal_simulator.high_yield_leverage_chapters.slice(0, 3).map((lev) => (
                  <Card key={lev.topic_id} className="flex flex-col justify-between gap-2 p-4">
                    <div>
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[11px] font-semibold uppercase tracking-wide text-midnight-700/60">
                          {lev.subject} · {lev.cluster}
                        </span>
                        <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-800">
                          +{lev.total_expected_marks_gain.toFixed(1)} NEET Marks
                        </span>
                      </div>
                      <p className="mt-1 font-medium text-midnight-950">{lev.topic_id}</p>
                      <p className="mt-1 text-xs text-midnight-700/75">
                        Direct Gain: <strong>+{lev.direct_marks_gain.toFixed(1)}</strong> · Downstream Diffusion:{" "}
                        <strong>+{lev.downstream_diffused_marks_gain.toFixed(1)}</strong> ({lev.unlocks_downstream_count} dependent chapters)
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => onStartPaper(lev.topic_id, 15, "diagnostic_15")}
                      className="self-start text-xs font-semibold text-accent hover:underline"
                    >
                      Unlock +{lev.total_expected_marks_gain.toFixed(0)} marks →
                    </button>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {v2Insights?.dina_cognitive_profile && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-end">
            <div>
              <h2 className="font-display text-lg font-semibold text-midnight-950">
                8-Skill DINA Cognitive Q-Matrix & van der Linden Response-Time (τ) Diagnosis
              </h2>
              <p className="text-xs text-midnight-700/70">
                {v2Insights.dina_cognitive_profile.primary_cognitive_bottleneck.diagnosis}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs font-medium text-midnight-800">
              <span className="rounded-xl bg-midnight-900/5 px-3 py-1">
                Latent Speed τ:{" "}
                <strong>
                  {v2Insights.dina_cognitive_profile.speed_ability_profile.latent_speed_tau >= 0
                    ? `+${v2Insights.dina_cognitive_profile.speed_ability_profile.latent_speed_tau.toFixed(2)}`
                    : v2Insights.dina_cognitive_profile.speed_ability_profile.latent_speed_tau.toFixed(2)}
                </strong>{" "}
                ({v2Insights.dina_cognitive_profile.speed_ability_profile.speed_percentile}%ile)
              </span>
              <span className="rounded-xl bg-midnight-900/5 px-3 py-1">
                Fluent: <strong>{v2Insights.dina_cognitive_profile.speed_ability_profile.behavior_classification.fluent_mastery}</strong> ·
                Slips: <strong>{v2Insights.dina_cognitive_profile.speed_ability_profile.behavior_classification.careless_slip}</strong> ·
                Struggle: <strong>{v2Insights.dina_cognitive_profile.speed_ability_profile.behavior_classification.conceptual_struggle}</strong>
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {v2Insights.dina_cognitive_profile.skills.map((sk) => (
              <Card key={sk.skill_id} className="flex flex-col justify-between gap-1.5 p-4">
                <div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-display text-lg font-semibold text-midnight-950">
                      {Math.round(sk.mastery_probability * 100)}%
                    </span>
                    <span className="rounded-full bg-midnight-900/5 px-2 py-0.5 text-[11px] font-medium capitalize text-midnight-800">
                      {sk.status.replace(/_/g, " ")}
                    </span>
                  </div>
                  <p className="mt-0.5 text-sm font-medium text-midnight-900">{sk.name}</p>
                  <p className="mt-0.5 text-[11px] text-midnight-700/65">{sk.description}</p>
                </div>
                <p className="border-t border-midnight-900/[0.06] pt-1.5 text-[11px] text-midnight-700/60">
                  {sk.attempts_count > 0
                    ? `${sk.correct_count}/${sk.attempts_count} items · ±${Math.round(sk.posterior_sd * 100)}% SD`
                    : `${sk.corpus_pyq_count ?? 0} NEET PYQs mapped · ±${Math.round(sk.posterior_sd * 100)}% SD`}
                </p>
              </Card>
            ))}
          </div>
        </div>
      )}

      {catAssessment && (
        <Card className="flex flex-col gap-4 border border-midnight-900/10 p-6">
          <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="font-display text-lg font-semibold text-midnight-950">
                  CAT Computerized Adaptive Testing Assessment & Cognitive Stamina
                </h2>
                <span className="rounded-full bg-accent/10 px-2.5 py-0.5 text-xs font-semibold text-accent-dim uppercase">
                  {catAssessment.assessment_status.replace(/_/g, " ")}
                </span>
              </div>
              <p className="mt-0.5 text-xs text-midnight-700/70">
                Item-level Fisher Information stopping rule (target SEM ≤ {catAssessment.target_sem_threshold}) with response latency stamina diagnosis.
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs font-medium text-midnight-800">
              <span className="rounded-xl bg-midnight-900/5 px-3 py-1">
                Items Evaluated: <strong>{catAssessment.items_administered}</strong>
              </span>
              <span className="rounded-xl bg-midnight-900/5 px-3 py-1">
                Fisher Info: <strong>{catAssessment.cumulative_fisher_information}</strong>
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-xl bg-midnight-900/5 p-3.5">
              <span className="text-[11px] font-medium text-midnight-700/70 uppercase">Standard Error (SEM θ)</span>
              <p className="font-display text-xl font-bold text-midnight-950">
                ±{catAssessment.sem_theta.toFixed(3)}
              </p>
              <p className="mt-1 text-[11px] text-midnight-700/60">
                {catAssessment.stopping_rule_met
                  ? "Confidence stopping threshold met"
                  : `Needs SEM ≤ ${catAssessment.target_sem_threshold} for 95% certainty`}
              </p>
            </div>

            <div className="rounded-xl bg-midnight-900/5 p-3.5">
              <span className="text-[11px] font-medium text-midnight-700/70 uppercase">95% Latent Ability CI</span>
              <p className="font-display text-xl font-bold text-midnight-950">
                [{catAssessment.confidence_interval_95[0].toFixed(2)}, {catAssessment.confidence_interval_95[1].toFixed(2)}]
              </p>
              <p className="mt-1 text-[11px] text-midnight-700/60">
                Estimated θ: {catAssessment.latent_theta >= 0 ? `+${catAssessment.latent_theta.toFixed(2)}` : catAssessment.latent_theta.toFixed(2)}
              </p>
            </div>

            <div className="rounded-xl bg-midnight-900/5 p-3.5">
              <span className="text-[11px] font-medium text-midnight-700/70 uppercase">Cognitive Fatigue Index</span>
              <p className="font-display text-xl font-bold text-midnight-950">
                {fatigueAnalysis ? `${Math.round(fatigueAnalysis.fatigue_index * 100)}%` : "0%"}
              </p>
              <p className="mt-1 text-[11px] text-midnight-700/60">
                Pacing Trend: <strong>{fatigueAnalysis?.latency_trend ?? "STABLE"}</strong>
              </p>
            </div>

            <div className="rounded-xl bg-midnight-900/5 p-3.5">
              <span className="text-[11px] font-medium text-midnight-700/70 uppercase">Speed vs Accuracy</span>
              <p className="font-display text-xl font-bold text-midnight-950">
                {fatigueAnalysis ? `${fatigueAnalysis.early_accuracy_pct}% → ${fatigueAnalysis.late_accuracy_pct}%` : "0%"}
              </p>
              <p className="mt-1 text-[11px] text-midnight-700/60">
                Rapid Guesses (&lt;15s): <strong>{fatigueAnalysis?.rapid_guess_count ?? 0}</strong>
              </p>
            </div>
          </div>

          {fatigueAnalysis?.pacing_coaching_advice && (
            <div className="rounded-xl border border-midnight-900/10 bg-white p-3 text-xs text-midnight-800">
              <strong>Coaching Advice:</strong> {fatigueAnalysis.pacing_coaching_advice}
            </div>
          )}
        </Card>
      )}

      {revisionCalendar && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col justify-between gap-1 sm:flex-row sm:items-end">
            <div>
              <h2 className="font-display text-lg font-semibold text-midnight-950">
                14-Day Adaptive Study Calendar & Revision Roadmap
              </h2>
              <p className="text-xs text-midnight-700/70">
                Synthesized daily drill schedule optimized by Ebbinghaus forgetting urgency, Prerequisite DAG leverage, and NEET blueprint weightage.
              </p>
            </div>
            <span className="rounded-xl bg-midnight-900/5 px-3 py-1 text-xs font-semibold text-midnight-800">
              14 Actionable Days Planned
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            {revisionCalendar.daily_schedule.map((item) => {
              let badgeColor = "bg-accent/10 text-accent-dim";
              if (item.action_type === "REMEDIATE_PREREQUISITE") badgeColor = "bg-amber-500/10 text-amber-900";
              if (item.action_type === "SPACED_REPETITION_REVIEW") badgeColor = "bg-purple-500/10 text-purple-900";

              return (
                <Card key={item.day} className="flex flex-col justify-between gap-2.5 p-4">
                  <div>
                    <div className="flex items-center justify-between gap-1">
                      <span className="font-display text-sm font-bold text-midnight-900">
                        {item.date_label}
                      </span>
                      <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${badgeColor}`}>
                        {item.action_type.replace(/_/g, " ")}
                      </span>
                    </div>
                    <p className="mt-1.5 font-medium text-midnight-950 text-sm">{item.chapter}</p>
                    <div className="mt-1 flex items-center justify-between text-[11px] text-midnight-700/70">
                      <span>{item.subject}</span>
                      <span className="font-semibold text-emerald-800">+{item.potential_marks_gain} Marks</span>
                    </div>
                    <p className="mt-1.5 text-[11px] text-midnight-700/65 line-clamp-2">
                      {item.reason}
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() => onStartPaper(item.chapter, 15, "diagnostic_15")}
                    className="self-start text-xs font-semibold text-accent hover:underline"
                  >
                    Start 15-Q Drill →
                  </button>
                </Card>
              );
            })}
          </div>
        </div>
      )}

      {studyPlan && (state === "ready" || state === "empty") && (
        <div className="flex flex-col gap-5">
          <Card className="flex flex-col gap-5 p-6">
            <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-end">
              <div>
                <h2 className="font-display text-xl font-semibold text-midnight-950">
                  ML Syllabus Coverage, Less-Learned Diagnostics & Improvement Plan
                </h2>
                <p className="text-xs text-midnight-700/70">
                  Real-time 65-chapter syllabus progress, Ebbinghaus decay tracking, missed-PYQ retest queue, and Graph Laplacian +Marks prescription.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs font-semibold">
                <span className="rounded-xl bg-midnight-950 px-3 py-1 text-white">
                  Covered: {studyPlan.syllabus_coverage.practiced_chapters} / {studyPlan.syllabus_coverage.total_chapters} ({studyPlan.syllabus_coverage.coverage_percent}%)
                </span>
                <span className="rounded-xl bg-emerald-500/10 px-2.5 py-1 text-emerald-800">
                  Mastered (≥75%): {studyPlan.syllabus_coverage.mastered_chapters}
                </span>
                <span className="rounded-xl bg-amber-500/10 px-2.5 py-1 text-amber-900">
                  Developing: {studyPlan.syllabus_coverage.developing_chapters}
                </span>
                <span className="rounded-xl bg-signal-incorrect/10 px-2.5 py-1 text-signal-incorrect">
                  Less-Learned (&lt;45%): {studyPlan.syllabus_coverage.less_learned_chapters}
                </span>
              </div>
            </div>

            <div className="h-2.5 w-full overflow-hidden rounded-full bg-midnight-900/10">
              <div
                className="h-full rounded-full bg-midnight-950 transition-all"
                style={{ width: `${Math.max(2, studyPlan.syllabus_coverage.coverage_percent)}%` }}
              />
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              {Object.entries(studyPlan.syllabus_coverage.subject_coverage).map(([subj, sc]) => (
                <div
                  key={subj}
                  className="rounded-xl border border-midnight-900/10 bg-midnight-900/[0.02] p-4"
                >
                  <div className="flex items-center justify-between text-xs font-semibold text-midnight-900">
                    <span>{subj}</span>
                    <span>
                      {sc.practiced_chapters}/{sc.total_chapters} chapters ({sc.coverage_percent}%)
                    </span>
                  </div>
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-midnight-900/10">
                    <div
                      className="h-full rounded-full bg-accent"
                      style={{ width: `${Math.max(2, sc.coverage_percent)}%` }}
                    />
                  </div>
                  <p className="mt-2 text-[11px] text-midnight-700/70">
                    Mean Mastery: <strong>{sc.mean_mastery_percent}%</strong> · Mastered: <strong>{sc.mastered_chapters}</strong> · Less-Learned: <strong>{sc.less_learned_chapters}</strong>
                  </p>
                </div>
              ))}
            </div>

            {studyPlan.less_learned_lessons.length > 0 && (
              <div className="flex flex-col gap-2.5 border-t border-midnight-900/[0.06] pt-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-signal-incorrect">
                  Less-Learned Practiced Chapters (&lt;45% Decayed BKT Mastery)
                </p>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {studyPlan.less_learned_lessons.slice(0, 6).map((ll) => (
                    <div
                      key={ll.topic_id}
                      className="flex flex-col justify-between gap-2 rounded-xl border border-signal-incorrect/20 bg-signal-incorrect/[0.04] p-3.5"
                    >
                      <div>
                        <div className="flex items-center justify-between gap-2 text-[11px] font-semibold text-midnight-700/70">
                          <span>{ll.subject} · {ll.current_level}</span>
                          <span className="text-signal-incorrect">{ll.decayed_mastery_percent}% mastery</span>
                        </div>
                        <p className="mt-1 text-sm font-semibold text-midnight-950">{ll.topic_id}</p>
                        {ll.blocking_prerequisites.length > 0 && (
                          <p className="mt-1 text-[11px] text-midnight-700/75">
                            Blocked by: <strong>{ll.blocking_prerequisites.join(", ")}</strong>
                          </p>
                        )}
                      </div>
                      <div className="flex items-center justify-between pt-1">
                        <PaceBadge pace={ll.pace_flag} />
                        <button
                          type="button"
                          onClick={() =>
                            onStartPaper(
                              ll.blocking_prerequisites.length > 0 ? ll.blocking_prerequisites[0] : ll.topic_id
                            )
                          }
                          className="text-xs font-semibold text-accent hover:underline"
                        >
                          {ll.blocking_prerequisites.length > 0 ? "Fix Prerequisite →" : "Remediate Chapter →"}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {studyPlan.mistake_retest_queue && studyPlan.mistake_retest_queue.length > 0 && (
              <div className="flex flex-col gap-2 border-t border-midnight-900/[0.06] pt-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-midnight-800">
                  Ebbinghaus Spaced Mistake Re-Test Queue ({studyPlan.mistake_retest_queue.length} Missed PYQs)
                </p>
                <div className="flex flex-wrap gap-2">
                  {studyPlan.mistake_retest_queue.slice(0, 6).map((mq) => (
                    <button
                      key={mq.question_id}
                      type="button"
                      onClick={() => onStartPaper(mq.topic_id)}
                      className="rounded-xl border border-midnight-900/10 bg-white px-3 py-2 text-left text-xs transition hover:border-accent"
                    >
                      <span className="font-mono font-semibold text-midnight-950">{mq.question_id}</span>
                      <span className="ml-2 text-midnight-700/75">{mq.topic_id}</span>
                      <span className="ml-2 rounded-full bg-midnight-900/5 px-2 py-0.5 text-[10px] font-semibold text-midnight-800">
                        Retest +{mq.ebbinghaus_retest_hours}h · {mq.error_mode.replace(/_/g, " ")}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {studyPlan.improvement_action_table.length > 0 && (
              <div className="flex flex-col gap-2.5 border-t border-midnight-900/[0.06] pt-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-midnight-800">
                  Ranked Actionable Improvement Table (Top Expected +NEET Marks ROI)
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-midnight-900/10 text-[11px] uppercase tracking-wide text-midnight-700/60">
                        <th className="py-2 pr-3">#</th>
                        <th className="py-2 pr-3">Chapter</th>
                        <th className="py-2 pr-3">Diagnosis</th>
                        <th className="py-2 pr-3 text-right">Mastery</th>
                        <th className="py-2 pr-3 text-right">+NEET Marks</th>
                        <th className="py-2 pr-3">Prescription</th>
                        <th className="py-2 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-midnight-900/[0.06]">
                      {studyPlan.improvement_action_table.slice(0, 8).map((row) => (
                        <tr key={row.topic_id}>
                          <td className="py-2.5 pr-3 font-mono font-semibold text-midnight-700">{row.rank}</td>
                          <td className="py-2.5 pr-3">
                            <p className="font-medium text-midnight-950">{row.topic_id}</p>
                            <p className="text-[11px] text-midnight-700/60">
                              {row.subject} · {row.recommended_drill_questions} Qs ({row.recommended_start_tier})
                              {typeof row.available_chapter_pyqs === "number"
                                ? ` · ${row.available_chapter_pyqs} PYQs`
                                : ""}
                            </p>
                          </td>
                          <td className="py-2.5 pr-3">
                            <span className="rounded-md bg-midnight-900/5 px-2 py-0.5 font-mono text-[10px] font-semibold text-midnight-800">
                              {row.diagnostic_category}
                            </span>
                          </td>
                          <td className="py-2.5 pr-3 text-right font-mono font-medium text-midnight-900">
                            {row.decayed_mastery_percent.toFixed(1)}%
                          </td>
                          <td className="py-2.5 pr-3 text-right font-mono font-semibold text-emerald-700">
                            +{row.expected_marks_gain_720.toFixed(1)}
                          </td>
                          <td className="max-w-xs py-2.5 pr-3 text-[11px] text-midnight-700/80">
                            {row.what_to_improve}
                          </td>
                          <td className="py-2.5 text-right">
                            <button
                              type="button"
                              onClick={() => onStartPaper(row.topic_id)}
                              className="rounded-lg bg-midnight-950 px-2.5 py-1 text-[11px] font-semibold text-white transition hover:bg-midnight-800"
                            >
                              Drill →
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </Card>
        </div>
      )}

      {allSyllabusChapters.length > 0 && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
            <div>
              <h2 className="font-display text-xl font-semibold text-midnight-950">
                Complete NCERT NEET Syllabus & Mastery Graph ({allSyllabusChapters.length} Chapters)
              </h2>
              <p className="text-xs text-midnight-700/70">
                All 65 canonical Physics, Chemistry, and Biology chapters ranked by LinUCB Zone of Proximal Development (ZPD) priority and 14-year NEET weightage.
              </p>
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search 65 chapters or clusters..."
              className="w-full rounded-xl border border-midnight-900/10 bg-surface-raised px-3.5 py-2 text-sm text-midnight-900 placeholder-midnight-700/40 outline-none focus:border-accent sm:w-64"
            />
          </div>

          <div className="flex flex-wrap gap-2">
            {(["All", "Biology", "Chemistry", "Physics"] as SubjectFilter[]).map((subj) => (
              <button
                key={subj}
                type="button"
                onClick={() => setSubjectFilter(subj)}
                className={`rounded-xl px-3.5 py-1.5 text-xs font-semibold transition ${
                  subjectFilter === subj
                    ? "bg-midnight-950 text-white"
                    : "bg-midnight-900/5 text-midnight-800 hover:bg-midnight-900/10"
                }`}
              >
                {subj} ({subjectCounts[subj]})
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {filteredChapters.map((ch) => (
              <Card key={ch.topic_id} className="flex flex-col justify-between gap-2.5 p-4">
                <div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wide text-midnight-700/60">
                      {ch.subject} {ch.cluster ? `· ${ch.cluster}` : ""}
                    </span>
                    <DifficultyBadge level={(ch.level ?? "basic") as Difficulty} />
                  </div>
                  <p className="mt-1 font-medium text-midnight-950">{ch.topic_id}</p>
                </div>

                <div className="flex flex-col gap-1.5 border-t border-midnight-900/[0.06] pt-2 text-xs text-midnight-700/75">
                  <div className="flex items-center justify-between">
                    <span>BKT Mastery: <strong>{Math.round(ch.current_mastery * 100)}%</strong></span>
                    <span>Readiness: <strong>{Math.round(ch.readiness * 100)}%</strong></span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-midnight-700/60">
                    <span>NEET Weightage: {ch.neet_weightage ?? 1.5}%</span>
                    <span>ZPD Priority: {ch.zpd_priority_score.toFixed(2)}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => onStartPaper(ch.topic_id, 15, "diagnostic_15")}
                    className="mt-1 self-start text-xs font-semibold text-accent hover:underline"
                  >
                    Practice chapter →
                  </button>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
      {reviewModal && (
        <PaperReviewModal
          review={reviewModal}
          onClose={() => setReviewModal(null)}
          onFetchHint={handleFetchSocraticHint}
          hints={hints}
          loadingHint={loadingHint}
          hintErrors={hintErrors}
        />
      )}
    </div>
  );
}

function PaperReviewModal({
  review,
  onClose,
  onFetchHint,
  hints,
  loadingHint,
  hintErrors,
}: {
  review: PaperReviewOut;
  onClose: () => void;
  onFetchHint: (questionId: string, selectedOption: string) => void;
  hints: Record<string, SocraticHintOut>;
  loadingHint: Record<string, boolean>;
  hintErrors: Record<string, string>;
}) {
  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-midnight-950/60 p-4 backdrop-blur-sm">
      <Card className="flex max-h-[88vh] w-full max-w-3xl flex-col gap-4 overflow-hidden p-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-midnight-900/[0.08] pb-3">
          <div>
            <h2 className="font-display text-xl font-semibold text-midnight-950">
              Sample Paper {review.paper_number} Review
            </h2>
            <p className="text-xs text-midnight-700/70">
              {review.correct_count} of {review.total_questions} correct · {review.score_percent}% accuracy
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-midnight-900/10 px-3 py-1.5 text-xs font-semibold text-midnight-800 hover:bg-midnight-900/5 transition"
          >
            Close ✕
          </button>
        </div>

        <div className="flex flex-1 flex-col gap-4 overflow-y-auto pr-1">
          {review.questions.map((q, i) => {
            const hint = hints[q.question_id];
            return (
              <Card key={q.question_id} className="p-4 bg-surface-raised">
                <div className="mb-2 flex items-center justify-between gap-2 border-b border-midnight-900/[0.06] pb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="inline-flex items-center rounded-md bg-emerald-500/10 px-2 py-0.5 text-[11px] font-bold text-emerald-800 border border-emerald-500/20">
                      {q.paper_source || parseExamYear(q.question_id)}
                    </span>
                    <span className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                      Question {i + 1} · {q.topic_id}
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-midnight-700/50">{q.question_id}</span>
                </div>
                <p className="mb-3 text-sm font-medium leading-relaxed text-midnight-950">{q.text}</p>
                <div className="flex flex-col gap-1.5">
                  {q.options.map((optText, optIdx) => {
                    const letter = OPTION_LETTERS[optIdx];
                    const isCorrect = letter === q.correct_answer;
                    const isSelected = letter === q.selected_option;
                    let cls = "border-midnight-900/10 bg-white/50";
                    if (isCorrect) cls = "border-signal-correct bg-signal-correct/10 font-semibold";
                    else if (isSelected) cls = "border-signal-incorrect bg-signal-incorrect/10";
                    return (
                      <div
                        key={optIdx}
                        className={`flex items-center gap-2.5 rounded-lg border px-3 py-2 text-xs ${cls}`}
                      >
                        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-midnight-900/20 text-[10px] font-bold uppercase text-midnight-700/70">
                          {letter}
                        </span>
                        <span className="text-midnight-900">{optText}</span>
                        {isCorrect && (
                          <span className="ml-auto text-[11px] font-bold text-signal-correct">
                            Correct answer
                          </span>
                        )}
                        {isSelected && !isCorrect && (
                          <span className="ml-auto text-[11px] font-bold text-signal-incorrect">
                            Your answer
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>

                {!q.is_correct && (
                  <div className="mt-3">
                    {!hint ? (
                      <div className="flex flex-col items-start gap-1">
                        <button
                          type="button"
                          onClick={() => onFetchHint(q.question_id, q.selected_option)}
                          disabled={loadingHint[q.question_id]}
                          className="rounded-lg bg-midnight-950 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-midnight-800 disabled:opacity-50"
                        >
                          {loadingHint[q.question_id]
                            ? "Retrieving NCERT Concept…"
                            : "Get NCERT Socratic Hint & Misconception Analysis"}
                        </button>
                        {hintErrors[q.question_id] && (
                          <p className="text-xs text-signal-incorrect">{hintErrors[q.question_id]}</p>
                        )}
                      </div>
                    ) : (
                      <div className="mt-2 rounded-xl border border-midnight-900/10 bg-midnight-900/5 p-3.5 text-xs leading-relaxed text-midnight-900">
                        <p className="font-semibold text-midnight-950">{hint.ncert_reference}</p>
                        {hint.concept_summary && (
                          <p className="mt-1">
                            <span className="font-semibold">NCERT Anchor:</span> {hint.concept_summary}
                          </p>
                        )}
                        <p className="mt-1">
                          <span className="font-semibold">Diagnosed Misconception:</span>{" "}
                          {hint.diagnosed_misconception}
                        </p>
                        <p className="mt-1">
                          <span className="font-semibold">Socratic Guidance:</span> {hint.socratic_hint}
                        </p>
                        {hint.key_formula && (
                          <p className="mt-1">
                            <span className="font-semibold">Key Formula:</span> {hint.key_formula}
                          </p>
                        )}
                        {hint.verified_step_by_step && (
                          <p className="mt-1">
                            <span className="font-semibold">Derivation Steps:</span>{" "}
                            {hint.verified_step_by_step}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </Card>
            );
          })}
        </div>

        <div className="flex justify-end border-t border-midnight-900/[0.08] pt-3">
          <Button variant="secondary" onClick={onClose}>
            Done Reviewing
          </Button>
        </div>
      </Card>
    </div>
  );
}

function SummaryStat({ label, value }: { label: string; value: number | string }) {
  return (
    <Card className="p-5">
      <p className="text-xs font-medium uppercase tracking-wide text-midnight-700/50">{label}</p>
      <p className="mt-1 font-display text-3xl font-semibold text-midnight-950">{value}</p>
    </Card>
  );
}
