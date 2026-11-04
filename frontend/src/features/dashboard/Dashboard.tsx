import { useEffect, useState } from "react";
import { api, ApiError } from "../../lib/api";
import type { Analytics, TopicState, V2StudentInsights } from "../../types";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { PaceBadge } from "../../components/ui/Badge";

interface DashboardProps {
  studentId: string;
  onStartPaper: () => void;
}

type LoadState = "loading" | "ready" | "error" | "empty";
type SubjectFilter = "All" | "Physics" | "Chemistry" | "Biology";

export function Dashboard({ studentId, onStartPaper }: DashboardProps) {
  const [topics, setTopics] = useState<TopicState[]>([]);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [v2Insights, setV2Insights] = useState<V2StudentInsights | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState("");
  const [subjectFilter, setSubjectFilter] = useState<SubjectFilter>("All");
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    Promise.all([
      api.getStudentState(studentId),
      api.getAnalytics(studentId),
      api.getV2Insights(studentId).catch(() => null),
    ])
      .then(([topicData, analyticsData, insightsData]) => {
        if (cancelled) return;
        setTopics(topicData);
        setAnalytics(analyticsData);
        setV2Insights(insightsData);
        setState(topicData.length === 0 ? "empty" : "ready");
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
            Your next paper adapts to your Zone of Proximal Development (target P ≈ 70%).
          </p>
        </div>
        <Button
          variant="secondary"
          onClick={onStartPaper}
          className="shrink-0 font-semibold"
        >
          Start sample paper
        </Button>
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
            <SummaryStat label="NCERT Chapters Mapped" value={allSyllabusChapters.length || 65} />
            <SummaryStat label="Initial BKT Prior" value="35%" />
            <SummaryStat
              label="Baseline NEET Projection"
              value={v2Insights ? `${v2Insights.predicted_neet_score_720} / 720` : "380 / 720"}
            />
            <SummaryStat
              label="3PL IRT Prior Percentile"
              value={v2Insights ? `${v2Insights.estimated_neet_percentile}%ile` : "50.0%ile"}
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
                  <Card key={rec.topic_id} className="flex flex-col gap-1.5 p-4">
                    <span className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                      {rec.subject} {rec.cluster ? `· ${rec.cluster}` : ""}
                    </span>
                    <p className="font-medium text-midnight-950">{rec.topic_id}</p>
                    <p className="text-xs text-midnight-700/70">
                      Readiness: {Math.round(rec.readiness * 100)}% · Weightage: {rec.neet_weightage ?? 1.5}% · ZPD: {rec.zpd_priority_score.toFixed(2)}
                    </p>
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
              value={v2Insights ? `${v2Insights.predicted_neet_score_720} / 720` : "—"}
            />
            <SummaryStat
              label="3PL IRT Percentile"
              value={v2Insights ? `${v2Insights.estimated_neet_percentile}%ile` : `${topics.length} topics`}
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
              <div className="mt-3 flex flex-col gap-2">
                {v2Insights.root_cause_bottlenecks.map((b, idx) => (
                  <p key={idx} className="text-sm text-midnight-800">
                    <span className="font-semibold">{b.weak_topic}</span> is bottlenecked by foundational chapter{" "}
                    <span className="font-semibold underline">{b.root_cause_prerequisite}</span> —{" "}
                    {b.recommendation}
                  </p>
                ))}
              </div>
            </Card>
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
                    <span className="rounded-full bg-midnight-900/5 px-2.5 py-0.5 text-xs font-medium capitalize text-midnight-800">
                      {t.level}
                    </span>
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
                </Card>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Complete 65-Chapter NCERT Syllabus & Mastery Explorer (Always visible in both empty and ready states) */}
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
                    <span className="rounded-full bg-midnight-900/5 px-2 py-0.5 text-[11px] font-medium capitalize text-midnight-800">
                      {ch.level ?? "basic"}
                    </span>
                  </div>
                  <p className="mt-1 font-medium text-midnight-950">{ch.topic_id}</p>
                </div>

                <div className="flex flex-col gap-1 border-t border-midnight-900/[0.06] pt-2 text-xs text-midnight-700/75">
                  <div className="flex items-center justify-between">
                    <span>BKT Mastery: <strong>{Math.round(ch.current_mastery * 100)}%</strong></span>
                    <span>Readiness: <strong>{Math.round(ch.readiness * 100)}%</strong></span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-midnight-700/60">
                    <span>NEET Weightage: {ch.neet_weightage ?? 1.5}%</span>
                    <span>ZPD Priority: {ch.zpd_priority_score.toFixed(2)}</span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
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
