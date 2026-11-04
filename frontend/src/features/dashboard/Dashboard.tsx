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

export function Dashboard({ studentId, onStartPaper }: DashboardProps) {
  const [topics, setTopics] = useState<TopicState[]>([]);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [v2Insights, setV2Insights] = useState<V2StudentInsights | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState("");

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

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-2">
        <h1 className="font-display text-3xl font-semibold tracking-tight text-midnight-950 md:text-4xl">
          Welcome back.
        </h1>
        <p className="text-midnight-700/70">
          Here's where your preparation stands right now, powered by 3PL IRT, Bayesian Knowledge Tracing & LinUCB.
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
            <SummaryStat label="Questions attempted" value={0} />
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

          {v2Insights && v2Insights.recommended_unlock_topics.length > 0 && (
            <div>
              <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
                Recommended Foundational Chapters (Zone of Proximal Development)
              </h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {v2Insights.recommended_unlock_topics.map((rec) => (
                  <Card key={rec.topic_id} className="flex flex-col gap-1.5 p-4">
                    <span className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                      {rec.subject}
                    </span>
                    <p className="font-medium text-midnight-950">{rec.topic_id}</p>
                    <p className="text-xs text-midnight-700/70">
                      Prerequisite Readiness: {Math.round(rec.readiness * 100)}% · ZPD Priority: {rec.zpd_priority_score.toFixed(2)}
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

          {v2Insights && v2Insights.recommended_unlock_topics.length > 0 && (
            <div>
              <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
                Next Recommended Unlock Topics (LinUCB & ZPD Priority)
              </h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {v2Insights.recommended_unlock_topics.map((rec) => (
                  <Card key={rec.topic_id} className="flex flex-col gap-1.5 p-4">
                    <span className="text-xs font-semibold uppercase tracking-wide text-midnight-700/60">
                      {rec.subject}
                    </span>
                    <p className="font-medium text-midnight-950">{rec.topic_id}</p>
                    <p className="text-xs text-midnight-700/70">
                      Mastery: {Math.round(rec.current_mastery * 100)}% · Readiness: {Math.round(rec.readiness * 100)}%
                    </p>
                  </Card>
                ))}
              </div>
            </div>
          )}

          <div>
            <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">
              Topics you're working on
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
