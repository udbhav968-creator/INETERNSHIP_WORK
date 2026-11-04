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
          Here's where your preparation stands right now, powered by 3PL IRT & Bayesian Knowledge Tracing.
        </p>
      </div>

      <Card className="flex flex-col items-start justify-between gap-4 bg-midnight-950 p-8 text-white md:flex-row md:items-center">
        <div>
          <p className="text-sm font-medium text-white/60">Ready for your next session?</p>
          <p className="mt-1 font-display text-xl font-semibold">
            Your next paper adapts to your Zone of Proximal Development.
          </p>
        </div>
        <Button onClick={onStartPaper} className="shrink-0 bg-white text-midnight-950 hover:bg-white/90">
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
        <Card className="text-center">
          <p className="font-medium text-midnight-900">No study history yet.</p>
          <p className="mt-1 text-sm text-midnight-700/70">
            Start your first practice paper and your per-topic progress will show up here.
          </p>
        </Card>
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
                    {typeof t.mastery_prob === "number" && (
                      <span className="text-xs font-medium text-midnight-700/70">
                        BKT Mastery: {Math.round(t.mastery_prob * 100)}%
                      </span>
                    )}
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
