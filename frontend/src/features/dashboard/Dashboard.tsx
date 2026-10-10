import { useEffect, useState } from "react";
import { api, ApiError } from "../../lib/api";
import type { TopicState } from "../../types";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { DifficultyBadge, PaceBadge } from "../../components/ui/Badge";

interface DashboardProps {
  studentId: string;
  onStartPaper: () => void;
}

type LoadState = "loading" | "ready" | "error" | "empty";

export function Dashboard({ studentId, onStartPaper }: DashboardProps) {
  const [topics, setTopics] = useState<TopicState[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    api
      .getStudentState(studentId)
      .then((data) => {
        if (cancelled) return;
        setTopics(data);
        setState(data.length === 0 ? "empty" : "ready");
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

  const counts = {
    basic: topics.filter((t) => t.level === "basic").length,
    medium: topics.filter((t) => t.level === "medium").length,
    advanced: topics.filter((t) => t.level === "advanced").length,
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-2">
        <h1 className="font-display text-3xl font-semibold tracking-tight text-midnight-950 md:text-4xl">
          Welcome back.
        </h1>
        <p className="text-midnight-700/70">
          Here's where your preparation stands right now, topic by topic.
        </p>
      </div>

      <Card className="flex flex-col items-start justify-between gap-4 bg-midnight-950 p-8 text-white md:flex-row md:items-center">
        <div>
          <p className="text-sm font-medium text-white/60">Ready for your next session?</p>
          <p className="mt-1 font-display text-xl font-semibold">
            Your next paper adapts to exactly where you stand.
          </p>
        </div>
        <Button onClick={onStartPaper} className="shrink-0 bg-white text-midnight-950 hover:bg-white/90">
          Start practice paper
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
          <div className="grid grid-cols-3 gap-4">
            <SummaryStat label="Basic" value={counts.basic} />
            <SummaryStat label="Medium" value={counts.medium} />
            <SummaryStat label="Advanced" value={counts.advanced} />
          </div>

          <div>
            <h2 className="mb-3 font-display text-lg font-semibold text-midnight-950">Topic-by-topic status</h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {topics.map((t) => (
                <Card key={t.topic_id} className="flex flex-col gap-2 p-5">
                  <p className="font-medium text-midnight-900">{t.topic_id}</p>
                  <div className="flex flex-wrap items-center gap-2">
                    <DifficultyBadge level={t.level} />
                    <PaceBadge pace={t.pace_flag} />
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

function SummaryStat({ label, value }: { label: string; value: number }) {
  return (
    <Card className="p-5">
      <p className="text-xs font-medium uppercase tracking-wide text-midnight-700/50">{label}</p>
      <p className="mt-1 font-display text-3xl font-semibold text-midnight-950">{value}</p>
      <p className="text-xs text-midnight-700/50">topics</p>
    </Card>
  );
}
