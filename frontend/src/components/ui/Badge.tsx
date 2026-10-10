import { ReactNode } from "react";
import type { Difficulty, PaceFlag } from "../../types";

const difficultyStyles: Record<Difficulty, string> = {
  basic: "bg-signal-correct/10 text-signal-correct",
  medium: "bg-signal-pending/15 text-amber-700",
  advanced: "bg-accent/10 text-accent-dim",
};

const difficultyLabels: Record<Difficulty, string> = {
  basic: "Basic",
  medium: "Medium",
  advanced: "Advanced",
};

export function DifficultyBadge({ level }: { level: Difficulty }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${difficultyStyles[level]}`}>
      {difficultyLabels[level]}
    </span>
  );
}

export function PaceBadge({ pace }: { pace: PaceFlag }) {
  if (pace === null) {
    return (
      <span className="inline-flex items-center rounded-full bg-midnight-900/5 px-2.5 py-1 text-xs font-medium text-midnight-700/60">
        Pace: not enough data yet
      </span>
    );
  }
  const styles = pace === "fast" ? "bg-signal-correct/10 text-signal-correct" : "bg-signal-incorrect/10 text-signal-incorrect";
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${styles}`}>
      {pace === "fast" ? "On pace" : "Slower than target"}
    </span>
  );
}

export function Badge({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span className={`inline-flex items-center rounded-full bg-midnight-900/5 px-2.5 py-1 text-xs font-medium text-midnight-700 ${className}`}>
      {children}
    </span>
  );
}
