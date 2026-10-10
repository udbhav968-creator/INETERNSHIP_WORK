import { HTMLAttributes } from "react";

export function Card({ className = "", ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`rounded-xl2 border border-midnight-900/[0.06] bg-surface-raised p-6 shadow-card ${className}`}
      {...props}
    />
  );
}
