import { HTMLAttributes } from "react";

export function Card({ className = "", ...props }: HTMLAttributes<HTMLDivElement>) {
  const hasCustomBg = /\bbg-/.test(className);
  const baseBg = hasCustomBg ? "" : "bg-surface-raised";
  return (
    <div
      className={`rounded-xl2 border border-midnight-900/[0.06] ${baseBg} p-6 shadow-card ${className}`}
      {...props}
    />
  );
}
