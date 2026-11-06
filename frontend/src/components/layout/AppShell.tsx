import { ReactNode } from "react";

type View = "dashboard" | "exam";

interface AppShellProps {
  children: ReactNode;
  activeView: View;
  onNavigate: (view: View) => void;
  studentId: string;
  onSwitchAccount?: () => void;
}

const navItems: { key: View; label: string; icon: string }[] = [
  { key: "dashboard", label: "Dashboard", icon: "M3 12l9-9 9 9M5 10v10a1 1 0 001 1h3v-6h6v6h3a1 1 0 001-1V10" },
  { key: "exam", label: "Practice Paper", icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" },
];

export function AppShell({ children, activeView, onNavigate, studentId, onSwitchAccount }: AppShellProps) {
  return (
    <div className="min-h-screen bg-surface">
      <aside className="fixed inset-y-0 left-0 hidden w-64 flex-col border-r border-midnight-900/[0.06] bg-surface-raised px-5 py-6 md:flex">
        <div className="mb-10 flex items-center gap-2 px-1">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-midnight-900">
            <span className="font-display text-sm font-semibold text-white">K</span>
          </div>
          <span className="font-display text-lg font-semibold tracking-tight">Klassed</span>
        </div>

        <nav className="flex flex-1 flex-col gap-1">
          {navItems.map((item) => (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              className={`focus-ring flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors ${
                activeView === item.key
                  ? "bg-midnight-900 text-white"
                  : "text-midnight-700 hover:bg-midnight-900/5"
              }`}
            >
              <svg className="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                <path strokeLinecap="round" strokeLinejoin="round" d={item.icon} />
              </svg>
              {item.label}
            </button>
          ))}
        </nav>

        <div className="rounded-xl border border-midnight-900/[0.06] bg-midnight-900/[0.03] px-3 py-3">
          <p className="text-xs font-medium text-midnight-700/60">Signed in as</p>
          <p className="truncate text-sm font-semibold text-midnight-900">{studentId}</p>
          <p className="mt-1 text-[11px] text-midnight-700/60">Enterprise v2 · 3PL IRT & BKT Active</p>
          {onSwitchAccount && (
            <button
              type="button"
              onClick={onSwitchAccount}
              className="mt-2 text-xs font-medium text-accent hover:underline"
            >
              Switch account
            </button>
          )}
        </div>
      </aside>

      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-midnight-900/[0.06] bg-surface-raised/90 px-4 py-3 backdrop-blur md:hidden">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-midnight-900">
            <span className="font-display text-xs font-semibold text-white">K</span>
          </div>
          <span className="font-display text-base font-semibold">Klassed</span>
        </div>
        <nav className="flex items-center gap-1">
          {navItems.map((item) => (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              className={`focus-ring rounded-lg px-2.5 py-1.5 text-xs font-medium ${
                activeView === item.key ? "bg-midnight-900 text-white" : "text-midnight-700"
              }`}
            >
              {item.label}
            </button>
          ))}
          {onSwitchAccount && (
            <button
              type="button"
              onClick={onSwitchAccount}
              className="ml-1 rounded-lg border border-midnight-900/10 px-2 py-1.5 text-xs font-medium text-accent"
            >
              Switch
            </button>
          )}
        </nav>
      </header>

      <main className="md:pl-64">
        <div className="mx-auto max-w-5xl px-4 py-8 md:px-10 md:py-12">{children}</div>
      </main>
    </div>
  );
}
