import { useState } from "react";
import { login, signup } from "../../lib/authApi";
import { saveSession } from "../../lib/session";
import type { Student } from "../../types/auth";

interface AuthPageProps {
  onAuthenticated: (student: Student) => void;
}

type Mode = "signup" | "login";

export function AuthPage({ onAuthenticated }: AuthPageProps) {
  const [mode, setMode] = useState<Mode>("signup");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const student =
        mode === "signup"
          ? await signup({ name, email, password })
          : await login({ email, password });
      saveSession(student);
      onAuthenticated(student);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0b0f1a] px-4">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-semibold text-white tracking-tight">Klassed</h1>
          <p className="mt-1 text-sm text-slate-400">Adaptive NEET prep</p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-[#11182a] p-6 shadow-xl">
          <div className="mb-6 flex rounded-lg bg-[#0b0f1a] p-1">
            <button
              type="button"
              onClick={() => setMode("signup")}
              className={`flex-1 rounded-md py-2 text-sm font-medium transition ${
                mode === "signup" ? "bg-violet-600 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Create account
            </button>
            <button
              type="button"
              onClick={() => setMode("login")}
              className={`flex-1 rounded-md py-2 text-sm font-medium transition ${
                mode === "login" ? "bg-violet-600 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Log in
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "signup" && (
              <div>
                <label className="mb-1 block text-xs font-medium text-slate-400">Name</label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-[#0b0f1a] px-3 py-2 text-sm text-white placeholder-slate-500 outline-none focus:border-violet-500"
                  placeholder="Your name"
                />
              </div>
            )}

            <div>
              <label className="mb-1 block text-xs font-medium text-slate-400">Email</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-[#0b0f1a] px-3 py-2 text-sm text-white placeholder-slate-500 outline-none focus:border-violet-500"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium text-slate-400">Password</label>
              <input
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-[#0b0f1a] px-3 py-2 text-sm text-white placeholder-slate-500 outline-none focus:border-violet-500"
                placeholder="At least 8 characters"
              />
            </div>

            {error && (
              <p className="rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-400">{error}</p>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="w-full rounded-lg bg-violet-600 py-2.5 text-sm font-medium text-white transition hover:bg-violet-500 disabled:opacity-50"
            >
              {submitting
                ? mode === "signup"
                  ? "Creating account..."
                  : "Logging in..."
                : mode === "signup"
                ? "Create account"
                : "Log in"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
