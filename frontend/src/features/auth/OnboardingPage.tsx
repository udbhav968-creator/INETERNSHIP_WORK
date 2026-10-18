import { useState } from "react";
import { completeOnboarding } from "../../lib/authApi";
import { saveSession } from "../../lib/session";
import type { ClassLevel, Student } from "../../types/auth";

interface OnboardingPageProps {
  student: Student;
  onComplete: (student: Student) => void;
}

const CLASS_OPTIONS: { value: ClassLevel; label: string }[] = [
  { value: "11", label: "Class 11" },
  { value: "12", label: "Class 12" },
  { value: "dropper", label: "Dropper" },
];

const CURRENT_YEAR = new Date().getFullYear();
const YEAR_OPTIONS = Array.from({ length: 4 }, (_, i) => CURRENT_YEAR + i);

/**
 * One-time questionnaire after signup: class + target NEET year.
 * Backed by POST /students/{id}/onboarding. Sets student.onboarded = true,
 * which is what gates a student out of this screen on future logins.
 */
export function OnboardingPage({ student, onComplete }: OnboardingPageProps) {
  const [classLevel, setClassLevel] = useState<ClassLevel | null>(null);
  const [targetYear, setTargetYear] = useState<number>(CURRENT_YEAR + 1);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!classLevel) {
      setError("Pick your class to continue");
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const updated = await completeOnboarding(student.student_id, {
        class_level: classLevel,
        target_neet_year: targetYear,
      });
      saveSession(updated);
      onComplete(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0b0f1a] px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-semibold text-white tracking-tight">
            Welcome, {student.name.split(" ")[0]}
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Two quick questions so we can build your adaptive paper correctly.
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="rounded-2xl border border-slate-800 bg-[#11182a] p-6 shadow-xl space-y-6"
        >
          <div>
            <label className="mb-2 block text-xs font-medium text-slate-400">
              Which class are you in?
            </label>
            <div className="grid grid-cols-3 gap-2">
              {CLASS_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setClassLevel(opt.value)}
                  className={`rounded-lg border py-2.5 text-sm font-medium transition ${
                    classLevel === opt.value
                      ? "border-violet-500 bg-violet-600/20 text-violet-300"
                      : "border-slate-700 text-slate-400 hover:border-slate-600"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-2 block text-xs font-medium text-slate-400">
              Target NEET year
            </label>
            <select
              value={targetYear}
              onChange={(e) => setTargetYear(Number(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-[#0b0f1a] px-3 py-2 text-sm text-white outline-none focus:border-violet-500"
            >
              {YEAR_OPTIONS.map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>
          </div>

          {error && (
            <p className="rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-400">{error}</p>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-lg bg-violet-600 py-2.5 text-sm font-medium text-white transition hover:bg-violet-500 disabled:opacity-50"
          >
            {submitting ? "Saving..." : "Start studying"}
          </button>
        </form>
      </div>
    </div>
  );
}
