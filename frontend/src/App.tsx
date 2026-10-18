import { useEffect, useState } from "react";
import { AppShell } from "./components/layout/AppShell";
import { Dashboard } from "./features/dashboard/Dashboard";
import { ExamScreen } from "./features/adaptive-paper/ExamScreen";
import { AuthPage } from "./features/auth/AuthPage";
import { OnboardingPage } from "./features/auth/OnboardingPage";
import { loadSession } from "./lib/session";
import type { Student } from "./types/auth";

type View = "dashboard" | "exam";

export default function App() {
  const [student, setStudent] = useState<Student | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);

  const [view, setView] = useState<View>("dashboard");
  // Bumping this forces ExamScreen to request a fresh paper.
  const [paperRunId, setPaperRunId] = useState(0);

  useEffect(() => {
    setStudent(loadSession());
    setCheckingSession(false);
  }, []);

  function handleStartPaper() {
    setPaperRunId((n) => n + 1);
    setView("exam");
  }

  // Still restoring (or failing to restore) a session from localStorage —
  // render nothing for a tick rather than flashing the auth screen.
  if (checkingSession) {
    return null;
  }

  // No real student yet -> signup/login. Replaces the old hardcoded
  // DEV_STUDENT_ID entirely; everything below this now uses a real row
  // from the students table.
  if (!student) {
    return <AuthPage onAuthenticated={setStudent} />;
  }

  // Signed up but hasn't answered the class / target-year questionnaire yet.
  if (!student.onboarded) {
    return <OnboardingPage student={student} onComplete={setStudent} />;
  }

  return (
    <AppShell activeView={view} onNavigate={setView} studentId={student.student_id}>
      {view === "dashboard" && (
        <Dashboard studentId={student.student_id} onStartPaper={handleStartPaper} />
      )}
      {view === "exam" && (
        <ExamScreen
          key={paperRunId}
          studentId={student.student_id}
          paperNumber={paperRunId + 1}
          onExit={() => setView("dashboard")}
          onStartNext={handleStartPaper}
        />
      )}
    </AppShell>
  );
}
