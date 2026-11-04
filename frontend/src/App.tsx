import { useEffect, useState } from "react";
import { AppShell } from "./components/layout/AppShell";
import { Dashboard } from "./features/dashboard/Dashboard";
import { ExamScreen } from "./features/adaptive-paper/ExamScreen";
import { AuthPage } from "./features/auth/AuthPage";
import { OnboardingPage } from "./features/auth/OnboardingPage";
import { clearSession, loadSession } from "./lib/session";
import type { Student } from "./types/auth";

type View = "dashboard" | "exam";

export default function App() {
  const [student, setStudent] = useState<Student | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);

  const [view, setView] = useState<View>("dashboard");
  const [paperRunId, setPaperRunId] = useState(0);

  useEffect(() => {
    setStudent(loadSession());
    setCheckingSession(false);
  }, []);

  function handleStartPaper() {
    setPaperRunId((n) => n + 1);
    setView("exam");
  }

  if (checkingSession) {
    return null;
  }

  if (!student) {
    return <AuthPage onAuthenticated={setStudent} />;
  }

  if (!student.onboarded) {
    return <OnboardingPage student={student} onComplete={setStudent} />;
  }

  return (
    <AppShell
      activeView={view}
      onNavigate={setView}
      studentId={student.student_id}
      onSwitchAccount={() => {
        clearSession();
        setStudent(null);
      }}
    >
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
