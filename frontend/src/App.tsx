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
  const [selectedTopicId, setSelectedTopicId] = useState<string | undefined>(undefined);

  useEffect(() => {
    setStudent(loadSession());
    setCheckingSession(false);
  }, []);

  function handleStartPaper(topicId?: string) {
    setSelectedTopicId(topicId);
    setPaperRunId((n) => n + 1);
    setView("exam");
  }

  function handleNavigate(nextView: View) {
    if (nextView === "exam") {
      handleStartPaper(undefined);
    } else {
      setView("dashboard");
    }
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
      onNavigate={handleNavigate}
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
          key={`${paperRunId}-${selectedTopicId ?? "full"}`}
          studentId={student.student_id}
          paperNumber={paperRunId || 1}
          topicId={selectedTopicId}
          onExit={() => setView("dashboard")}
          onStartNext={() => handleStartPaper(selectedTopicId)}
        />
      )}
    </AppShell>
  );
}
