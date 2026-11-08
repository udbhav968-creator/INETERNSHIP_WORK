import { useEffect, useState } from "react";
import { AppShell, type AppView } from "./components/layout/AppShell";
import { Dashboard } from "./features/dashboard/Dashboard";
import { ExamScreen } from "./features/adaptive-paper/ExamScreen";
import { ModelAuditView } from "./features/mlops/ModelAuditView";
import { AuthPage } from "./features/auth/AuthPage";
import { OnboardingPage } from "./features/auth/OnboardingPage";
import { api } from "./lib/api";
import { clearSession, loadSession } from "./lib/session";
import type { Student } from "./types/auth";

export default function App() {
  const [student, setStudent] = useState<Student | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);

  const [view, setView] = useState<AppView>("dashboard");
  const [paperRunId, setPaperRunId] = useState(0);
  const [nextPaperNumber, setNextPaperNumber] = useState(1);
  const [selectedTopicId, setSelectedTopicId] = useState<string | undefined>(undefined);
  const [selectedTotalLength, setSelectedTotalLength] = useState<number>(45);
  const [selectedExamMode, setSelectedExamMode] = useState<"mock_45" | "neet_180" | "diagnostic_15">("mock_45");

  useEffect(() => {
    setStudent(loadSession());
    setCheckingSession(false);
  }, []);

  useEffect(() => {
    if (!student || !student.onboarded) {
      setNextPaperNumber(1);
      return;
    }
    let cancelled = false;
    api
      .getPaperHistory(student.student_id)
      .then((history) => {
        if (cancelled) return;
        const maxExisting = history.reduce((acc, p) => Math.max(acc, p.paper_number), 0);
        setNextPaperNumber(maxExisting + 1);
      })
      .catch(() => {
        if (!cancelled) setNextPaperNumber(1);
      });
    return () => {
      cancelled = true;
    };
  }, [student]);

  function handleStartPaper(
    topicId?: string,
    totalLength = 45,
    mode: "mock_45" | "neet_180" | "diagnostic_15" = "mock_45"
  ) {
    setSelectedTopicId(topicId);
    setSelectedTotalLength(totalLength);
    setSelectedExamMode(mode);
    setPaperRunId((n) => n + 1);
    setView("exam");
  }

  function handleNavigate(nextView: AppView) {
    if (nextView === "exam") {
      handleStartPaper(undefined, 45, "mock_45");
    } else {
      setView(nextView);
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
        setPaperRunId(0);
        setNextPaperNumber(1);
        setView("dashboard");
      }}
    >
      {view === "dashboard" && (
        <Dashboard studentId={student.student_id} onStartPaper={handleStartPaper} />
      )}
      {view === "exam" && (
        <ExamScreen
          key={`${paperRunId}-${selectedTopicId ?? "full"}-${selectedExamMode}`}
          studentId={student.student_id}
          paperNumber={nextPaperNumber}
          topicId={selectedTopicId}
          totalLength={selectedTotalLength}
          mode={selectedExamMode}
          onPaperSubmitted={(submittedNumber) => {
            setNextPaperNumber((prev) => Math.max(prev, submittedNumber + 1));
          }}
          onExit={() => setView("dashboard")}
          onStartNext={() => handleStartPaper(selectedTopicId, selectedTotalLength, selectedExamMode)}
        />
      )}
      {view === "audit" && <ModelAuditView />}
    </AppShell>
  );
}
