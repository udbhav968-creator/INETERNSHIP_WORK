import { useState } from "react";
import { AppShell } from "./components/layout/AppShell";
import { Dashboard } from "./features/dashboard/Dashboard";
import { ExamScreen } from "./features/adaptive-paper/ExamScreen";

// Development-only identifier. TEST_STUDENT is reserved for local testing —
// this is explicitly NOT an authentication system. Replace this with real
// auth-derived student identity before any production use.
const DEV_STUDENT_ID = "TEST_STUDENT";

type View = "dashboard" | "exam";

export default function App() {
  const [view, setView] = useState<View>("dashboard");
  // Bumping this forces ExamScreen to request a fresh paper.
  const [paperRunId, setPaperRunId] = useState(0);

  function handleStartPaper() {
    setPaperRunId((n) => n + 1);
    setView("exam");
  }

  return (
    <AppShell activeView={view} onNavigate={setView} studentId={DEV_STUDENT_ID}>
      {view === "dashboard" && <Dashboard studentId={DEV_STUDENT_ID} onStartPaper={handleStartPaper} />}
      {view === "exam" && <ExamScreen key={paperRunId} studentId={DEV_STUDENT_ID} paperNumber={paperRunId + 1} />}
    </AppShell>
  );
}
