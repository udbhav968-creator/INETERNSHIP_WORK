import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../../lib/api";
import {
  OPTION_LETTERS,
  type AttemptPayload,
  type OptionLetter,
  type PaperReviewOut,
  type QuestionOut,
  type SocraticHintOut,
} from "../../types";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { DifficultyBadge } from "../../components/ui/Badge";

interface ExamScreenProps {
  studentId: string;
  paperNumber: number;
  topicId?: string;
  totalLength?: number;
  mode?: "mock_45" | "neet_180" | "diagnostic_15";
  onPaperSubmitted?: (submittedNumber: number) => void;
  onExit: () => void;
  onStartNext: () => void;
}

type LoadState = "loading" | "ready" | "error" | "empty" | "submitting" | "submit-error" | "results";

export function ExamScreen({
  studentId,
  paperNumber,
  topicId,
  totalLength = 45,
  mode = "mock_45",
  onPaperSubmitted,
  onExit,
  onStartNext,
}: ExamScreenProps) {
  const [questions, setQuestions] = useState<QuestionOut[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState("");
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [review, setReview] = useState<PaperReviewOut | null>(null);
  const [confirmingSubmit, setConfirmingSubmit] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [showOmr, setShowOmr] = useState(false);
  const [sectionFilter, setSectionFilter] = useState<"all" | "physics" | "chemistry" | "biology">("all");

  useEffect(() => {
    if (loadState !== "ready") return;
    const interval = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [loadState]);

  const firstVisitedAt = useRef<Record<number, number>>({});
  const firstInteractionAt = useRef<Record<number, number>>({});
  const optionSwitchCounts = useRef<Record<number, number>>({});
  const activeDwellMs = useRef<Record<number, number>>({});
  const lastActiveTimestamp = useRef<number>(Date.now());
  const currentIndexRef = useRef<number>(0);

  useEffect(() => {
    let cancelled = false;
    const cacheKey = `klassed_active_${studentId}_${paperNumber}_${topicId ?? "full"}_${mode}`;
    const cached = localStorage.getItem(cacheKey);
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        if (parsed.answers) setAnswers(parsed.answers);
        if (typeof parsed.index === "number") setIndex(parsed.index);
        if (typeof parsed.elapsedSeconds === "number") setElapsedSeconds(parsed.elapsedSeconds);
        if (parsed.activeDwell && typeof parsed.activeDwell === "object") activeDwellMs.current = parsed.activeDwell;
      } catch {
        localStorage.removeItem(cacheKey);
      }
    }
    api
      .getNextPaper(studentId, topicId, totalLength, mode)
      .then((data) => {
        if (cancelled) return;
        setQuestions(data);
        setLoadState(data.length === 0 ? "empty" : "ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setErrorMessage(err instanceof ApiError ? err.message : "Something went wrong.");
        setLoadState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [studentId, topicId, totalLength, mode, paperNumber]);

  useEffect(() => {
    if (loadState !== "ready" || questions.length === 0) return;
    const cacheKey = `klassed_active_${studentId}_${paperNumber}_${topicId ?? "full"}_${mode}`;
    localStorage.setItem(
      cacheKey,
      JSON.stringify({ answers, index, elapsedSeconds, activeDwell: activeDwellMs.current })
    );
  }, [answers, index, elapsedSeconds, loadState, studentId, paperNumber, topicId, mode, questions.length]);

  useEffect(() => {
    if (loadState !== "ready") return;
    const now = Date.now();
    const prevIdx = currentIndexRef.current;
    if (prevIdx !== undefined && prevIdx >= 0) {
      const delta = Math.max(0, now - lastActiveTimestamp.current);
      activeDwellMs.current[prevIdx] = (activeDwellMs.current[prevIdx] ?? 0) + delta;
    }
    currentIndexRef.current = index;
    lastActiveTimestamp.current = now;
    if (!(index in firstVisitedAt.current)) {
      firstVisitedAt.current[index] = now;
    }
  }, [index, loadState]);

  const current = questions[index];
  const answeredCount = Object.keys(answers).length;
  const unansweredCount = questions.length - answeredCount;

  function selectOptionFor(qIndex: number, optionIndex: number) {
    if (!(qIndex in firstInteractionAt.current)) {
      firstInteractionAt.current[qIndex] = Date.now();
    }
    if (answers[qIndex] === optionIndex) {
      optionSwitchCounts.current[qIndex] = (optionSwitchCounts.current[qIndex] ?? 0) + 1;
      setAnswers((prev) => {
        const next = { ...prev };
        delete next[qIndex];
        return next;
      });
      return;
    }
    optionSwitchCounts.current[qIndex] = (optionSwitchCounts.current[qIndex] ?? 0) + 1;
    setAnswers((prev) => ({ ...prev, [qIndex]: optionIndex }));
  }

  function selectOption(optionIndex: number) {
    selectOptionFor(index, optionIndex);
  }

  function clearCurrentOption() {
    if (!(index in answers)) return;
    optionSwitchCounts.current[index] = (optionSwitchCounts.current[index] ?? 0) + 1;
    setAnswers((prev) => {
      const next = { ...prev };
      delete next[index];
      return next;
    });
  }

  function goTo(i: number) {
    if (i < 0 || i >= questions.length) return;
    setIndex(i);
  }

  useEffect(() => {
    if (loadState !== "ready" || confirmingSubmit) return;
    function onKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setIndex((prev) => Math.max(0, prev - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setIndex((prev) => Math.min(questions.length - 1, prev + 1));
      } else {
        const k = e.key.toLowerCase();
        const optMap: Record<string, number> = {
          a: 0,
          b: 1,
          c: 2,
          d: 3,
          "1": 0,
          "2": 1,
          "3": 2,
          "4": 3,
        };
        if (k in optMap && current && optMap[k] < current.options.length) {
          e.preventDefault();
          selectOption(optMap[k]);
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [loadState, confirmingSubmit, index, questions.length, answers, current]);

  async function handleSubmitPaper() {
    setConfirmingSubmit(false);
    setLoadState("submitting");
    setErrorMessage("");

    try {
      const finalNow = Date.now();
      const currIdx = currentIndexRef.current;
      if (currIdx !== undefined && currIdx >= 0) {
        const delta = Math.max(0, finalNow - lastActiveTimestamp.current);
        activeDwellMs.current[currIdx] = (activeDwellMs.current[currIdx] ?? 0) + delta;
        lastActiveTimestamp.current = finalNow;
      }

      const batchPayloads: AttemptPayload[] = [];
      for (const [qIndexStr, optionIndex] of Object.entries(answers)) {
        const qIndex = Number(qIndexStr);
        const question = questions[qIndex];
        if (!question) continue;

        const visitedAt = firstVisitedAt.current[qIndex] ?? finalNow;
        const firstClickedAt = firstInteractionAt.current[qIndex] ?? finalNow;
        const dwellMs = Math.max(1000, activeDwellMs.current[qIndex] ?? 1000);
        const hesitationMs = Math.max(0, Math.min(dwellMs, firstClickedAt - visitedAt));
        const switches = optionSwitchCounts.current[qIndex] ?? 0;
        const timeTaken = Math.max(1, Math.round(dwellMs / 1000));
        const letter: OptionLetter = OPTION_LETTERS[optionIndex];

        batchPayloads.push({
          student_id: studentId,
          question_id: question.question_id,
          paper_number: paperNumber,
          selected_option: letter,
          time_taken_seconds: timeTaken,
          option_switches: switches,
          hesitation_ms: hesitationMs,
          dwell_time_ms: dwellMs,
        });
      }

      const reviewData = await api.submitPaper(studentId, paperNumber, batchPayloads);
      const cacheKey = `klassed_active_${studentId}_${paperNumber}_${topicId ?? "full"}_${mode}`;
      localStorage.removeItem(cacheKey);
      onPaperSubmitted?.(reviewData.paper_number);
      setReview(reviewData);
      setLoadState("results");
    } catch (err) {
      setErrorMessage(
        err instanceof ApiError ? err.message : "Couldn't submit your paper. Try again."
      );
      setLoadState("submit-error");
    }
  }

  if (loadState === "loading") {
    return (
      <div className="mx-auto max-w-3xl">
        <Card className="h-80 animate-pulse bg-midnight-900/5" />
      </div>
    );
  }

  if (loadState === "error") {
    return (
      <Card className="mx-auto max-w-xl border-signal-incorrect/20 bg-signal-incorrect/5">
        <p className="font-medium text-signal-incorrect">Couldn't load your next paper.</p>
        <p className="mt-1 text-sm text-midnight-700/70">{errorMessage}</p>
      </Card>
    );
  }

  if (loadState === "empty") {
    return (
      <Card className="mx-auto max-w-xl text-center">
        <p className="font-medium text-midnight-900">No questions available right now.</p>
        <p className="mt-1 text-sm text-midnight-700/70">
          This usually means topic importance scores or tagged questions haven't been loaded yet.
        </p>
      </Card>
    );
  }

  if (loadState === "results" && review) {
    return <ResultsView review={review} totalInPaper={questions.length} onExit={onExit} onStartNext={onStartNext} />;
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 lg:flex-row">
      <aside className="order-2 flex flex-col gap-3 lg:order-1 lg:w-56 lg:shrink-0">
        <Card className="p-4">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-midnight-700/50">
            {topicId ? `Chapter Drill · ${topicId}` : `Sample Paper ${paperNumber}`}
          </p>

          {questions.length === 180 && (
            <div className="mb-2.5 flex flex-wrap gap-1">
              {(["all", "physics", "chemistry", "biology"] as const).map((sec) => (
                <button
                  key={sec}
                  type="button"
                  onClick={() => setSectionFilter(sec)}
                  className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${
                    sectionFilter === sec
                      ? "bg-midnight-950 text-white"
                      : "bg-midnight-900/5 text-midnight-700/70 hover:bg-midnight-900/10"
                  }`}
                >
                  {sec}
                </button>
              ))}
            </div>
          )}

          <div className="grid grid-cols-6 gap-2 lg:grid-cols-5">
            {questions
              .map((q, i) => ({ q, i }))
              .filter(({ i }) => {
                if (questions.length !== 180 || sectionFilter === "all") return true;
                if (sectionFilter === "physics") return i < 45;
                if (sectionFilter === "chemistry") return i >= 45 && i < 90;
                if (sectionFilter === "biology") return i >= 90;
                return true;
              })
              .map(({ i }) => {
                const answered = i in answers;
                const isCurrent = i === index;
                let classes = "border-midnight-900/10 text-midnight-700/60 hover:border-midnight-900/25";
                if (answered) classes = "border-accent bg-accent/10 text-accent-dim font-semibold";
                if (isCurrent) classes = "border-accent bg-accent text-white font-semibold";
                return (
                  <button
                    key={i}
                    onClick={() => goTo(i)}
                    className={`focus-ring flex h-9 w-9 items-center justify-center rounded-lg border text-xs transition-colors ${classes}`}
                  >
                    {i + 1}
                  </button>
                );
              })}
          </div>
          <p className="mt-3 text-xs text-midnight-700/60">
            {answeredCount} of {questions.length} answered · {unansweredCount} skipped (0 marks)
          </p>
        </Card>

        <Button
          onClick={() => setConfirmingSubmit(true)}
          disabled={answeredCount === 0 || loadState === "submitting"}
          className="w-full"
        >
          {loadState === "submitting" ? "Submitting…" : "Submit paper"}
        </Button>

        {loadState === "submit-error" && (
          <p className="text-sm text-signal-incorrect">{errorMessage}</p>
        )}
      </aside>

      <div className="order-1 flex-1 lg:order-2">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm font-medium text-midnight-700/60">
            Question {index + 1} of {questions.length} · +4 / -1 NEET Marking
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowOmr(true)}
              className="flex items-center gap-1 rounded-lg border border-midnight-900/15 bg-white px-2.5 py-1 text-xs font-semibold text-midnight-800 shadow-sm hover:border-accent hover:text-accent"
            >
              Digital OMR
            </button>
            <span className="rounded-lg bg-midnight-900/5 px-2.5 py-1 text-xs font-mono font-medium text-midnight-800">
              Time {Math.floor(elapsedSeconds / 60).toString().padStart(2, "0")}:{(elapsedSeconds % 60).toString().padStart(2, "0")}
            </span>
            <span className="rounded-lg bg-midnight-900/5 px-2.5 py-1 text-xs font-medium text-midnight-700/70">
              Target: {current.expected_time_seconds ?? 90}s
            </span>
          </div>
        </div>

        <Card className="p-7">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs font-medium uppercase tracking-wide text-midnight-700/50">
              {current.topic_id}
            </p>
            <DifficultyBadge level={current.difficulty} />
          </div>
          <p className="mb-6 text-lg leading-relaxed text-midnight-950">{current.text}</p>

          <div className="flex flex-col gap-3">
            {current.options.map((optionText, i) => {
              const isSelected = answers[index] === i;
              return (
                <button
                  key={i}
                  onClick={() => selectOption(i)}
                  className={`focus-ring flex items-center gap-3 rounded-xl border px-4 py-3.5 text-left text-sm transition-colors ${
                    isSelected
                      ? "border-accent bg-accent/5"
                      : "border-midnight-900/10 hover:border-midnight-900/25"
                  }`}
                >
                  <span
                    className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-xs font-semibold uppercase ${
                      isSelected ? "border-accent bg-accent text-white" : "border-midnight-900/20 text-midnight-700/60"
                    }`}
                  >
                    {OPTION_LETTERS[i]}
                  </span>
                  <span className="text-midnight-900">{optionText}</span>
                </button>
              );
            })}
          </div>

          <div className="mt-6 flex items-center justify-between gap-2">
            <Button variant="secondary" onClick={() => goTo(index - 1)} disabled={index === 0}>
              Previous
            </Button>

            {index in answers && (
              <Button variant="ghost" size="sm" onClick={clearCurrentOption}>
                Clear response (Skip)
              </Button>
            )}

            <Button
              variant="secondary"
              onClick={() => goTo(index + 1)}
              disabled={index >= questions.length - 1}
            >
              Next
            </Button>
          </div>
        </Card>
      </div>

      {confirmingSubmit && (
        <ConfirmSubmitDialog
          unansweredCount={unansweredCount}
          onCancel={() => setConfirmingSubmit(false)}
          onConfirm={handleSubmitPaper}
        />
      )}

      {showOmr && (
        <DigitalOmrModal
          questions={questions}
          answers={answers}
          currentIndex={index}
          onSelect={(qIdx, optIdx) => selectOptionFor(qIdx, optIdx)}
          onGoTo={(qIdx) => {
            goTo(qIdx);
            setShowOmr(false);
          }}
          onClose={() => setShowOmr(false)}
        />
      )}
    </div>
  );
}

function DigitalOmrModal({
  questions,
  answers,
  currentIndex,
  onSelect,
  onGoTo,
  onClose,
}: {
  questions: QuestionOut[];
  answers: Record<number, number>;
  currentIndex: number;
  onSelect: (qIndex: number, optionIndex: number) => void;
  onGoTo: (qIndex: number) => void;
  onClose: () => void;
}) {
  const answeredCount = Object.keys(answers).length;
  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-midnight-950/60 p-4">
      <Card className="flex max-h-[90vh] w-full max-w-3xl flex-col p-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-midnight-900/10 pb-4">
          <div>
            <h3 className="font-display text-lg font-bold text-midnight-950">
              Digital OMR Sheet Simulator
            </h3>
            <p className="text-xs text-midnight-700/70">
              Answered: <strong>{answeredCount}</strong> / {questions.length} · Skipped:{" "}
              <strong>{questions.length - answeredCount}</strong>
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>

        <div className="my-4 max-h-[60vh] overflow-y-auto pr-2">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 md:grid-cols-3">
            {questions.map((q, qIdx) => {
              const isCurrent = qIdx === currentIndex;
              const selectedOpt = answers[qIdx];
              return (
                <div
                  key={q.question_id}
                  className={`flex items-center justify-between rounded-lg border px-3 py-2 ${
                    isCurrent
                      ? "border-accent bg-accent/5"
                      : "border-midnight-900/10 bg-white"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => onGoTo(qIdx)}
                    className="font-mono text-xs font-semibold text-midnight-800 hover:text-accent"
                  >
                    Q{qIdx + 1}
                  </button>
                  <div className="flex items-center gap-1.5">
                    {OPTION_LETTERS.map((letter, optIdx) => {
                      const isFilled = selectedOpt === optIdx;
                      return (
                        <button
                          key={letter}
                          type="button"
                          onClick={() => onSelect(qIdx, optIdx)}
                          className={`flex h-6 w-6 items-center justify-center rounded-full border text-[11px] font-bold uppercase transition-all ${
                            isFilled
                              ? "border-accent bg-accent text-white shadow-sm"
                              : "border-midnight-900/20 text-midnight-700 hover:border-midnight-900/40"
                          }`}
                        >
                          {letter}
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="flex items-center justify-between border-t border-midnight-900/10 pt-4">
          <span className="text-xs text-midnight-700/60">
            Click bubbles to fill or clear · Click Q# to jump directly
          </span>
          <Button onClick={onClose}>Back to Paper</Button>
        </div>
      </Card>
    </div>
  );
}

function ConfirmSubmitDialog({
  unansweredCount,
  onCancel,
  onConfirm,
}: {
  unansweredCount: number;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-midnight-950/50 px-4">
      <Card className="w-full max-w-sm p-6">
        <p className="font-display text-lg font-semibold text-midnight-950">Submit this paper?</p>
        <p className="mt-2 text-sm text-midnight-700/70">
          {unansweredCount > 0
            ? `You have ${unansweredCount} skipped question${unansweredCount === 1 ? "" : "s"} (0 marks, avoiding -1 negative marking). ` +
              "You won't be able to change your answers after submitting."
            : "You won't be able to change your answers after submitting."}
        </p>
        <div className="mt-5 flex justify-end gap-3">
          <Button variant="ghost" onClick={onCancel}>
            Keep reviewing
          </Button>
          <Button onClick={onConfirm}>Submit paper</Button>
        </div>
      </Card>
    </div>
  );
}

function ResultsView({
  review,
  totalInPaper,
  onExit,
  onStartNext,
}: {
  review: PaperReviewOut;
  totalInPaper: number;
  onExit: () => void;
  onStartNext: () => void;
}) {
  const unanswered = totalInPaper - review.total_questions;
  const [hints, setHints] = useState<Record<string, SocraticHintOut>>({});
  const [loadingHint, setLoadingHint] = useState<Record<string, boolean>>({});
  const [hintErrors, setHintErrors] = useState<Record<string, string>>({});

  async function handleFetchSocraticHint(questionId: string, selectedOption: string) {
    setLoadingHint((prev) => ({ ...prev, [questionId]: true }));
    setHintErrors((prev) => ({ ...prev, [questionId]: "" }));
    try {
      const hint = await api.getSocraticHint(questionId, selectedOption);
      setHints((prev) => ({ ...prev, [questionId]: hint }));
    } catch (err) {
      setHintErrors((prev) => ({
        ...prev,
        [questionId]: err instanceof ApiError ? err.message : "Failed to load Socratic hint.",
      }));
    } finally {
      setLoadingHint((prev) => ({ ...prev, [questionId]: false }));
    }
  }

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      <Card className="flex flex-col items-center gap-2 bg-midnight-950 p-8 text-center text-white">
        <p className="text-sm font-medium text-white/60">Sample Paper {review.paper_number} — result</p>
        <p className="font-display text-4xl font-semibold">
          {review.correct_count} / {review.total_questions}
        </p>
        <p className="text-white/70">{review.score_percent}% correct</p>
        {unanswered > 0 && (
          <p className="mt-1 text-xs text-white/50">
            {unanswered} question{unanswered === 1 ? "" : "s"} strategically skipped (0 marks)
          </p>
        )}
      </Card>

      <div className="flex flex-col gap-3">
        <h2 className="font-display text-lg font-semibold text-midnight-950">Answer review</h2>
        {review.questions.map((q, i) => {
          const hint = hints[q.question_id];
          return (
            <Card key={q.question_id} className="p-5">
              <p className="mb-1 text-xs font-medium uppercase tracking-wide text-midnight-700/50">
                Question {i + 1} · {q.topic_id}
              </p>
              <p className="mb-3 text-sm leading-relaxed text-midnight-950">{q.text}</p>
              <div className="flex flex-col gap-2">
                {q.options.map((optionText, optIdx) => {
                  const letter = OPTION_LETTERS[optIdx];
                  const isCorrectAnswer = letter === q.correct_answer;
                  const isStudentAnswer = letter === q.selected_option;
                  let classes = "border-midnight-900/10";
                  if (isCorrectAnswer) classes = "border-signal-correct bg-signal-correct/5";
                  else if (isStudentAnswer) classes = "border-signal-incorrect bg-signal-incorrect/5";
                  return (
                    <div
                      key={optIdx}
                      className={`flex items-center gap-3 rounded-lg border px-3 py-2 text-sm ${classes}`}
                    >
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-midnight-900/20 text-[11px] font-semibold uppercase text-midnight-700/60">
                        {letter}
                      </span>
                      <span className="text-midnight-900">{optionText}</span>
                      {isCorrectAnswer && (
                        <span className="ml-auto text-xs font-semibold text-signal-correct">Correct answer</span>
                      )}
                      {isStudentAnswer && !isCorrectAnswer && (
                        <span className="ml-auto text-xs font-semibold text-signal-incorrect">Your answer</span>
                      )}
                    </div>
                  );
                })}
              </div>

              {!q.is_correct && (
                <div className="mt-4">
                  {!hint ? (
                    <div className="flex flex-col items-start gap-1.5">
                      <Button
                        variant="secondary"
                        onClick={() => handleFetchSocraticHint(q.question_id, q.selected_option)}
                        disabled={loadingHint[q.question_id]}
                      >
                        {loadingHint[q.question_id]
                          ? "Retrieving NCERT Concept…"
                          : "Get NCERT Socratic Hint & Misconception Analysis"}
                      </Button>
                      {hintErrors[q.question_id] && (
                        <p className="text-xs text-signal-incorrect">{hintErrors[q.question_id]}</p>
                      )}
                    </div>
                  ) : (
                    <div className="rounded-xl border border-midnight-900/10 bg-midnight-900/5 p-4 text-xs leading-relaxed text-midnight-900">
                      <p className="font-semibold text-midnight-950">{hint.ncert_reference}</p>
                      {hint.concept_summary && (
                        <p className="mt-1">
                          <span className="font-semibold">NCERT Core Concept:</span> {hint.concept_summary}
                        </p>
                      )}
                      <p className="mt-1">
                        <span className="font-semibold">Diagnosed Misconception:</span>{" "}
                        {hint.diagnosed_misconception}
                      </p>
                      <p className="mt-1">
                        <span className="font-semibold">Socratic Prompt:</span> {hint.socratic_hint}
                      </p>
                      <p className="mt-1">
                        <span className="font-semibold">Key Formula:</span> {hint.key_formula}
                      </p>
                      {hint.verified_step_by_step && (
                        <p className="mt-1">
                          <span className="font-semibold">Step-by-Step Derivation:</span>{" "}
                          {hint.verified_step_by_step}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </Card>
          );
        })}
      </div>

      <div className="flex justify-center gap-3 pb-4">
        <Button variant="secondary" onClick={onExit}>
          Back to dashboard
        </Button>
        <Button onClick={onStartNext}>Start another paper</Button>
      </div>
    </div>
  );
}
