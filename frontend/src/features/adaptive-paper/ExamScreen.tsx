import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../../lib/api";
import { OPTION_LETTERS, type OptionLetter, type PaperReviewOut, type QuestionOut } from "../../types";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";

interface ExamScreenProps {
  studentId: string;
  paperNumber: number;
  /** Called when the student leaves the exam/results screen for the dashboard. */
  onExit: () => void;
  /** Called when the student asks to start a fresh paper from the results screen. */
  onStartNext: () => void;
}

type LoadState = "loading" | "ready" | "error" | "empty" | "submitting" | "submit-error" | "results";

/**
 * Real paper-taking flow: load the full paper up front, let the student
 * move freely between questions and change answers, then submit the whole
 * paper at once. No per-question correctness and no difficulty tier is
 * ever shown while solving — that only happens for a developer tool, not
 * the student. Scoring + the correct-answer review come from
 * GET /students/{id}/papers/{paperNumber}/review, which only reveals
 * answers for questions already submitted.
 */
export function ExamScreen({ studentId, paperNumber, onExit, onStartNext }: ExamScreenProps) {
  const [questions, setQuestions] = useState<QuestionOut[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState("");
  const [index, setIndex] = useState(0);
  // questionIndex -> selected option index (0-3)
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [review, setReview] = useState<PaperReviewOut | null>(null);
  const [confirmingSubmit, setConfirmingSubmit] = useState(false);

  // First time each question is viewed, remember when — used to compute
  // time_taken_seconds per question at submit time.
  const firstVisitedAt = useRef<Record<number, number>>({});

  useEffect(() => {
    let cancelled = false;
    api
      .getNextPaper(studentId)
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
  }, [studentId]);

  useEffect(() => {
    if (loadState !== "ready") return;
    if (!(index in firstVisitedAt.current)) {
      firstVisitedAt.current[index] = Date.now();
    }
  }, [index, loadState]);

  const current = questions[index];
  const answeredCount = Object.keys(answers).length;
  const unansweredCount = questions.length - answeredCount;

  function selectOption(optionIndex: number) {
    setAnswers((prev) => ({ ...prev, [index]: optionIndex }));
  }

  function goTo(i: number) {
    setIndex(i);
  }

  async function handleSubmitPaper() {
    setConfirmingSubmit(false);
    setLoadState("submitting");
    setErrorMessage("");

    try {
      // Sequential, not parallel — submissions update the same per-topic
      // mastery state machine, and we want them applied in the order the
      // student actually answered, not racing each other.
      for (const [qIndexStr, optionIndex] of Object.entries(answers)) {
        const qIndex = Number(qIndexStr);
        const question = questions[qIndex];
        if (!question) continue;

        const visitedAt = firstVisitedAt.current[qIndex] ?? Date.now();
        const timeTaken = Math.max(1, Math.round((Date.now() - visitedAt) / 1000));
        const letter: OptionLetter = OPTION_LETTERS[optionIndex];

        await api.submitAttempt({
          student_id: studentId,
          question_id: question.question_id,
          paper_number: paperNumber,
          selected_option: letter,
          time_taken_seconds: timeTaken,
        });
      }

      const reviewData = await api.getPaperReview(studentId, paperNumber);
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

  // "ready", "submitting", "submit-error" all render the paper itself —
  // submitting/submit-error just overlay state on top of it.
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 lg:flex-row">
      {/* Question navigator */}
      <aside className="order-2 flex flex-col gap-3 lg:order-1 lg:w-56 lg:shrink-0">
        <Card className="p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wide text-midnight-700/50">
            Sample Paper {paperNumber}
          </p>
          <div className="grid grid-cols-6 gap-2 lg:grid-cols-5">
            {questions.map((_, i) => {
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
            {answeredCount} of {questions.length} answered
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

      {/* Current question */}
      <div className="order-1 flex-1 lg:order-2">
        <div className="mb-3 flex items-center justify-between">
          <p className="text-sm font-medium text-midnight-700/60">
            Question {index + 1} of {questions.length}
          </p>
        </div>

        <Card className="p-7">
          <p className="mb-1 text-xs font-medium uppercase tracking-wide text-midnight-700/50">
            {current.topic_id}
          </p>
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

          <div className="mt-6 flex items-center justify-between">
            <Button variant="secondary" onClick={() => goTo(index - 1)} disabled={index === 0}>
              Previous
            </Button>
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
            ? `You have ${unansweredCount} unanswered question${unansweredCount === 1 ? "" : "s"}. ` +
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
            {unanswered} question{unanswered === 1 ? "" : "s"} left unanswered
          </p>
        )}
      </Card>

      <div className="flex flex-col gap-3">
        <h2 className="font-display text-lg font-semibold text-midnight-950">Answer review</h2>
        {review.questions.map((q, i) => (
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
          </Card>
        ))}
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
