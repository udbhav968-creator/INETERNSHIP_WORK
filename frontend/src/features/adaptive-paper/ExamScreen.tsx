import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../../lib/api";
import { OPTION_LETTERS, type AttemptResult, type OptionLetter, type QuestionOut } from "../../types";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { DifficultyBadge } from "../../components/ui/Badge";

interface ExamScreenProps {
  studentId: string;
  paperNumber: number;
}

type LoadState = "loading" | "ready" | "error" | "empty";

export function ExamScreen({ studentId, paperNumber }: ExamScreenProps) {
  const [questions, setQuestions] = useState<QuestionOut[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState("");
  const [index, setIndex] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [result, setResult] = useState<AttemptResult | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const startedAt = useRef<number>(Date.now());

  useEffect(() => {
    let cancelled = false;
    api
      .getNextPaper(studentId)
      .then((data) => {
        if (cancelled) return;
        setQuestions(data);
        setLoadState(data.length === 0 ? "empty" : "ready");
        startedAt.current = Date.now();
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

  const current = questions[index];

  async function handleSubmit() {
    if (selected === null || !current) return;
    setSubmitting(true);
    setSubmitError("");

    const timeTaken = Math.max(1, Math.round((Date.now() - startedAt.current) / 1000));
    const letter: OptionLetter = OPTION_LETTERS[selected];

    try {
      const res = await api.submitAttempt({
        student_id: studentId,
        question_id: current.question_id,
        paper_number: paperNumber,
        selected_option: letter,
        time_taken_seconds: timeTaken,
      });
      setResult(res);
    } catch (err) {
      setSubmitError(err instanceof ApiError ? err.message : "Couldn't submit your answer. Try again.");
    } finally {
      setSubmitting(false);
    }
  }

  function handleNext() {
    setIndex((i) => i + 1);
    setSelected(null);
    setResult(null);
    setSubmitError("");
    startedAt.current = Date.now();
  }

  if (loadState === "loading") {
    return (
      <div className="mx-auto max-w-2xl">
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

  if (index >= questions.length) {
    return (
      <Card className="mx-auto max-w-xl text-center">
        <p className="font-display text-xl font-semibold text-midnight-950">Paper complete.</p>
        <p className="mt-2 text-sm text-midnight-700/70">
          You've worked through every question in this set. Your dashboard reflects the updated levels.
        </p>
      </Card>
    );
  }

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-5">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-midnight-700/60">
          Question {index + 1} of {questions.length}
        </p>
        <div className="flex items-center gap-2">
          <DifficultyBadge level={current.difficulty} />
        </div>
      </div>

      <div className="h-1.5 w-full overflow-hidden rounded-full bg-midnight-900/5">
        <div
          className="h-full rounded-full bg-accent transition-all"
          style={{ width: `${((index + (result ? 1 : 0)) / questions.length) * 100}%` }}
        />
      </div>

      <Card className="p-7">
        <p className="mb-1 text-xs font-medium uppercase tracking-wide text-midnight-700/50">{current.topic_id}</p>
        <p className="mb-6 text-lg leading-relaxed text-midnight-950">{current.text}</p>

        <div className="flex flex-col gap-3">
          {current.options.map((optionText, i) => {
            const isSelected = selected === i;
            const showResult = result !== null;
            let stateClasses = "border-midnight-900/10 hover:border-midnight-900/25";
            if (showResult && isSelected) {
              stateClasses = result.is_correct
                ? "border-signal-correct bg-signal-correct/5"
                : "border-signal-incorrect bg-signal-incorrect/5";
            } else if (isSelected) {
              stateClasses = "border-accent bg-accent/5";
            }

            return (
              <button
                key={i}
                disabled={showResult}
                onClick={() => setSelected(i)}
                className={`focus-ring flex items-center gap-3 rounded-xl border px-4 py-3.5 text-left text-sm transition-colors disabled:cursor-default ${stateClasses}`}
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

        {submitError && <p className="mt-4 text-sm text-signal-incorrect">{submitError}</p>}

        {result && (
          <div
            className={`mt-5 rounded-xl px-4 py-3 text-sm font-medium ${
              result.is_correct ? "bg-signal-correct/10 text-signal-correct" : "bg-signal-incorrect/10 text-signal-incorrect"
            }`}
          >
            {result.is_correct ? "Correct." : "Not quite."} Topic level is now{" "}
            <span className="font-semibold">{result.new_level}</span>.
          </div>
        )}

        <div className="mt-6 flex justify-end">
          {!result ? (
            <Button onClick={handleSubmit} disabled={selected === null || submitting}>
              {submitting ? "Checking…" : "Submit answer"}
            </Button>
          ) : (
            <Button onClick={handleNext}>
              {index + 1 < questions.length ? "Next question" : "Finish paper"}
            </Button>
          )}
        </div>
      </Card>
    </div>
  );
}
