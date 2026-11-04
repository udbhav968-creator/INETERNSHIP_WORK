import type {
  Analytics,
  AttemptPayload,
  AttemptResult,
  PaperReviewOut,
  QuestionOut,
  SocraticHintOut,
  TopicState,
  V2StudentInsights,
} from "../types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  (typeof window !== "undefined" &&
  window.location.hostname !== "localhost" &&
  window.location.hostname !== "127.0.0.1"
    ? ""
    : "http://127.0.0.1:8000");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError("Could not reach the backend. Is the server running?", 0);
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON — fall back to statusText
    }
    throw new ApiError(detail, res.status);
  }

  return res.json() as Promise<T>;
}

export const api = {
  getStudentState: (studentId: string) =>
    request<TopicState[]>(`/students/${encodeURIComponent(studentId)}/state`),

  getNextPaper: (studentId: string) =>
    request<QuestionOut[]>(`/students/${encodeURIComponent(studentId)}/next_paper`),

  submitAttempt: (payload: AttemptPayload) =>
    request<AttemptResult>("/submit_attempt", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getPaperReview: (studentId: string, paperNumber: number) =>
    request<PaperReviewOut>(
      `/students/${encodeURIComponent(studentId)}/papers/${paperNumber}/review`
    ),

  getAnalytics: (studentId: string) =>
    request<Analytics>(`/students/${encodeURIComponent(studentId)}/analytics`),

  getV2Insights: (studentId: string) =>
    request<V2StudentInsights>(`/students/${encodeURIComponent(studentId)}/v2_insights`),

  getSocraticHint: (questionId: string, selectedOption: string, timeTakenSeconds = 60, optionSwitches = 0) =>
    request<SocraticHintOut>("/remediation/socratic_hint", {
      method: "POST",
      body: JSON.stringify({
        question_id: questionId,
        selected_option: selectedOption,
        time_taken_seconds: timeTakenSeconds,
        option_switches: optionSwitches,
      }),
    }),
};
