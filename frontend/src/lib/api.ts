import type {
  Analytics,
  AttemptPayload,
  AttemptResult,
  PaperReviewOut,
  QuestionOut,
  TopicState,
} from "../types";

// Configurable base URL — defaults to local dev backend. Set
// VITE_API_BASE_URL in a .env file to point elsewhere.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

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
    // Network-level failure — backend likely not running at all.
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
};
