import { appendJournalAttempts, encodeJournalHeader, loadSession } from "./session";
import type {
  Analytics,
  AttemptPayload,
  AttemptResult,
  DashboardSnapshot,
  PaperHistorySummary,
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
  const session = loadSession();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (session?.access_token) {
    headers.Authorization = `Bearer ${session.access_token}`;
  }
  if (session?.student_id) {
    const journalHdr = encodeJournalHeader(session.student_id);
    if (journalHdr) {
      headers["X-Klassed-Journal"] = journalHdr;
    }
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        ...headers,
        ...(options?.headers as Record<string, string> | undefined),
      },
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
      detail = res.statusText;
    }
    throw new ApiError(detail, res.status);
  }

  return res.json() as Promise<T>;
}

export const api = {
  getDashboardSnapshot: (studentId: string) =>
    request<DashboardSnapshot>(`/students/${encodeURIComponent(studentId)}/dashboard_snapshot`),

  getStudentState: (studentId: string) =>
    request<TopicState[]>(`/students/${encodeURIComponent(studentId)}/state`),

  getNextPaper: (studentId: string, topicId?: string) => {
    const query = topicId ? `?topic_id=${encodeURIComponent(topicId)}` : "";
    return request<QuestionOut[]>(`/students/${encodeURIComponent(studentId)}/next_paper${query}`);
  },

  submitAttempt: async (payload: AttemptPayload) => {
    const res = await request<AttemptResult>("/submit_attempt", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    appendJournalAttempts(payload.student_id, payload.paper_number, [payload]);
    return res;
  },

  submitPaper: async (studentId: string, paperNumber: number, attempts: AttemptPayload[]) => {
    const review = await request<PaperReviewOut>("/submit_paper", {
      method: "POST",
      body: JSON.stringify({
        student_id: studentId,
        paper_number: paperNumber,
        attempts,
      }),
    });
    appendJournalAttempts(studentId, review.paper_number, attempts);
    return review;
  },

  getPaperReview: (studentId: string, paperNumber: number) =>
    request<PaperReviewOut>(
      `/students/${encodeURIComponent(studentId)}/papers/${paperNumber}/review`
    ),

  getPaperHistory: (studentId: string) =>
    request<PaperHistorySummary[]>(
      `/students/${encodeURIComponent(studentId)}/papers/history`
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
