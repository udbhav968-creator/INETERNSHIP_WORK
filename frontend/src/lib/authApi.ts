import { loadSession } from "./session";
import type { LoginPayload, OnboardingPayload, SignupPayload, Student } from "../types/auth";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  (typeof window !== "undefined" &&
  window.location.hostname !== "localhost" &&
  window.location.hostname !== "127.0.0.1"
    ? ""
    : "http://127.0.0.1:8000");

async function parseOrThrow(res: Response) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return body;
}

export async function signup(payload: SignupPayload): Promise<Student> {
  const res = await fetch(`${API_BASE}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseOrThrow(res);
}

export async function login(payload: LoginPayload): Promise<Student> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseOrThrow(res);
}

export async function completeOnboarding(
  studentId: string,
  payload: OnboardingPayload
): Promise<Student> {
  const token = loadSession()?.access_token;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await fetch(`${API_BASE}/students/${encodeURIComponent(studentId)}/onboarding`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  return parseOrThrow(res);
}
