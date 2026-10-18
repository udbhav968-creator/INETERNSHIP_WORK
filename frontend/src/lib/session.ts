/**
 * Local persistence for "who is logged in right now".
 *
 * There's no JWT / server session yet — the backend just returns the
 * Student row on signup/login. We cache that in localStorage so a page
 * refresh doesn't drop the student back to a hardcoded test account.
 * Swap this for real token storage when the backend grows auth tokens.
 */
import type { Student } from "../types/auth";

const STORAGE_KEY = "klassed.student";

export function saveSession(student: Student): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(student));
}

export function loadSession(): Student | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Student;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function clearSession(): void {
  localStorage.removeItem(STORAGE_KEY);
}
