import type { AttemptPayload } from "../types";
import type { Student } from "../types/auth";

const STORAGE_KEY = "klassed.student";
const JOURNAL_KEY_PREFIX = "klassed.journal.";

export type CompactJournalItem = [number, string, string, number, number, number, number];

export function saveSession(student: Student): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(student));
}

export function loadSession(): Student | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as Student;
    if (!parsed || !parsed.student_id || !parsed.access_token) {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return parsed;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function clearSession(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function loadJournal(studentId: string): CompactJournalItem[] {
  if (!studentId) return [];
  const raw = localStorage.getItem(`${JOURNAL_KEY_PREFIX}${studentId}`);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CompactJournalItem[]) : [];
  } catch {
    return [];
  }
}

export function appendJournalAttempts(
  studentId: string,
  paperNumber: number,
  attempts: AttemptPayload[]
): void {
  if (!studentId || !attempts.length) return;
  const existing = loadJournal(studentId);
  const seen = new Set(existing.map((it) => `${it[0]}:${it[1]}`));
  for (const att of attempts) {
    const pNum = paperNumber || att.paper_number || 1;
    const key = `${pNum}:${att.question_id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    existing.push([
      pNum,
      att.question_id,
      att.selected_option,
      att.time_taken_seconds,
      att.option_switches ?? 0,
      att.hesitation_ms ?? 0,
      att.dwell_time_ms ?? att.time_taken_seconds * 1000,
    ]);
  }
  const trimmed = existing.slice(-75);
  localStorage.setItem(`${JOURNAL_KEY_PREFIX}${studentId}`, JSON.stringify(trimmed));
}

export function encodeJournalHeader(studentId: string): string | null {
  const items = loadJournal(studentId);
  if (!items.length) return null;
  try {
    const b64 = btoa(JSON.stringify(items))
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");
    return b64;
  } catch {
    return null;
  }
}
