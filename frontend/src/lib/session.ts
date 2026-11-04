import type { Student } from "../types/auth";

const STORAGE_KEY = "klassed.student";
const LOGOUT_FLAG_KEY = "klassed.logged_out";

const DEFAULT_DEMO_STUDENT: Student = {
  student_id: "UDBHAVYADAV",
  name: "Udbhav Yadav",
  email: "udbhav@klassed.in",
  class_level: "12",
  target_neet_year: 2027,
  onboarded: true,
};

export function saveSession(student: Student): void {
  localStorage.removeItem(LOGOUT_FLAG_KEY);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(student));
}

export function loadSession(): Student | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (raw) {
    try {
      return JSON.parse(raw) as Student;
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
  }
  if (localStorage.getItem(LOGOUT_FLAG_KEY) === "1") {
    return null;
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_DEMO_STUDENT));
  return DEFAULT_DEMO_STUDENT;
}

export function clearSession(): void {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.setItem(LOGOUT_FLAG_KEY, "1");
}
