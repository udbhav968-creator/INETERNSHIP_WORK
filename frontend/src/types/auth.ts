// Merge these into your existing src/types/index.ts (or keep this file
// and `export * from "./auth"` from index.ts — whichever matches your
// current setup).

export type ClassLevel = "11" | "12" | "dropper";

export interface Student {
  student_id: string;
  name: string;
  email: string;
  class_level: ClassLevel | null;
  target_neet_year: number | null;
  onboarded: boolean;
}

export interface SignupPayload {
  name: string;
  email: string;
  password: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface OnboardingPayload {
  class_level: ClassLevel;
  target_neet_year: number;
}
