// Mirrors backend/app/schemas.py and db_models.py exactly. If the backend
// schema changes, update here first — the API client depends on this.

export type Difficulty = "basic" | "medium" | "advanced";
export type PaceFlag = "fast" | "slow" | null;
export type OptionLetter = "a" | "b" | "c" | "d";

export const OPTION_LETTERS: OptionLetter[] = ["a", "b", "c", "d"];

export interface TopicState {
  topic_id: string;
  level: Difficulty;
  pace_flag: PaceFlag;
  avg_time_ratio: number | null;
}

export interface QuestionOut {
  question_id: string;
  topic_id: string;
  difficulty: Difficulty;
  text: string;
  // Options arrive as a plain ordered list — index 0 is option "a", 1 is
  // "b", 2 is "c", 3 is "d". The backend's correct_answer field is a
  // lowercase letter, not option text, so submission must send the letter
  // corresponding to the clicked index, never the visible option text.
  options: string[];
}

export interface AttemptPayload {
  student_id: string;
  question_id: string;
  paper_number: number;
  selected_option: OptionLetter;
  time_taken_seconds: number;
}

export interface AttemptResult {
  is_correct: boolean;
  new_level: Difficulty;
  pace_flag: PaceFlag;
}
