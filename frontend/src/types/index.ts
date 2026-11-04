// Mirrors backend/app/schemas.py and db_models.py (v1 + Enterprise v2 Psychometric & Socratic RAG).

export type Difficulty = "basic" | "medium" | "advanced";
export type PaceFlag = "fast" | "slow" | null;
export type OptionLetter = "a" | "b" | "c" | "d";

export const OPTION_LETTERS: OptionLetter[] = ["a", "b", "c", "d"];

export interface TopicState {
  topic_id: string;
  level: Difficulty;
  pace_flag: PaceFlag;
  avg_time_ratio: number | null;
  mastery_prob?: number;
  confidence_interval?: number;
  irt_theta?: number;
}

export interface QuestionOut {
  question_id: string;
  topic_id: string;
  difficulty: Difficulty;
  text: string;
  options: string[];
  expected_time_seconds?: number;
}

export interface AttemptPayload {
  student_id: string;
  question_id: string;
  paper_number: number;
  selected_option: OptionLetter;
  time_taken_seconds: number;
  option_switches?: number;
  hesitation_ms?: number;
  dwell_time_ms?: number;
}

export interface AttemptResult {
  is_correct: boolean;
  new_level: Difficulty;
  pace_flag: PaceFlag;
  mastery_prob?: number;
  confidence_interval?: number;
  irt_theta?: number;
}

// --- paper review (post-submission only) ---

export interface QuestionReviewOut {
  question_id: string;
  topic_id: string;
  text: string;
  options: string[];
  correct_answer: OptionLetter;
  selected_option: OptionLetter;
  is_correct: boolean;
}

export interface PaperReviewOut {
  paper_number: number;
  total_questions: number;
  correct_count: number;
  score_percent: number;
  questions: QuestionReviewOut[];
}

// --- analytics ---

export interface SubjectBreakdown {
  subject: string;
  attempts: number;
  correct: number;
  accuracy: number;
}

export interface RecentAttempt {
  question_id: string;
  topic_id: string;
  is_correct: boolean;
  time_taken_seconds: number;
  timestamp: string;
}

export interface Analytics {
  total_attempts: number;
  correct_attempts: number;
  accuracy: number;
  topics_practiced: number;
  subject_breakdown: SubjectBreakdown[];
  recent_attempts: RecentAttempt[];
}

// --- v2 Psychometric Insights & Socratic RAG Remediation ---

export interface SocraticHintOut {
  question_id: string;
  chapter: string;
  ncert_reference: string;
  concept_summary: string;
  key_formula: string;
  diagnosed_misconception: string;
  socratic_hint: string;
  verified_step_by_step: string;
  retrieval_similarity: number;
}

export interface V2StudentInsights {
  student_id: string;
  global_theta: number;
  estimated_neet_percentile: number;
  predicted_neet_score_720: number;
  zpd_target_success_prob: number;
  root_cause_bottlenecks: Array<{
    weak_topic: string;
    weak_topic_mastery: number;
    root_cause_prerequisite: string;
    prerequisite_mastery: number;
    recommendation: string;
  }>;
  recommended_unlock_topics: Array<{
    topic_id: string;
    subject: string;
    current_mastery: number;
    readiness: number;
    zpd_priority_score: number;
  }>;
  topic_mastery_details: Array<{
    topic_id: string;
    level: Difficulty;
    mastery_prob: number;
    confidence_interval: number;
    irt_theta: number;
    readiness_score: number;
    pace_flag: PaceFlag;
  }>;
}
