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

export interface PaperHistorySummary {
  paper_number: number;
  total_questions: number;
  correct_count: number;
  score_percent: number;
  total_time_seconds: number;
  timestamp: string;
  topics: string[];
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

export interface DinaSkillItem {
  skill_id: string;
  name: string;
  description: string;
  mastery_probability: number;
  posterior_sd: number;
  attempts_count: number;
  correct_count: number;
  status: "uncalibrated" | "mastered" | "developing" | "bottleneck";
}

export interface DinaCognitiveProfile {
  model_type: string;
  skills: DinaSkillItem[];
  primary_cognitive_bottleneck: {
    skill_id: string;
    name: string;
    mastery_probability: number;
    diagnosis: string;
  };
  speed_ability_profile: {
    latent_speed_tau: number;
    latent_speed_se: number;
    speed_percentile: number;
    avg_time_ratio: number;
    behavior_classification: {
      fluent_mastery: number;
      deep_derivation: number;
      careless_slip: number;
      rapid_guess: number;
      conceptual_struggle: number;
    };
  };
}

export interface MonteCarloConformalSimulator {
  model_type: string;
  n_trials: number;
  expected_score_attempt_all_720: number;
  conformal_90_interval: [number, number];
  interquartile_50_interval: [number, number];
  score_std_dev: number;
  negative_marking_optimizer: {
    ev_formula: string;
    optimal_skip_threshold_prob: number;
    optimal_skip_expected_score_720: number;
    expected_marks_gained_by_skipping: number;
    recommended_skip_questions_per_180: number;
    recommended_attempt_questions_per_180: number;
  };
  cutoff_probabilities: {
    qualifying_cutoff_165_percent: number;
    state_govt_mbbs_550_percent: number;
    top_aiims_650_percent: number;
  };
  high_yield_leverage_chapters: Array<{
    topic_id: string;
    subject: string;
    cluster: string;
    current_mastery: number;
    direct_marks_gain: number;
    downstream_diffused_marks_gain: number;
    total_expected_marks_gain: number;
    unlocks_downstream_count: number;
    downstream_chapters: string[];
  }>;
}

export interface V2StudentInsights {
  student_id: string;
  global_theta: number;
  estimated_neet_percentile: number;
  predicted_neet_score_720: number;
  zpd_target_success_prob: number;
  mirt_3d_profile?: {
    theta_conceptual: number;
    theta_formula: number;
    theta_numerical: number;
    attempts_conceptual: number;
    attempts_formula: number;
    attempts_numerical: number;
  };
  dina_cognitive_profile?: DinaCognitiveProfile;
  monte_carlo_conformal_simulator?: MonteCarloConformalSimulator;
  spaced_repetition_reviews?: Array<{
    topic_id: string;
    peak_mastery: number;
    decayed_mastery: number;
    elapsed_hours: number;
    retention_urgency: number;
  }>;
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
    cluster?: string;
    level?: Difficulty;
    current_mastery: number;
    readiness: number;
    neet_weightage?: number;
    prerequisites?: string[];
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
