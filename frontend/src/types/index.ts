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

export interface DashboardSnapshot {
  topics: TopicState[];
  analytics: Analytics;
  v2_insights: V2StudentInsights;
  paper_history: PaperHistorySummary[];
}

export interface ColdStartItemInput {
  question_text: string;
  options?: string[];
  has_diagram?: boolean;
  subject?: string;
  chapter?: string;
  topic_tag?: string;
  question_type?: string;
  ideal_time_sec?: number;
}

export interface ColdStartItemPrediction {
  predicted_difficulty: Difficulty;
  rf_predicted_difficulty: Difficulty | null;
  rf_engine: string;
  complexity_score: number;
  irt_params: {
    a: number;
    b: number;
    c: number;
  };
  nlp_features: {
    word_count: number;
    formula_density: number;
    numerical_complexity: number;
    negation_trap: number;
    distractor_similarity: number;
  };
  rf_probabilities: Record<string, number> | null;
  resolved_metadata: {
    subject: string;
    chapter: string;
    topic_tag: string;
    question_type: string;
    ideal_time_sec: number;
    has_diagram: boolean;
  };
  mirt_dimension: "conceptual" | "formula" | "numerical";
  dina_q_vector: Record<string, number>;
  active_cognitive_skills: Array<{
    skill_id: string;
    name: string;
    description: string;
  }>;
}

export interface MlopsHealthAndDrift {
  feature_store: {
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    cache_hit_rate: number;
    cached_students: number;
  };
  drift_and_calibration: {
    psi_difficulty: number;
    psi_response_time: number;
    effective_psi: number;
    brier_score: number;
    expected_calibration_error: number;
    drift_status: string;
    recalibration_required: boolean;
    sample_size: number;
  };
  models_loaded: Record<string, boolean | number>;
}

export interface MlopsAuditOverview {
  health_and_drift: MlopsHealthAndDrift;
  corpus_integrity: {
    verified_questions_in_db: number;
    canonical_chapters_in_db: number;
    subject_question_counts: Record<string, number>;
    difficulty_tier_counts: Record<string, number>;
    rag_ncert_chapters: number;
    calibrated_irt_items: number;
    dag_nodes_count: number;
    dag_edges_count: number;
    dag_acyclic: boolean;
  };
  validation_report: {
    pipeline_status?: string;
    execution_time_seconds?: number;
    random_forest_holdout_evaluation?: {
      training_rows: number;
      testing_rows: number;
      correct_predictions: number;
      holdout_accuracy: number;
      macro_f1: number;
      weighted_f1: number;
      confusion_matrix_labels: string[];
      confusion_matrix: number[][];
      per_class_metrics: Record<
        string,
        {
          precision: number;
          recall: number;
          f1_score: number;
          support: number;
        }
      >;
    };
    stratified_5fold_cross_validation?: {
      n_splits: number;
      fold_accuracies: number[];
      mean_cv_accuracy: number;
      std_cv_accuracy: number;
      mean_cv_macro_f1: number;
      mean_cv_log_loss: number;
      oof_multiclass_brier_score: number;
    };
    irt_3pl_calibration_summary?: {
      calibrated_questions: number;
      discrimination_a_range: [number, number];
      difficulty_b_range: [number, number];
      guessing_c_range: [number, number];
    };
  };
  harness_benchmark: {
    harness_status?: string;
    execution_time_seconds?: number;
    persona_simulations?: {
      strong_master?: {
        final_level: string;
        final_mastery_prob: number;
        final_theta: number;
        estimated_percentile: number;
        invariants_satisfied: boolean;
      };
      foundational_struggler?: {
        final_level: string;
        final_mastery_prob: number;
        final_theta: number;
        detected_bottlenecks: number;
        invariants_satisfied: boolean;
      };
      impulsive_guesser_telemetry_audit?: {
        deliberate_posterior: number;
        impulsive_guess_posterior: number;
        damping_delta: number;
        invariants_satisfied: boolean;
      };
      spaced_repetition_learner?: {
        peak_mastery: number;
        decayed_mastery_after_168h: number;
        retention_urgency: number;
        invariants_satisfied: boolean;
      };
    };
    linucb_vs_random_ablation?: {
      n_rounds: number;
      linucb_mean_zpd_alignment: number;
      random_mean_zpd_alignment: number;
      linucb_mean_fisher_info: number;
      random_mean_fisher_info: number;
      linucb_cumulative_regret: number;
      random_cumulative_regret: number;
      linucb_outperforms_random: boolean;
    };
    latency_sla_benchmarks?: Record<
      string,
      | boolean
      | {
          p50_ms: number;
          p95_ms: number;
          p99_ms: number;
        }
    >;
  };
  chapters_by_subject: Record<string, string[]>;
}
