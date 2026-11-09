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
  prior_mastery?: number;
  corpus_pyq_count?: number;
  posterior_sd: number;
  attempts_count: number;
  correct_count: number;
  status: "prior_calibrated" | "uncalibrated" | "mastered" | "developing" | "bottleneck";
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
  national_air_projection?: {
    total_national_candidates: number;
    conformal_air_interval: [number, number];
    projected_air_median: number;
    conformal_percentile_interval: [number, number];
    projected_percentile_median: number;
    college_admission_probabilities: Array<{
      college_id: string;
      college_tier: string;
      target_score_720: number;
      target_air_cutoff: number;
      admission_probability_percent: number;
      status: string;
    }>;
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

export interface SubjectCoverageItem {
  total_chapters: number;
  practiced_chapters: number;
  coverage_percent: number;
  mastered_chapters: number;
  developing_chapters: number;
  less_learned_chapters: number;
  untouched_chapters: number;
  mean_mastery_percent: number;
}

export interface LessLearnedLessonItem {
  topic_id: string;
  subject: string;
  cluster: string;
  current_level: Difficulty;
  peak_mastery_percent: number;
  decayed_mastery_percent: number;
  irt_theta: number;
  neet_weightage_percent: number;
  pace_flag: PaceFlag;
  blocking_prerequisites: string[];
  deficit_severity: number;
}

export interface ImprovementActionRow {
  rank: number;
  topic_id: string;
  subject: string;
  cluster: string;
  lesson_status: string;
  diagnostic_category: string;
  current_mastery_percent: number;
  decayed_mastery_percent: number;
  readiness_percent: number;
  neet_weightage_percent: number;
  expected_marks_gain_720: number;
  blocking_prerequisites: string[];
  recommended_start_tier: Difficulty;
  recommended_drill_questions: number;
  available_chapter_pyqs?: number;
  remediation_priority_score: number;
  what_to_improve: string;
}

export interface MistakeRetestItem {
  question_id: string;
  topic_id: string;
  subject: string;
  difficulty: Difficulty;
  time_taken_seconds: number;
  error_mode: string;
  ebbinghaus_retest_hours: number;
  chapter_mastery_percent: number;
}

export interface StudentStudyPlan {
  student_id: string;
  syllabus_coverage: {
    total_chapters: number;
    practiced_chapters: number;
    untouched_chapters: number;
    coverage_percent: number;
    mastered_chapters: number;
    developing_chapters: number;
    less_learned_chapters: number;
    retention_decay_alerts: number;
    subject_coverage: Record<string, SubjectCoverageItem>;
  };
  primary_cognitive_skill_to_improve: {
    skill_id: string;
    name: string;
    mastery_probability: number;
    diagnosis: string;
  };
  projected_neet_score_interval_90: [number, number];
  less_learned_lessons: LessLearnedLessonItem[];
  improvement_action_table: ImprovementActionRow[];
  mistake_retest_queue?: MistakeRetestItem[];
}

export interface CATAssessment {
  student_id: string;
  items_administered: number;
  latent_theta: number;
  sem_theta: number;
  confidence_interval_95: [number, number];
  cumulative_fisher_information: number;
  target_sem_threshold: number;
  stopping_rule_met: boolean;
  predicted_neet_score: number;
  assessment_status: string;
}

export interface RevisionDayPlan {
  day: number;
  date_label: string;
  chapter: string;
  subject: string;
  action_type: string;
  target_questions: number;
  urgency_score: number;
  potential_marks_gain: number;
  reason: string;
}

export interface RevisionCalendar {
  student_id: string;
  generated_at: string;
  total_days: number;
  daily_schedule: RevisionDayPlan[];
}

export interface FatigueAnalysis {
  student_id: string;
  total_attempts_analyzed: number;
  fatigue_index: number;
  latency_trend: string;
  mean_early_latency_sec: number;
  mean_late_latency_sec: number;
  early_accuracy_pct: number;
  late_accuracy_pct: number;
  accuracy_drop_pct: number;
  rapid_guess_count: number;
  pacing_coaching_advice: string;
}

export interface DashboardSnapshot {
  topics: TopicState[];
  analytics: Analytics;
  v2_insights: V2StudentInsights;
  paper_history: PaperHistorySummary[];
  study_plan?: StudentStudyPlan;
  cat_assessment?: CATAssessment;
  fatigue_analysis?: FatigueAnalysis;
  revision_calendar?: RevisionCalendar;
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
  irt_estimator?: string;
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
  calibrated_probabilities?: Record<string, number> | null;
  calibration_temperature?: number;
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
    champion_challenger_tournament?: Record<
      string,
      {
        holdout_accuracy: number;
        macro_f1: number;
        log_loss: number;
      }
    >;
    serverless_compact_rf_parity?: {
      verified_rows: number;
      exact_class_matches: number;
      max_abs_probability_diff: number;
    };
    irt_3pl_calibration_summary?: {
      calibrated_questions: number;
      discrimination_a_range: [number, number];
      difficulty_b_range: [number, number];
      guessing_c_range: [number, number];
    };
  };
  auxiliary_models_training?: {
    irt_cold_start_ridge_regressor?: {
      n_samples: number;
      ridge_alpha: number;
      a_5fold_rmse: number;
      a_5fold_r2: number;
      b_5fold_rmse: number;
      b_5fold_r2: number;
      c_5fold_rmse: number;
      c_5fold_r2: number;
    };
    bkt_65_chapter_empirical_bayes?: {
      total_chapters: number;
      weighted_global_prior: number;
      min_chapter_prior: number;
      max_chapter_prior: number;
    };
    dina_8_skill_calibrated?: {
      total_corpus_items: number;
      skill_priors: Record<string, number>;
    };
    ncert_pyq_svd_rag?: {
      model_type: string;
      n_chapters_indexed: number;
      vocab_size: number;
      lsa_dimensions: number;
      lsa_explained_variance_ratio: number;
    };
    linucb_and_laplacian?: {
      linucb_prior_r2: number;
      laplacian_spectral_gap: number;
    };
  };
  chapter_wise_summary?: {
    total_chapters: number;
    total_clean_pyqs: number;
    chapters_missing_basic: number;
    chapters_missing_medium: number;
    chapters_missing_advanced: number;
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
  artifact_registry?: {
    registry_version?: string;
    champion_classifier?: string;
    serverless_runtime?: string;
    artifacts?: Record<
      string,
      {
        exists: boolean;
        size_bytes: number;
        sha256: string;
      }
    >;
  };
  live_integrity_verification?: {
    integrity_status: string;
    verified_artifacts_count: number;
    calibrated_irt_items: number;
    canonical_chapters: number;
    cold_start_verification_ms: number;
  };
  distribution_summary?: {
    subject_difficulty_matrix: Record<
      string,
      {
        basic: number;
        medium: number;
        advanced: number;
        total: number;
        chapter_count: number;
      }
    >;
    gap_summary: {
      topics_missing_basic_count: number;
      topics_missing_medium_count: number;
      topics_missing_advanced_count: number;
      adjacent_tier_fallback_resolves_100_percent_topics: boolean;
    };
    blueprint_readiness: {
      max_non_repeating_45q_mocks?: number;
      max_non_repeating_180q_mocks?: number;
      current_adaptive_45q_blueprint?: {
        max_full_non_repeating_mocks: number;
      };
      target_neet_180q_blueprint?: {
        max_full_non_repeating_mocks: number;
      };
    };
  };
  tier_recalibration_audit?: {
    gap_resolution_summary: {
      chapters_missing_basic_before: number;
      chapters_resolvable_via_irt_anchors: number;
      physics_chapters_missing_basic_before: number;
      physics_chapters_resolvable_via_irt_anchors: number;
      chapters_missing_advanced_before: number;
      advanced_chapters_resolvable_via_irt_anchors: number;
      strict_global_basic_candidates_count: number;
      strict_global_advanced_candidates_count: number;
    };
    top_basic_recalibration_candidates: Array<{
      question_id: string;
      topic_id: string;
      subject: string;
      irt_b: number;
      ideal_time_sec: number;
      proposed_tier: string;
    }>;
  };
  question_similarity_audit?: {
    near_duplicate_pairs_count: number;
    near_duplicate_clusters_count: number;
    unique_effective_question_clusters: number;
    high_concept_overlap_pairs_count: number;
    top_near_duplicate_pairs: Array<{
      question_id_a: string;
      question_id_b: string;
      subject: string;
      chapter_a: string;
      chapter_b: string;
      cosine_similarity: number;
      snippet_a: string;
    }>;
  };
  probability_calibration?: {
    optimal_temperature: number;
    pre_calibration: {
      multiclass_nll: number;
      multiclass_brier_score: number;
      ece_10_bin: number;
    };
    post_calibration: {
      multiclass_nll: number;
      multiclass_brier_score: number;
      ece_10_bin: number;
    };
    nll_improvement_percent: number;
    ece_improvement_percent: number;
    serverless_compression_benchmark: {
      sklearn_joblib_size_bytes: number;
      serverless_bundle_total_bytes: number;
      compression_ratio_x: number;
      size_reduction_percent: number;
    };
  };
  data_integrity_audit?: {
    status?: string;
    blocking_failures?: string[];
    warnings?: string[];
    totals?: {
      questions: number;
      chapters: number;
      official_pyq_provenance_verified: number;
    };
    chi_square_uniformity?: Record<
      string,
      {
        chi2: number;
        p_value: number;
        significant_skew: boolean;
        counts: Record<string, number>;
      }
    >;
    synthetic_label_check?: {
      heuristic_quantiles: boolean;
      synthetic_ideal_time: boolean;
    };
    zero_tier_cells_current?: {
      count: number;
      cells: Array<{ subject: string; chapter: string; tier: string }>;
    };
    zero_tier_cells_original?: {
      count: number;
      cells: Array<{ subject: string; chapter: string; tier: string }>;
    };
  };
  label_validity_audit?: {
    what_the_rf_accuracy_measures?: string;
    baselines_5fold_cv_accuracy?: {
      majority_class: number;
      depth4_tree_on_ideal_time_sec_only: number;
      depth4_tree_on_ideal_time_sec_and_subject: number;
    };
    tier_vs_original_source_tags?: {
      available: boolean;
      n?: number;
      exact_agreement_percent?: number;
      cohens_kappa?: number;
      changed_from_original?: number;
    };
    student_responses_available?: number;
    empirical_difficulty_validation_possible?: boolean;
    interpretation?: string;
  };
}
