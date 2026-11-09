# Klassed Adaptive Engine — ML Intern Proposal: v2 Probabilistic & Psychometric Layer

**Author:** ML Intern  
**Date:** November 9, 2026  
**Context:** Review of `backend/app/state_machine.py` and `backend/app/time_tracking.py`  
**Status:** Proposal Only (For Team Torque & Lead Review; No Live Swaps Without Prior Approval)

---

## 1. Deep Technical Review of Existing Rule-Based Engine

### 1.1 Per-Topic Mastery Logic (`backend/app/state_machine.py`)
The existing engine implements a deterministic finite state machine over three discrete mastery tiers: `basic`, `medium`, and `advanced`.

```
                    2 of last 3 correct
       [ basic ] --------------------------> [ medium ]
                 <--------------------------
                    2 of last 3 incorrect

                    streak of 3 correct
       [ medium ] --------------------------> [ advanced ]
                  <--------------------------
                    2 consecutive incorrect
```

#### Key Characteristics:
1. **Window Size:** Evaluates a strict moving window of $W = 3$ items (`last_3_results`).
2. **Promotion Conditions:**
   - `basic` $\to$ `medium`: $\sum \text{correct} \ge 2$ in the last 3 results.
   - `medium` $\to$ `advanced`: Requires a consecutive correct streak $\ge 3$.
3. **Demotion Conditions:**
   - `medium` $\to$ `basic`: Requires $\ge 2$ incorrect in the last 3 results.
   - `advanced` $\to$ `medium`: Requires 2 consecutive incorrect attempts (`[False, False]`).
4. **State Resets:** Upon any tier transition, both `streak` and `last_3_results` are cleared to 0 / empty.

#### Strengths:
- **Interpretability:** Deterministic, easily explainable to students and educators.
- **Fast Response:** Transitions occur rapidly within 2 to 3 attempts.
- **Zero Computational Overhead:** Requires only small integer/boolean array operations.

#### Observed Limitations:
- **Sensitivity to Guessing & Slips:** In a 4-option multiple-choice exam like NEET, random guessing yields a 25% base probability. Two lucky guesses in a row can prematurely promote a basic student to medium. Conversely, a clerical slip on an advanced item can trigger an abrupt demotion.
- **Uniform Item Difficulty Assumption:** All questions within a tier are treated as having identical difficulty and discrimination. A challenging multi-step numerical item counts identically to a simple recall question.
- **Lack of Memory Decay:** Does not account for time elapsed between sessions. A student who achieved `advanced` mastery 3 months ago is treated identically to one who practiced yesterday.

---

### 1.2 Pace Tracking Logic (`backend/app/time_tracking.py`)
The pace tracker monitors solve velocity relative to calibrated benchmark response times.

$$\text{ratio} = \frac{\text{time\_taken\_seconds}}{\text{expected\_time(difficulty, expected\_time\_seconds)}}$$

$$\text{avg\_time\_ratio}_{n+1} = \frac{\text{avg\_time\_ratio}_n \cdot n + \text{ratio}}{n + 1}$$

$$\text{pace\_flag} = \begin{cases} \text{"slow"}, & \text{if } \text{avg\_time\_ratio} > 1.2 \\ \text{"fast"}, & \text{otherwise} \end{cases}$$

#### Key Characteristics:
1. **Conditional Update:** Updates are executed **strictly when `is_correct == True`**. Incorrect attempts are completely ignored (`if not is_correct: return state`).
2. **Cumulative Mean:** Maintains a running arithmetic mean over all historical correct attempts (`attempts_counted`).
3. **Binary Cutoff:** Uses a static threshold multiplier of $1.2\times$ benchmark time.

#### Strengths:
- **Noise Rejection:** Filtering out incorrect answers prevents random wrong guesses or question-abandonment from artificially deflating expected solve latency.
- **Stability:** The cumulative average converges to a stable representation of the student's characteristic solve speed.

#### Observed Limitations:
- **Blindness to Impulsive Guessing:** If a student impulsively clicks an option in 4 seconds and gets it wrong, `time_tracking.py` ignores the attempt. The system cannot distinguish between a thoughtful error and rash time-wasting.
- **Historical Inertia:** After 30 attempts, an arithmetic mean requires numerous slow or fast responses to shift the flag, failing to reflect recent improvements or fatigue.
- **Binary Classification:** A student at $1.21\times$ is tagged `"slow"` identically to a student at $3.5\times$, losing nuance for adaptive intervention.

---

## 2. Proposed v2 Layer: Hybrid Probabilistic & Psychometric Architecture

The proposed v2 layer is designed as an **additive statistical overlay** that preserves the rule-based state machine as a deterministic safety rail while providing continuous probabilistic calibration.

### 2.1 Continuous Bayesian Knowledge Tracing (BKT)
Rather than a binary tier, represent student mastery in topic $k$ as a continuous probability $P(L_t) \in [0, 1]$ updated via standard Bayes rule with slip ($S$) and guess ($G$) parameters:

$$P(L_t \mid \text{correct}) = \frac{P(L_{t-1}) \cdot (1 - P(S))}{P(L_{t-1}) \cdot (1 - P(S)) + (1 - P(L_{t-1})) \cdot P(G)}$$

$$P(L_t \mid \text{incorrect}) = \frac{P(L_{t-1}) \cdot P(S)}{P(L_{t-1}) \cdot P(S) + (1 - P(L_{t-1})) \cdot (1 - P(G))}$$

$$P(L_{t+1}) = P(L_t) + (1 - P(L_t)) \cdot P(T)$$

- **Hysteresis Guard:** Tiers transition only when posterior mastery crosses confidence bands (e.g., $P(L) > 0.75$ with upper 95% CI for promotion, $P(L) < 0.35$ with lower 95% CI for demotion), preventing rapid oscillating transitions.

### 2.2 3PL Item Response Theory (IRT) Latent Ability Estimation
Model item-student interactions using the standard Three-Parameter Logistic (3PL) model:

$$P(Y_{ij} = 1 \mid \theta_i) = c_j + \frac{1 - c_j}{1 + \exp(-a_j(\theta_i - b_j))}$$

- $a_j$: Item discrimination parameter.
- $b_j$: Item difficulty parameter.
- $c_j$: Pseudo-guessing asymptote ($c_j \approx 0.20$ for 4-option NEET items).
- $\theta_i$: Student latent ability on the standard normal scale $\theta \sim \mathcal{N}(0, 1)$.

Ability is estimated online via Maximum A Posteriori (MAP) with Fisher Information maximization for Computerized Adaptive Testing (CAT).

### 2.3 Response-Time Modulated Telemetry (van der Linden RT Model)
Extend pace tracking by logging response latency across **both** correct and incorrect answers using a log-normal response time model:

$$\ln(T_{ij}) = \beta_j - \tau_i + \epsilon_{ij}, \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma_j^2)$$

- **Impulsive Guess Trap:** If $T_{ij} < 15\text{s}$ and $Y_{ij} = 0$, classify as `IMPULSIVE_GUESS`. Rather than reducing estimated mastery, flag a pacing deficit and prescribe deliberate step-by-step reading drills.
- **Cognitive Exhaustion:** If $T_{ij} > 2.0 \times \text{expected}$ and late in an exam session, classify as `COGNITIVE_FATIGUE` rather than conceptual failure.

### 2.4 Ebbinghaus Forgetting Curve Integration
Incorporate retention decay between practice sessions:

$$R(t) = \exp\left(-\frac{\Delta t}{S}\right)$$

Where $S$ is memory stability proportional to historical streak and mastery, and $\Delta t$ is elapsed hours. When retention falls below a threshold ($R < 0.60$), the paper generator schedules a targeted spaced-repetition review.

---

## 3. Comparison Matrix: Rule-Based (v1) vs Hybrid Overlay (v2)

| Dimension | Current Rule-Based Engine (v1) | Proposed Hybrid Overlay (v2) |
| :--- | :--- | :--- |
| **Mastery Representation** | Discrete: `basic`, `medium`, `advanced` | Dual: Discrete tier + continuous $P(L) \in [0, 1]$ and $\theta \in [-3, +3]$ |
| **Observation Window** | Last 3 attempts | Full Bayesian sequence with exponential recency weighting |
| **Guess & Slip Handling** | Uniform assumptions; vulnerable to 2-guess streaks | Explicit slip ($P(S)$) and guess ($P(G)$) conditioning |
| **Item Differentiation** | Tier label only | Calibrated IRT parameters ($a, b, c$) per question |
| **Pace Handling** | Cumulative mean of correct answers only | Log-normal response time on all attempts; detects impulsive guesses & fatigue |
| **Retention Decay** | Static state persisted indefinitely | Ebbinghaus exponential decay over elapsed session hours |
| **Paper Generation** | Heuristic weightage with adjacent fallback | Contextual Multi-Armed Bandit (LinUCB) optimizing information gain in ZPD |

---

## 4. Proposed Implementation Roadmap

```mermaid
flowchart TD
    subgraph Phase 1 [Phase 1: Shadow Telemetry - Weeks 1 to 2]
        P1A["Log full response telemetry (time, option switches, correctness)"]
        P1B["Run BKT & 3PL IRT in shadow mode (read-only)"]
        P1C["Compute agreement metrics between v1 rules and v2 posteriors"]
    end

    subgraph Phase 2 [Phase 2: Offline Calibration - Weeks 3 to 4]
        P2A["Gather first 1,000 real student test attempts"]
        P2B["Calibrate empirical IRT b-parameters from response data"]
        P2C["Validate item difficulty baselines against shallow tree benchmarks"]
    end

    subgraph Phase 3 [Phase 3: Soft Gating & A/B Trial - Weeks 5 to 6]
        P3A["Use v2 continuous mastery to break ties in question selection"]
        P3B["Retain v1 state machine as hard boundary and fallback rail"]
        P3C["Measure student score improvement and retention in controlled cohort"]
    end

    Phase 1 --> Phase 2
    Phase 2 --> Phase 3
```

---

## 5. Ground Rules & Safety Commitments
1. **No Live Swap Without Approval:** The v1 rule-based state machine remains authoritative. The v2 probabilistic engine runs purely in observer/shadow mode until reviewed by the team lead.
2. **Preserve Fallbacks:** The adjacent-tier fallback in `paper_generator.py` remains active so that missing tiers in any topic never cause a subject to drop out.
3. **No Fabricated Data:** Empirical difficulty parameters ($b$) will be estimated strictly from logged student response telemetry, with uncertainty bounds reported until sample sizes achieve convergence.
