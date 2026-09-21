# Lab 2: ReVA evaluation and Qwen LoRA

## Setup

I used Qwen3-VL-4B-Instruct and VILA1.5-3b, as specified in the assignment. Qwen was fine-tuned on the first 200 official training questions, using one epoch, batch size 1, gradient accumulation 4, and the supplied learning rate of `2e-7`. LoRA used rank 8, alpha 16, and dropout 0 on the query, key, value, and output projections. This trained 5,898,240 parameters in 50 steps. The mean training loss was 0.4115.

All three evaluations use the same 4,000 official test questions from 1,014 videos, with at most four frames per video. All three use greedy decoding. The supplied pipelines use different prompts and output limits: Qwen is asked for reasoning and an answer tag, with up to 1,024 new tokens, while VILA is asked for a letter, with up to 128 new tokens. The score is correct answers divided by all 4,000 questions; missing or unparseable answers count as wrong. Completion is reported separately.

The 200 training questions come from 17 VisDrone videos and cover all 11 question types. There are no repeated video/question pairs between this training subset and the test set. However, 16 training videos also occur in the test set, accounting for 68 test questions. ReVA uses a question-level split, so this is not a test entirely on unseen videos.

## Results

| Model | Correct / total | Accuracy | Parseable answers |
|---|---:|---:|---:|
| Qwen base | 2,675 / 4,000 | 66.88% | 3,971 / 4,000 |
| Qwen LoRA | 2,673 / 4,000 | 66.83% | 3,964 / 4,000 |
| VILA base | 1,940 / 4,000 | 48.50% | 4,000 / 4,000 |

All three runs generated a response for every test question. Qwen base had 29 unparseable answers and Qwen LoRA had 36; these were counted as wrong.

LoRA did not improve overall accuracy in this run: it corrected 93 previously wrong answers but changed 95 correct answers to wrong ones, leaving two fewer correct answers, or a decrease of 0.05 percentage points. The low training loss did not translate into better test accuracy. This was a small, one-epoch run with the supplied low learning rate, so the result does not show that LoRA cannot help with more suitable training settings. Qwen scored higher than VILA under these evaluation settings; their different prompts and output limits should be kept in mind when comparing them.

The saved comparison is `outputs/model_comparison.csv`.

## Three examples

### 1. Counting cars: QA-000005

Video: `data/ReVA/VisDrone/uav0000009_03358_v_01.mp4`

Question: How many cars are visible in the parking lot at 00:00?

Options: A: 40–50; B: 10–20; C: 20–30; D: 30–40.

| Official answer | Qwen base | Qwen LoRA | VILA |
|---|---|---|---|
| B | D | C | B |

**Likely issue: visual perception.** Both Qwen runs overcounted the cars. Fine-tuning changed the estimated range but did not make this answer correct. The first frame contains small and partly cropped cars. More cars enter the view as the camera moves, so the question requires counting specifically at 00:00. VILA selected the correct range.

### 2. A time interval: QA-000010

Video: `data/ReVA/VisDrone/uav0000009_03358_v_01.mp4`

Question: How long does the group of people remain visible in the parking lot?

Options: A: 1–5 s; B: 0–4 s; C: 0–5 s; D: 2–5 s.

| Official answer | Qwen base | Qwen LoRA | VILA |
|---|---|---|---|
| C | C | C | D |

**Likely issue: temporal reasoning.** VILA selected a later starting time and a shorter interval than the official answer. Both Qwen runs selected C. Four sampled frames provide limited evidence for exact start and end times. The group is not clearly visible in the first decoded frame, so the annotation's 0-second boundary should also be treated cautiously. Scores here still follow the official label.

### 3. Matching the explanation to a letter: QA-000248

Video: `data/ReVA/VisDrone/uav0000161_00000_v_02.mp4`

Question: What is the camera viewpoint classification?

Options: A: high angle (>45°); B: top-down (90°); C: high angle (<45°); D: ground-level (0°).

| Official answer | Qwen base | Qwen LoRA | VILA |
|---|---|---|---|
| A | C | A | B |

**Likely issue: option parsing / answer mapping.** The base model's explanation says “high angle (>45 degrees),” which matches A, but its final tag contains C. The scorer correctly uses that final letter. The LoRA run selects A, while VILA selects top-down. This example shows why the final answer must be scored consistently even when the explanation sounds plausible.

The three examples illustrate individual behaviors; they do not by themselves show an overall improvement or establish overfitting. The full comparison table provides the overall result. Commands are listed in `README.md`.
