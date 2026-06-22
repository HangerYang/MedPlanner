# MediQ: Question-Asking LLMs for Adaptive and Reliable Clinical Reasoning

## [[paper](https://arxiv.org/abs/2406.00922)] [[website](https://stellalisy.com/projects/mediQ/)] [[data](https://github.com/stellali7/mediQ/tree/main/data)]

## Overview

MediQ simulates a turn-by-turn clinical conversation between a **Patient** (who holds the full case context) and an **Expert** (a doctor who must reach a diagnosis). Each turn the Expert either asks a clarifying question or commits to a final multiple-choice answer (A/B/C/D). The benchmark measures how well an Expert system uses targeted questions to improve accuracy while minimising unnecessary questions.

---

## Installation

```bash
git clone https://github.com/stellali7/MediQ.git
cd MediQ
conda env create -f environment.yml
```

If you plan to use an OpenAI-compatible API, add your key to `src/keys.py`.

---

## Quick Start

```bash
# Fast smoke-test — no model calls required
cd src && python mediQ_benchmark.py \
  --expert_class RandomExpert \
  --patient_class RandomPatient \
  --data_dir ../data/med_data \
  --dev_filename all_dev_convo.jsonl \
  --output_filename results/smoke_test.jsonl \
  --max_questions 3

# Non-interactive baseline (full context, no questions)
cd src && python mediQ_benchmark.py \
  --expert_class FixedExpert  --expert_model meta-llama/Llama-3.1-8B-Instruct \
  --patient_class FullContextPatient \
  --data_dir ../data/med_data --dev_filename all_dev_convo.jsonl \
  --output_filename results/noninteractive_full.jsonl \
  --max_questions 0 --use_vllm

# Interactive run
cd src && python mediQ_benchmark.py \
  --expert_class FixedExpert  --expert_model meta-llama/Llama-3.1-8B-Instruct \
  --patient_class FactSelectPatient --patient_model meta-llama/Llama-3.1-8B-Instruct \
  --data_dir ../data/med_data --dev_filename all_dev_convo.jsonl \
  --output_filename results/interactive.jsonl \
  --max_questions 30 --use_vllm
```

See `run.sh` for the full suite of benchmark runs.

---

## The Interaction Loop

```
for each case:
    patient = PatientClass(case)          # holds full context internally
    expert  = ExpertClass(inquiry, opts)  # sees only initial_info at first

    while True:
        state = patient.get_state()       # {initial_info, interaction_history}
        response = expert.respond(state)  # {type, question/letter_choice, confidence, ...}

        if response["type"] == "choice":
            break                         # final answer reached
        if len(history) >= max_questions:
            break                         # question budget exhausted

        answer = patient.respond(response["question"])
        patient.update_state(question, answer)
```

`initial_info` is what the Patient reveals upfront (one sentence to full paragraph, depending on Patient class). `interaction_history` grows as the conversation proceeds.

---

## Expert Classes (`src/expert.py`)

The Expert is the doctor. Every Expert class must implement `respond(patient_state)` and return either:
- `{"type": "question", "question": "...", "letter_choice": "A", "confidence": 0.4}` — ask more
- `{"type": "choice", "letter_choice": "B", "confidence": 0.9}` — commit to answer

The classes differ only in their **abstention strategy** — how they decide whether to ask another question or answer now.

### `RandomExpert`
- **No model calls.** Flips a coin each turn (50 % question, 50 % answer). The question is always the same toy string; the answer is a random option.
- **Use for**: quick pipeline smoke-tests.

### `BasicExpert`
- **1 LLM call per turn.** A single prompt asks the model to output either a follow-up question or a letter choice. If the response contains `?` it is treated as a question (abstain); otherwise it is parsed as an answer.
- **Self-consistency** (`--self_consistency N`): runs the same call N times; confidence = fraction of runs that chose to answer vs. ask.
- **Use for**: simplest LLM-based interactive expert.

### `FixedExpert`
- **Rule-based abstention, no LLM for the decision.** Simply asks questions until `len(interaction_history) == max_questions`, then answers. Abstention requires 0 LLM calls; the answer itself requires 1 call; if still abstaining, question generation is a second call.
- **Use for**: non-interactive baselines (`--max_questions 0`) and as a simple interactive ceiling check.
- This is the Expert used in the original paper's main experiments.

### `BinaryExpert`
- **2 LLM calls per turn.** Call 1: "Do you have enough information to answer? YES/NO." `NO` means abstain. Call 2 (always, for evaluation): generate the letter choice. Call 3 (if abstaining): question generator.
- **Use for**: explicit binary confidence signal.

### `NumericalExpert`
- **3 LLM calls per turn.** Call 1: produce a numerical confidence score (0.0–1.0). Call 2: given that score, YES/NO to proceed? `NO` means abstain. Call 3 (always): letter choice. Call 4 (if abstaining): question generator.
- Confidence score and YES/NO are decoupled, making the model reason about its own score.
- **Use for**: richer two-step numerical abstention.

### `NumericalCutOffExpert`
- **2 LLM calls per turn.** Call 1: produce a numerical confidence score (0.0–1.0). Abstain if `score < --abstain_threshold` (default `0.8`). Call 2 (always): letter choice. Call 3 (if abstaining): question generator.
- **Use for**: deterministic threshold over a numerical score; avoids the second YES/NO judgment.

### `ScaleExpert`
- **2 LLM calls per turn.** Call 1: produce a Likert-scale confidence (1 = very unconfident … 5 = very confident). Abstain if `score < --abstain_threshold` (default `4.0`). Call 2 (always): letter choice. Call 3 (if abstaining): question generator.
- **Use for**: ordinal confidence abstention.

### `HumanExpert`
- **No model calls.** Prints the full patient state, inquiry, and options to the terminal. The human types a question to ask the patient, or a letter (A/B/C/D) to commit to an answer.
- **Use for**: human-in-the-loop evaluation and prompt debugging.

> **Note:** All LLM-backed experts (except `BasicExpert`) always compute a `letter_choice` even when abstaining. This "shadow answer" is logged for analysis but is not returned to the patient.

---

## Patient Classes (`src/patient.py`)

The Patient holds the full clinical case. It controls how much information is visible to the Expert upfront and how it responds to questions.

### `RandomPatient`
- **No model calls.** Returns a random sentence from the context list 50 % of the time, or "The patient cannot answer this question" 50 % of the time.
- **Use for**: fast pipeline testing with no GPU required.

### `DirectPatient`
- **1 LLM call per question.** The model is given only `initial_info` (the first sentence of context) and asked to answer the question.
- **Use for**: testing how well an expert does when the patient can only recall a single initial fact.

### `InstructPatient`
- **1 LLM call per question.** The model is given the **full context paragraph** with strict instructions: answer only what is asked, using only the context; if the question is not answered by the context, say "cannot answer."
- **Use for**: non-interactive lower bound (pair with `--max_questions 0`), or interactive with a cooperative LLM patient.

### `FactSelectPatient`
- **1 LLM call on first question** (decompose context into numbered atomic facts). **1 LLM call per subsequent question** (select and return verbatim only the facts that answer the question).
- The fact list is cached for the lifetime of the case, so decomposition only happens once.
- **Use for**: the main interactive benchmark. The patient answers only with discrete, verbatim facts, making information disclosure controlled and reproducible.

### `FullContextPatient`
- **No `respond()` method** (raises `NotImplementedError`). Overrides `initial_info` to be the entire context paragraph, so the Expert sees everything upfront.
- **Use for**: non-interactive upper bound — pair with `--max_questions 0` and `FixedExpert`.

---

## Evaluation Modes

| Mode | Expert | Patient | `--max_questions` | Notes |
|---|---|---|---|---|
| Non-interactive upper bound | `FixedExpert` | `FullContextPatient` | `0` | Expert sees full context, no questions |
| Non-interactive lower bound | `FixedExpert` | `InstructPatient` | `0` | Expert sees only initial_info |
| Interactive | any LLM Expert | `FactSelectPatient` | `≥1` | Main benchmark |
| Human-in-the-loop | `HumanExpert` | any | any | Manual evaluation |
| SCOPE (MCTS) | `scope_mediq_runner.py` | `FactSelectPatient` | `5` | Monte-Carlo Tree Search planner |
| vLLM server eval | `vllm_eval.py` | — | — | Direct eval against a running vLLM endpoint |

---

## Command-Line Arguments (`src/args.py`)

### Required
| Argument | Description |
|---|---|
| `--expert_class` | Expert class name (e.g. `FixedExpert`) |
| `--patient_class` | Patient class name (e.g. `FactSelectPatient`) |
| `--data_dir` | Directory containing data files |
| `--dev_filename` | Data filename (e.g. `all_dev_convo.jsonl`) |

### Models
| Argument | Default | Description |
|---|---|---|
| `--expert_model` | `meta-llama/Llama-3.1-8B-Instruct` | Model for expert decision-making (abstention + answer) |
| `--expert_model_question_generator` | same as `--expert_model` | Separate model for follow-up question generation; useful to use a lighter/faster model here |
| `--patient_model` | `meta-llama/Llama-3.1-8B-Instruct` | Model used by the patient to answer questions |
| `--expert_module` | `expert` | Python file (without `.py`) containing the Expert class |
| `--patient_module` | `patient` | Python file (without `.py`) containing the Patient class |

### Inference Backend
| Argument | Default | Description |
|---|---|---|
| `--use_vllm` | off | Use vLLM for batched local inference (recommended for GPUs) |
| `--use_api` | `None` | Use an external API; currently only `openai` is supported |
| `--tensor_parallel_size` | `1` | Number of GPUs for vLLM tensor parallelism |
| `--batch_size` | `256` | vLLM `max_num_seqs` (concurrent sequences) |
| `--temperature` | `0.6` | Sampling temperature |
| `--top_p` | `0.9` | Nucleus sampling top-p |
| `--max_tokens` | `256` | Maximum tokens to generate per call |
| `--top_logprobs` | `0` | Number of top log-probabilities to return |
| `--api_account` | `mediQ` | Key name to look up in `src/keys.py` |

### Benchmark Control
| Argument | Default | Description |
|---|---|---|
| `--max_questions` | `30` | Maximum clarifying questions before forcing a final answer; `0` = non-interactive |
| `--abstain_threshold` | `0.8` | Confidence threshold for `NumericalCutOffExpert` (0–1) and `ScaleExpert` (1–5 Likert) |
| `--self_consistency` | `1` | Number of independent runs to aggregate for confidence estimation |
| `--rationale_generation` | off | Prompt the expert to generate chain-of-thought rationale before deciding |
| `--independent_modules` | off | Question generator does not see the abstention conversation history (fresh context each turn) |

### Output & Logging
| Argument | Default | Description |
|---|---|---|
| `--output_filename` | `results.jsonl` | Path for per-example output (predictions, confidence, token usage) |
| `--log_filename` | `None` | General benchmark summary log |
| `--history_log_filename` | `None` | Full interaction history log |
| `--detail_log_filename` | `None` | Detailed prompt/response log for abstention decisions |
| `--message_log_filename` | `None` | Raw API message log |

---

## Data Format

Each line in the `.jsonl` data file is a JSON object:

```json
{
  "id": 0,
  "context": ["Sentence one.", "Sentence two.", ...],
  "initial_info": "First sentence (optional override).",
  "atomic_facts": ["Fact 1.", "Fact 2.", ...],
  "question": "What is the most likely diagnosis?",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "Pancreatitis",
  "answer_idx": "B"
}
```

`atomic_facts` is pre-computed and used by `FactSelectPatient` (skips decomposition if present). `initial_info` overrides the auto-extracted first sentence.

---

## Implementing Your Own Expert

Subclass `Expert` in a new file and implement `respond`. Use `self.ask_question()` and `self.get_abstain_kwargs()` inherited helpers.

```python
# src/my_expert.py
from expert import Expert
import expert_functions

class MyExpert(Expert):
    def respond(self, patient_state):
        kwargs = self.get_abstain_kwargs(patient_state)
        # pick any abstention strategy, or write your own logic
        result = expert_functions.fixed_abstention_decision(**kwargs)
        if not result["abstain"]:
            return {"type": "choice", "letter_choice": result["letter_choice"],
                    "confidence": result["confidence"]}
        q = self.ask_question(patient_state, result["messages"])
        return {"type": "question", "question": q["atomic_question"],
                "letter_choice": result["letter_choice"],
                "confidence": result["confidence"]}
```

```bash
python mediQ_benchmark.py \
  --expert_module my_expert --expert_class MyExpert \
  --patient_class FactSelectPatient \
  --data_dir ../data/med_data --dev_filename all_dev_convo.jsonl \
  --output_filename results/my_expert.jsonl --max_questions 10 --use_vllm
```

---

## SCOPE Integration (`src/scope_mediq_runner.py`)

The SCOPE runner replaces the Expert with a Monte-Carlo Tree Search (MCTS) planner from the `convo-plan-SCOPE` submodule. SCOPE uses its own internal transition model to simulate future conversation states and chooses the question predicted to maximise diagnostic accuracy.

```bash
cd src && python scope_mediq_runner.py \
  --data_file ../data/med_data/all_dev_convo.jsonl \
  --output_filename results/scope.jsonl \
  --max_questions 5
```

SCOPE requires two GPUs (transition model + reward model).

---

## vLLM Server Evaluation (`src/vllm_eval.py`)

For non-interactive evaluation against an already-running vLLM server:

```bash
# 1. Start the server
bash run_vllm_server.sh

# 2. Run evaluation
python src/vllm_eval.py \
  --config vllm_eval_config.json \
  --prompt_style mediq \          # "default" or "mediq" prompt template
  --output_file src/results/vllm_eval.jsonl \
  --mode all                      # "infer", "eval", or "all"
```

Edit `vllm_eval_config.json` to point at the correct model, host, port, and data file.

---

## How to Cite

```bibtex
@inproceedings{li2024mediq,
  title={MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning},
  author={Li, Shuyue Stella and Balachandran, Vidhisha and Feng, Shangbin and Ilgen, Jonathan S
          and Pierson, Emma and Koh, Pang Wei and Tsvetkov, Yulia},
  journal={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

---

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
