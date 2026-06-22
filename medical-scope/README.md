# medical-scope

`medical-scope` is a self-contained SCOPE-style planner wrapped as a mediQ expert.
It does not import code from `convo-plan-SCOPE`. It does import mediQ's prompt and
benchmark utilities from `src/`, so the doctor/expert prompt surface stays the
same as mediQ.

## Runtime flow

For each mediQ turn:

1. `ScopeMedicalExpert.respond(patient_state)` receives the current mediQ state.
2. Candidate actions are generated through mediQ prompt functions:
   - `scale_abstention_decision`
   - `Expert.ask_question` / `question_generation`
   - `final_choice_with_options` when committing to an answer
3. Candidate text actions are embedded with Qwen3.
4. SCOPE-style semantic MCTS plans with the saved transition model and reward MLP.
5. The temporary Q table scores the original candidates.
6. The best candidate is returned to mediQ as either a question or a final choice.
7. mediQ's patient model responds and mediQ decides whether to continue.

There is no SCOPE evaluation depth. Conversation length is controlled by mediQ's
own loop, usually `--max_questions` plus the expert's confidence/choice logic.

## Run

```bash
medical-scope/run_scope_medical.sh
```

Useful environment variables:

```bash
MAX_EXAMPLES=1
SCOPE_MEDICAL_MCTS_TIME=5
SCOPE_MEDICAL_PLANNING_DEPTH=8
SCOPE_MEDICAL_NUM_CANDIDATES=5
SCOPE_MEDICAL_TRANSITION_DIR=/home/hyang/mediQ/scope_saved/transition_models
SCOPE_MEDICAL_REWARD_PATH=/home/hyang/mediQ/scope_saved/reward/embedding_mediQ_reward_cumulative.pt
SCOPE_MEDICAL_TRACE_JSONL=/home/hyang/mediQ/new_outputs/medical_scope_trace.jsonl
```

## Boundary

Only files inside `medical-scope/` are needed for this integration. Existing
mediQ and SCOPE folders are left untouched.
