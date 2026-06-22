from __future__ import annotations

import time

import expert_functions
import helper as mediq_helper
from expert import Expert

from .candidate_generator import MediQPromptCandidateGenerator, parse_scope_action
from .config import ScopeMedicalConfig
from .conversation import MedicalConversation, condensed_patient_state
from .planner import get_planner


def _patch_qwen_chat_content() -> None:
    if getattr(mediq_helper.ModelCache, "_scope_medical_qwen_patch", False):
        return
    original = mediq_helper.ModelCache._to_content_list

    def _to_content_list(self, messages):
        model_name = str(getattr(self, "model_name", "")).lower()
        if "qwen" in model_name:
            return [
                {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else str(m["content"])}
                for m in messages
            ]
        return original(self, messages)

    mediq_helper.ModelCache._to_content_list = _to_content_list
    mediq_helper.ModelCache._scope_medical_qwen_patch = True


_patch_qwen_chat_content()


class ScopeMedicalExpert(Expert):
    """mediQ Expert that uses mediQ prompts and SCOPE semantic planning."""

    def set_trace_context(self, **context):
        self._trace_context = context

    def _build_conversation(self, patient_state):
        return MedicalConversation.from_patient_state(patient_state, self.inquiry, self.options)

    @staticmethod
    def _merge_usage(*usages):
        merged = {"input_tokens": 0, "output_tokens": 0}
        for usage in usages:
            if not usage:
                continue
            merged["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            merged["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        return merged

    def _base_meta(self, confidence, start_time, **extra):
        meta = {
            "confidence": confidence.get("confidence"),
            "confidence_rationale": confidence.get("confidence_rationale"),
            "shadow_answer": confidence.get("shadow_answer"),
            "boxed_answer": confidence.get("boxed_answer"),
            "usage": confidence.get("usage", {"input_tokens": 0, "output_tokens": 0}),
            "time_taken_scope_medical": time.time() - start_time,
        }
        meta.update(extra)
        return meta

    def respond(self, patient_state):
        turn_index = len(patient_state.get("interaction_history") or [])
        print(f"==================== SCOPE-Medical Turn {turn_index + 1} ====================")
        start = time.time()

        condensed_state = condensed_patient_state(patient_state, self.inquiry, self.options)

        max_questions = getattr(self.args, "max_questions", None)
        if max_questions is not None and turn_index >= max_questions:
            print("++++++++++++++++++++ Max turns reached — forcing final answer with condensed state ++++++++++++++++++++")
            letter, final_usage = expert_functions.final_choice_with_options(
                condensed_state,
                self.inquiry,
                self.options,
                **self.get_inference_kwargs(),
            )
            action = f"FINAL ANSWER: {letter}" if letter else "FINAL ANSWER:"
            base = self._base_meta(
                {},
                start,
                raw_action=action,
                scope_candidate_generation=[{"type": "choice", "action": action, "letter_choice": letter, "usage": final_usage}],
                scope_candidate_rewards=[],
                scope_selected_action_index=None,
                scope_planning_skipped=True,
                condensed_evidence=condensed_state["initial_info"],
                usage=final_usage,
            )
            response = {**base, "type": "choice", "letter_choice": letter}
            print(f"[SCOPE-Medical Expert System]: {response}")
            return response

        print("++++++++++++++++++++ Start of SCOPE-Medical Confidence Pass ++++++++++++++++++++")
        confidence = expert_functions.scale_abstention_decision(
            **self.get_abstain_kwargs(patient_state)
        )
        print(
            f"[SCOPE-Medical Confidence]: confidence={confidence.get('confidence')}, "
            f"abstain={confidence.get('abstain')}, letter={confidence.get('letter_choice')}"
        )

        if not confidence.get("abstain"):
            print("++++++++++++++++++++ Start of SCOPE-Medical Final Answer Pass ++++++++++++++++++++")
            letter, final_usage = expert_functions.final_choice_with_options(
                condensed_state,
                self.inquiry,
                self.options,
                **self.get_inference_kwargs(),
            )
            action = f"FINAL ANSWER: {letter}" if letter else "FINAL ANSWER:"
            base = self._base_meta(
                confidence,
                start,
                raw_action=action,
                scope_candidate_generation=[{"type": "choice", "action": action, "letter_choice": letter, "usage": final_usage}],
                scope_candidate_rewards=[],
                scope_selected_action_index=None,
                scope_planning_skipped=True,
                condensed_evidence=condensed_state["initial_info"],
                usage=self._merge_usage(confidence.get("usage"), final_usage),
            )
            response = {**base, "type": "choice", "letter_choice": letter or confidence.get("letter_choice")}
            print(f"[SCOPE-Medical Expert System]: {response}")
            return response

        cfg = ScopeMedicalConfig()
        candidate_generator = MediQPromptCandidateGenerator(self, num_candidates=cfg.num_candidates)
        candidate_result = candidate_generator.generate_questions(patient_state, confidence)

        convo = self._build_conversation(patient_state)
        planner = get_planner()
        seed = hash((turn_index, str(patient_state))) % (2**32)
        trace_context = {
            **getattr(self, "_trace_context", {}),
            "turn_index": turn_index,
            "num_observed_questions": turn_index,
        }
        action, planning = planner.choose_action(
            convo,
            candidate_result.actions,
            seed=seed,
            trace_context=trace_context,
        )
        print(f"\n  [SCOPE action]: {action!r}")

        action_type, letter = parse_scope_action(action)
        candidate_rewards = []
        for idx, candidate_action in enumerate(planning.get("possible_actions") or []):
            candidate_rewards.append(
                {
                    "index": idx,
                    "action": candidate_action,
                    "embedding_reward": (planning.get("greedy_rewards") or [None])[idx]
                    if idx < len(planning.get("greedy_rewards") or [])
                    else None,
                    "mcts_q": (planning.get("possible_actions_reward") or [None])[idx]
                    if idx < len(planning.get("possible_actions_reward") or [])
                    else None,
                    "selected": idx == planning.get("selected_action_index"),
                }
            )

        base = self._base_meta(
            confidence,
            start,
            raw_action=action,
            scope_candidate_generation=candidate_result.records,
            scope_candidate_rewards=candidate_rewards,
            scope_selected_action_index=planning.get("selected_action_index"),
            scope_planning_skipped=False,
            condensed_evidence=condensed_state["initial_info"],
        )

        if action_type == "choice":
            response = {**base, "type": "choice", "letter_choice": letter}
        else:
            response = {
                **base,
                "type": "question",
                "question": action,
                "letter_choice": confidence.get("letter_choice") or next(iter(self.options.keys())),
            }
        print(f"[SCOPE-Medical Expert System]: {response}")
        return response
