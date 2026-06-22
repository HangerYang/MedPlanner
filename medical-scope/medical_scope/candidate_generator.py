from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch

import helper as mediq_helper
import prompts
from .config import ScopeMedicalConfig


@dataclass
class CandidateResult:
    actions: list[str]
    confidence_result: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)


def normalize_question_action(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^(?:QUESTION|ATOMIC QUESTION)\s*\d*\s*[:.-]\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def parse_atomic_question(text: str) -> str | None:
    for line in reversed(str(text or "").splitlines()):
        if "?" not in line:
            continue
        question = line.split(":", 1)[-1].strip().strip("'\"")
        return question if question else None
    return None


class MediQPromptCandidateGenerator:
    """Generate real-text expert actions through mediQ's atomic question prompt."""

    def __init__(self, expert, num_candidates: int = 5) -> None:
        self.expert = expert
        self.num_candidates = max(1, int(num_candidates))

    def _build_messages(self, patient_state: dict) -> list[dict[str, str]]:
        patient_info = patient_state["initial_info"]
        conv_log = "\n".join(
            f"{prompts.expert_system['question_word']}: {qa['question']}\n"
            f"{prompts.expert_system['answer_word']}: {qa['answer']}"
            for qa in patient_state["interaction_history"]
        )
        task_key = "atomic_question_improved_RG" if self.expert.args.rationale_generation else "atomic_question_improved"
        task_prompt = prompts.expert_system[task_key]
        if self.expert.args.option_mode in ("no-option", "option-in-the-end"):
            prompt = prompts.expert_system["curr_template_no_options"].format(
                patient_info, conv_log if conv_log else "None", self.expert.inquiry, task_prompt
            )
        else:
            options_text = (
                f"A: {self.expert.options['A']}, B: {self.expert.options['B']}, "
                f"C: {self.expert.options['C']}, D: {self.expert.options['D']}"
            )
            prompt = prompts.expert_system["curr_template"].format(
                patient_info, conv_log if conv_log else "None", self.expert.inquiry, options_text, task_prompt
            )
        return [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user", "content": prompt},
        ]

    def _get_hf_model_cache(self, q_kwargs: dict[str, Any]):
        model_name = q_kwargs["model_name"]
        if q_kwargs.get("use_api") or q_kwargs.get("use_vllm"):
            raise RuntimeError("SCOPE-Medical beam candidate generation requires a local HF model, not API/vLLM.")
        model_cache = mediq_helper.models.get(model_name)
        if model_cache is None:
            model_cache = mediq_helper.ModelCache(
                model_name,
                use_vllm=q_kwargs.get("use_vllm", False),
                use_api=q_kwargs.get("use_api"),
                **{k: v for k, v in q_kwargs.items() if k not in {"model_name", "use_vllm", "use_api"}},
            )
            mediq_helper.models[model_name] = model_cache
        elif hasattr(model_cache, "args"):
            model_cache.args.update(q_kwargs)
        return model_cache

    def _model_and_tokenizer(self, model_cache):
        if hasattr(model_cache, "_local_llm"):
            return model_cache._local_llm.model, model_cache._local_llm.tokenizer
        model = getattr(model_cache, "model", None)
        tokenizer = getattr(model_cache, "tokenizer", None)
        if model is None or tokenizer is None:
            raise RuntimeError("SCOPE-Medical beam candidate generation could not access model/tokenizer.")
        return model, tokenizer

    def _tokenize_messages(self, model_cache, tokenizer, messages, model_name: str):
        tmpl_kwargs = mediq_helper._chat_template_kwargs(model_name)
        token_messages = model_cache._to_content_list(messages) if hasattr(model_cache, "_to_content_list") else messages
        try:
            return tokenizer.apply_chat_template(
                token_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_attention_mask=True,
                return_dict=True,
                **tmpl_kwargs,
            )
        except Exception:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_attention_mask=True,
                return_dict=True,
                **tmpl_kwargs,
            )

    def _beam_generate_questions(self, messages: list[dict[str, str]], q_kwargs: dict[str, Any]):
        model_name = q_kwargs["model_name"]
        model_cache = self._get_hf_model_cache(q_kwargs)
        model, tokenizer = self._model_and_tokenizer(model_cache)
        tokens = self._tokenize_messages(model_cache, tokenizer, messages, model_name).to(model.device)
        input_len = int(tokens["input_ids"].shape[-1])
        cfg = ScopeMedicalConfig()
        max_new = int(cfg.candidate_max_new_tokens)
        num_beam_groups = cfg.candidate_num_beam_groups or self.num_candidates
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if hasattr(tokenizer, "tokenizer"):
            pad_token_id = getattr(tokenizer.tokenizer, "eos_token_id", pad_token_id)

        with torch.inference_mode():
            outputs = model.generate(
                **tokens,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=self.num_candidates,
                num_beam_groups=num_beam_groups,
                num_return_sequences=self.num_candidates,
                diversity_penalty=cfg.candidate_diversity_penalty,
                repetition_penalty=cfg.candidate_repetition_penalty,
                early_stopping=True,
                pad_token_id=pad_token_id,
                trust_remote_code=True,
                custom_generate="transformers-community/group-beam-search",
            )

        records = []
        for beam_idx, output in enumerate(outputs):
            generated = output[input_len:]
            raw_text = tokenizer.decode(generated, skip_special_tokens=True)
            response_text = mediq_helper._strip_thinking(raw_text)
            question = parse_atomic_question(response_text)
            action = normalize_question_action(question or "")
            records.append(
                {
                    "type": "question",
                    "rank": beam_idx + 1,
                    "action": action,
                    "raw_response": response_text,
                    "usage": {"input_tokens": input_len, "output_tokens": int(generated.shape[-1])},
                }
            )
        return records

    def generate_questions(self, patient_state: dict, confidence_result: dict[str, Any]) -> CandidateResult:
        print("++++++++++++++++++++ Start of SCOPE-Medical Question Candidate Generation ++++++++++++++++++++")
        q_kwargs = dict(
            self.expert.get_inference_kwargs(),
            model_name=self.expert.args.expert_model_question_generator or self.expert.args.expert_model,
        )
        messages = self._build_messages(patient_state)
        records = self._beam_generate_questions(messages, q_kwargs)

        actions: list[str] = []
        for record in records:
            record.update(
                {
                    "confidence": confidence_result.get("confidence"),
                    "letter_choice": confidence_result.get("letter_choice"),
                    "shadow_answer": confidence_result.get("shadow_answer"),
                }
            )
            action = record.get("action") or ""
            if action and action not in actions:
                actions.append(action)

        if not actions:
            raise RuntimeError("SCOPE-Medical beam candidate generation produced no valid questions.")

        print("possible question actions generated by mediQ prompt path: ", actions)
        return CandidateResult(actions=actions, confidence_result=confidence_result, records=records)


def parse_scope_action(text: str):
    text = (text or "").strip()
    match = re.search(r"FINAL\s*ANSWER\s*[:\-]?\s*([ABCD])", text.upper())
    if match:
        return "choice", match.group(1)
    match = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])\s*$", text.upper())
    if match and len(text) < 30:
        return "choice", match.group(1)
    return "question", None
