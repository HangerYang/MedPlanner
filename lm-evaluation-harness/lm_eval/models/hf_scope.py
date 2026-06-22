"""HF model wrapper that uses Code-SCOPE MCTS planning to select among beam candidates.

Usage
-----
    lm_eval run \\
        --model hf-scope \\
        --model_args pretrained=Qwen/Qwen3-4B \\
        --tasks humaneval \\
        --batch_size 1

All CODE_SCOPE_* environment variables from code-scope/code_scope/config.py are honoured.
Individual fields can be overridden via model_args (e.g. scope_num_candidates=5).

Trajectory / entropy logging
-----------------------------
Set CODE_SCOPE_ENTROPY_LOGGING=1 (or scope_enable_entropy_logging=1 in model_args) to
record per-candidate entropy, per-rollout entropy steps, and correlation statistics to a
JSONL file.  The output path defaults to <output_dir>/trajectory.jsonl and can be
overridden with CODE_SCOPE_TRAJECTORY_JSONL or scope_trajectory_jsonl in model_args.

Note: because the harness runs code execution after generate_until returns, the
`selected_passed` / `oracle_passed` / per-candidate `passed` fields are written as null.
They can be filled in post-hoc by joining against the main results JSONL on task_id.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    handle_stop_sequences,
    normalize_gen_kwargs,
    postprocess_generated_text,
)
from lm_eval.models.utils_hf import stop_sequences_criteria

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)

# /home/hyang/mediQ/code-scope  (three parents up from this file's location)
_CODE_SCOPE_ROOT = Path(__file__).resolve().parents[3] / "code-scope"


def _ensure_scope_on_path() -> None:
    if str(_CODE_SCOPE_ROOT) not in sys.path:
        sys.path.insert(0, str(_CODE_SCOPE_ROOT))


def _mean(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None



def _build_trajectory_record(
    task_id: str,
    candidates: list[str],
    planning: dict,
) -> dict | None:
    """Build a trajectory record from planner output.

    Pass/fail fields are null because execution happens outside generate_until
    in the harness; they can be joined in post-hoc from the results JSONL.
    """
    traj = planning.get("trajectory")
    if traj is None:
        return None

    candidate_records = traj.get("candidates", [])
    for i, rec in enumerate(candidate_records):
        rec["passed"] = None
        rec["candidate_index"] = i
        rec["text"] = candidates[i] if i < len(candidates) else None

    # Correlation stubs — no execution results available at this point
    correlation = {
        "passed_entropy_mean": None,
        "failed_entropy_mean": None,
        "entropy_delta": None,
        "passed_features_mean": None,
        "failed_features_mean": None,
    }

    return {
        "task_id": task_id,
        "n_candidates": len(candidates),
        "selected_index": planning["selected_index"],
        "selected_passed": None,
        "oracle_passed": None,
        "prompt_entropy": traj.get("prompt_entropy"),
        "prompt_features": traj.get("prompt_features"),
        "candidates": candidate_records,
        "rollouts": traj.get("rollouts", []),
        "rollout_summary": traj.get("summary", {}),
        "correlation": correlation,
    }


@register_model("hf-scope")
class HFScopeLM(HFLM):
    """HFLM extended with Code-SCOPE MCTS planning for candidate selection.

    For each prompt the model generates ``scope_num_candidates`` diverse beam
    completions, embeds both the prompt and each completion, then runs the
    SCOPE MCTS planner to pick the best one.  Everything else (task handling,
    metrics, output) is driven by lm-evaluation-harness as normal.
    """

    def __init__(
        self,
        pretrained: str,
        *,
        # SCOPE overrides — fall back to CODE_SCOPE_* env vars / config defaults
        scope_transition_dir: str | None = None,
        scope_reward_path: str | None = None,
        scope_transition_device: str | None = None,
        scope_reward_device: str | None = None,
        scope_num_candidates: int | str = 5,
        scope_planning_rounds: int | str = 10,
        scope_mcts_time: float | str = 30.0,
        scope_diversity_penalty: float | str = 1.0,
        scope_repetition_penalty: float | str = 1.0,
        scope_transition_samples: int | str = 4,
        scope_transition_noise: float | str = 0.005,
        scope_seed: int | str = 42,
        scope_enable_entropy_logging: int | str = 0,
        scope_trajectory_jsonl: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, **kwargs)

        _ensure_scope_on_path()
        from code_scope.config import CodeScopeConfig
        from code_scope.planner import CodeScopePlanner

        cfg = CodeScopeConfig()
        if scope_transition_dir is not None:
            cfg.transition_dir = scope_transition_dir
        if scope_reward_path is not None:
            cfg.reward_path = scope_reward_path
        if scope_transition_device is not None:
            cfg.transition_device = scope_transition_device
        if scope_reward_device is not None:
            cfg.reward_device = scope_reward_device
        cfg.num_candidates = int(scope_num_candidates)
        cfg.planning_rounds = int(scope_planning_rounds)
        cfg.mcts_time = float(scope_mcts_time)
        cfg.diversity_penalty = float(scope_diversity_penalty)
        cfg.repetition_penalty = float(scope_repetition_penalty)
        cfg.transition_samples = int(scope_transition_samples)
        cfg.transition_noise = float(scope_transition_noise)

        # Entropy logging: model_arg overrides env var
        if int(scope_enable_entropy_logging):
            cfg.enable_entropy_logging = True
        # trajectory_jsonl: model_arg overrides env var
        if scope_trajectory_jsonl is not None:
            cfg.trajectory_jsonl = scope_trajectory_jsonl

        # Pass lm_head to planner only when entropy logging is on
        lm_head = self.model.lm_head if cfg.enable_entropy_logging else None

        self._scope_cfg = cfg
        self._planner = CodeScopePlanner(cfg, lm_head=lm_head)
        self._scope_seed_base = int(scope_seed)
        self._scope_problem_counter = 0

        # Open trajectory file if entropy logging is on
        self._traj_file = None
        if cfg.enable_entropy_logging:
            traj_path = Path(cfg.trajectory_jsonl) if cfg.trajectory_jsonl else None
            if traj_path is None:
                eval_logger.warning(
                    "scope_enable_entropy_logging=1 but no trajectory path set. "
                    "Set scope_trajectory_jsonl=<path> or CODE_SCOPE_TRAJECTORY_JSONL."
                )
            else:
                traj_path.parent.mkdir(parents=True, exist_ok=True)
                self._traj_file = traj_path.open("a")
                eval_logger.info("SCOPE trajectory logging → %s", traj_path)

    def __del__(self) -> None:
        if getattr(self, "_traj_file", None) is not None:
            self._traj_file.close()

    # ------------------------------------------------------------------
    # Embedding  (mirrors QwenCodeModel.embed_turn)
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed *text* as a single user turn; return the last-token hidden state."""
        messages = [{"role": "user", "content": text}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            backbone = getattr(self.model, "model", self.model)
            outputs = backbone(input_ids=input_ids)
        return outputs.last_hidden_state[:, -1, :].detach().float().cpu().numpy()[0]

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _decode_candidate(self, cont_toks: list[int], stop: list[str]) -> str:
        """Decode token ids, applying think_end_token stripping and stop handling."""
        if isinstance(self.think_end_token, int):
            indices = [i for i, t in enumerate(cont_toks) if t == self.think_end_token]
            if indices:
                cont_toks = cont_toks[indices[-1] + 1:]
        s = self.tok_decode(cont_toks)
        if isinstance(self.think_end_token, int):
            s = s.lstrip()
        return postprocess_generated_text(
            s,
            stop=stop,
            think_end_token=self.think_end_token if isinstance(self.think_end_token, str) else None,
        )

    def _generate_candidates(
        self,
        context_enc: torch.Tensor,
        attn_masks: torch.Tensor,
        max_length: int,
        stop: list[str],
    ) -> list[str]:
        """Run diverse group beam search and return deduplicated decoded completions."""
        cfg = self._scope_cfg
        input_len = context_enc.shape[1]
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, input_len, context_enc.shape[0]
        )
        beam_kwargs: dict = dict(
            input_ids=context_enc,
            attention_mask=attn_masks,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            do_sample=False,
            num_beams=cfg.num_candidates,
            num_beam_groups=cfg.num_candidates,
            num_return_sequences=cfg.num_candidates,
            diversity_penalty=cfg.diversity_penalty,
            repetition_penalty=cfg.repetition_penalty,
            early_stopping=True,
        )
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ):
                try:
                    outputs = self.model.generate(
                        **beam_kwargs,
                        custom_generate="transformers-community/group-beam-search",
                        trust_remote_code=True,
                    )
                except Exception as exc:
                    eval_logger.warning(
                        "Custom group beam search unavailable (%s); using built-in.", exc
                    )
                    outputs = self.model.generate(**beam_kwargs)

        candidates: list[str] = []
        seen: set[str] = set()
        for output in outputs:
            s = self._decode_candidate(output[input_len:].tolist(), stop)
            if s not in seen:
                seen.add(s)
                candidates.append(s)
        return candidates

    # ------------------------------------------------------------------
    # Main override
    # ------------------------------------------------------------------

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        res: list[str] = []
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or self.rank != 0),
            desc="Running generate_until (SCOPE)",
        )

        for req in requests:
            context, gen_kwargs = req.args
            # task_id for trajectory logging (HumanEval/N); fall back to counter
            task_id: str = (
                req.doc.get("task_id", f"unknown/{self._scope_problem_counter}")
                if hasattr(req, "doc") and req.doc
                else f"unknown/{self._scope_problem_counter}"
            )

            kwargs = normalize_gen_kwargs(dict(gen_kwargs), self.max_gen_toks)
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            max_gen_toks = kwargs.pop("max_gen_toks")
            kwargs.pop("max_length", None)

            max_ctx_len = self.max_length - max_gen_toks
            context_enc, attn_masks = self.tok_batch_encode(
                [context],
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)
            max_length = context_enc.shape[1] + max_gen_toks

            seed = self._scope_seed_base + self._scope_problem_counter
            self._scope_problem_counter += 1

            candidates = self._generate_candidates(context_enc, attn_masks, max_length, until)

            if len(candidates) <= 1:
                selected = candidates[0] if candidates else ""
            else:
                prompt_emb = self._embed_text(context)
                candidate_embs = [self._embed_text(c) for c in candidates]
                planning = self._planner.choose(prompt_emb, candidate_embs, seed)
                selected = candidates[planning["selected_index"]]

                if self._traj_file is not None:
                    record = _build_trajectory_record(task_id, candidates, planning)
                    if record is not None:
                        self._traj_file.write(json.dumps(record) + "\n")
                        self._traj_file.flush()

            res.append(selected)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), selected)
            pbar.update(1)

        pbar.close()
        return res
