from __future__ import annotations

from dataclasses import dataclass, field


CANNOT_ANSWER_MARKERS = (
    "i cannot answer this question",
    "please do not ask this question again",
    "this question is probably irrelevant",
    "please ask something else",
)


@dataclass
class MedicalConversation:
    """MediQ-style conversation used by the SCOPE planner.

    Roles match the training data used for the medical SCOPE artifacts:
    user = patient, assistant = expert/doctor.
    """

    messages: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_patient_state(
        cls,
        patient_state: dict,
        inquiry: str | None = None,
        options: dict[str, str] | None = None,
    ) -> "MedicalConversation":
        convo = cls()
        initial_info = patient_state.get("initial_info") or ""
        first_message = cls._starter_text(initial_info, inquiry, options)
        convo.add_patient(first_message)
        for qa in patient_state.get("interaction_history") or []:
            convo.add_expert(qa.get("question") or "")
            convo.add_patient(qa.get("answer") or "")
        return convo

    @classmethod
    def from_condensed_evidence(
        cls,
        patient_state: dict,
        inquiry: str | None = None,
        options: dict[str, str] | None = None,
    ) -> "MedicalConversation":
        return cls([{"role": "user", "content": condensed_evidence_text(patient_state, inquiry, options)}])

    @staticmethod
    def _starter_text(initial_info: str, inquiry: str | None, options: dict[str, str] | None) -> str:
        parts = [str(initial_info).strip()]
        if inquiry:
            parts.append(f"Clinical question: {str(inquiry).strip()}")
        if options:
            option_text = ", ".join(f"{key}: {value}" for key, value in options.items())
            parts.append(f"Options: {option_text}")
        return "\n\n".join(part for part in parts if part)

    def add_patient(self, text: str) -> "MedicalConversation":
        self.messages.append({"role": "user", "content": str(text).strip()})
        return self

    def add_expert(self, text: str) -> "MedicalConversation":
        self.messages.append({"role": "assistant", "content": str(text).strip()})
        return self

    def with_expert_action(self, text: str) -> "MedicalConversation":
        clone = self.copy()
        clone.add_expert(text)
        return clone

    def copy(self) -> "MedicalConversation":
        return MedicalConversation([dict(m) for m in self.messages])

    def as_chat(self) -> list[dict[str, str]]:
        return [dict(m) for m in self.messages]

    @property
    def full_convo(self) -> list[str]:
        return [m["content"] for m in self.messages]

    def __str__(self) -> str:
        lines = []
        for msg in self.messages:
            label = "Patient" if msg["role"] == "user" else "Expert"
            lines.append(f'{label}\t: "{msg["content"]}"')
        return "\n".join(lines)


@dataclass(frozen=True)
class SemanticState:
    conversation: tuple[float, ...]
    depth: int = 1
    embedding_history: tuple = ()  # growing list of state embeddings, one per turn

    @property
    def response(self) -> tuple[float, ...]:
        return self.conversation

    def __str__(self) -> str:
        return f"Depth: {self.depth}, Conversation dim: {len(self.conversation)}"


def is_useful_patient_fact(answer: str) -> bool:
    answer = str(answer or "").strip()
    if not answer:
        return False
    lowered = answer.lower()
    return not any(marker in lowered for marker in CANNOT_ANSWER_MARKERS)


def useful_patient_facts(patient_state: dict) -> list[str]:
    facts = []
    seen = set()
    for qa in patient_state.get("interaction_history") or []:
        answer = str(qa.get("answer") or "").strip()
        if not is_useful_patient_fact(answer):
            continue
        key = " ".join(answer.lower().split())
        if key in seen:
            continue
        seen.add(key)
        facts.append(answer)
    return facts


def condensed_evidence_text(
    patient_state: dict,
    inquiry: str | None = None,
    options: dict[str, str] | None = None,
) -> str:
    initial_info = str(patient_state.get("initial_info") or "").strip()
    parts = [initial_info]
    facts = useful_patient_facts(patient_state)
    if facts:
        parts.append("Known useful patient facts:\n" + "\n".join(facts))
    if inquiry:
        parts.append(f"Clinical question:\n{str(inquiry).strip()}")
    if options:
        option_lines = [f"{key}: {value}" for key, value in options.items()]
        parts.append("Options:\n" + "\n".join(option_lines))
    return "\n\n".join(part for part in parts if part)


def condensed_patient_state(
    patient_state: dict,
    inquiry: str | None = None,
    options: dict[str, str] | None = None,
) -> dict:
    return {
        "initial_info": condensed_evidence_text(patient_state, inquiry, options),
        "interaction_history": [],
    }
