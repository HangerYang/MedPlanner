import random
import re
import logging
from helper import get_response

def _log(message):
    logging.getLogger("detail_logger").info(message)

# FactSelectPatient — fixed reply when no fact answers the doctor's question.
FACTSELECT_CANNOT_ANSWER = (
    "I cannot answer this question. Please do not ask it again; ask something else instead."
)


def _parse_fact_index_line(raw: str, n_facts: int):
    """Parse LLM output that should only list fact indices or NONE.

    Returns:
        []  — no applicable facts (patient should use FACTSELECT_CANNOT_ANSWER)
        list of int — 1-based indices into self.facts (deduped, in order)
        None — unparseable / no valid in-range indices (treat as cannot answer)
    """
    if n_facts <= 0:
        return []
    t = (raw or "").strip()
    t = re.sub(r"<[^>]*>", "", t)  # drop stray special tokens
    if not t:
        return None
    first = t.splitlines()[0].strip()
    # Strip common label noise the model may prepend
    first = re.sub(
        r"^\s*(?:\*\*)?\s*(?:mode\s*\(?\s*[ab]\s*\)?|output|answer|indices?)\s*[:\-]?\s*",
        "",
        first,
        flags=re.IGNORECASE,
    ).strip()
    low = first.lower()
    if low in ("none", "n/a", "na", "null", "0", "-", "nil"):
        return []
    nums = [int(x) for x in re.findall(r"\b(\d+)\b", first)]
    nums = [i for i in nums if 1 <= i <= n_facts]
    seen = set()
    ordered = []
    for i in nums:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    if not ordered:
        return None
    return ordered


class Patient:
    def __init__(self, args, sample):
        # Assuming 'context' is a list or a long string of historical or background information
        if isinstance(sample['context'], list) and len(sample['context']) > 0:
            if 'initial_info' in sample: self.initial_info = sample['initial_info']
            else: self.initial_info = sample['context'][0]  # Taking the first item if it's a list
            self.context_list = sample['context']
            self.context_para = " ".join(sample['context'])
        elif isinstance(sample['context'], str):
            # Assuming sentences are separated by periods, taking the first sentence
            if 'initial_info' in sample: self.initial_info = sample['initial_info']
            else: self.initial_info = sample['context'].split(". ")[0]
            temp = sample['context'].split(". ")
            self.context_list = [temp[i]+'.' if i!=len(temp)-1 and not temp[i].endswith('.') else temp[i] for i in range(len(temp))]
            self.context_para = sample['context']
        else:
            if 'initial_info' in sample: self.initial_info = sample['initial_info']
            else: self.initial_info = ""  # Default fallback
            self.context_list = []
            self.context_para = 'None'
        
        self.model_name = args.patient_model
        self.history = []  # To track the interaction history of questions and answers
        self.facts = sample['atomic_facts'] if 'atomic_facts' in sample else None  # To store atomic facts after initial processing, you can choose to store this somewhere locally to avoid repeated processing

        self.max_length = 50  # Maximum length of the response (different from the expert system)
        self.use_vllm = args.use_vllm
        self.use_api = args.use_api  # Use an API to generate responses
        self.tensor_parallel_size = args.tensor_parallel_size
        self.batch_size = args.batch_size
        self.gpu_memory_utilization = getattr(args, "gpu_memory_utilization", None)
        self.vllm_max_model_len = getattr(args, "vllm_max_model_len", None)
        self.vllm_max_num_seqs = getattr(args, "vllm_max_num_seqs", None)
        self.vllm_enforce_eager = getattr(args, "vllm_enforce_eager", False)

    def update_state(self, question, answer):
        # Update the internal history with the new question and the corresponding answer
        self.history.append({"question": question, "answer": answer})

    def get_state(self):
        # Return the initial context and the history of interactions
        return {
            "initial_info": self.initial_info,
            "interaction_history": self.history
        }
    
    def get_questions(self):
        # Return the list of questions asked so far
        return [qa["question"] for qa in self.history]
    
    def get_answers(self):
        # Return the list of answers provided so far
        return [qa["answer"] for qa in self.history]
    
    def get_response(self, messages, max_length=None):
        if max_length is None: max_length = self.max_length
        return get_response(
            messages,
            self.model_name,
            use_vllm=self.use_vllm,
            use_api=self.use_api,
            max_length=max_length,
            tensor_parallel_size=self.tensor_parallel_size,
            batch_size=self.batch_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            vllm_max_model_len=self.vllm_max_model_len,
            vllm_max_num_seqs=self.vllm_max_num_seqs,
            vllm_enforce_eager=self.vllm_enforce_eager,
        )
    
    def respond(self, question):
        raise NotImplementedError
    

class FullContextPatient(Patient):
    """Patient that exposes the full context as initial_info, for non-interactive evaluation."""
    def __init__(self, args, sample):
        super().__init__(args, sample)
        self.initial_info = self.context_para  # override: show expert all context upfront

    def respond(self, question):
        raise NotImplementedError("FullContextPatient is for non-interactive use only.")


class RandomPatient(Patient):
    def respond(self, question):
        # Randomly select a response mode
        if random.random() < 0.5 or len(self.context_list) == 0:
            answer = "This question is probably irrelevant to the case. Please ask something else instead."
        else:
            answer = random.choice(self.context_list)
        self.update_state(question, answer)
        return answer

class DirectPatient(Patient):
    def respond(self, question):
        system_prompt = "Answer the question with the given context."
        user_prompt = f"Context: \"{self.initial_info}\"\nQuestion: \"{question}\"\n"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        _log(f"[PATIENT PROMPT (DirectPatient)]: {messages}")
        response, log_probs, num_tokens = self.get_response(messages)
        _log(f"[PATIENT RESPONSE (DirectPatient)]: {response}")
        self.update_state(question, response)
        return response

class InstructPatient(Patient):
    def respond(self, question):
        system_prompt = "You are a truthful assistant that understands the patient's information, and you are trying to answer questions from a medical doctor about the patient."
        user_prompt = f"Below is a context paragraph describing the patient and their conditions:\n\"{self.context_para}\"\nQuestion from the doctor: \"{question}\"\nUse the context paragraph to answer the doctor's question. If the paragraph does not answer the question, simply say \"This question is probably irrelevant to the case. Please ask something else instead.\" Answer only what the question asks for. Do not provide any analysis, inference, or implications. Respond with a straightforward answer to the question ONLY and NOTHING ELSE."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        _log(f"[PATIENT PROMPT (InstructPatient)]: {messages}")
        response, log_probs, num_tokens = self.get_response(messages)
        _log(f"[PATIENT RESPONSE (InstructPatient)]: {response}")
        self.update_state(question, response)
        return response
    
class FactSelectPatient(Patient):
    def respond(self, question):
        if not self.facts:
            system_prompt = "You are a truthful medical assistant that understands the patient's information."
            user_prompt = f"Break the following patient information into a list of independent atomic facts, with one piece of information in each statement. Each fact should only include the smallest unit of information, but should be self-contained.\n\"{self.context_para}\"\nResponse with the list of atomic facts and nothing else, prepend each fact by an index starting from 1. No sub-list allowed."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            _log(f"[PATIENT PROMPT (FactSelectPatient/decompose)]: {messages}")
            response_text, log_probs, num_tokens = self.get_response(messages, max_length=1000)
            _log(f"[PATIENT RESPONSE (FactSelectPatient/decompose)]: {response_text}")
            response_text = [s.strip() for s in response_text.splitlines() if s.strip()]
            self.facts = response_text

        n = len(self.facts)
        # Renumber 1..n for the classifier (display only); answers use original fact lines.
        display_lines = []
        for i, line in enumerate(self.facts, start=1):
            body = re.sub(r"^\s*\d+\.\s*", "", (line or "").strip()).strip()
            display_lines.append(f"{i}. {body}")
        facts_prompt = "\n".join(display_lines)

        system_prompt = (
            "You are a classifier. You will see a numbered list of atomic patient facts "
            f"(lines 1–{n}) and one doctor question.\n\n"
            "Your task: decide which fact line numbers (if any) are needed to answer that "
            "question. Do NOT copy or paraphrase any fact text. Do NOT explain. Output "
            "exactly one of the following on a single line:\n"
            "  • NONE — if no fact helps answer the question\n"
            "  • One or more integers separated by commas (e.g. 3 or 3,7,12) — the 1-based "
            "line numbers of all relevant facts, in ascending order, each between 1 and "
            f"{n}, no duplicates.\n\n"
            "Do not output words like MODE, bullets, or JSON. Only NONE or a comma-separated list of integers."
        )
        user_prompt = (
            f"FACTS:\n{facts_prompt}\n\n"
            f'Doctor question: "{question}"\n\n'
            "Your line (NONE or integers only):"
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        _log(f"[PATIENT PROMPT (FactSelectPatient/select-indices)]: {messages}")
        raw_indices, log_probs, num_tokens = self.get_response(messages)
        _log(f"[PATIENT RESPONSE (FactSelectPatient/select-indices, raw)]: {raw_indices}")

        parsed = _parse_fact_index_line(raw_indices, n)
        if parsed is None:
            answer = FACTSELECT_CANNOT_ANSWER
        elif len(parsed) == 0:
            answer = FACTSELECT_CANNOT_ANSWER
        else:
            parsed = sorted(parsed)
            answer = "\n".join(self.facts[i - 1] for i in parsed)

        _log(f"[PATIENT RESPONSE (FactSelectPatient/assembled)]: {answer}")
        self.update_state(question, answer)
        return answer