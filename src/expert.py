import random
import expert_functions

class Expert:
    """
    Expert system skeleton
    """
    def __init__(self, args, inquiry, options):
        # Initialize the expert with necessary parameters and the initial context or inquiry
        self.args = args
        self.inquiry = inquiry
        self.options = options

    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        raise NotImplementedError
    
    def ask_question(self, patient_state, prev_messages):
        # Generate a question based on the current patient state
        kwargs = {
            "patient_state": patient_state,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "messages": prev_messages,
            "independent_modules": self.args.independent_modules,
            "option_mode": self.args.option_mode,
            "rationale_generation": self.args.rationale_generation,
            "model_name": self.args.expert_model_question_generator or self.args.expert_model,
            "use_vllm": self.args.use_vllm,
            "use_api": self.args.use_api,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
            "api_account": self.args.api_account,
            "tensor_parallel_size": self.args.tensor_parallel_size,
            "batch_size": self.args.batch_size,
            "gpu_memory_utilization": getattr(self.args, "gpu_memory_utilization", None),
            "vllm_max_model_len": getattr(self.args, "vllm_max_model_len", None),
            "vllm_max_num_seqs": getattr(self.args, "vllm_max_num_seqs", None),
            "vllm_enforce_eager": getattr(self.args, "vllm_enforce_eager", False),
        }
        return expert_functions.question_generation(**kwargs)

    def get_abstain_kwargs(self, patient_state):
        kwargs = {
            "max_depth": self.args.max_questions,
            "patient_state": patient_state,
            "rationale_generation": self.args.rationale_generation,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "abstain_threshold": self.args.abstain_threshold,
            "option_mode": self.args.option_mode,
            "self_consistency": self.args.self_consistency,
            "model_name": self.args.expert_model,
            "use_vllm": self.args.use_vllm,
            "use_api": self.args.use_api,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
            "api_account": self.args.api_account,
            "tensor_parallel_size": self.args.tensor_parallel_size,
            "batch_size": self.args.batch_size,
            "gpu_memory_utilization": getattr(self.args, "gpu_memory_utilization", None),
            "vllm_max_model_len": getattr(self.args, "vllm_max_model_len", None),
            "vllm_max_num_seqs": getattr(self.args, "vllm_max_num_seqs", None),
            "vllm_enforce_eager": getattr(self.args, "vllm_enforce_eager", False),
        }
        return kwargs

    def get_inference_kwargs(self):
        return {
            "model_name": self.args.expert_model,
            "use_vllm": self.args.use_vllm,
            "use_api": self.args.use_api,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
            "api_account": self.args.api_account,
            "tensor_parallel_size": self.args.tensor_parallel_size,
            "batch_size": self.args.batch_size,
            "gpu_memory_utilization": getattr(self.args, "gpu_memory_utilization", None),
            "vllm_max_model_len": getattr(self.args, "vllm_max_model_len", None),
            "vllm_max_num_seqs": getattr(self.args, "vllm_max_num_seqs", None),
            "vllm_enforce_eager": getattr(self.args, "vllm_enforce_eager", False),
        }


class RandomExpert(Expert):
    """
    Below is an example Expert system that randomly asks a question or makes a choice based on the current patient state.
    This should be replaced with a more sophisticated expert system that can make informed decisions based on the patient state.
    """

    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        initial_info = patient_state['initial_info']  # not use because it's random
        history = patient_state['interaction_history']  # not use because it's random

        # randomly decide to ask a question or make a choice
        abstain = random.random() < 0.5
        toy_question = "Can you describe your symptoms more?"
        toy_decision = self.choice(patient_state)
        conf_score = random.random()/2 if abstain else random.random()

        return {
            "type": "question" if abstain else "choice",
            "question": toy_question,
            "letter_choice": toy_decision,
            "confidence": conf_score,  # Optional confidence score
            "urgent": True,  # Example of another optional flag
            "additional_info": "Check for any recent changes."  # Any other optional data
        }

    def choice(self, patient_state):
        # Generate a choice or intermediate decision based on the current patient state
        # randomly choose an option
        return random.choice(list(self.options.keys()))


class BasicExpert(Expert):
    def respond(self, patient_state):
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.implicit_abstention_decision(**kwargs)
        return {
            "type": "question" if abstain_response_dict["abstain"] else "choice",
            "question": abstain_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class FixedExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.fixed_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }
        

class BinaryExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.binary_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class NumericalExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.numerical_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class NumericalCutOffExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.numcutoff_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class CondensedScaleExpert(Expert):
    """Scale-confidence expert that commits on condensed evidence (initial_info + extracted facts, no conv log)."""

    def respond(self, patient_state):
        max_questions = getattr(self.args, "max_questions", None)
        turn_index = len(patient_state.get("interaction_history") or [])

        if max_questions is not None and turn_index >= max_questions:
            condensed = expert_functions.condensed_patient_state(patient_state)
            letter_choice, num_tokens = expert_functions.final_choice_with_options(
                condensed, self.inquiry, self.options, **self.get_inference_kwargs())
            return {
                "type": "choice",
                "letter_choice": letter_choice,
                "confidence": None,
                "usage": num_tokens,
            }

        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.scale_abstention_decision(**kwargs)

        if not abstain_response_dict["abstain"]:
            condensed = expert_functions.condensed_patient_state(patient_state)
            letter_choice, num_tokens_final = expert_functions.final_choice_with_options(
                condensed, self.inquiry, self.options, **self.get_inference_kwargs())
            abstain_response_dict["usage"]["input_tokens"] += num_tokens_final["input_tokens"]
            abstain_response_dict["usage"]["output_tokens"] += num_tokens_final["output_tokens"]
            return {
                "type": "choice",
                "letter_choice": letter_choice,
                "confidence": abstain_response_dict["confidence"],
                "confidence_rationale": abstain_response_dict["confidence_rationale"],
                "shadow_answer": abstain_response_dict["shadow_answer"],
                "usage": abstain_response_dict["usage"],
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "confidence_rationale": abstain_response_dict["confidence_rationale"],
            "shadow_answer": abstain_response_dict["shadow_answer"],
            "usage": abstain_response_dict["usage"],
        }


class HumanExpert(Expert):
    def respond(self, patient_state):
        print("\n" + "="*60)
        print("PATIENT INFO:", patient_state["initial_info"])
        print("-"*60)
        for i, qa in enumerate(patient_state["interaction_history"], 1):
            print(f"  Q{i}: {qa['question']}")
            print(f"  A{i}: {qa['answer']}")
        print("-"*60)
        print("CLINICAL QUESTION:", self.inquiry)
        print("OPTIONS:")
        for k, v in self.options.items():
            print(f"  {k}: {v}")
        print("="*60)
        while True:
            action = input("Your move — type a question to ask the patient, or A/B/C/D to give final answer: ").strip()
            if action.upper() in self.options:
                return {"type": "choice", "letter_choice": action.upper()}
            elif action:
                return {"type": "question", "question": action, "letter_choice": list(self.options.keys())[0]}
            print("Please enter a question or a letter choice (A/B/C/D).")


class ScaleExpert(Expert):
    def respond(self, patient_state):
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.scale_abstention_decision(**kwargs)

        if not abstain_response_dict["abstain"]:
            if self.args.option_mode == "option-in-the-end":
                letter_choice, num_tokens_final = expert_functions.final_choice_with_options(
                    patient_state, self.inquiry, self.options, **self.get_inference_kwargs())
                abstain_response_dict["letter_choice"] = letter_choice
                abstain_response_dict["usage"]["input_tokens"] += num_tokens_final["input_tokens"]
                abstain_response_dict["usage"]["output_tokens"] += num_tokens_final["output_tokens"]
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "boxed_answer": abstain_response_dict["boxed_answer"],
                "confidence": abstain_response_dict["confidence"],
                "confidence_rationale": abstain_response_dict["confidence_rationale"],
                "shadow_answer": abstain_response_dict["shadow_answer"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "question_rationale": question_response_dict.get("question_rationale"),
            "letter_choice": abstain_response_dict["letter_choice"],
            "boxed_answer": abstain_response_dict["boxed_answer"],
            "confidence": abstain_response_dict["confidence"],
            "confidence_rationale": abstain_response_dict["confidence_rationale"],
            "shadow_answer": abstain_response_dict["shadow_answer"],
            "usage": abstain_response_dict["usage"]
        }