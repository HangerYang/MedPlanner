expert_system = {

    # ── System message ───────────────────────────────────────────────────────
    "meditron_system_msg": (
        "You are a medical doctor trying to reason through a real-life clinical case. "
        "Based on your understanding of basic and clinical science, medical knowledge, "
        "and mechanisms underlying health, disease, patient care, and modes of therapy, "
        "respond according to the task specified by the user. Base your response on the "
        "current and standard practices referenced in medical guidelines."
    ),

    # ── Conversation scaffolding ─────────────────────────────────────────────
    "question_word": "Doctor Question",
    "answer_word": "Patient Response",

    "curr_template": (
        "A patient comes into the clinic presenting with a symptom as described in the "
        "conversation log below:\n    \n"
        "PATIENT INFORMATION: {}\n"
        "CONVERSATION LOG:\n{}\n"
        "QUESTION: {}\n"
        "OPTIONS: {}\n"
        "YOUR TASK: {}"
    ),

    "curr_template_no_options": (
        "A patient comes into the clinic presenting with a symptom as described in the "
        "conversation log below:\n    \n"
        "PATIENT INFORMATION: {}\n"
        "CONVERSATION LOG:\n{}\n"
        "QUESTION: {}\n"
        "YOUR TASK: {}"
    ),

    # ── Final answer task ─────────────────────────────────────────────────────
    "answer": (
        "Assume that you already have enough information from the above question-answer "
        "pairs to answer the patient inquiry, use the above information to produce a "
        "factual conclusion. Respond with the correct letter choice (A, B, C, or D) and "
        "NOTHING ELSE.\nLETTER CHOICE: "
    ),

    "answer_boxed": (
        "Assume that you already have enough information from the above question-answer "
        "pairs to answer the patient inquiry, use the above information to produce a "
        "factual conclusion. Be concise — your answer should be a short phrase (a "
        "diagnosis, medication name, finding, etc.), not a sentence.\n"
        "OUTPUT FORMAT — exactly one line:\n"
        "    \\box{your answer}\n"
        "Do NOT use markdown code fences (```), JSON, lists, or any other formatting. "
        "Do NOT wrap the answer in quotes. Do NOT add any text before or after the "
        "\\box{...}. Just the \\box{...} line and nothing else.\n"
        "ANSWER: "
    ),

    # ── Judge ─────────────────────────────────────────────────────────────────
    "judge": (
        "The clinical question is: {}\n"
        "The correct answer is: {}\n"
        "The doctor's free-text answer is: {}\n"
        "Is the doctor's answer correct? Consider whether the meaning matches the "
        "correct answer, even if the wording differs. "
        "Answer in the following format:\n"
        "JUDGMENT: YES or NO\n"
        "REASON: one sentence explaining why.\n"
        "JUDGMENT: "
    ),

    # ── Follow-up question generator ─────────────────────────────────────────
    "atomic_question_improved": (
        "If there are missing features that prevent you from picking a confident and "
        "factual answer to the inquiry, consider which features are not yet asked about "
        "in the conversation log; then, consider which missing feature is the most "
        "important to ask the patient in order to provide the most helpful information "
        "toward a correct medical decision. You can ask about any relevant information "
        "about the patient's case, such as family history, tests and exams results, "
        "treatments already done, etc. Consider what are the common questions asked in "
        "the specific subject relating to the patient's known symptoms, and what the "
        "best and most intuitive doctor would ask. Ask ONE SPECIFIC ATOMIC QUESTION to "
        "address this feature. The question should be bite-sized, and NOT ask for too "
        "much at once. Make sure to NOT repeat any questions from the above conversation "
        "log. Answer in the following format:\n"
        "ATOMIC QUESTION: the atomic question and NOTHING ELSE.\n"
        "ATOMIC QUESTION: "
    ),

    # Same as atomic_question_improved, but the model must first output a one-sentence
    # rationale explaining why this is the most useful next question. The rationale is
    # logged for analysis; only the parsed question is forwarded to the patient model.
    "atomic_question_improved_RG": (
        "If there are missing features that prevent you from picking a confident and "
        "factual answer to the inquiry, consider which features are not yet asked about "
        "in the conversation log; then, consider which missing feature is the most "
        "important to ask the patient in order to provide the most helpful information "
        "toward a correct medical decision. You can ask about any relevant information "
        "about the patient's case, such as family history, tests and exams results, "
        "treatments already done, etc. Consider what are the common questions asked in "
        "the specific subject relating to the patient's known symptoms, and what the "
        "best and most intuitive doctor would ask. Ask ONE SPECIFIC ATOMIC QUESTION to "
        "address this feature. The question should be bite-sized, and NOT ask for too "
        "much at once. Make sure to NOT repeat any questions from the above conversation "
        "log. Answer strictly in the following two-line format:\n"
        "REASON: a one-sentence explanation of which missing feature you are targeting "
        "and why this question is the most useful one to ask right now.\n"
        "ATOMIC QUESTION: the atomic question and NOTHING ELSE."
    ),

    "top_k_questions_RG": (
        "If there are missing features that prevent you from picking a confident and "
        "factual answer to the inquiry, propose the TWO best next atomic questions to "
        "ask the patient. The two questions should target different useful missing "
        "features when possible, should be bite-sized, and must not repeat questions "
        "already present in the conversation log. For each question, briefly explain "
        "why that question is useful for distinguishing the answer choices. Answer "
        "strictly in this format:\n"
        "QUESTION 1 REASON: one sentence explaining why question 1 is useful.\n"
        "QUESTION 1: the first atomic question.\n"
        "QUESTION 2 REASON: one sentence explaining why question 2 is useful.\n"
        "QUESTION 2: the second atomic question."
    ),

    # ── Abstention strategies ─────────────────────────────────────────────────

    # Implicit: model outputs a question OR a letter in one shot (BasicExpert)
    "implicit": (
        "Given the information so far, if you are confident to pick an option correctly "
        "and factually, respond with the letter choice and NOTHING ELSE. Otherwise, if "
        "you are not confident to pick an option and need more information, ask ONE "
        "SPECIFIC ATOMIC QUESTION to the patient. The question should be bite-sized, "
        "NOT ask for too much at once, and NOT repeat what has already been asked. In "
        "this case, respond with the atomic question and NOTHING ELSE."
    ),

    "implicit_RG": (
        "Given the information so far, if you are confident to pick an option correctly "
        "and factually, respond in the format:\n"
        "REASON: a one-sentence explanation of why you are choosing a particular option.\n"
        "ANSWER: the letter choice and NOTHING ELSE. Otherwise, if you are not confident "
        "to pick an option and need more information, ask ONE SPECIFIC ATOMIC QUESTION "
        "to the patient. The question should be bite-sized, NOT ask for too much at "
        "once, and NOT repeat what has already been asked. In this case, respond in the "
        "format:\n"
        "REASON: a one-sentence explanation of why you should ask the particular "
        "question.\n"
        "QUESTION: the atomic question and NOTHING ELSE."
    ),

    # Binary: model says YES/NO to "do you have enough info?" (BinaryExpert)
    "binary": (
        "Medical conditions are complex, so you should seek to understand their "
        "situations across many features. First, consider which medical specialty is "
        "this patient's case; then, consider a list of necessary features a doctor "
        "would need to make the right medical judgment; finally, consider whether all "
        "necessary information is given in the conversation above. Now, are you "
        "confident to pick the correct option to the inquiry factually using the "
        "conversation log? Answer with YES or NO and NOTHING ELSE."
    ),

    "binary_RG": (
        "Medical conditions are complex, so you should seek to understand their "
        "situations across many features. First, consider which medical specialty is "
        "this patient's case; then, consider a list of necessary features a doctor "
        "would need to make the right medical judgment; finally, consider whether all "
        "necessary information is given in the conversation above. Up to this point, "
        "are you confident to pick the correct option to the inquiry factually using "
        "the conversation log? Answer in the following format:\n"
        "REASON: a one-sentence explanation of why you are or are not confident and "
        "what other information is needed.\n"
        "DECISION: YES or NO."
    ),

    # YES/NO follow-up used by NumericalExpert after producing a confidence score
    "yes_no": (
        "Now, are you confident to pick the correct option to the inquiry factually "
        "using the conversation log? Answer with YES or NO and NOTHING ELSE."
    ),

    # Numerical: model outputs a 0.0–1.0 score (NumericalExpert + NumericalCutOffExpert)
    "numerical": (
        "Medical conditions are complex, so you should seek to understand their "
        "situations across many features. First, consider which medical specialty is "
        "this patient's case; then, consider a list of necessary features a doctor "
        "would need to make the right medical judgment; finally, consider whether all "
        "necessary information is given in the conversation above. What is your "
        "confidence score to pick the correct option to the inquiry factually using the "
        "conversation log? Answer with the probability as a float from 0.0 to 1.0 and "
        "NOTHING ELSE."
    ),

    "numerical_RG": (
        "Medical conditions are complex, so you should seek to understand their "
        "situations across many features. First, consider which medical specialty is "
        "this patient's case; then, consider a list of necessary features a doctor "
        "would need to make the right medical judgment; finally, consider whether all "
        "necessary information is given in the conversation above. What is your "
        "confidence score to pick the correct option to the inquiry factually using the "
        "conversation log? Answer strictly in the following format:\n"
        "REASON: a one-sentence explanation of why you are or are not confident and "
        "what other information is needed.\n"
        "SCORE: your confidence score written as a float from 0.0 to 1.0."
    ),

    # Numcutoff: same score prompt as numerical, abstention decided by threshold
    "numcutoff": (
        "Medical conditions are complex, so you should seek to understand their "
        "situations across many features. First, consider which medical specialty is "
        "this patient's case; then, consider a list of necessary features a doctor "
        "would need to make the right medical judgment; finally, consider whether all "
        "necessary information is given in the conversation above. What is your "
        "confidence score to pick the correct option to the inquiry factually using the "
        "conversation log? Answer with the probability as a float from 0.0 to 1.0 and "
        "NOTHING ELSE."
    ),

    "numcutoff_RG": (
        "Medical conditions are complex, so you should seek to understand their "
        "situations across many features. First, consider which medical specialty is "
        "this patient's case; then, consider a list of necessary features a doctor "
        "would need to make the right medical judgment; finally, consider whether all "
        "necessary information is given in the conversation above. What is your "
        "confidence score to pick the correct option to the inquiry factually using the "
        "conversation log? Answer strictly in the following format:\n"
        "REASON: a one-sentence explanation of why you are or are not confident and "
        "what other information is needed.\n"
        "SCORE: your confidence score written as a float from 0.0 to 1.0."
    ),

    # Scale: model outputs a 1–5 Likert rating (ScaleExpert)
    "scale": (
        "Medical conditions are complex, so you should seek to understand their "
        "situations across many features. First, consider which medical specialty is "
        "this patient's case; then, consider a list of necessary features a doctor "
        "would need to make the right medical judgment; finally, consider whether all "
        "necessary information is given in the conversation above. How confident are "
        "you to pick the correct option to the problem factually using the conversation "
        "log? Choose between the following ratings:\n"
        '"Very Confident" - The correct option is supported by all evidence, and there '
        "is enough evidence to eliminate the rest of the answers, so the option can be "
        "confirmed conclusively.\n"
        '"Somewhat Confident" - I have reasonably enough information to tell that the '
        "correct option is more likely than other options, more information is helpful "
        "to make a conclusive decision.\n"
        '"Neither Confident or Unconfident" - There are evident supporting the correct '
        "option, but further evidence is needed to be sure which one is the correct "
        "option.\n"
        '"Somewhat Unconfident" - There are evidence supporting more than one options, '
        "therefore more questions are needed to further distinguish the options.\n"
        '"Very Unconfident" - There are not enough evidence supporting any of the '
        "options, the likelihood of picking the correct option at this point is near "
        "random guessing.\n\n"
        "Think carefully step by step, respond with the chosen confidence rating ONLY "
        "and NOTHING ELSE."
    ),

    "scale_RG": (
        "Medical conditions are complex, so you should seek to understand their "
        "situations across many features. First, consider which medical specialty is "
        "this patient's case; then, consider a list of necessary features a doctor "
        "would need to make the right medical judgment; finally, consider whether all "
        "necessary information is given in the conversation above. How confident are "
        "you to pick the correct option to the problem factually using the conversation "
        "log? Choose between the following ratings:\n"
        '"Very Confident" - The correct option is supported by all evidence, and there '
        "is enough evidence to eliminate the rest of the answers, so the option can be "
        "confirmed conclusively.\n"
        '"Somewhat Confident" - I have reasonably enough information to tell that the '
        "correct option is more likely than other options, more information is helpful "
        "to make a conclusive decision.\n"
        '"Neither Confident or Unconfident" - There are evident supporting the correct '
        "option, but further evidence is needed to be sure which one is the correct "
        "option.\n"
        '"Somewhat Unconfident" - There are evidence supporting more than one options, '
        "therefore more questions are needed to further distinguish the options.\n"
        '"Very Unconfident" - There are not enough evidence supporting any of the '
        "options, the likelihood of picking the correct option at this point is near "
        "random guessing.\n\n"
        "Answer in the following format:\n"
        "REASON: a one-sentence explanation of why you are or are not confident and "
        "what other information is needed.\n"
        "DECISION: chosen rating from the above list."
    ),

    # ── UNUSED — kept as comments for reference ───────────────────────────────
    # "meditron_system_msg_old": "You are a medical doctor answering real-world medical
    #     entrance exam questions ... Task: You will be asked to reason through the
    #     current patient's information and either ask an information seeking question
    #     or choose an option.",
    # "meditron_system_msg_original": "You are a medical doctor answering real-world
    #     medical entrance exam questions ...",
    # "basic_system_msg": "You are an experienced doctor trying to make a medical
    #     decision about a patient.",
    # "empty_system_msg": "",
    # "only_choice": "Please answer with ONLY the correct letter choice ...",
    # "system": "You are an experienced doctor ...",
    # "starter": "A patient comes into the clinic ...\n\nCONVERSATION LOG:\n",
    # "task": "Given the information from above, your task is to choose one of four
    #     options that best answers the inquiry.",
    # "prompt": (multi-turn prompt combining confidence reasoning + answer/question),
    # "implicit_abstain": (older version of implicit prompt),
    # "atomic_question": (older version of atomic_question_improved),
    # "verbal_abstain_llama": (binary YES/NO without REASON field),
    # "non_interactive": {"starter": ..., "question_prompt": ..., "response": ...},
}

# ── UNUSED SECTIONS ───────────────────────────────────────────────────────────
# patient_system: prompt templates for patient-side LLM calls.
#   Not referenced in code — patient classes build their prompts inline.
#
# conformal_scores: per-option probability scoring prompt.
#   Not referenced anywhere in the current codebase.
