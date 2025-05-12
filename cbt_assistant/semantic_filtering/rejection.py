import random

DEFAULT_REJECTION_PHRASES = [
    "Sorry, I am not able to help with this request.",
    "This question seems to be out of my area of expertise.",
    "Please ask a question related to cognitive behavioral therapy.",
    "I am designed to assist only with CBT-related topics.",
    "Unfortunately, I cannot process this request.",
    "Could you please rephrase your question to focus on therapy-related topics?",
    "I'm specialized in mental health and CBT, not this subject.",
    "I'm here to help with psychological well-being, not with this kind of query.",
    "Let's focus on cognitive and behavioral therapy topics, please.",
    "Sorry, this doesn't seem to be a relevant topic for this assistant."
]


class RejectionMessageGenerator:

    def generate(self) -> str:
        return random.choice(DEFAULT_REJECTION_PHRASES)