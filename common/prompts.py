"""Prompt template baked into training targets.
"""

THINKING_PROMPT_TEMPLATE = (
    "You are an expert document OCR assistant.\n"
    "Carefully read the provided image(s) of the document.\n\n"
    "Question: {question}\n\n"
    "Reason step by step between <think> and </think> tags: identify the "
    "candidate fields and values in the image(s) relevant to the question, "
    "then determine which is the required one.\n"
    "Then, immediately after the closing </think> tag, write your final "
    "answer by itself on the next line — just the answer text, with no "
    "labels, headers, brackets, or additional commentary, and do not "
    "start a new round of reasoning after it."
)
