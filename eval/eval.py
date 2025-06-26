from prometheus_eval import PrometheusEval
from prometheus_eval.vllm import VLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT

model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

instruction = " "
response = " "
reference_answer = " "

rubric = """
Criteria: Given a student's query, the chatbot's response
1. Accuracy (1–5)
2. Groundedness (1–5)
3. Relevance (1–5)
4. Completeness (1–5)
5. Conciseness (1–5)
6. Fluency (1–5)
7. Hallucination (Yes/No)
"""

feedback, score = judge.single_absolute_grade(
    instruction=instruction,
    response=response,
    rubric=rubric,
    reference_answer=reference_answer
)

print("Feedback:", feedback)
print("Score:", score)
