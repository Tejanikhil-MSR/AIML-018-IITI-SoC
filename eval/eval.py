from prometheus_eval import PrometheusEval
from prometheus_eval.vllm import VLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT

model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

instruction = "Explain gravity to a 10-year-old."
response = "Gravity is what pulls things down to Earth, like when you drop your toy."
reference_answer = "Gravity is a force that pulls objects toward each other. It's what makes things fall to the ground."

rubric = """
Criteria: Simplicity and clarity in explanation for a child.
1: Confusing, technical, not suitable for kids.
3: Understandable, but lacks clarity or vividness.
5: Clear, simple, and age-appropriate.
"""

feedback, score = judge.single_absolute_grade(
    instruction=instruction,
    response=response,
    rubric=rubric,
    reference_answer=reference_answer
)

print("Feedback:", feedback)
print("Score:", score)
