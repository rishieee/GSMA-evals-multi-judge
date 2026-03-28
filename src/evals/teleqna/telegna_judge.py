from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import scorer, Score, accuracy, mean, stderr
from inspect_ai.solver import chain_of_thought, generate, prompt_template
from inspect_ai.model import get_model, GenerateConfig
import re

DEFAULT_SUBJECT = "full"
DEFAULT_DATASET = "rishieee/ORAN_TeleQNA_filtered"
DEFAULT_DATASET_NAME = "default"
DEFAULT_SPLIT = "train"

# ── Judge prompt ──────────────────────────────────────────────────────────────
GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[reasoning]: {reasoning}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. The [response] is the candidate's Exact Answer field. Put the extracted answer as 'None' if there is no exact, final answer to extract from [response].

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

GRADE: Answer 'C' only if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'I' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
There is no partial category for correctness. Only output 'C' or 'I'.

reasoning_quality: <write ONLY a single integer from 0 to 10 rating the quality of the reasoning in [reasoning]. No letters, no explanation.>
    0    = No reasoning or completely wrong/contradictory
    1-4  = Weak reasoning, major gaps or unsupported claims
    5-6  = Mixed reasoning, some correct elements but meaningful flaws
    7-8  = Sound reasoning, minor gaps or imprecision
    9-10 = Strong, precise, technically correct reasoning

Output format (strict; output exactly these four lines and nothing else):
extracted_final_answer: <answer text or None>
reasoning: <1-3 concise sentences>
GRADE: <C|I>
reasoning_quality: <single integer 0-10>

Do not output a standalone number, do not output bullet points, and do not replace the label 'GRADE' with any other label such as correct, verdict, or correctness.
""".strip()

# ── Primary model prompt ──────────────────────────────────────────────────────
QUERY_TEMPLATE = """
{prompt}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# ── Regex patterns ────────────────────────────────────────────────────────────
GRADE_PATTERN = re.compile(
    r"(?im)^\s*grade\s*:\s*([CI])\b"
)

# Primary: labelled match — wide net for common judge variants
REASONING_QUALITY_PATTERN = re.compile(
    r"(?im)^\s*(?:line\s*2\s*:?\s*)?(?:reasoning[_\s]*quality|reasoning\s+confidence)?\s*[:\-]?\s*(\d{1,2})(?:\s*/\s*10)?\s*$"
)
EXPLANATION_PATTERN = re.compile(
    r"(?is)^\s*\*{0,2}(?:explanation|reasoning)\*{0,2}\s*[:\-]\s*(.+?)(?=^\s*\*{0,2}exact\s*answer\*{0,2}\s*[:\-]|\Z)",
    re.DOTALL | re.MULTILINE,
)
ANSWER_PATTERN = re.compile(
    r"(?is)^\s*\*{0,2}exact\s*answer\*{0,2}\s*[:\-]\s*(.+?)(?=^\s*\*{0,2}confidence\*{0,2}\s*[:\-]|\Z)",
    re.DOTALL | re.MULTILINE,
)
CONFIDENCE_PATTERN = re.compile(
    r"(?im)^\s*\*{0,2}confidence\*{0,2}\s*[:\-]\s*([^\n]+)\s*$"
)


def _clean_extracted_field(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^\*{1,2}\s*", "", cleaned)
    cleaned = re.sub(r"\s*\*{1,2}$", "", cleaned)
    return cleaned.strip()


# ── Custom scorer ─────────────────────────────────────────────────────────────
def teleqna_scorer(judge_model: str):
    @scorer(metrics={
        "accuracy": [accuracy(), stderr()],
        "reasoning_quality": [mean(), stderr()],
    })
    def _scorer():
        async def score(state, target):
            # ── 1. Parse primary model output ─────────────────────────────
            output = state.output.completion
            explanation_match = EXPLANATION_PATTERN.search(output)
            answer_match = ANSWER_PATTERN.search(output)
            confidence_match = CONFIDENCE_PATTERN.search(output)

            reasoning = _clean_extracted_field(explanation_match.group(1)) if explanation_match else output.strip()
            answer = _clean_extracted_field(answer_match.group(1)) if answer_match else output.strip()
            confidence = confidence_match.group(1).strip() if confidence_match else None
            parse_failed = not explanation_match or not answer_match

            # ── 2. Single judge call ───────────────────────────────────────
            judge = get_model(judge_model, config=GenerateConfig(max_tokens=512))
            judge_result = await judge.generate(
                GRADER_TEMPLATE.format(
                    question=state.input_text,
                    reasoning=reasoning,
                    response=answer,
                    correct_answer=target.text,
                )
            )
            judge_output = judge_result.completion

            # ── 3. Parse GRADE: C/I ───────────────────────────────────────
            grade_match = GRADE_PATTERN.search(judge_output)
            grade = grade_match.group(1).upper() if grade_match else "I"
            correctness = 1.0 if grade == "C" else 0.0

            # ── 4. Parse reasoning_quality ────────────────────────────────
            # Primary: labelled match
            rq_match = REASONING_QUALITY_PATTERN.search(judge_output)

            if rq_match:
                raw = int(rq_match.group(1))
                rq_parse_failed = False
                raw = max(0, min(10, raw))
                reasoning_quality = float(raw)
            else:
                raw = None
                rq_parse_failed = True
                # Use NaN so mean/stderr metrics can skip unparsed values cleanly.
                reasoning_quality = float("nan")

            # ── 5. Return score ───────────────────────────────────────────
            return Score(
                value={
                    "accuracy": correctness,
                    "reasoning_quality": reasoning_quality,
                },
                answer=answer,
                explanation=judge_output,
                metadata={
                    "grade": grade,
                    "reasoning_quality_raw": raw,
                    "rq_parse_failed": rq_parse_failed,
                    "primary_parse_failed": parse_failed,
                    "student_confidence": confidence,
                    "student_reasoning": reasoning,
                    "judge_output": judge_output,
                },
            )
        return score
    return _scorer()


# ── Dataset ───────────────────────────────────────────────────────────────────
def record_to_sample(record: dict) -> Sample:
    import ast

    choices_raw = record.get("choices", [])
    if isinstance(choices_raw, str):
        try:
            choices = ast.literal_eval(choices_raw)
        except (ValueError, SyntaxError):
            choices = [c.strip() for c in choices_raw.strip("[]").split(", ") if c.strip("'\"")]
    else:
        choices = choices_raw

    answer_raw = record.get("answer")
    if isinstance(answer_raw, str) and answer_raw.isalpha() and len(answer_raw) == 1:
        answer_index = ord(answer_raw.upper()) - ord("A")
    elif answer_raw is None:
        raise ValueError("Record is missing an 'answer' field")
    else:
        answer_index = int(str(answer_raw))
    correct_answer_text = str(choices[answer_index])

    return Sample(
        input=str(record.get("Question", record.get("question", ""))),
        target=correct_answer_text,
        metadata={"subject": record.get("subject")},
    )


# ── Task ──────────────────────────────────────────────────────────────────────
@task
def teleqna_generative(
    subject: str = DEFAULT_SUBJECT,
    dataset_path: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    judge_model: str = "llama-cpp-python/tele-it-q8",
) -> Task:
    dataset = hf_dataset(
        dataset_path,
        name=DEFAULT_DATASET_NAME,
        sample_fields=record_to_sample,
        split=split,
    )
    if subject != DEFAULT_SUBJECT:
        dataset = dataset.filter(
            lambda sample: sample.metadata is not None
            and sample.metadata.get("subject") == subject
        )

    return Task(
        dataset=dataset,
        solver=[
            prompt_template(
                "You are an expert Telecommunication Engineer. Provide short, accurate, "
                "professional and standard-compliant answers.\n\nQuestion:\n{prompt}"
            ),
            chain_of_thought(template=QUERY_TEMPLATE),
            generate(),
        ],
        scorer=teleqna_scorer(judge_model=judge_model),
    )
