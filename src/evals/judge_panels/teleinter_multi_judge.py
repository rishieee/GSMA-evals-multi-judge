import math
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, accuracy, mean, scorer, stderr
from inspect_ai.solver import chain_of_thought, generate, prompt_template

DEFAULT_DATASET = "rishieee/5G_interview_questions_OEM"
DEFAULT_DATASET_NAME = "default"
DEFAULT_SPLIT = "train"

# ── Judge prompt ──────────────────────────────────────────────────────────────
GRADER_TEMPLATE = """
You are a 5G Telecommunications Domain Expert (3GPP, NR, 5GC, O-RAN). Judge whether the [Student Answer] is technically correct using your expert domain knowledge. Use [correct_answer] as a reference guide, not as the sole truth.

[question]: {question}

[Student Explanation]: {reasoning}

[Student Answer]: {answer}

[correct_answer]: {correct_answer}

Output ONLY these fields and must produce, nothing else and make sure the Judge_reasoning, GRADE, and reasoning_quality are there in order.

Judge_reasoning: One brief sentence on why the [Student Answer] is correct or incorrect based on your expert 3GPP/5G domain knowledge. Reference [correct_answer] only as a guide. Do not solve the problem.

GRADE: C if the [Student Answer] is technically correct per your expert 3GPP/5G domain knowledge and consistent with [correct_answer]. Make sure your GRADE aligns with your reasoning. If not technically correct, GRADE: I.

reasoning_quality: Integer 0-10 rating how factually correct and well-supported the student's reasoning in [Student Explanation] is, judged primarily by your expert 3GPP/5G knowledge, with [correct_answer] as a reference guide.
  0=no reasoning/contradictory | 1-4=weak, major gaps | 5-6=partial, missing key 3GPP mechanisms/interfaces/concepts from [correct_answer] | 7-8=sound, covers all key concepts, only minor wording gaps | 9-10=strong, precise, explicitly names correct 3GPP mechanisms and interfaces
reasoning_evidence: <=40-char quote from [Student Explanation] justifying the score, or None.
""".strip()

# ── Primary model prompt ──────────────────────────────────────────────────────
QUERY_TEMPLATE = """
{prompt}

Your response should be in the following format:
Student Explanation: {{your explanation for your final answer in 2 lines}}
Student Answer: {{your succinct, final answer in 1 line}}
""".strip()

# ── Regex patterns ────────────────────────────────────────────────────────────
GRADE_PATTERN = re.compile(
    r"(?i)\bgrade\s*[:\-]?\s*(correct|incorrect|[ci]\b|[ci](?=reasoning))"
)

REASONING_QUALITY_PATTERN = re.compile(
    r"(?i)\breasoning[_\s]*quality\s*[:\-]?\s*(\d{1,2})"
)
EXPLANATION_PATTERN = re.compile(
    r"(?is)^\s*\*{0,2}(?:student\s*explanation|explanation|reasoning)\*{0,2}\s*[:\-]\s*(.+?)(?=^\s*\*{0,2}(?:student\s*answer|exact\s*answer)\*{0,2}\s*[:\-]|\Z)",
    re.DOTALL | re.MULTILINE,
)
ANSWER_PATTERN = re.compile(
    r"(?is)^\s*\*{0,2}(?:student\s*answer|exact\s*answer)(?:\*{0,2}\s*[:\-]|[:\-]\s*\*{0,2})\s*(.+?)\s*\Z",
    re.DOTALL | re.MULTILINE,
)

REASONING_QUALITY_FALLBACK_PATTERN = re.compile(
    r"(?is)reasoning[_\s]*quality[^0-9]{0,24}(10|[0-9])\b"
)

STANDALONE_INT_LINE_PATTERN = re.compile(
    r"(?im)^\s*(10|[0-9])\s*$"
)


def _clean_extracted_field(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^\*{1,2}\s*", "", cleaned)
    cleaned = re.sub(r"\s*\*{1,2}$", "", cleaned)
    return cleaned.strip()


def _parse_reasoning_quality(judge_output: str) -> tuple[float, int | None, bool]:
    rq_match = REASONING_QUALITY_PATTERN.search(judge_output)
    if rq_match:
        raw = int(rq_match.group(1))
        raw = max(0, min(10, raw))
        return float(raw), raw, False

    rq_fallback = REASONING_QUALITY_FALLBACK_PATTERN.search(judge_output)
    if rq_fallback:
        raw = int(rq_fallback.group(1))
        raw = max(0, min(10, raw))
        return float(raw), raw, False

    standalone_lines = STANDALONE_INT_LINE_PATTERN.findall(judge_output)
    if standalone_lines:
        raw = int(standalone_lines[-1])
        raw = max(0, min(10, raw))
        return float(raw), raw, False

    return float("nan"), None, True


def _sanitize_model_name(model_name: str) -> str:
    name = model_name.split("/")[-1]
    name = name.replace(":latest", "").replace("-gguf", "")
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _infer_grade_from_reasoning(judge_output: str) -> str | None:
    txt = judge_output.lower()
    negative_patterns = [
        r"not semantically equivalent",
        r"not equivalent",
        r"not semantically equal",
        r"not equivalent to",
        r"does not convey",
        r"does not match",
        r"is incorrect",
        r"not semantically",
    ]
    for p in negative_patterns:
        if re.search(p, txt):
            return "I"

    positive_patterns = [
        r"semantically equivalent",
        r"is semantically equivalent",
        r"semantically equal",
    ]
    for p in positive_patterns:
        if re.search(p, txt):
            return "C"

    return None


# ── Dataset ───────────────────────────────────────────────────────────────────
def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=str(record.get("Questions", "")),
        target=str(record.get("Refference Answers", "")),
        metadata={
            "category": record.get("Category"),
            "target_company": record.get("Target Company"),
            "id": record.get("ID"),
            "web_reference": record.get("Web Reference Link"),
        },
    )


# ── Custom scorer ─────────────────────────────────────────────────────────────
def teleinter_scorer(
    judge_models: list[str] | str,
):
    if isinstance(judge_models, str):
        if judge_models.startswith("[") and judge_models.endswith("]"):
            try:
                import ast
                judge_models = ast.literal_eval(judge_models)
            except Exception:
                judge_models = [judge_models]
        elif "," in judge_models:
            judge_models = [m.strip() for m in judge_models.split(",")]
        else:
            judge_models = [judge_models]

    metric_keys = {
        "accuracy": [accuracy(), stderr()],
        "reasoning_quality": [mean(), stderr()],
    }
    for j_model in judge_models:
        key = _sanitize_model_name(j_model)
        metric_keys[f"accuracy_{key}"] = [accuracy(), stderr()]
        metric_keys[f"reasoning_quality_{key}"] = [mean(), stderr()]

    @scorer(metrics=metric_keys)
    def _scorer():
        async def score(state, target):
            # ── 1. Parse primary model output ─────────────────────────────
            output = state.output.completion
            explanation_match = EXPLANATION_PATTERN.search(output)
            answer_match = ANSWER_PATTERN.search(output)

            reasoning = _clean_extracted_field(explanation_match.group(1)) if explanation_match else output.strip()
            answer = _clean_extracted_field(answer_match.group(1)) if answer_match else output.strip()
            parse_failed = not explanation_match or not answer_match or not answer

            # ── 2. Run all judges ─────────────────────────────────────────
            judge_results = []

            for judge_model in judge_models:
                supports_reasoning = any(x in judge_model.lower() for x in ["gemini", "o1", "o3", "claude-3-7", "claude-3.7"])
                effort = "low" if supports_reasoning else "none"

                judge = get_model(
                    judge_model,
                    config=GenerateConfig(
                        max_tokens=2048,
                        temperature=0.1,
                        verbosity="low",
                        reasoning_effort=effort,
                    ),
                )
                judge_output = ""
                grade_match = None
                reasoning_quality = float("nan")
                raw = None
                rq_parse_failed = True

                for attempt in range(3):
                    prompt_text = GRADER_TEMPLATE.format(
                        question=state.input_text,
                        reasoning=reasoning,
                        answer=answer,
                        correct_answer=target.text,
                    )
                    if attempt > 0:
                        prompt_text += (
                            "\n\nIMPORTANT: Your previous response was malformed/truncated. "
                            "Make sure to output your response in the specified format, "
                            "including exactly 'GRADE: C' or 'GRADE: I' and 'reasoning_quality: <integer>'."
                        )

                    judge_result = await judge.generate(prompt_text)
                    judge_output = judge_result.completion

                    grade_match = GRADE_PATTERN.search(judge_output)
                    reasoning_quality, raw, rq_parse_failed = _parse_reasoning_quality(judge_output)

                    if grade_match and not rq_parse_failed:
                        break

                if grade_match:
                    raw_grade = grade_match.group(1).upper()
                    grade = "C" if raw_grade.startswith("C") else "I"
                else:
                    grade = "I"
                correctness = 1.0 if grade == "C" else 0.0

                inferred = _infer_grade_from_reasoning(judge_output)
                grade_correction = None
                if inferred and inferred != grade:
                    grade_correction = {"original_grade": grade, "corrected_grade": inferred}
                    grade = inferred
                    correctness = 1.0 if grade == "C" else 0.0

                entry = {
                    "model": judge_model,
                    "grade": grade,
                    "correctness": correctness,
                    "reasoning_quality": reasoning_quality,
                    "raw_rq": raw,
                    "rq_parse_failed": rq_parse_failed,
                    "output": judge_output,
                }
                if grade_correction:
                    entry.update({"grade_consistency_fixed": True, **grade_correction})

                judge_results.append(entry)

            # ── 3. Populate dynamic score values ──────────────────────────
            score_values = {}
            for r in judge_results:
                key = _sanitize_model_name(r["model"])
                score_values[f"accuracy_{key}"] = r["correctness"]
                score_values[f"reasoning_quality_{key}"] = r["reasoning_quality"]

            avg_correctness = sum(r["correctness"] for r in judge_results) / len(judge_results)

            valid_rqs = [
                r["reasoning_quality"]
                for r in judge_results
                if not math.isnan(r["reasoning_quality"])
            ]
            avg_rq = sum(valid_rqs) / len(valid_rqs) if valid_rqs else float("nan")

            score_values["accuracy"] = avg_correctness
            score_values["reasoning_quality"] = avg_rq

            # ── 4. Return score ───────────────────────────────────────────
            explanation_parts = []
            for r in judge_results:
                explanation_parts.append(
                    f"### Judge Model: {r['model']}\n"
                    f"**Grade**: {r['grade']} | **Reasoning Quality**: {r['raw_rq']}\n\n"
                    f"**Output**:\n{r['output']}"
                )

            return Score(
                value=score_values,
                answer=answer,
                explanation="\n\n---\n\n".join(explanation_parts),
                metadata={
                    "judges": judge_results,
                    "primary_parse_failed": parse_failed,
                    "student_reasoning": reasoning,
                },
            )
        return score
    return _scorer()


# ── Task ──────────────────────────────────────────────────────────────────────
@task
def teleinter(
    dataset_path: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    category: str | None = None,
    judge_models: list[str] | str = [
        "openrouter/openai/gpt-5.5",
        "openrouter/google/gemini-3.1-pro-preview",
        "openrouter/google/gemini-3.5-flash",
    ],
) -> Task:
    dataset = hf_dataset(
        dataset_path,
        name=DEFAULT_DATASET_NAME,
        sample_fields=record_to_sample,
        split=split,
    )
    if category is not None:
        dataset = dataset.filter(
            lambda s: s.metadata is not None
            and s.metadata.get("category") == category
        )

    return Task(
        dataset=dataset,
        solver=[
            prompt_template(
                "You are an expert 5G Telecommunications Engineer. "
                "Provide a concise, accurate, and technically precise answer to the following interview question.\n\n"
                "Question:\n{prompt}"
            ),
            chain_of_thought(template=QUERY_TEMPLATE),
            generate(),
        ],
        scorer=teleinter_scorer(judge_models=judge_models),
    )
