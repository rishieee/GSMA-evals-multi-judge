import re
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, chain_of_thought, generate
from inspect_ai.model import get_model

DEFAULT_SUBJECT = "full"
DEFAULT_DATASET = "rishieee/ORAN_TeleQNA_filtered"
DEFAULT_DATASET_NAME = "ORAN_TeleQNA_filtered"
DEFAULT_SPLIT = "train"

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[student_reasoning]: {student_reasoning}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

reasoning_score: A score from 1 to 10 evaluating the logical soundness and clarity of the student's reasoning process.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()


def record_to_sample(record: dict) -> Sample:
    """Convert dataset record to Sample with subject metadata."""
    import ast
    # Handle choices which might be a stringified list depending on HF loading
    choices_raw = record["choices"]
    if isinstance(choices_raw, str):
        try:
            choices = ast.literal_eval(choices_raw)
        except (ValueError, SyntaxError):
            choices = [c.strip() for c in choices_raw.strip("[]").split(", ") if c.strip("'\"")]
    else:
        choices = choices_raw

    # Extract the full text for the correct answer
    correct_answer_text = str(choices[int(record["answer"])])
    
    return Sample(
        input=str(record["Question"]),
        target=correct_answer_text,
        metadata={
            "subject": record.get("subject"),
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def teleqna_grader(judge_model: str = "ollama/phi4-mini-reasoning:latest"):
    """Custom scorer that calls a judge model using GRADER_TEMPLATE, parses the response, and computes a score."""
    async def score(state: TaskState, target: Target) -> Score:
        question = state.input_text
        student_output = state.output.completion
        correct_answer = target.text
        
        # Format the prompt using the provided template
        prompt = GRADER_TEMPLATE.format(
            question=question,
            student_reasoning=student_output, # Pass the entire output as reasoning for now
            response=student_output,
            correct_answer=correct_answer
        )
        
        # Call the judge model to evaluate the result
        model = get_model(judge_model)
        result = await model.generate(prompt)
        judge_text = result.completion
        
        # Initialize extracted fields
        extracted_final_answer = ""
        reasoning = ""
        reasoning_score = ""
        correct_answer_eval = ""
        confidence = ""
        
        # Parse output fields using Regex. We use re.DOTALL to match across newlines in explanations.
        ans_match = re.search(r"extracted_final_answer:\s*(.*?)(?=\n\[correct_answer\]|\nreasoning:|\nreasoning_score:|\Z)", judge_text, re.IGNORECASE | re.DOTALL)
        if ans_match: extracted_final_answer = ans_match.group(1).strip()
            
        rs_match = re.search(r"reasoning:\s*(.*?)(?=\nreasoning_score:|\ncorrect:|\nconfidence:|\Z)", judge_text, re.IGNORECASE | re.DOTALL)
        if rs_match: reasoning = rs_match.group(1).strip()
            
        rscore_match = re.search(r"reasoning_score:\s*(.*?)(?=\ncorrect:|\nconfidence:|\Z)", judge_text, re.IGNORECASE | re.DOTALL)
        if rscore_match: reasoning_score = rscore_match.group(1).strip()
            
        corr_match = re.search(r"correct:\s*(.*?)(?=\nconfidence:|\Z)", judge_text, re.IGNORECASE | re.DOTALL)
        if corr_match: correct_answer_eval = corr_match.group(1).strip().lower()
            
        conf_match = re.search(r"confidence:\s*(.*?)(?=\Z)", judge_text, re.IGNORECASE | re.DOTALL)
        if conf_match: confidence = conf_match.group(1).strip()
            
        # Determine actual score boolean (yes=CORRECT, no=INCORRECT)
        is_correct = "yes" in correct_answer_eval
        
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=extracted_final_answer,
            explanation=reasoning, # Natively supported for displaying scorer explanations
            metadata={
                "reasoning_score": reasoning_score,
                "confidence": confidence,
                "judge_raw_output": judge_text
            }
        )
    return score


@task
def teleqna_generative(
    subject: str = DEFAULT_SUBJECT,
    dataset_path: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    judge_model: str = "ollama/phi4-mini-reasoning:latest",
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
        solver=[chain_of_thought(), generate()],
        scorer=teleqna_grader(judge_model=judge_model),
    )
