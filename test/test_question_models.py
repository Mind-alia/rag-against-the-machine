from src.structdata import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    UnansweredQuestion,
)


def test_create_unanswered_question() -> None:
    question = UnansweredQuestion(
        question_id="question-001",
        question="How does vLLM generate text?",
    )

    assert question.question_id == "question-001"
    assert question.question == "How does vLLM generate text?"


def test_create_answered_question() -> None:
    source = MinimalSource(
        file_path="vllm/entrypoints/llm.py",
        first_character_index=100,
        last_character_index=500,
    )

    question = AnsweredQuestion(
        question_id="question-002",
        question="Where is the LLM class implemented?",
        answer="The LLM class is implemented in vllm/entrypoints/llm.py.",
        sources=[source],
    )

    assert question.answer
    assert len(question.sources) == 1
    assert question.sources[0].file_path == "vllm/entrypoints/llm.py"


def test_create_complete_dataset() -> None:
    unanswered_question = UnansweredQuestion(
        question_id="question-001",
        question="How does vLLM generate text?",
    )

    dataset = RagDataset(
        rag_questions=[unanswered_question],
    )

    assert len(dataset.rag_questions) == 1
    assert dataset.rag_questions[0].question_id == "question-001"
