import unittest

from interview_coach.agents import InterviewAgents, format_evidence
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from interview_coach.routing import Intent, detect_intent, should_retrieve


class RoutingTests(unittest.TestCase):
    def test_explicit_intents_take_priority(self):
        self.assertEqual(detect_intent("Please coach me", has_active_question=True), Intent.COACH)
        self.assertEqual(detect_intent("next question please", has_active_question=True), Intent.INTERVIEW)
        self.assertEqual(detect_intent("show my report", has_active_question=True), Intent.REPORT)
        self.assertEqual(detect_intent("score this response", has_active_question=True), Intent.EVALUATE)

    def test_unlabelled_message_is_answer_only_when_question_is_open(self):
        self.assertEqual(
            detect_intent("I would first validate the metric definition.", has_active_question=True),
            Intent.EVALUATE,
        )
        self.assertEqual(detect_intent("hello", has_active_question=False), Intent.INTERVIEW)

    def test_report_retrieval_requires_an_evaluated_answer(self):
        self.assertFalse(
            should_retrieve(
                Intent.REPORT, company="Acme", role="Analyst", has_evaluations=False
            )
        )
        self.assertTrue(
            should_retrieve(
                Intent.REPORT, company="Acme", role="Analyst", has_evaluations=True
            )
        )
        self.assertFalse(
            should_retrieve(Intent.COACH, company="", role="Analyst", has_evaluations=True)
        )

    def test_evidence_label_includes_match_quality_and_source(self):
        evidence = format_evidence(
            [
                {
                    "content": "Clarify the metric before proposing a test.",
                    "metadata": {
                        "company": "Acme",
                        "role": "Analyst",
                        "kind": "feedback",
                        "source": "Archive row 4",
                    },
                    "exact_role": True,
                }
            ]
        )
        self.assertIn("exact role match", evidence)
        self.assertIn("Archive row 4", evidence)

    def test_evidence_without_matches_is_explicit(self):
        self.assertIn("No matching archive evidence", format_evidence([]))

    def test_interviewer_uses_archived_question_verbatim_when_available(self):
        agents = InterviewAgents(FakeListChatModel(responses=["Unrelated generated question."]))
        question = agents.ask_question(
            company="Acme",
            role="Analyst",
            focus="metrics",
            records=[
                {
                    "content": "How would you validate a sudden change in a metric?",
                    "metadata": {"kind": "question"},
                    "exact_role": True,
                }
            ],
        )
        self.assertEqual(question, "How would you validate a sudden change in a metric?")

    def test_interviewer_selects_an_unasked_archived_question(self):
        agents = InterviewAgents(FakeListChatModel(responses=["A fresh generated question?"]))
        question = agents.ask_question(
            company="Acme",
            role="Analyst",
            focus="metrics",
            records=[
                {
                    "content": "How would you validate a sudden change in a metric?",
                    "metadata": {"kind": "question"},
                    "exact_role": True,
                },
                {
                    "content": "How would you explain a confidence interval to a stakeholder?",
                    "metadata": {"kind": "question"},
                    "exact_role": True,
                },
            ],
            previous_questions=["How would you validate a sudden change in a metric?"],
        )

        self.assertEqual(question, "How would you explain a confidence interval to a stakeholder?")

    def test_interviewer_generates_new_question_after_archive_questions_are_used(self):
        agents = InterviewAgents(FakeListChatModel(responses=["How would you prioritize an analysis?"]))
        previous_question = "How would you validate a sudden change in a metric?"
        question = agents.ask_question(
            company="Acme",
            role="Analyst",
            focus="metrics",
            records=[
                {
                    "content": previous_question,
                    "metadata": {
                        "company": "Acme",
                        "role": "Analyst",
                        "kind": "question",
                        "source": "test archive",
                    },
                    "exact_role": True,
                }
            ],
            previous_questions=[previous_question],
        )

        self.assertEqual(question, "How would you prioritize an analysis?")

    def test_interviewer_does_not_ask_a_company_only_question_verbatim(self):
        agents = InterviewAgents(
            FakeListChatModel(responses=["How would you analyze a model's false negatives?"])
        )
        question = agents.ask_question(
            company="Acme",
            role="AI/ML Engineer",
            focus="Role Fundamentals",
            records=[
                {
                    "content": "How would you validate a marketing conversion metric?",
                    "metadata": {
                        "company": "Acme",
                        "role": "Data Analyst",
                        "kind": "question",
                        "source": "test archive",
                    },
                    "exact_role": False,
                }
            ],
        )

        self.assertEqual(
            question,
            "How would you analyze a model's false negatives?",
        )


if __name__ == "__main__":
    unittest.main()
