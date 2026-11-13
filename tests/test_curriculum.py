import csv
import io
import unittest

from interview_coach.curriculum import (
    INTERVIEW_ROUNDS,
    QUESTIONS_PER_ROUND,
    ROLE_QUESTION_BANK,
    build_role_round_csv,
)


class CurriculumTests(unittest.TestCase):
    def test_every_supported_role_has_the_fixed_question_count_in_each_round(self):
        self.assertTrue(ROLE_QUESTION_BANK)
        for role, round_questions in ROLE_QUESTION_BANK.items():
            with self.subTest(role=role):
                self.assertEqual(len(round_questions), len(INTERVIEW_ROUNDS))
                self.assertTrue(
                    all(
                        len(questions) == QUESTIONS_PER_ROUND
                        for questions in round_questions
                    )
                )

    def test_role_pack_is_company_scoped_and_contains_round_labels(self):
        packed_csv = build_role_round_csv("Example Co", "AI/ML Engineer")
        self.assertIsNotNone(packed_csv)
        rows = list(csv.DictReader(io.StringIO(packed_csv.decode("utf-8"))))

        self.assertEqual(len(rows), len(INTERVIEW_ROUNDS) * QUESTIONS_PER_ROUND)
        self.assertEqual({row["company"] for row in rows}, {"Example Co"})
        self.assertEqual({row["role"] for row in rows}, {"AI/ML Engineer"})
        self.assertEqual(
            {row["round"] for row in rows},
            {interview_round.name for interview_round in INTERVIEW_ROUNDS},
        )


if __name__ == "__main__":
    unittest.main()
