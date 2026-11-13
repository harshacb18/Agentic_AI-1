import gc
import tempfile
import unittest

from chromadb.api.client import SharedSystemClient

from interview_coach.archive import InterviewArchive
from interview_coach.curriculum import INTERVIEW_ROUNDS, build_role_round_csv


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.archive = InterviewArchive(self.temporary_directory.name)

    def tearDown(self):
        self.archive.client._system.stop()
        self.archive = None
        SharedSystemClient.clear_system_cache()
        gc.collect()
        self.temporary_directory.cleanup()

    def test_ingest_deduplicates_and_retrieves_exact_role_evidence(self):
        csv_content = (
            "company,role,type,content,source\n"
            "Acme,Analyst,question,How would you validate a metric?,Archive A\n"
            "Acme,Engineer,question,How would you design a service?,Archive A\n"
        ).encode()

        self.assertEqual(
            self.archive.ingest_csv(csv_content, source_label="upload", dataset="test"), 2
        )
        self.assertEqual(
            self.archive.ingest_csv(csv_content, source_label="upload", dataset="test"), 0
        )
        results = self.archive.retrieve(
            company="Acme", role="Analyst", query="metric validation", kind="question"
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["exact_role"])
        self.assertIn("validate a metric", results[0]["content"])

    def test_missing_role_evidence_falls_back_to_company_evidence(self):
        self.archive.ingest_csv(
            b"company,role,type,content\nAcme,Engineer,feedback,Clarify trade-offs.\n",
            source_label="upload",
            dataset="test",
        )

        results = self.archive.retrieve(
            company="Acme", role="Analyst", query="trade-offs", kind="feedback"
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["exact_role"])

    def test_retrieval_skips_questions_already_asked(self):
        first_question = "How would you validate a sudden change in a metric?"
        second_question = "How would you explain a confidence interval to a stakeholder?"
        third_question = "How would you assess a dashboard's data quality?"
        self.archive.ingest_csv(
            (
                "company,role,type,content\n"
                f"Acme,Analyst,question,{first_question}\n"
                f"Acme,Analyst,question,{second_question}\n"
                f"Acme,Analyst,question,\"{third_question}\"\n"
            ).encode(),
            source_label="upload",
            dataset="test",
        )

        results = self.archive.retrieve(
            company="Acme",
            role="Analyst",
            query="metric analysis interview",
            kind="question",
            limit=10,
            exclude_content=[first_question, second_question],
        )

        self.assertTrue(results)
        content = [item["content"] for item in results]
        self.assertNotIn(first_question, content)
        self.assertNotIn(second_question, content)
        self.assertIn(third_question, content)

    def test_retrieval_can_be_scoped_to_an_interview_round(self):
        self.archive.ingest_csv(
            (
                "company,role,type,content,round\n"
                "Acme,Analyst,question,Estimate the conversion change.,Aptitude\n"
                "Acme,Analyst,question,Explain a design trade-off.,Technical\n"
            ).encode(),
            source_label="practice pack",
            dataset="illustrative",
        )

        results = self.archive.retrieve(
            company="Acme",
            role="Analyst",
            query="interview question",
            kind="question",
            round_name="Technical",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["round"], "Technical")
        self.assertIn("design trade-off", results[0]["content"])

    def test_role_pack_retrieves_a_fixed_unasked_set_for_each_round(self):
        packed_csv = build_role_round_csv("Example Co", "AI/ML Engineer")
        self.assertIsNotNone(packed_csv)
        self.archive.ingest_csv(
            packed_csv,
            source_label="Illustrative role-round practice pack",
            dataset="illustrative",
        )

        for interview_round in INTERVIEW_ROUNDS:
            with self.subTest(round=interview_round.name):
                questions = self.archive.retrieve(
                    company="Example Co",
                    role="AI/ML Engineer",
                    query=f"AI/ML Engineer {interview_round.focus}",
                    kind="question",
                    round_name=interview_round.name,
                )
                self.assertEqual(len(questions), 2)
                next_questions = self.archive.retrieve(
                    company="Example Co",
                    role="AI/ML Engineer",
                    query=f"AI/ML Engineer {interview_round.focus}",
                    kind="question",
                    round_name=interview_round.name,
                    exclude_content=[questions[0]["content"]],
                )
                self.assertEqual(len(next_questions), 1)
                self.assertNotEqual(
                    questions[0]["content"], next_questions[0]["content"]
                )

    def test_rejects_unsupported_archive_record_types(self):
        with self.assertRaisesRegex(ValueError, "unsupported type"):
            self.archive.ingest_csv(
                b"company,role,type,content\nAcme,Analyst,profile,Candidate bio.\n",
                source_label="upload",
                dataset="test",
            )


if __name__ == "__main__":
    unittest.main()
