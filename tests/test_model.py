import json
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from interview_coach.model import ModelServiceError, OpenAIHttpChatModel, build_chat_model


class OpenAIHttpChatModelTests(unittest.TestCase):
    def test_sends_chat_messages_and_parses_completion_without_network(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {"content": "ready"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 2},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(respond))
        model = OpenAIHttpChatModel(
            model_name="test-model",
            api_key=SecretStr("test-key"),
        )
        try:
            with patch("interview_coach.model.httpx.Client", return_value=client):
                result = model.invoke(
                    [
                        SystemMessage(content="Be concise."),
                        HumanMessage(content="Say ready."),
                    ]
                )
        finally:
            client.close()

        self.assertEqual(result.content, "ready")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer test-key")
        payload = json.loads(requests[0].read())
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say ready."},
            ],
        )

    def test_api_provider_requires_a_key(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            build_chat_model(provider="OpenAI API", model_name="test-model")

    def test_quota_error_is_actionable_without_exposing_provider_response(self):
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    429,
                    json={
                        "error": {
                            "code": "insufficient_quota",
                            "message": "sensitive provider diagnostic",
                        }
                    },
                )
            )
        )
        model = OpenAIHttpChatModel(
            model_name="test-model",
            api_key=SecretStr("test-key"),
        )
        try:
            with patch("interview_coach.model.httpx.Client", return_value=client):
                with self.assertRaises(ModelServiceError) as raised:
                    model.invoke("Say ready.")
        finally:
            client.close()

        self.assertIn("no remaining API quota", str(raised.exception))
        self.assertNotIn("sensitive provider diagnostic", str(raised.exception))

    def test_rate_limit_error_recommends_waiting_and_checking_usage(self):
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})
            )
        )
        model = OpenAIHttpChatModel(
            model_name="test-model",
            api_key=SecretStr("test-key"),
        )
        try:
            with patch("interview_coach.model.httpx.Client", return_value=client):
                with self.assertRaises(ModelServiceError) as raised:
                    model.invoke("Say ready.")
        finally:
            client.close()

        self.assertIn("rate-limiting", str(raised.exception))
        self.assertIn("account usage and billing", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
