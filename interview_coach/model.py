"""LangChain chat-model adapters for local Ollama and the OpenAI HTTP API."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import ChatOllama
from pydantic import SecretStr


class ModelServiceError(RuntimeError):
    """A safe, actionable error returned by the hosted chat service."""


class OpenAIHttpChatModel(BaseChatModel):
    """OpenAI-compatible chat completions client using pure-Python HTTPX."""

    model_name: str
    api_key: SecretStr
    temperature: float = 0.25
    timeout: float = 120

    @property
    def _llm_type(self) -> str:
        return "openai-http-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        api_messages = [self._format_message(message) for message in messages]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if not response.is_success:
                try:
                    error_body = response.json().get("error", {})
                except (ValueError, AttributeError):
                    error_body = {}
                error_code = error_body.get("code") if isinstance(error_body, dict) else None
                if not isinstance(error_code, str):
                    error_code = None
                if error_code in {"insufficient_quota", "billing_hard_limit_reached"}:
                    raise ModelServiceError(
                        "This account has no remaining API quota. Check its usage, billing, "
                        "and spending limits, then try again."
                    )
                if response.status_code == 429:
                    raise ModelServiceError(
                        "The service is temporarily rate-limiting requests. Wait a little and "
                        "try again; if it continues, check account usage and billing."
                    )
                if response.status_code in {401, 403}:
                    raise ModelServiceError(
                        "The service rejected its credentials. Check the API key in the project "
                        ".env file."
                    )
                if response.status_code == 404:
                    raise ModelServiceError(
                        "The configured model was not found or is unavailable to this account."
                    )
                if response.status_code >= 500:
                    raise ModelServiceError(
                        "The service is temporarily unavailable. Please try again shortly."
                    )
                raise ModelServiceError(
                    f"The service could not complete the request (HTTP {response.status_code})."
                )

        result = response.json()
        choices = result.get("choices")
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise ValueError("The OpenAI API response did not contain a chat completion.")

        completion = choices[0]
        content = completion["message"].get("content")
        if not isinstance(content, str):
            raise ValueError("The OpenAI API returned an empty or unsupported message.")

        generation = ChatGeneration(
            message=AIMessage(content=content),
            generation_info={
                "finish_reason": completion.get("finish_reason"),
                "model_name": result.get("model", self.model_name),
            },
        )
        return ChatResult(
            generations=[generation],
            llm_output={"token_usage": result.get("usage", {})},
        )

    @staticmethod
    def _format_message(message: BaseMessage) -> dict[str, str]:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ChatMessage):
            role = message.role
        else:
            raise ValueError(f"Unsupported message type for OpenAI chat: {message.type}")

        if not isinstance(message.content, str):
            raise ValueError("OpenAI chat prompts must contain text messages.")
        return {"role": role, "content": message.content}


def build_chat_model(
    *,
    provider: str,
    model_name: str,
    api_key: str = "",
    base_url: str = "http://localhost:11434",
) -> BaseChatModel:
    if provider == "OpenAI API":
        if not api_key.strip():
            raise ValueError("Add an OpenAI API key to the project .env file.")
        return OpenAIHttpChatModel(
            model_name=model_name,
            api_key=SecretStr(api_key.strip()),
            temperature=0.25,
        )
    if provider == "Ollama (local)":
        return ChatOllama(
            model=model_name,
            base_url=base_url.rstrip("/"),
            temperature=0.25,
        )
    raise ValueError(f"Unsupported model provider: {provider}")
