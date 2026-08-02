from unittest.mock import MagicMock, patch

from physics_agent.llm_client import LLMClient, DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT_SECONDS


def _make_client_with_stubbed_openai(**client_kwargs):
    """Constructs an LLMClient with the underlying openai.OpenAI class
    replaced by a stub that records how it was called, so we can verify
    timeout/max_tokens threading without any real network access."""
    with patch("physics_agent.llm_client.OpenAI") as mock_openai_cls:
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance
        client = LLMClient(**client_kwargs)
        return client, mock_openai_cls, mock_instance


def test_default_timeout_is_passed_to_openai_client():
    client, mock_openai_cls, _ = _make_client_with_stubbed_openai()
    _, call_kwargs = mock_openai_cls.call_args
    assert call_kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS


def test_custom_timeout_is_passed_to_openai_client():
    client, mock_openai_cls, _ = _make_client_with_stubbed_openai(timeout=30.0)
    _, call_kwargs = mock_openai_cls.call_args
    assert call_kwargs["timeout"] == 30.0


def test_default_max_tokens_included_in_chat_request():
    client, _, mock_instance = _make_client_with_stubbed_openai()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello"))]
    mock_instance.chat.completions.create.return_value = fake_response

    client.chat([{"role": "user", "content": "hi"}])

    _, call_kwargs = mock_instance.chat.completions.create.call_args
    assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS


def test_custom_max_tokens_included_in_chat_request():
    client, _, mock_instance = _make_client_with_stubbed_openai(max_tokens=500)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello"))]
    mock_instance.chat.completions.create.return_value = fake_response

    client.chat([{"role": "user", "content": "hi"}])

    _, call_kwargs = mock_instance.chat.completions.create.call_args
    assert call_kwargs["max_tokens"] == 500


def test_max_tokens_none_omits_the_field_entirely():
    # Explicitly disabling the cap (max_tokens=None) should not send a
    # max_tokens kwarg at all, rather than sending max_tokens=None (which
    # some backends might not handle the same way as "no limit").
    client, _, mock_instance = _make_client_with_stubbed_openai(max_tokens=None)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello"))]
    mock_instance.chat.completions.create.return_value = fake_response

    client.chat([{"role": "user", "content": "hi"}])

    _, call_kwargs = mock_instance.chat.completions.create.call_args
    assert "max_tokens" not in call_kwargs


def test_per_call_max_tokens_overrides_client_default():
    client, _, mock_instance = _make_client_with_stubbed_openai(max_tokens=2048)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello"))]
    mock_instance.chat.completions.create.return_value = fake_response

    client.chat([{"role": "user", "content": "hi"}], max_tokens=100)

    _, call_kwargs = mock_instance.chat.completions.create.call_args
    assert call_kwargs["max_tokens"] == 100


def test_per_call_temperature_overrides_client_default():
    client, _, mock_instance = _make_client_with_stubbed_openai()
    client.temperature = 0.2
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello"))]
    mock_instance.chat.completions.create.return_value = fake_response

    client.chat([{"role": "user", "content": "hi"}], temperature=0.9)

    _, call_kwargs = mock_instance.chat.completions.create.call_args
    assert call_kwargs["temperature"] == 0.9


def test_chat_returns_message_content():
    client, _, mock_instance = _make_client_with_stubbed_openai()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="the answer"))]
    mock_instance.chat.completions.create.return_value = fake_response

    result = client.chat([{"role": "user", "content": "hi"}])
    assert result == "the answer"
