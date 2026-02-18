import pytest
from unittest.mock import AsyncMock, patch
from zvecsearch.compact import compact_chunks, COMPACT_PROMPT


def test_default_prompt_has_placeholder():
    assert "{chunks}" in COMPACT_PROMPT


@pytest.mark.asyncio
async def test_compact_joins_chunks():
    chunks = [
        {"content": "chunk one"},
        {"content": "chunk two"},
    ]
    with patch("zvecsearch.compact._compact_openai", new_callable=AsyncMock) as mock:
        mock.return_value = "summary"
        result = await compact_chunks(chunks, llm_provider="openai")
        call_prompt = mock.call_args[0][0]
        assert "chunk one" in call_prompt
        assert "chunk two" in call_prompt
        assert result == "summary"


@pytest.mark.asyncio
async def test_compact_custom_template():
    chunks = [{"content": "data"}]
    template = "Summarize: {chunks}"
    with patch("zvecsearch.compact._compact_openai", new_callable=AsyncMock) as mock:
        mock.return_value = "done"
        await compact_chunks(chunks, prompt_template=template)
        assert mock.call_args[0][0] == "Summarize: data"
