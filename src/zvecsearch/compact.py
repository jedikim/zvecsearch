from __future__ import annotations

COMPACT_PROMPT = """You are a knowledge compression assistant. Given a set of memory chunks, produce a concise, well-structured summary in Markdown that preserves key facts, decisions, and action items.

Chunks:

{chunks}

Produce a clear, concise summary:"""


async def compact_chunks(
    chunks: list[dict],
    llm_provider: str = "openai",
    model: str | None = None,
    prompt_template: str | None = None,
) -> str:
    combined = "\n\n---\n\n".join(c["content"] for c in chunks)
    template = prompt_template or COMPACT_PROMPT
    prompt = template.format(chunks=combined)

    if llm_provider == "openai":
        return await _compact_openai(prompt, model or "gpt-4o-mini")
    elif llm_provider == "anthropic":
        return await _compact_anthropic(prompt, model or "claude-sonnet-4-5-20250929")
    elif llm_provider == "gemini":
        return await _compact_gemini(prompt, model or "gemini-2.0-flash")
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider!r}")


async def _compact_openai(prompt: str, model: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


async def _compact_anthropic(prompt: str, model: str) -> str:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()
    resp = await client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


async def _compact_gemini(prompt: str, model: str) -> str:
    from google import genai
    client = genai.Client()
    resp = await client.aio.models.generate_content(model=model, contents=prompt)
    return resp.text or ""
