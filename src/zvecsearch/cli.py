"""CLI interface for zvecsearch."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from .config import (
    ZvecSearchConfig,
    config_to_dict,
    get_config_value,
    load_config_file,
    resolve_config,
    save_config,
    set_config_value,
    _GLOBAL_CFG,
    _PROJECT_CFG,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# -- CLI param name -> dotted config key mapping --
_PARAM_MAP = {
    "model": "embedding.model",
    "collection": "zvec.collection",
    "zvec_path": "zvec.path",
    "llm_provider": "compact.llm_provider",
    "llm_model": "compact.llm_model",
    "prompt_file": "compact.prompt_file",
    "max_chunk_size": "chunking.max_chunk_size",
    "overlap_lines": "chunking.overlap_lines",
    "debounce_ms": "watch.debounce_ms",
}


def _build_cli_overrides(**kwargs) -> dict:
    """Map flat CLI params to a nested config override dict."""
    result: dict = {}
    for param, dotted_key in _PARAM_MAP.items():
        val = kwargs.get(param)
        if val is None:
            continue
        section, field = dotted_key.split(".")
        result.setdefault(section, {})[field] = val
    return result


def _cfg_to_zvecsearch_kwargs(cfg: ZvecSearchConfig) -> dict:
    """Extract ZvecSearch constructor kwargs from a resolved config."""
    return {
        "zvec_path": cfg.zvec.path,
        "collection": cfg.zvec.collection,
        "embedding_provider": cfg.embedding.provider,
        "embedding_model": cfg.embedding.model,
        "max_chunk_size": cfg.chunking.max_chunk_size,
        "overlap_lines": cfg.chunking.overlap_lines,
        "enable_mmap": cfg.zvec.enable_mmap,
        "hnsw_m": cfg.zvec.hnsw_m,
        "hnsw_ef": cfg.zvec.hnsw_ef,
        "quantize_type": cfg.zvec.quantize_type,
        "query_ef": cfg.search.query_ef,
        "reranker": cfg.search.reranker,
        "dense_weight": cfg.search.dense_weight,
        "sparse_weight": cfg.search.sparse_weight,
    }


def _cfg_to_store_kwargs(cfg: ZvecSearchConfig) -> dict:
    """Extract ZvecStore constructor kwargs for store-only commands."""
    return {
        "path": cfg.zvec.path,
        "collection": cfg.zvec.collection,
        "embedding_provider": cfg.embedding.provider,
        "embedding_model": cfg.embedding.model,
        "enable_mmap": cfg.zvec.enable_mmap,
    }


# -- Common CLI options --

def _common_options(f):
    """Shared options for commands that create a ZvecSearch instance."""
    f = click.option("--model", "-m", default=None, help="Override embedding model.")(f)
    f = click.option("--collection", "-c", default=None, help="Zvec collection name.")(f)
    f = click.option("--zvec-path", default=None, help="Zvec database path.")(f)
    return f


@click.group(invoke_without_command=True)
@click.version_option(package_name="zvecsearch")
@click.pass_context
def cli(ctx) -> None:
    """zvecsearch -- semantic memory search for markdown knowledge bases."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ======================================================================
# Index command
# ======================================================================

@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@_common_options
@click.option("--force", is_flag=True, help="Re-index all files.")
def index(
    paths: tuple[str, ...],
    model: str | None,
    collection: str | None,
    zvec_path: str | None,
    force: bool,
) -> None:
    """Index markdown files from PATHS."""
    from .core import ZvecSearch

    cfg = resolve_config(_build_cli_overrides(
        model=model, collection=collection, zvec_path=zvec_path,
    ))
    ms = ZvecSearch(list(paths), **_cfg_to_zvecsearch_kwargs(cfg))
    try:
        n = ms.index(force=force)
        click.echo(f"Indexed {n} chunks.")
    finally:
        ms.close()


# ======================================================================
# Search command
# ======================================================================

@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=None, type=int, help="Number of results.")
@_common_options
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON.")
def search(
    query: str,
    top_k: int | None,
    model: str | None,
    collection: str | None,
    zvec_path: str | None,
    json_output: bool,
) -> None:
    """Search indexed memory for QUERY."""
    from .core import ZvecSearch

    cfg = resolve_config(_build_cli_overrides(
        model=model, collection=collection, zvec_path=zvec_path,
    ))
    ms = ZvecSearch(**_cfg_to_zvecsearch_kwargs(cfg))
    try:
        results = ms.search(query, top_k=top_k or 5)
        if json_output:
            click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            if not results:
                click.echo("No results found.")
                return
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                source = r.get("source", "?")
                heading = r.get("heading", "")
                content = r.get("content", "")
                click.echo(f"\n--- Result {i} (score: {score:.4f}) ---")
                click.echo(f"Source: {source}")
                if heading:
                    click.echo(f"Heading: {heading}")
                if len(content) > 500:
                    click.echo(content[:500])
                    chunk_hash = r.get("chunk_hash", "")
                    click.echo(f"  ... [truncated, run 'zvecsearch expand {chunk_hash}' for full content]")
                else:
                    click.echo(content)
    finally:
        ms.close()


# ======================================================================
# Expand command (progressive disclosure L2)
# ======================================================================

@cli.command()
@click.argument("chunk_hash")
@click.option("--section/--no-section", default=True, help="Show full heading section (default).")
@click.option("--lines", "-n", default=None, type=int, help="Show N lines before/after instead of full section.")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON.")
@click.option("--collection", "-c", default=None, help="Zvec collection name.")
@click.option("--zvec-path", default=None, help="Zvec database path.")
def expand(
    chunk_hash: str,
    section: bool,
    lines: int | None,
    json_output: bool,
    collection: str | None,
    zvec_path: str | None,
) -> None:
    """Expand a memory chunk to show full context."""
    from .store import ZvecStore

    cfg = resolve_config(_build_cli_overrides(
        collection=collection, zvec_path=zvec_path,
    ))
    store = ZvecStore(**_cfg_to_store_kwargs(cfg))
    try:
        chunks = store.query(filter_expr=f'chunk_hash == "{chunk_hash}"')
        if not chunks:
            click.echo(f"Chunk not found: {chunk_hash}", err=True)
            sys.exit(1)

        chunk = chunks[0]
        source = chunk["source"]
        start_line = chunk["start_line"]
        end_line = chunk["end_line"]
        heading = chunk.get("heading", "")
        heading_level = chunk.get("heading_level", 0)

        source_path = Path(source)
        if not source_path.exists():
            click.echo(f"Source file not found: {source}", err=True)
            sys.exit(1)

        all_lines = source_path.read_text(encoding="utf-8").splitlines()

        if lines is not None:
            ctx_start = max(0, start_line - 1 - lines)
            ctx_end = min(len(all_lines), end_line + lines)
            expanded = "\n".join(all_lines[ctx_start:ctx_end])
            expanded_start = ctx_start + 1
            expanded_end = ctx_end
        else:
            expanded, expanded_start, expanded_end = _extract_section(
                all_lines, start_line, heading_level,
            )

        import re
        anchor_match = re.search(
            r"<!--\s*session:(\S+)\s+turn:(\S+)\s+transcript:(\S+)\s*-->",
            expanded,
        )
        anchor = {}
        if anchor_match:
            anchor = {
                "session": anchor_match.group(1),
                "turn": anchor_match.group(2),
                "transcript": anchor_match.group(3),
            }

        if json_output:
            result = {
                "chunk_hash": chunk_hash,
                "source": source,
                "heading": heading,
                "start_line": expanded_start,
                "end_line": expanded_end,
                "content": expanded,
            }
            if anchor:
                result["anchor"] = anchor
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            click.echo(f"Source: {source} (lines {expanded_start}-{expanded_end})")
            if heading:
                click.echo(f"Heading: {heading}")
            if anchor:
                click.echo(f"Session: {anchor['session']}  Turn: {anchor['turn']}")
                click.echo(f"Transcript: {anchor['transcript']}")
            click.echo(f"\n{expanded}")
    finally:
        store.close()


def _extract_section(
    all_lines: list[str], start_line: int, heading_level: int,
) -> tuple[str, int, int]:
    """Extract the full section containing the chunk."""
    section_start = start_line - 1
    if heading_level > 0:
        for i in range(start_line - 2, -1, -1):
            line = all_lines[i]
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                if level <= heading_level:
                    section_start = i
                    break

    section_end = len(all_lines)
    if heading_level > 0:
        for i in range(start_line, len(all_lines)):
            line = all_lines[i]
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                if level <= heading_level:
                    section_end = i
                    break

    content = "\n".join(all_lines[section_start:section_end])
    return content, section_start + 1, section_end


# ======================================================================
# Transcript command (progressive disclosure L3)
# ======================================================================

@cli.command()
@click.argument("jsonl_path", type=click.Path(exists=True))
@click.option("--turn", "-t", default=None, help="Target turn UUID (prefix match).")
@click.option("--context", "-c", "ctx", default=3, type=int, help="Number of turns before/after target.")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON.")
def transcript(
    jsonl_path: str,
    turn: str | None,
    ctx: int,
    json_output: bool,
) -> None:
    """View original conversation turns from a JSONL transcript."""
    from .transcript import (
        parse_transcript,
        find_turn_context,
        format_turns,
        format_turn_index,
        turns_to_dicts,
    )

    turns = parse_transcript(jsonl_path)
    if not turns:
        click.echo("No conversation turns found.")
        return

    if turn:
        context_turns, highlight = find_turn_context(turns, turn, context=ctx)
        if not context_turns:
            click.echo(f"Turn not found: {turn}", err=True)
            sys.exit(1)
        if json_output:
            click.echo(json.dumps(turns_to_dicts(context_turns), indent=2, ensure_ascii=False))
        else:
            click.echo(f"Showing {len(context_turns)} turns around {turn[:12]}:\n")
            click.echo(format_turns(context_turns, highlight_idx=highlight))
    else:
        if json_output:
            click.echo(json.dumps(turns_to_dicts(turns), indent=2, ensure_ascii=False))
        else:
            click.echo(f"All turns ({len(turns)}):\n")
            click.echo(format_turn_index(turns))


# ======================================================================
# Watch command
# ======================================================================

@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@_common_options
@click.option("--debounce-ms", default=None, type=int, help="Debounce delay in ms.")
def watch(
    paths: tuple[str, ...],
    model: str | None,
    collection: str | None,
    zvec_path: str | None,
    debounce_ms: int | None,
) -> None:
    """Watch PATHS for markdown changes and auto-index."""
    from .core import ZvecSearch

    cfg = resolve_config(_build_cli_overrides(
        model=model, collection=collection,
        zvec_path=zvec_path, debounce_ms=debounce_ms,
    ))
    ms = ZvecSearch(list(paths), **_cfg_to_zvecsearch_kwargs(cfg))

    n = ms.index()
    if n:
        click.echo(f"Indexed {n} chunks.")

    def _on_event(event_type: str, summary: str, file_path) -> None:
        click.echo(summary)

    click.echo(f"Watching {len(paths)} path(s) for changes... (Ctrl+C to stop)")
    watcher = ms.watch(on_event=_on_event, debounce_ms=cfg.watch.debounce_ms)
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping watcher.")
    finally:
        watcher.stop()
        ms.close()


# ======================================================================
# Compact command
# ======================================================================

@cli.command()
@click.option("--source", "-s", default=None, help="Only compact chunks from this source.")
@click.option("--output-dir", "-o", default=None, type=click.Path(), help="Directory to write the compact summary into.")
@click.option("--llm-provider", default=None, help="LLM for summarization.")
@click.option("--llm-model", default=None, help="Override LLM model.")
@click.option("--prompt", default=None, help="Custom prompt template (must contain {chunks}).")
@click.option("--prompt-file", default=None, type=click.Path(exists=True), help="Read prompt template from file.")
@_common_options
def compact(
    source: str | None,
    output_dir: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    prompt: str | None,
    prompt_file: str | None,
    model: str | None,
    collection: str | None,
    zvec_path: str | None,
) -> None:
    """Compress stored memories into a summary."""
    from .core import ZvecSearch

    cfg = resolve_config(_build_cli_overrides(
        model=model, collection=collection,
        zvec_path=zvec_path, llm_provider=llm_provider,
        llm_model=llm_model, prompt_file=prompt_file,
    ))

    prompt_template = prompt
    if cfg.compact.prompt_file and not prompt_template:
        prompt_template = Path(cfg.compact.prompt_file).read_text(encoding="utf-8")

    ms = ZvecSearch(**_cfg_to_zvecsearch_kwargs(cfg))
    try:
        summary = _run(ms.compact(
            source=source,
            llm_provider=cfg.compact.llm_provider,
            llm_model=cfg.compact.llm_model or None,
            prompt_template=prompt_template,
            output_dir=output_dir,
        ))
        if summary:
            click.echo("Compact complete. Summary:\n")
            click.echo(summary)
        else:
            click.echo("No chunks to compact.")
    finally:
        ms.close()


# ======================================================================
# Optimize command
# ======================================================================

@cli.command()
@click.option("--collection", "-c", default=None, help="Zvec collection name.")
@click.option("--zvec-path", default=None, help="Zvec database path.")
def optimize(
    collection: str | None,
    zvec_path: str | None,
) -> None:
    """Optimize the index (segment merge + rebuild)."""
    from .store import ZvecStore

    cfg = resolve_config(_build_cli_overrides(
        collection=collection, zvec_path=zvec_path,
    ))
    store = ZvecStore(**_cfg_to_store_kwargs(cfg))
    try:
        store.optimize()
        click.echo("Optimization complete.")
    finally:
        store.close()


# ======================================================================
# Stats command
# ======================================================================

@cli.command()
@click.option("--collection", "-c", default=None, help="Zvec collection name.")
@click.option("--zvec-path", default=None, help="Zvec database path.")
def stats(
    collection: str | None,
    zvec_path: str | None,
) -> None:
    """Show statistics about the index."""
    from .store import ZvecStore

    cfg = resolve_config(_build_cli_overrides(
        collection=collection, zvec_path=zvec_path,
    ))
    try:
        store = ZvecStore(**_cfg_to_store_kwargs(cfg))
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    try:
        count = store.count()
        click.echo(f"Total indexed chunks: {count}")
    finally:
        store.close()


# ======================================================================
# Reset command
# ======================================================================

@cli.command()
@click.option("--collection", "-c", default=None, help="Zvec collection name.")
@click.option("--zvec-path", default=None, help="Zvec database path.")
@click.confirmation_option(prompt="This will delete all indexed data. Continue?")
def reset(
    collection: str | None,
    zvec_path: str | None,
) -> None:
    """Drop all indexed data."""
    from .store import ZvecStore

    cfg = resolve_config(_build_cli_overrides(
        collection=collection, zvec_path=zvec_path,
    ))
    store = ZvecStore(**_cfg_to_store_kwargs(cfg))
    try:
        store.drop()
        click.echo("Dropped collection.")
    finally:
        store.close()


# ======================================================================
# Config command group
# ======================================================================

@cli.group("config")
def config_group() -> None:
    """Manage zvecsearch configuration."""


@config_group.command("init")
@click.option("--project", is_flag=True, help="Write to .zvecsearch.toml (project-level) instead of global.")
def config_init(project: bool) -> None:
    """Interactive configuration wizard."""
    target = _PROJECT_CFG if project else _GLOBAL_CFG
    current = resolve_config()

    result: dict = {}

    click.echo("zvecsearch configuration wizard")
    click.echo(f"Writing to: {target}\n")

    # Zvec
    click.echo("-- Zvec --")
    result["zvec"] = {}
    result["zvec"]["path"] = click.prompt(
        "  Database path", default=current.zvec.path,
    )
    result["zvec"]["collection"] = click.prompt(
        "  Collection name", default=current.zvec.collection,
    )
    result["zvec"]["quantize_type"] = click.prompt(
        "  Quantization (none/int8/int4/fp16)", default=current.zvec.quantize_type,
    )

    # Embedding
    click.echo("\n-- Embedding --")
    result["embedding"] = {}
    result["embedding"]["model"] = click.prompt(
        "  Embedding model", default=current.embedding.model,
    )

    # Search
    click.echo("\n-- Search --")
    result["search"] = {}
    result["search"]["reranker"] = click.prompt(
        "  Reranker (rrf/weighted)", default=current.search.reranker,
    )

    # Chunking
    click.echo("\n-- Chunking --")
    result["chunking"] = {}
    result["chunking"]["max_chunk_size"] = click.prompt(
        "  Max chunk size (chars)", default=current.chunking.max_chunk_size, type=int,
    )
    result["chunking"]["overlap_lines"] = click.prompt(
        "  Overlap lines", default=current.chunking.overlap_lines, type=int,
    )

    # Watch
    click.echo("\n-- Watch --")
    result["watch"] = {}
    result["watch"]["debounce_ms"] = click.prompt(
        "  Debounce (ms)", default=current.watch.debounce_ms, type=int,
    )

    # Compact
    click.echo("\n-- Compact --")
    _compact_defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-5-20250929",
        "gemini": "gemini-2.0-flash",
    }
    result["compact"] = {}
    result["compact"]["llm_provider"] = click.prompt(
        "  LLM provider (openai/anthropic/gemini)", default=current.compact.llm_provider,
    )
    _compact_provider = result["compact"]["llm_provider"]
    _compact_model_default = current.compact.llm_model or _compact_defaults.get(_compact_provider, "")
    result["compact"]["llm_model"] = click.prompt(
        "  LLM model", default=_compact_model_default,
    )
    result["compact"]["prompt_file"] = click.prompt(
        "  Prompt file path (empty for built-in)", default=current.compact.prompt_file,
    )

    save_config(result, target)
    click.echo(f"\nConfig saved to {target}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--project", is_flag=True, help="Write to project config.")
def config_set(key: str, value: str, project: bool) -> None:
    """Set a config value (e.g. zvecsearch config set embedding.model text-embedding-3-large)."""
    try:
        set_config_value(key, value, project=project)
        target = _PROJECT_CFG if project else _GLOBAL_CFG
        click.echo(f"Set {key} = {value} in {target}")
    except (KeyError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@config_group.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a resolved config value (e.g. zvecsearch config get embedding.model)."""
    try:
        val = get_config_value(key)
        click.echo(val)
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@config_group.command("list")
@click.option("--resolved", "mode", flag_value="resolved", default=True, help="Show fully resolved config (default).")
@click.option("--global", "mode", flag_value="global", help="Show global config file only.")
@click.option("--project", "mode", flag_value="project", help="Show project config file only.")
def config_list(mode: str) -> None:
    """Show configuration."""
    import tomli_w

    if mode == "global":
        data = load_config_file(_GLOBAL_CFG)
        label = f"Global ({_GLOBAL_CFG})"
    elif mode == "project":
        data = load_config_file(_PROJECT_CFG)
        label = f"Project ({_PROJECT_CFG})"
    else:
        cfg = resolve_config()
        data = config_to_dict(cfg)
        label = "Resolved (all sources merged)"

    click.echo(f"# {label}\n")
    if data:
        click.echo(tomli_w.dumps(data))
    else:
        click.echo("(empty)")
