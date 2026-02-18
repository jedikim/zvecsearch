from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    content: str
    source: str
    heading: str
    heading_level: int
    start_line: int
    end_line: int
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            h = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            object.__setattr__(self, "content_hash", h)


def compute_chunk_id(
    source: str,
    start_line: int,
    end_line: int,
    content_hash: str,
    model: str,
) -> str:
    raw = f"markdown:{source}:{start_line}:{end_line}:{content_hash}:{model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_markdown(
    text: str,
    source: str = "",
    max_chunk_size: int = 1500,
    overlap_lines: int = 2,
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    sections: list[tuple[str, int, int, int]] = []  # heading, level, start, end

    heading_positions = []
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            heading_positions.append((i, len(m.group(1)), m.group(2).strip()))

    if not heading_positions or heading_positions[0][0] > 0:
        end = heading_positions[0][0] if heading_positions else len(lines)
        sections.append(("", 0, 0, end))

    for idx, (pos, level, title) in enumerate(heading_positions):
        next_pos = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
        sections.append((title, level, pos, next_pos))

    chunks: list[Chunk] = []
    for heading, level, start, end in sections:
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        if len(body) <= max_chunk_size:
            chunks.append(Chunk(
                content=body, source=source, heading=heading,
                heading_level=level, start_line=start + 1, end_line=end,
            ))
        else:
            chunks.extend(_split_large_section(
                lines[start:end], source, heading, level,
                start, max_chunk_size, overlap_lines,
            ))

    return chunks


def _split_large_section(
    lines: list[str],
    source: str,
    heading: str,
    heading_level: int,
    base_line: int,
    max_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = 0

    for i, line in enumerate(lines):
        buf.append(line)
        current = "\n".join(buf).strip()
        if len(current) >= max_size and len(buf) > 1:
            text = "\n".join(buf[:-1]).strip()
            if text:
                chunks.append(Chunk(
                    content=text, source=source, heading=heading,
                    heading_level=heading_level,
                    start_line=base_line + buf_start + 1,
                    end_line=base_line + buf_start + len(buf) - 1,
                ))
            buf_start = max(0, i - overlap)
            buf = list(lines[buf_start:i + 1])

    if buf:
        text = "\n".join(buf).strip()
        if text:
            chunks.append(Chunk(
                content=text, source=source, heading=heading,
                heading_level=heading_level,
                start_line=base_line + buf_start + 1,
                end_line=base_line + buf_start + len(buf),
            ))

    return chunks
