import pytest
from zvecsearch.chunker import Chunk, chunk_markdown, compute_chunk_id


class TestChunk:
    def test_chunk_has_content_hash(self):
        c = Chunk(content="hello", source="a.md", heading="", heading_level=0,
                  start_line=1, end_line=1)
        assert len(c.content_hash) == 16
        assert c.content_hash == Chunk(content="hello", source="a.md", heading="",
                                        heading_level=0, start_line=1, end_line=1).content_hash

    def test_different_content_different_hash(self):
        c1 = Chunk(content="aaa", source="a.md", heading="", heading_level=0,
                   start_line=1, end_line=1)
        c2 = Chunk(content="bbb", source="a.md", heading="", heading_level=0,
                   start_line=1, end_line=1)
        assert c1.content_hash != c2.content_hash


class TestChunkMarkdown:
    def test_simple_heading_split(self):
        md = "# H1\nParagraph one.\n## H2\nParagraph two."
        chunks = chunk_markdown(md, source="test.md")
        assert len(chunks) == 2
        assert chunks[0].heading == "H1"
        assert chunks[0].heading_level == 1
        assert chunks[1].heading == "H2"
        assert chunks[1].heading_level == 2

    def test_preamble_without_heading(self):
        md = "Some intro text.\n# Title\nBody."
        chunks = chunk_markdown(md, source="t.md")
        assert len(chunks) == 2
        assert chunks[0].heading == ""
        assert chunks[0].heading_level == 0

    def test_empty_input(self):
        assert chunk_markdown("", source="e.md") == []

    def test_whitespace_only(self):
        assert chunk_markdown("   \n\n  ", source="e.md") == []

    def test_large_section_splitting(self):
        big = "# Big\n" + ("word " * 400 + "\n\n") * 3
        chunks = chunk_markdown(big, source="big.md", max_chunk_size=500)
        assert len(chunks) > 1
        for c in chunks:
            assert c.heading == "Big"

    def test_source_and_lines(self):
        md = "# A\nLine1\nLine2\n# B\nLine3"
        chunks = chunk_markdown(md, source="s.md")
        assert chunks[0].source == "s.md"
        assert chunks[0].start_line == 1
        assert chunks[1].start_line == 4


class TestComputeChunkId:
    def test_deterministic(self):
        a = compute_chunk_id("s.md", 1, 5, "abc123", "openai")
        b = compute_chunk_id("s.md", 1, 5, "abc123", "openai")
        assert a == b

    def test_different_model_different_id(self):
        a = compute_chunk_id("s.md", 1, 5, "abc123", "openai")
        b = compute_chunk_id("s.md", 1, 5, "abc123", "google")
        assert a != b
