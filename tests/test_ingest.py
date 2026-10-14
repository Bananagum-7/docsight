from pathlib import Path

from docsight.ingest.chunk import chunk_sections, pack, split_units
from docsight.ingest.markdown import page_url, parse_markdown

DOC = """# Background Tasks { #background-tasks }

You can define background tasks to run after returning a response.

## Using `BackgroundTasks` { #using-backgroundtasks }

Import it and declare a parameter.

{* ../../docs_src/bg/tutorial001.py hl[1] *}

```python
# a heading inside code is not a heading
x = 1
```

### Details

More text.
"""


def write_doc(tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "docs" / "en" / "docs"
    (docs / "tutorial").mkdir(parents=True)
    md = docs / "tutorial" / "background-tasks.md"
    md.write_text(DOC)
    code = tmp_path / "docs_src" / "bg"
    code.mkdir(parents=True)
    (code / "tutorial001.py").write_text("from fastapi import BackgroundTasks\n")
    return md, docs


def test_parse_sections_breadcrumbs_and_includes(tmp_path):
    md, docs = write_doc(tmp_path)
    sections = parse_markdown(md, docs, code_root=tmp_path)
    crumbs = [s.breadcrumb for s in sections]
    assert crumbs == [
        "Background Tasks",
        "Background Tasks > Using BackgroundTasks",
        "Background Tasks > Using BackgroundTasks > Details",
    ]
    using = sections[1]
    assert using.anchor == "using-backgroundtasks"
    assert "from fastapi import BackgroundTasks" in using.text  # include was inlined
    assert "# a heading inside code" in using.text  # fenced line not treated as heading
    assert all(s.title == "Background Tasks" for s in sections)


def test_page_url():
    base = "https://fastapi.tiangolo.com/"
    assert page_url(base, "tutorial/background-tasks.md", "x") == "https://fastapi.tiangolo.com/tutorial/background-tasks/#x"
    assert page_url(base, "tutorial/index.md", "") == "https://fastapi.tiangolo.com/tutorial/"
    assert page_url(base, "index.md", "") == "https://fastapi.tiangolo.com/"


def test_code_blocks_never_split():
    code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(80)) + "\n```"
    units = split_units("Intro paragraph.\n\n" + code + "\n\nOutro.")
    assert len(units) == 3 and units[1] == code
    chunks = pack(units, size=50, overlap=5)
    assert any(c.count("```") == 2 for c in chunks)
    assert all(c.count("```") % 2 == 0 for c in chunks)


def test_chunks_have_header_and_url(tmp_path):
    md, docs = write_doc(tmp_path)
    chunks = chunk_sections(parse_markdown(md, docs, tmp_path), "https://fastapi.tiangolo.com/")
    c = chunks[1]
    assert c.text.startswith("[Background Tasks > Using BackgroundTasks]")
    assert c.url.endswith("/tutorial/background-tasks/#using-backgroundtasks")
    assert c.has_code
