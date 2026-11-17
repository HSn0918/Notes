#!/usr/bin/env python3
"""Zero-dependency SQLite FTS5 search CLI and deterministic catalog/tag index builder."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_DIRS = {".git", ".idea", "node_modules", ".obsidian", ".traces", "__pycache__"}
IGNORED_ROOT_DOCS = {"AGENTS.md", "CLAUDE.md"}
GENERATED_DOCS = {"docs/catalog.md", "docs/tags.md"}
SCHEMA_VERSION = "2"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TAG_LINE_RE = re.compile(r"^#[^\s#]")
TAG_TOKEN_RE = re.compile(r"(?:^|\s)#([^\s#`\[\](),;:\"'<>]+)")
WIKILINK_ALIAS_RE = re.compile(r"\[\[[^\[\]\|\n]+\|([^\[\]\n]+)\]\]")
WIKILINK_PLAIN_RE = re.compile(r"\[\[([^\[\]\n]+)\]\]")
MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]+\)")

ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_.-]+")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

DOMAIN_ORDER = [
    "ai",
    "cloud-native",
    "go",
    "foundations",
    "docs",
]

DOMAIN_TITLES = {
    "ai": "AI / LLM Systems",
    "cloud-native": "Cloud Native & Kubernetes",
    "go": "Go & Runtime",
    "foundations": "CS & Engineering Foundations",
    "docs": "Repository Docs",
}


@dataclass(frozen=True)
class NoteRecord:
    path: Path
    rel_path: str
    stem: str
    domain: str
    topic: str
    title: str
    headings_h1: tuple[str, ...]
    headings_h2: tuple[str, ...]
    headings_other: tuple[str, ...]
    tags: tuple[str, ...]
    summary: str
    raw_lines: tuple[str, ...]
    body_text: str


def strip_inline_code(line: str) -> str:
    """Remove inline code spans outside wikilinks."""
    out: list[str] = []
    i = 0
    while i < len(line):
        if line[i] == "`":
            marker_len = 1
            while i + marker_len < len(line) and line[i + marker_len] == "`":
                marker_len += 1
            marker = "`" * marker_len
            close = line.find(marker, i + marker_len)
            if close != -1:
                out.append(" ")
                i = close + marker_len
                continue
        out.append(line[i])
        i += 1
    return "".join(out)


def clean_markup(text: str) -> str:
    """Convert Markdown/wikilink markup into plain summary text safe for tables."""
    text = WIKILINK_ALIAS_RE.sub(r"\1", text)
    text = WIKILINK_PLAIN_RE.sub(r"\1", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = text.replace("|", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_fts(text: str) -> str:
    """Emit split ASCII identifiers plus CJK unigrams and bigrams for FTS5."""
    tokens: list[str] = []
    for match in ASCII_WORD_RE.finditer(text):
        raw = match.group(0).lower().strip(".-_")
        if not raw:
            continue
        tokens.append(raw)
        parts = [p for p in re.split(r"[_.-]+", raw) if p]
        if len(parts) > 1:
            tokens.extend(parts)
    for match in CJK_RUN_RE.finditer(text):
        run = match.group(0)
        tokens.extend(list(run))
        if len(run) > 1:
            for i in range(len(run) - 1):
                tokens.append(run[i : i + 2])
    return " ".join(tokens)


def infer_domain_and_topic(rel_path: str) -> tuple[str, str]:
    parts = Path(rel_path).parts
    if len(parts) <= 1:
        return ("", "")
    domain = parts[0]
    if domain == "cloud-native" and len(parts) >= 3 and parts[1] == "kubernetes":
        if len(parts) == 3:
            return (domain, "cloud-native/kubernetes")
        return (domain, "/".join(parts[:3]))
    if len(parts) >= 3:
        return (domain, "/".join(parts[:2]))
    return (domain, domain)


def parse_note(path: Path, root: Path) -> NoteRecord:
    rel_path = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    raw_lines = tuple(text.splitlines())

    headings_h1: list[str] = []
    headings_h2: list[str] = []
    headings_other: list[str] = []
    tags: list[str] = []
    seen_tags: set[str] = set()
    summary_candidates: list[str] = []

    fence: str | None = None
    in_frontmatter = False
    frontmatter_checked = False

    for idx, raw_line in enumerate(raw_lines):
        stripped = raw_line.lstrip()

        if idx == 0 and stripped.strip() == "---":
            in_frontmatter = True
            frontmatter_checked = True
            continue
        if in_frontmatter:
            if stripped.strip() == "---":
                in_frontmatter = False
            else:
                fm_tags = re.match(r"^tags:\s*\[([^\]]*)\]", stripped)
                if fm_tags:
                    for item in fm_tags.group(1).split(","):
                        tag = item.strip().strip("\"'#")
                        if tag and tag.lower() not in seen_tags:
                            seen_tags.add(tag.lower())
                            tags.append(tag)
            continue
        if not frontmatter_checked:
            frontmatter_checked = True

        fence_candidate = stripped
        while fence_candidate.startswith(">"):
            fence_candidate = fence_candidate[1:].lstrip()
        marker = fence_candidate[:3]
        if fence is None and marker in {"```", "~~~"}:
            fence = marker
            continue
        if fence is not None:
            if fence_candidate.startswith(fence):
                fence = None
            continue

        clean_line = strip_inline_code(raw_line).strip()
        if not clean_line:
            continue

        heading_match = HEADING_RE.match(clean_line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = clean_markup(heading_match.group(2).rstrip("#").strip())
            if level == 1:
                headings_h1.append(heading_text)
            elif level == 2:
                headings_h2.append(heading_text)
            else:
                headings_other.append(heading_text)
            continue

        if TAG_LINE_RE.match(clean_line):
            for match in TAG_TOKEN_RE.finditer(clean_line):
                tag = match.group(1).strip()
                if tag and tag.lower() not in seen_tags:
                    seen_tags.add(tag.lower())
                    tags.append(tag)
            continue

        if (
            clean_line.startswith("相关笔记")
            or clean_line.startswith("![")
            or clean_line.startswith("|")
            or clean_line in {"---", "***"}
            or clean_line.startswith("> ⬆ 返回")
            or clean_line.startswith("> 返回")
        ):
            continue

        prose_line = raw_line.strip()
        if prose_line.startswith(">"):
            prose_line = prose_line.lstrip(">").strip()
            if prose_line.startswith(("**状态**", "**TL;DR**", "维护提示", "[!question]")):
                prose_line = ""
        if prose_line and not prose_line.startswith(("- ", "* ", "+ ")) and not re.match(r"^\d+\.\s", prose_line):
            cleaned = clean_markup(prose_line)
            if cleaned:
                summary_candidates.append(cleaned)

    title = (
        headings_h1[0]
        if headings_h1
        else (headings_h2[0] if headings_h2 else path.stem)
    )

    if summary_candidates:
        summary = summary_candidates[0]
    elif headings_h2:
        summary = " / ".join(headings_h2[:3])
    else:
        summary = title

    if len(summary) > 100:
        summary = summary[:97].rstrip() + "..."

    domain, topic = infer_domain_and_topic(rel_path)
    return NoteRecord(
        path=path,
        rel_path=rel_path,
        stem=path.stem,
        domain=domain,
        topic=topic,
        title=title,
        headings_h1=tuple(headings_h1),
        headings_h2=tuple(headings_h2),
        headings_other=tuple(headings_other),
        tags=tuple(tags),
        summary=summary,
        raw_lines=raw_lines,
        body_text=text,
    )


def iter_markdown_paths(root: Path, exclude_generated: bool = True) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue
        if EXCLUDED_DIRS.intersection(path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        if exclude_generated and rel in GENERATED_DOCS:
            continue
        paths.append(path)
    return paths


def compute_vault_fingerprint(paths: list[Path]) -> str:
    max_mtime = 0
    total_size = 0
    for path in paths:
        st = path.stat()
        if st.st_mtime_ns > max_mtime:
            max_mtime = st.st_mtime_ns
        total_size += st.st_size
    return f"{SCHEMA_VERSION}:{len(paths)}:{max_mtime}:{total_size}"


def scan_notes(root: Path, exclude_generated: bool = True) -> list[NoteRecord]:
    root = root.resolve()
    return [parse_note(path, root) for path in iter_markdown_paths(root, exclude_generated=exclude_generated)]


def fts5_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
        conn.close()
        return True
    except sqlite3.Error:
        return False


def db_path_for(root: Path) -> Path:
    return root / ".traces" / "notes_fts.sqlite3"


def rebuild_fts_index(root: Path, notes: list[NoteRecord], fingerprint: str) -> Path:
    db_path = db_path_for(root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".sqlite3.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(tmp_path)
    try:
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY,
                rel_path TEXT UNIQUE NOT NULL,
                stem TEXT NOT NULL,
                domain TEXT NOT NULL,
                topic TEXT NOT NULL,
                title TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                h1_text TEXT NOT NULL,
                h2_text TEXT NOT NULL,
                other_h_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                raw_lines_json TEXT NOT NULL,
                body_text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE notes_fts USING fts5(
                rel_path,
                stem,
                tags,
                h1,
                h2,
                headings_other,
                body,
                tokenize='unicode61'
            );
            """
        )
        conn.execute("INSERT INTO meta (key, value) VALUES ('fingerprint', ?)", (fingerprint,))
        for note in notes:
            h1_text = "\n".join(note.headings_h1)
            h2_text = "\n".join(note.headings_h2)
            other_h_text = "\n".join(note.headings_other)
            tags_space = " ".join(note.tags)
            cur = conn.execute(
                """
                INSERT INTO notes (
                    rel_path, stem, domain, topic, title, tags_json,
                    h1_text, h2_text, other_h_text, summary, raw_lines_json, body_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.rel_path,
                    note.stem,
                    note.domain,
                    note.topic,
                    note.title,
                    json.dumps(list(note.tags), ensure_ascii=False),
                    h1_text,
                    h2_text,
                    other_h_text,
                    note.summary,
                    json.dumps(list(note.raw_lines), ensure_ascii=False),
                    note.body_text,
                ),
            )
            rowid = cur.lastrowid
            conn.execute(
                """
                INSERT INTO notes_fts (rowid, rel_path, stem, tags, h1, h2, headings_other, body)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rowid,
                    tokenize_for_fts(note.rel_path),
                    tokenize_for_fts(note.stem),
                    tokenize_for_fts(tags_space),
                    tokenize_for_fts(h1_text),
                    tokenize_for_fts(h2_text),
                    tokenize_for_fts(other_h_text),
                    tokenize_for_fts(note.body_text),
                ),
            )
        conn.commit()
    finally:
        conn.close()

    tmp_path.replace(db_path)
    return db_path


def ensure_fresh_fts_db(root: Path, force_rebuild: bool = False) -> tuple[sqlite3.Connection, bool]:
    paths = iter_markdown_paths(root, exclude_generated=True)
    fingerprint = compute_vault_fingerprint(paths)
    db_path = db_path_for(root)

    rebuilt = False
    need_rebuild = force_rebuild or not db_path.exists()
    if not need_rebuild:
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
            if not row or row[0] != fingerprint:
                conn.close()
                need_rebuild = True
            else:
                return conn, False
        except sqlite3.Error:
            need_rebuild = True

    if need_rebuild:
        notes = [parse_note(p, root) for p in paths]
        rebuild_fts_index(root, notes, fingerprint)
        rebuilt = True
    return sqlite3.connect(db_path), rebuilt


def build_fts_query(query: str) -> str:
    """Build a safe FTS5 query string supporting prefix matching on ASCII tokens."""
    toks = [t for t in tokenize_for_fts(query).split() if t]
    if not toks:
        return ""
    seen: list[str] = []
    for tok in toks:
        if tok not in seen:
            seen.append(tok)
    parts: list[str] = []
    for tok in seen:
        safe = tok.replace('"', '""')
        if tok.isascii() and len(tok) >= 3:
            parts.append(f'"{safe}"*')
        else:
            parts.append(f'"{safe}"')
    return " AND ".join(parts)


def extract_snippet(raw_lines: list[str], summary: str, terms: list[str], phrase: str) -> str:
    fallback_line = ""
    best_line = ""
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("```") or stripped == "---":
            continue
        lower = stripped.lower()
        phrase_hit = bool(phrase and phrase in lower)
        term_hit = any(t in lower for t in terms)
        if not (phrase_hit or term_hit):
            continue
        is_meta = TAG_LINE_RE.match(stripped) or stripped.startswith(("相关笔记", "> ⬆ 返回", "> 返回"))
        if not is_meta:
            if phrase_hit:
                best_line = stripped
                break
            if not best_line:
                best_line = stripped
        elif not fallback_line:
            fallback_line = stripped

    chosen = best_line or fallback_line
    if not chosen:
        return summary

    compact = re.sub(r"\s+", " ", chosen).strip()
    if len(compact) <= 120:
        return compact

    anchor = -1
    compact_lower = compact.lower()
    if phrase:
        anchor = compact_lower.find(phrase)
    if anchor == -1:
        for t in terms:
            pos = compact_lower.find(t)
            if pos != -1:
                anchor = pos
                break
    if anchor == -1:
        return compact[:117] + "..."
    start = max(0, anchor - 35)
    end = min(len(compact), start + 120)
    snippet = compact[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(compact):
        snippet = snippet + "..."
    return snippet


def score_candidate(
    rel_path: str,
    stem: str,
    tags: list[str],
    h1_text: str,
    h2_text: str,
    other_h_text: str,
    body_text: str,
    terms: list[str],
    phrase: str,
    bm25_score: float = 0.0,
) -> tuple[float, list[str]]:
    if not terms:
        base = 20.0 if Path(rel_path).name != "README.md" else 10.0
        return base, ["filter"]

    score = max(0.0, -bm25_score * 5.0)
    matched_fields: list[str] = []

    stem_lower = stem.lower()
    rel_lower = rel_path.lower()
    tags_lower = [t.lower() for t in tags]
    h1_lower = h1_text.lower()
    h2_lower = h2_text.lower()
    other_h_lower = other_h_text.lower()
    body_lower = body_text.lower()

    if phrase:
        if phrase == stem_lower:
            score += 140.0
            matched_fields.append("filename:exact")
        elif phrase in stem_lower:
            score += 80.0
            matched_fields.append("filename")
        elif phrase in rel_lower:
            score += 45.0
            matched_fields.append("path")

        if any(phrase == t for t in tags_lower):
            score += 75.0
            matched_fields.append("tag:exact")
        elif any(phrase in t for t in tags_lower):
            score += 35.0
            matched_fields.append("tag")

        if phrase in h1_lower:
            score += 65.0
            matched_fields.append("H1")
        if phrase in h2_lower:
            score += 40.0
            matched_fields.append("H2")
        if phrase in other_h_lower:
            score += 20.0
            matched_fields.append("heading")
        if phrase in body_lower:
            score += 25.0
            matched_fields.append("body")

    for term in terms:
        if (term in stem_lower or term in rel_lower) and "filename" not in matched_fields:
            matched_fields.append("filename")
        if any(term == t or term in t for t in tags_lower) and "tag" not in matched_fields:
            matched_fields.append("tag")
        if term in h1_lower and "H1" not in matched_fields:
            matched_fields.append("H1")
        if term in h2_lower and "H2" not in matched_fields:
            matched_fields.append("H2")
        if term in body_lower and "body" not in matched_fields:
            matched_fields.append("body")

    if not matched_fields:
        matched_fields.append("fts5")

    if Path(rel_path).name != "README.md":
        score += 2.0

    return round(score, 1), matched_fields


def run_search(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    query = " ".join(args.keyword).strip()
    domain_filter = args.domain.strip("/").lower() if args.domain else ""
    tag_filter = args.tag.lstrip("#").strip().lower() if args.tag else ""

    if not query and not domain_filter and not tag_filter:
        print("Error: provide a search <keyword>, --domain, --tag, or --rebuild-catalog.", file=sys.stderr)
        return 2

    if not fts5_available():
        print("Error: SQLite FTS5 is required.", file=sys.stderr)
        return 2

    conn, _ = ensure_fresh_fts_db(root, force_rebuild=args.rebuild)
    try:
        where_clauses: list[str] = []
        params: list[object] = []

        if domain_filter:
            where_clauses.append("(LOWER(n.rel_path) = ? OR LOWER(n.rel_path) LIKE ?)")
            params.extend([domain_filter, domain_filter + "/%"])

        terms = [t.lower() for t in query.split() if t.strip()]
        phrase = query.lower()
        fts_query = build_fts_query(query) if query else ""

        rows: list[tuple[object, ...]] = []
        if fts_query:
            sql = """
                SELECT n.rel_path, n.stem, n.title, n.tags_json, n.h1_text, n.h2_text,
                       n.other_h_text, n.summary, n.raw_lines_json, n.body_text,
                       bm25(notes_fts, 8.0, 12.0, 10.0, 9.0, 6.0, 3.0, 1.5) AS rank_score
                FROM notes_fts
                JOIN notes n ON n.id = notes_fts.rowid
                WHERE notes_fts MATCH ?
            """
            q_params: list[object] = [fts_query, *params]
            if where_clauses:
                sql += " AND " + " AND ".join(where_clauses)
            sql += " ORDER BY rank_score LIMIT 200"
            rows = conn.execute(sql, q_params).fetchall()

        # Fallback to exact substring scan inside SQLite if FTS query returned nothing (e.g. non-token punctuation).
        if not rows:
            sql = """
                SELECT n.rel_path, n.stem, n.title, n.tags_json, n.h1_text, n.h2_text,
                       n.other_h_text, n.summary, n.raw_lines_json, n.body_text, 0.0 AS rank_score
                FROM notes n
            """
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            rows = conn.execute(sql, params).fetchall()

        results: list[tuple[float, str, str, list[str], list[str], str]] = []
        for row in rows:
            rel_path, stem, title, tags_json, h1_text, h2_text, other_h_text, summary, raw_lines_json, body_text, rank_score = row
            tags: list[str] = json.loads(str(tags_json))
            if tag_filter and tag_filter not in {t.lower() for t in tags}:
                continue
            if query and float(rank_score) == 0.0:
                haystack = f"{rel_path}\n{h1_text}\n{h2_text}\n{' '.join(tags)}\n{body_text}".lower()
                if not all(t in haystack for t in terms):
                    continue
            score, fields = score_candidate(
                str(rel_path),
                str(stem),
                tags,
                str(h1_text),
                str(h2_text),
                str(other_h_text),
                str(body_text),
                terms,
                phrase,
                float(rank_score),
            )
            raw_lines: list[str] = json.loads(str(raw_lines_json))
            snippet = extract_snippet(raw_lines, str(summary), terms, phrase)
            results.append((score, str(rel_path), str(title), tags, fields, snippet))
    finally:
        conn.close()

    results.sort(key=lambda item: (-item[0], 0 if Path(item[1]).name != "README.md" else 1, item[1]))
    shown = results[: max(1, args.limit)]

    if args.json:
        payload = [
            {
                "score": score,
                "path": rel_path,
                "title": title,
                "tags": tags,
                "matched": fields,
                "snippet": snippet,
            }
            for score, rel_path, title, tags, fields, snippet in shown
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Found {len(results)} matching note(s) (showing top {len(shown)}):")
    for idx, (score, rel_path, title, tags, fields, snippet) in enumerate(shown, start=1):
        tag_str = " ".join(f"#{t}" for t in tags) if tags else "(no tags)"
        field_str = ", ".join(fields)
        print(f"{idx}. [{score}] {rel_path} — {title}")
        print(f"   tags: {tag_str} | matched: {field_str}")
        print(f"   snippet: {snippet}")
    return 0


def format_catalog_row(note: NoteRecord, catalog_dir: Path, stem_counts: Counter[str]) -> str:
    rel_link = Path(os.path.relpath(note.path, catalog_dir)).as_posix()
    title = note.title.rstrip("?？").strip() or note.stem
    summary = note.summary.rstrip("?？").strip() or title
    if note.path.name not in {"README.md", *IGNORED_ROOT_DOCS} and stem_counts[note.stem] == 1:
        wikilink = f"[[{note.stem}]]"
    else:
        wikilink = "—"
    tags_col = " ".join(f"`#{t}`" for t in note.tags) if note.tags else "—"
    return f"| [{title}]({rel_link}) | {wikilink} | {tags_col} | {summary} |"


def rebuild_catalog_and_tags(root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    paths = iter_markdown_paths(root, exclude_generated=True)
    fingerprint = compute_vault_fingerprint(paths)
    notes = [parse_note(p, root) for p in paths]
    db_path = rebuild_fts_index(root, notes, fingerprint)

    catalog_path = root / "docs" / "catalog.md"
    tags_path = root / "docs" / "tags.md"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    stem_counts = Counter(
        n.stem for n in notes if n.path.name not in {"README.md", *IGNORED_ROOT_DOCS}
    )

    root_entries = [n for n in notes if not n.domain]
    domain_notes: dict[str, dict[str, list[NoteRecord]]] = defaultdict(lambda: defaultdict(list))
    for note in notes:
        if note.domain:
            domain_notes[note.domain][note.topic].append(note)

    ordered_domains = [d for d in DOMAIN_ORDER if d in domain_notes]
    ordered_domains.extend(sorted(d for d in domain_notes if d not in DOMAIN_ORDER))

    catalog_lines: list[str] = [
        "# 全库知识速查索引",
        "",
        "> ⬆ 返回 [知识库首页](../README.md) · 架构说明见 [仓库架构](architecture.md) · 按标签查看 [全库标签聚合索引](tags.md)",
        ">",
        "> 本文件由 `python3 scripts/search_notes.py --rebuild-catalog` 自动扫描生成并同步重建 `.traces/notes_fts.sqlite3` FTS5 索引（不手工维护脆弱的精确篇数统计）。",
        "",
    ]

    if root_entries:
        catalog_lines.extend(
            [
                "## 仓库入口与规范",
                "",
                "| 文档 | Wikilink | 标签 | 摘要 |",
                "| --- | --- | --- | --- |",
            ]
        )
        root_sorted = sorted(
            root_entries,
            key=lambda n: (0 if n.path.name == "README.md" else 1, n.rel_path),
        )
        for note in root_sorted:
            catalog_lines.append(format_catalog_row(note, catalog_path.parent, stem_counts))
        catalog_lines.append("")

    for domain in ordered_domains:
        domain_title = DOMAIN_TITLES.get(domain, domain)
        catalog_lines.append(f"## {domain_title}")
        catalog_lines.append("")

        topics = sorted(
            domain_notes[domain].keys(),
            key=lambda t: (0 if t == domain else 1, t),
        )
        for topic in topics:
            catalog_lines.append(f"### {topic}")
            catalog_lines.append("")
            catalog_lines.append("| 文档 | Wikilink | 标签 | 摘要 |")
            catalog_lines.append("| --- | --- | --- | --- |")
            entries = sorted(
                domain_notes[domain][topic],
                key=lambda n: (0 if n.path.name == "README.md" else 1, n.rel_path),
            )
            for note in entries:
                catalog_lines.append(format_catalog_row(note, catalog_path.parent, stem_counts))
            catalog_lines.append("")

    catalog_path.write_text("\n".join(catalog_lines).rstrip() + "\n", encoding="utf-8")

    # Build docs/tags.md
    tag_buckets: dict[str, list[NoteRecord]] = defaultdict(list)
    tag_display: dict[str, str] = {}
    for note in notes:
        for tag in note.tags:
            key = tag.lower().rstrip("?？")
            if not key:
                continue
            tag_display.setdefault(key, tag.rstrip("?？"))
            tag_buckets[key].append(note)

    sorted_tags = sorted(tag_buckets.keys(), key=lambda k: (-len(tag_buckets[k]), k))
    tag_lines: list[str] = [
        "# 全库标签聚合索引",
        "",
        "> ⬆ 返回 [知识库首页](../README.md) · 按目录查看 [全库知识速查索引](catalog.md)",
        ">",
        "> 本文件由 `python3 scripts/search_notes.py --rebuild-catalog` 自动聚合全库 `#tag` 生成。",
        "",
    ]
    for key in sorted_tags:
        display = tag_display[key]
        tag_lines.append(f"## `#{display}`")
        tag_lines.append("")
        tag_lines.append("| 所属领域 | 文档 | Wikilink |")
        tag_lines.append("| --- | --- | --- |")
        entries = sorted(tag_buckets[key], key=lambda n: n.rel_path)
        for note in entries:
            rel_link = Path(os.path.relpath(note.path, tags_path.parent)).as_posix()
            title = note.title.rstrip("?？").strip() or note.stem
            wikilink = f"[[{note.stem}]]" if stem_counts[note.stem] == 1 else "—"
            tag_lines.append(f"| `{note.topic or note.domain}` | [{title}]({rel_link}) | {wikilink} |")
        tag_lines.append("")

    tags_path.write_text("\n".join(tag_lines).rstrip() + "\n", encoding="utf-8")
    return catalog_path, tags_path, db_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Learning Notes via persistent SQLite FTS5 or rebuild docs/catalog.md & docs/tags.md."
    )
    parser.add_argument("keyword", nargs="*", help="Keyword or multi-word query to search.")
    parser.add_argument("--domain", help="Filter by top-level domain or subpath prefix (e.g. ai, foundations, cloud-native/kubernetes).")
    parser.add_argument("--tag", help="Filter by #tag (with or without leading #).")
    parser.add_argument("--limit", type=int, default=15, help="Maximum number of search results (default: 15).")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of .traces/notes_fts.sqlite3 before searching.")
    parser.add_argument("--rebuild-catalog", action="store_true", help="Rebuild FTS5 index, docs/catalog.md, and docs/tags.md.")
    parser.add_argument("--json", action="store_true", help="Emit search results as JSON.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Vault root directory (defaults to repository root).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rebuild_catalog:
        catalog_path, tags_path, db_path = rebuild_catalog_and_tags(args.root)
        print(f"Rebuilt catalog: {catalog_path}")
        print(f"Rebuilt tags:    {tags_path}")
        print(f"Rebuilt FTS5 DB: {db_path}")
        return 0
    return run_search(args)


if __name__ == "__main__":
    sys.exit(main())
