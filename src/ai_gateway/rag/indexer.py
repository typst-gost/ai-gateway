from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import lancedb
import yaml
from lancedb.index import FTS
from loguru import logger
from openai import BadRequestError, OpenAI
from slugify import slugify

from ai_gateway.config import Settings
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.rag.retriever import DocsRetriever

HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
EXPLICIT_ID_RE = re.compile(r"\s*(?:\{#([^}]+)\}|\[#([^\]]+)\])\s*$")
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")

CALLOUT_LABELS = {
    "info": "Примечание",
    "tip": "Совет",
    "warning": "Важно",
    "error": "Предупреждение",
    "success": "Готово",
}


def _text_hash(embedding_model: str, text: str) -> str:
    return hashlib.sha256(f"{embedding_model}\0{text}".encode("utf-8")).hexdigest()


def _classify_doc(rel_path: str) -> str:
    norm = rel_path.replace("\\", "/")
    if norm.startswith("typst-gost/"):
        return "typst-gost"
    if norm.startswith("typst/changelog/"):
        return "typst-changelog"
    if norm.startswith("typst/symbols/"):
        return "typst-symbols"
    if norm.startswith("typst/guides/"):
        return "typst-guides"
    if norm.startswith("typst/tutorial/"):
        return "typst-tutorial"
    return "typst-reference"


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", raw, re.S)
    if not match:
        return {}, raw
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        metadata = {}
    return (metadata if isinstance(metadata, dict) else {}), raw[match.end() :]


def _update_fence(line: str, opened: tuple[str, int] | None) -> tuple[str, int] | None:
    match = FENCE_RE.match(line)
    if not match:
        return opened
    marks, rest = match.groups()
    if opened is None:
        return (marks[0], len(marks))
    if marks[0] == opened[0] and len(marks) >= opened[1] and not rest.strip():
        return None
    return opened


def _skip_quoted(src: str, idx: int, quote: str) -> tuple[str, int]:
    out: list[str] = []
    i = idx + 1
    n = len(src)
    while i < n:
        ch = src[i]
        if ch == "\\" and i + 1 < n:
            nxt = src[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt in (quote, "\\", "$"):
                out.append(nxt)
            else:
                out.append("\\" + nxt)
            i += 2
            continue
        if ch == quote:
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    return "".join(out), n


def _extract_balanced_braces(src: str, idx: int) -> tuple[str, int]:
    depth = 0
    i = idx
    n = len(src)
    while i < n:
        ch = src[i]
        if ch in ('"', "'", "`"):
            _, i = _skip_quoted(src, i, ch)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[idx : i + 1], i + 1
        i += 1
    return src[idx:], n


def _find_self_closing_tag_end(src: str, idx: int) -> int:
    i = idx
    n = len(src)
    while i < n:
        ch = src[i]
        if ch in ('"', "'", "`"):
            _, i = _skip_quoted(src, i, ch)
            continue
        if ch == "{":
            _, i = _extract_balanced_braces(src, i)
            continue
        if src.startswith("/>", i):
            return i + 2
        if ch == ">":
            return i + 1
        i += 1
    return n


def _parse_js_object_fields(obj_body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(obj_body)
    while i < n:
        while i < n and obj_body[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        if obj_body.startswith("//", i):
            nl = obj_body.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        if obj_body[i] in ('"', "'"):
            key, i = _skip_quoted(obj_body, i, obj_body[i])
        else:
            match = re.match(r"[A-Za-z_][\w-]*", obj_body[i:])
            if not match:
                i += 1
                continue
            key = match.group(0)
            i += len(key)
        while i < n and obj_body[i] in " \t\r\n":
            i += 1
        if i >= n or obj_body[i] != ":":
            continue
        i += 1
        while i < n and obj_body[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        if obj_body[i] in ('"', "'", "`"):
            val, i = _skip_quoted(obj_body, i, obj_body[i])
            fields[key] = re.sub(r"\s+", " ", val).strip()
        elif obj_body[i] == "{":
            raw_obj, i = _extract_balanced_braces(obj_body, i)
            fields[key] = re.sub(r"\s+", " ", raw_obj).strip()
        else:
            match = re.match(r"[^,\n}]+", obj_body[i:])
            if match:
                fields[key] = match.group(0).strip()
                i += len(match.group(0))
            else:
                i += 1
    return fields


def _format_type_table(type_attr_raw: str) -> str:
    inner = type_attr_raw.strip()
    if inner.startswith("{{") and inner.endswith("}}"):
        inner = inner[2:-2]
    elif inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]

    lines: list[str] = []
    i = 0
    n = len(inner)
    while i < n:
        while i < n and inner[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        if inner.startswith("//", i):
            nl = inner.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        if inner[i] in ('"', "'"):
            param_name, i = _skip_quoted(inner, i, inner[i])
        else:
            match = re.match(r"[A-Za-z_][\w-]*", inner[i:])
            if not match:
                i += 1
                continue
            param_name = match.group(0)
            i += len(param_name)

        while i < n and inner[i] in " \t\r\n":
            i += 1
        if i >= n or inner[i] != ":":
            continue
        i += 1
        while i < n and inner[i] in " \t\r\n":
            i += 1
        if i >= n or inner[i] != "{":
            continue

        raw_param_obj, i = _extract_balanced_braces(inner, i)
        fields = _parse_js_object_fields(raw_param_obj[1:-1])

        meta: list[str] = []
        if fields.get("type"):
            meta.append(f"тип: `{fields['type']}`")
        if fields.get("required", "").lower() == "true":
            meta.append("обязательный")
        if fields.get("default"):
            meta.append(f"по умолчанию: `{fields['default']}`")

        meta_str = f" ({', '.join(meta)})" if meta else ""
        desc = fields.get("description", "").strip()
        if desc:
            lines.append(f"- **`{param_name}`**{meta_str}: {desc}")
        else:
            lines.append(f"- **`{param_name}`**{meta_str}")

    if not lines:
        return ""
    return "\n\nПараметры:\n" + "\n".join(lines) + "\n\n"


def _extract_jsx_attr_template(attrs: str, attr_name: str) -> str | None:
    match = re.search(rf"\b{attr_name}\s*=\s*\{{", attrs)
    if not match:
        return None
    raw_brace, _ = _extract_balanced_braces(attrs, match.end() - 1)
    inner = raw_brace[1:-1].strip()
    if inner.startswith("`") and inner.endswith("`") and len(inner) >= 2:
        val, _ = _skip_quoted(inner, 0, "`")
        return val
    if (inner.startswith('"') and inner.endswith('"')) or (
        inner.startswith("'") and inner.endswith("'")
    ):
        val, _ = _skip_quoted(inner, 0, inner[0])
        return val
    return None


def _replace_typst_renders_and_type_tables(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        if i == 0 or text[i - 1] == "\n":
            line_end = text.find("\n", i)
            line = text[i:] if line_end == -1 else text[i:line_end]
            fm = FENCE_RE.match(line)
            if fm:
                fence_char, fence_len = fm.group(1)[0], len(fm.group(1))
                close_re = re.compile(
                    rf"(?m)^ {{0,3}}{re.escape(fence_char)}{{{fence_len},}}[ \t]*$"
                )
                search_from = len(text) if line_end == -1 else line_end + 1
                cm = close_re.search(text, search_from)
                end_pos = cm.end() if cm else n
                out.append(text[i:end_pos])
                i = end_pos
                continue

        if text.startswith("<TypstRender", i) and i + 12 < n and text[i + 12] in " \t\r\n/>":
            end_pos = _find_self_closing_tag_end(text, i)
            tag_src = text[i:end_pos]
            code = _extract_jsx_attr_template(tag_src, "code")
            hidden_prefix = _extract_jsx_attr_template(tag_src, "hiddenPrefix")
            if code is not None:
                cleaned = textwrap.dedent(code).strip("\n")
                if hidden_prefix and "#import" in hidden_prefix and "#import" not in cleaned:
                    imports = [
                        ln.strip()
                        for ln in hidden_prefix.splitlines()
                        if ln.strip().startswith("#import")
                    ]
                    if imports:
                        cleaned = "\n".join(imports) + "\n\n" + cleaned
                out.append(f"\n\n```typst\n{cleaned}\n```\n\n")
            i = end_pos
            continue

        if text.startswith("<TypeTable", i) and i + 10 < n and text[i + 10] in " \t\r\n/>":
            end_pos = _find_self_closing_tag_end(text, i)
            tag_src = text[i:end_pos]
            match = re.search(r"\btype\s*=\s*\{", tag_src)
            if match:
                raw_type, _ = _extract_balanced_braces(tag_src, match.end() - 1)
                out.append(_format_type_table(raw_type[1:-1]))
            i = end_pos
            continue

        out.append(text[i])
        i += 1

    return "".join(out)


def _format_gost_quote(match: re.Match[str]) -> str:
    attrs = match.group(1) or ""
    body = match.group(2).strip()
    page_m = re.search(r"\bpage\s*=\s*\{?(\d+)\}?", attrs)
    id_m = re.search(r'\bid\s*=\s*["\']([^"\']+)["\']', attrs)
    meta_parts = ["ГОСТ 7.32-2017"]
    if id_m:
        meta_parts.append(f"п. {id_m.group(1)}")
    if page_m:
        meta_parts.append(f"стр. {page_m.group(1)}")
    quoted_lines = [f"> {line}" if line.strip() else ">" for line in body.splitlines()]
    return f"\n\n> **[{', '.join(meta_parts)}]**\n" + "\n".join(quoted_lines) + "\n\n"


def _format_callout(match: re.Match[str]) -> str:
    attrs = match.group(1) or ""
    body = textwrap.dedent(match.group(2)).strip()
    type_m = re.search(r'\btype\s*=\s*["\']([^"\']+)["\']', attrs)
    label = CALLOUT_LABELS.get((type_m.group(1) if type_m else "").lower(), "Примечание")
    lines = [f"> {ln}" if ln.strip() else ">" for ln in body.splitlines()]
    return f"\n\n> **{label}:**\n" + "\n".join(lines) + "\n\n"


def normalize_mdx(body: str) -> str:
    body = _replace_typst_renders_and_type_tables(body)
    body = re.sub(r"<GostQuote\b([^>]*)>([\s\S]*?)</GostQuote>", _format_gost_quote, body)
    body = re.sub(r"<Callout\b([^>]*)>([\s\S]*?)</Callout>", _format_callout, body)
    body = re.sub(r"<summary[^>]*>([\s\S]*?)</summary>", r"\n**\1**\n", body)

    result: list[str] = []
    fence: tuple[str, int] | None = None
    for line in body.splitlines():
        if fence is None:
            if re.match(r"^\s*import\s+.+?\s+from\s+['\"][^'\"]+['\"]\s*;?\s*$", line):
                continue
            line = re.sub(r"</?[A-Z][\w.]*\b[^>]*>", "", line)
            line = re.sub(
                r"</?(?:details|span|img|br|sup|sub|div)\b[^>]*>",
                "",
                line,
                flags=re.IGNORECASE,
            )
        result.append(line)
        fence = _update_fence(line, fence)

    cleaned = "\n".join(result).replace("\x00", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _get_sections(body: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    parents: list[tuple[int, str]] = []
    current: dict[str, Any] = {
        "path": ["Введение"],
        "heading": "Введение",
        "anchor": "",
        "lines": [],
    }
    fence: tuple[str, int] | None = None
    slug_counts: dict[str, int] = {}

    for line in body.splitlines():
        match = HEADING_RE.match(line) if fence is None else None
        if match:
            if any(ln.strip() for ln in current["lines"]):
                sections.append(current)
            level = len(match.group(1))
            raw_title = re.sub(r"\s+#+\s*$", "", match.group(2)).strip()
            explicit_match = EXPLICIT_ID_RE.search(raw_title)
            title = EXPLICIT_ID_RE.sub("", raw_title).strip()
            while parents and parents[-1][0] >= level:
                parents.pop()
            parents.append((level, title))
            auto_slug = slugify(title, allow_unicode=True)
            count = slug_counts.get(auto_slug, 0)
            slug_counts[auto_slug] = count + 1
            if explicit_match:
                anchor = explicit_match.group(1) or explicit_match.group(2)
            else:
                anchor = f"{auto_slug}-{count}" if count else auto_slug
            current = {
                "path": [name for _, name in parents],
                "heading": title,
                "anchor": anchor,
                "lines": [],
            }
        else:
            current["lines"].append(line)
            fence = _update_fence(line, fence)

    if any(ln.strip() for ln in current["lines"]):
        sections.append(current)
    return sections


def _get_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines: list[str] = []
    fence: tuple[str, int] | None = None
    for line in text.splitlines():
        if not line.strip() and fence is None:
            if lines:
                blocks.append("\n".join(lines).strip())
                lines = []
        else:
            lines.append(line)
            fence = _update_fence(line, fence)
    if lines:
        blocks.append("\n".join(lines).strip())
    return [b for b in blocks if b]


def _join_examples(blocks: list[str], budget: int) -> list[str]:
    result: list[str] = []
    for block in blocks:
        if (
            result
            and block.startswith(("```", "~~~"))
            and not result[-1].startswith(("```", "~~~"))
            and len(result[-1]) <= 600
            and len(result[-1]) + len(block) + 2 <= budget
        ):
            result[-1] += "\n\n" + block
        else:
            result.append(block)
    return result


def _split_plain(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9#`])", text)
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= limit:
            pieces.append(sentence)
        else:
            pieces.extend(
                textwrap.wrap(
                    sentence,
                    width=limit,
                    break_long_words=False,
                    break_on_hyphens=False,
                    replace_whitespace=False,
                )
            )
    result: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > limit:
            result.append(current)
            current = ""
        current = (current + " " + piece).strip()
    if current:
        result.append(current)
    return result


def _split_long_block(block: str, limit: int) -> list[str]:
    if len(block) <= limit:
        return [block]
    match = re.search(r"(?m)^(`{3,}|~{3,})[^\n]*\n", block)
    if not match:
        return _split_plain(block, limit)
    prefix = block[: match.start()].strip()
    opener, closing = match.group(0).rstrip("\n"), match.group(1)
    rest = block[match.end() :]
    end = re.search(rf"(?m)^\s*{re.escape(closing)}\s*$", rest)
    if not end:
        return _split_plain(block, limit)
    code, suffix = rest[: end.start()].strip("\n"), rest[end.end() :].strip()
    capacity = limit - len(opener) - len(closing) - 3
    if capacity < 100:
        return [block]

    groups: list[str] = []
    group: list[str] = []
    size = 0
    for line in code.splitlines():
        if group and size + len(line) + 1 > capacity:
            groups.append("\n".join(group))
            group, size = [], 0
        group.append(line)
        size += len(line) + 1
    if group:
        groups.append("\n".join(group))

    result = _split_plain(prefix, limit) if prefix else []
    result.extend(f"{opener}\n{g}\n{closing}" for g in groups)
    result.extend(_split_plain(suffix, limit) if suffix else [])
    return result


def _pack_blocks(blocks: list[str], limit: int, overlap: int) -> list[str]:
    pieces = [
        p for b in _join_examples(blocks, limit) for p in _split_long_block(b, limit)
    ]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        addition = len(piece) + (2 if current else 0)
        if current and size + addition > limit:
            chunks.append("\n\n".join(current))
            carry = current[-1] if len(current[-1]) <= overlap else None
            if carry and len(carry) + len(piece) + 2 <= limit:
                current, size = [carry], len(carry)
            else:
                current, size = [], 0
        size += len(piece) + (2 if current else 0)
        current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(
    file_path: Path,
    rel_path: Path,
    *,
    docs_base_url: str,
    max_chars: int = 2500,
    overlap: int = 250,
) -> list[dict[str, Any]]:
    raw = file_path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(raw)
    doc_title = str(fm.get("title") or file_path.stem)
    description = re.sub(r"\s+", " ", str(fm.get("description") or "")).strip()

    rel = str(rel_path).replace("\\", "/")
    category = _classify_doc(rel)
    route = re.sub(r"\.(md|mdx)$", "", rel, flags=re.I)
    if route.endswith("/index"):
        route = route[:-6]
    elif route == "index":
        route = ""

    site_docs_root = f"{docs_base_url.rstrip('/')}/docs"
    doc_url = f"{site_docs_root}/{route}".rstrip("/")

    chunks: list[dict[str, Any]] = []
    chunk_order = 0
    for section in _get_sections(normalize_mdx(body)):
        content = "\n".join(section["lines"]).strip()
        if not content:
            continue
        path = section["path"]
        header = f"Документ: {doc_title} [{category}] > " + " > ".join(path)
        if description:
            header += f"\nОписание страницы: {description[:240]}"
        budget = max_chars - len(header) - 1
        if budget < 200:
            raise ValueError(f"Header too long: {rel}")
        parts = _pack_blocks(_get_blocks(content), budget, overlap)
        url = doc_url + ("#" + section["anchor"] if section["anchor"] else "")
        for idx, part in enumerate(parts):
            full_text = header + "\n" + part
            chunk_id = hashlib.sha256(
                f"{rel}\0{header}\0{idx}\0{part}".encode("utf-8")
            ).hexdigest()[:20]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "title": doc_title,
                    "section": section["heading"],
                    "heading_path": " > ".join(path),
                    "description": description,
                    "category": category,
                    "doc_url": doc_url,
                    "chunk_order": chunk_order,
                    "part_index": idx + 1,
                    "parts_total": len(parts),
                    "url": url,
                    "text": full_text,
                    "source": rel,
                }
            )
            chunk_order += 1
    return chunks


def _extract_subdir(tar: tarfile.TarFile, subdir: str, dest_dir: Path) -> int:
    normalized = subdir.strip("/")
    prefix = f"{normalized}/"
    count = 0

    for member in tar.getmembers():
        parts = member.name.split("/", 1)
        if len(parts) < 2:
            continue
        rel_in_repo = parts[1]
        if not rel_in_repo.startswith(prefix):
            continue
        inner_rel = rel_in_repo[len(prefix) :]
        if not inner_rel or member.issym() or member.islnk():
            continue

        dest = (dest_dir / inner_rel).resolve()
        if not dest.is_relative_to(dest_dir.resolve()):
            raise ValueError(f"Unsafe archive entry: {member.name}")

        if member.isdir():
            dest.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            dest.parent.mkdir(parents=True, exist_ok=True)
            src_file = tar.extractfile(member)
            if src_file is not None:
                with src_file, dest.open("wb") as dst_file:
                    shutil.copyfileobj(src_file, dst_file)
                count += 1
    return count


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    try:
        for root, dirs, files in os.walk(str(path), topdown=False):
            for f in files:
                p = os.path.join(root, f)
                try:
                    os.chmod(p, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
                    os.unlink(p)
                except Exception:
                    pass
            for d in dirs:
                p = os.path.join(root, d)
                try:
                    os.chmod(p, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
                    os.rmdir(p)
                except Exception:
                    pass
        os.chmod(str(path), stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        shutil.rmtree(str(path), ignore_errors=True)
    except Exception as exc:
        logger.warning(f"Failed to cleanly remove directory {path}: {exc}")


def _atomic_swap(staging_dir: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    for old_backup in target_dir.parent.glob(f".{target_dir.name}.backup*"):
        _safe_rmtree(old_backup)

    backup_dir = target_dir.parent / f".{target_dir.name}.backup.{int(time.time())}"

    try:
        if target_dir.exists():
            target_dir.rename(backup_dir)
        staging_dir.rename(target_dir)
    except Exception:
        if backup_dir.exists() and not target_dir.exists():
            backup_dir.rename(target_dir)
        raise
    finally:
        if backup_dir.exists():
            _safe_rmtree(backup_dir)


@dataclass(frozen=True, slots=True)
class ReindexResult:
    files_processed: int
    chunks_count: int
    embeddings_fetched: int
    assets_synced: int


class DocsIndexer:
    def __init__(
        self,
        settings: Settings,
        db: GatewayDatabase,
        retriever: DocsRetriever,
    ) -> None:
        self._settings = settings
        self._db = db
        self._retriever = retriever
        self._lock = asyncio.Lock()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def _load_cache(self, cache_path: Path) -> dict[str, list[float]]:
        cache: dict[str, list[float]] = {}
        if not cache_path.exists():
            return cache
        with cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    cache[item["hash"]] = item["vector"]
                except (ValueError, KeyError, TypeError):
                    continue
        return cache

    def _append_cache(self, cache_path: Path, hash_value: str, vector: list[float]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"hash": hash_value, "vector": vector}) + "\n")

    def _fetch_embeddings_sync(
        self,
        client: OpenAI,
        model: str,
        texts: list[str],
    ) -> list[list[float] | None]:
        try:
            response = client.embeddings.create(model=model, input=texts)
            vectors: list[list[float] | None] = [None] * len(texts)
            for item in response.data:
                vectors[item.index] = item.embedding
            return vectors
        except BadRequestError as exc:
            logger.warning(f"Batch embedding error ({exc}), retrying item by item")
            vectors = []
            for text in texts:
                try:
                    res = client.embeddings.create(model=model, input=[text])
                    vectors.append(res.data[0].embedding)
                except BadRequestError as item_exc:
                    logger.error(f"Embedding failed for chunk ({item_exc}): {text[:120]!r}")
                    vectors.append(None)
            return vectors

    def _sync_github_archive_sync(
        self,
        *,
        repo_slug: str,
        branch: str,
        docs_subdir: str,
        assets_subdir: str,
    ) -> int:
        archive_url = f"https://codeload.github.com/{repo_slug}/tar.gz/refs/heads/{branch}"
        logger.info(f"Downloading documentation archive: {archive_url}")

        with tempfile.TemporaryDirectory(prefix="ai_gw_sync_") as tmp_root:
            tmp_path = Path(tmp_root)
            archive_path = tmp_path / "repo.tar.gz"
            with httpx.stream(
                "GET",
                archive_url,
                timeout=120.0,
                follow_redirects=True,
                headers={"User-Agent": "ai-gateway-indexer/1.0"},
            ) as resp:
                resp.raise_for_status()
                with archive_path.open("wb") as out_file:
                    for chunk in resp.iter_bytes():
                        out_file.write(chunk)

            assets_count = 0
            with tarfile.open(archive_path, "r:gz") as tar:
                docs_target = self._settings.docs_cache_dir
                docs_target.parent.mkdir(parents=True, exist_ok=True)
                staging_docs = Path(
                    tempfile.mkdtemp(dir=docs_target.parent, prefix=f".{docs_target.name}.")
                )
                try:
                    docs_files = _extract_subdir(tar, docs_subdir, staging_docs)
                    if docs_files == 0:
                        raise RuntimeError(f"No documentation files found in '{docs_subdir}'")
                    _atomic_swap(staging_docs, docs_target)
                    logger.info(f"Synced {docs_files} docs files to {docs_target}")
                except Exception:
                    _safe_rmtree(staging_docs)
                    raise

                if assets_subdir:
                    assets_target = self._settings.assets_dir
                    assets_target.parent.mkdir(parents=True, exist_ok=True)
                    staging_assets = Path(
                        tempfile.mkdtemp(
                            dir=assets_target.parent,
                            prefix=f".{assets_target.name}.",
                        )
                    )
                    try:
                        assets_count = _extract_subdir(tar, assets_subdir, staging_assets)
                        _atomic_swap(staging_assets, assets_target)
                        logger.info(f"Synced {assets_count} compile assets to {assets_target}")
                    except Exception:
                        _safe_rmtree(staging_assets)
                        raise

            return assets_count

    def _run_reindex_sync(
        self,
        gw_settings: dict[str, Any],
        *,
        sync_github: bool,
        force_reembed: bool,
    ) -> ReindexResult:
        assets_synced = 0
        if sync_github:
            assets_synced = self._sync_github_archive_sync(
                repo_slug=str(gw_settings["docs_repo_slug"]),
                branch=str(gw_settings["docs_repo_branch"]),
                docs_subdir=str(gw_settings["docs_repo_docs_subdir"]),
                assets_subdir=str(gw_settings["docs_repo_assets_subdir"]),
            )

        docs_dir = self._settings.docs_cache_dir
        if not docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        docs_base_url = str(gw_settings["docs_base_url"])
        max_chars = int(gw_settings["chunk_max_chars"])
        overlap = int(gw_settings["chunk_overlap"])
        batch_size = int(gw_settings["index_batch_size"])
        embedding_model = str(gw_settings["embedding_model"])

        doc_files = sorted(
            path
            for path in docs_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in (".md", ".mdx")
        )
        all_chunks: list[dict[str, Any]] = []
        for file_path in doc_files:
            all_chunks.extend(
                chunk_document(
                    file_path,
                    file_path.relative_to(docs_dir),
                    docs_base_url=docs_base_url,
                    max_chars=max_chars,
                    overlap=overlap,
                )
            )

        if not all_chunks:
            raise RuntimeError("No chunks produced from documentation files")

        cache_path = self._settings.embeddings_cache_path
        cache = {} if force_reembed else self._load_cache(cache_path)
        missing = {
            _text_hash(embedding_model, c["text"]): c["text"]
            for c in all_chunks
            if _text_hash(embedding_model, c["text"]) not in cache
        }

        fetched_count = 0
        if missing:
            client = OpenAI(
                base_url=self._settings.polza_base_url,
                api_key=self._settings.polza_api_key,
            )
            items = list(missing.items())
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                embeddings = self._fetch_embeddings_sync(
                    client,
                    embedding_model,
                    [text for _, text in batch],
                )
                for (h, _), vector in zip(batch, embeddings):
                    if vector is not None:
                        cache[h] = vector
                        self._append_cache(cache_path, h, vector)
                        fetched_count += 1

        rows = []
        for chunk in all_chunks:
            vector = cache.get(_text_hash(embedding_model, chunk["text"]))
            if vector is not None:
                rows.append({**chunk, "vector": vector})

        if not rows:
            raise RuntimeError("No embeddings available to build LanceDB table")

        db_target = self._settings.db_path
        db_target.parent.mkdir(parents=True, exist_ok=True)
        staging_db = Path(
            tempfile.mkdtemp(dir=db_target.parent, prefix=f".{db_target.name}.")
        )
        try:
            db = lancedb.connect(str(staging_db))
            table = db.create_table("docs", data=rows, mode="overwrite")
            table.create_index(
                "text",
                config=FTS(language="Russian", with_position=False),
                replace=True,
            )

            self._retriever.unload()
            gc.collect()

            _atomic_swap(staging_db, db_target)
        except Exception:
            _safe_rmtree(staging_db)
            raise

        self._retriever.reload()
        logger.info(
            f"Reindex completed: files={len(doc_files)} chunks={len(rows)} "
            f"fetched_embeddings={fetched_count} assets={assets_synced}"
        )
        return ReindexResult(
            files_processed=len(doc_files),
            chunks_count=len(rows),
            embeddings_fetched=fetched_count,
            assets_synced=assets_synced,
        )

    async def reindex(
        self,
        *,
        trigger: str = "manual",
        sync_github: bool = True,
        force_reembed: bool = False,
    ) -> tuple[int, ReindexResult]:
        if self._lock.locked():
            raise RuntimeError("Reindex job is already running")

        async with self._lock:
            self._is_running = True
            job_id = await self._db.create_reindex_job(trigger)
            try:
                gw_settings = await self._db.get_gateway_settings()
                result = await asyncio.to_thread(
                    self._run_reindex_sync,
                    gw_settings,
                    sync_github=sync_github,
                    force_reembed=force_reembed,
                )
                await self._db.finish_reindex_job(
                    job_id,
                    status="completed",
                    files_processed=result.files_processed,
                    chunks_count=result.chunks_count,
                    embeddings_fetched=result.embeddings_fetched,
                    assets_synced=result.assets_synced,
                )
                return job_id, result
            except Exception as exc:
                logger.exception(f"Reindex job {job_id} failed")
                await self._db.finish_reindex_job(
                    job_id,
                    status="failed",
                    error=str(exc),
                )
                raise
            finally:
                self._is_running = False