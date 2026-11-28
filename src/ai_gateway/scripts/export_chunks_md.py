from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from ai_gateway.rag.indexer import _extract_subdir, chunk_document

EVAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Схема эталонного датасета пользовательских запросов для оценки RAG-метрик (Recall@k и MRR)",
    "properties": {
        "questions": {
            "type": "array",
            "description": "Список эталонных пользовательских запросов с привязкой к chunk_id",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Уникальный идентификатор вопроса, например 'q_001'",
                    },
                    "query": {
                        "type": "string",
                        "description": "Текст естественного, самодостаточного вопроса пользователя на русском языке",
                    },
                    "relevant_chunk_ids": {
                        "type": "array",
                        "description": "Список от 1 до 4 точных 20-символьных chunk_id в порядке убывания релевантности",
                        "items": {
                            "type": "string",
                            "description": "20-символьный chunk_id из заголовка чанка",
                        },
                    },
                    "category": {
                        "type": "string",
                        "description": "Тематическая категория документации",
                        "enum": [
                            "typst-gost",
                            "typst-reference",
                            "typst-guides",
                            "typst-tutorial",
                            "typst-symbols",
                            "typst-changelog",
                            "mixed",
                        ],
                    },
                    "query_type": {
                        "type": "string",
                        "description": "Тип и паттерн пользовательского запроса",
                        "enum": [
                            "howto",
                            "troubleshooting",
                            "params",
                            "short_natural",
                            "concept_compare",
                            "multi_chunk",
                        ],
                    },
                    "difficulty": {
                        "type": "string",
                        "description": "Сложность перефразировки относительно исходного текста чанков",
                        "enum": ["easy", "medium", "hard"],
                    },
                },
                "required": [
                    "id",
                    "query",
                    "relevant_chunk_ids",
                    "category",
                    "query_type",
                    "difficulty",
                ],
            },
        }
    },
    "required": ["questions"],
}

SYSTEM_PROMPT_EVAL_GENERATION = """\
Ты — эксперт по оценке качества RAG-систем и составитель эталонных поисковых бенчмарков (Golden Evaluation Dataset) для метрик Recall@k и MRR.

Твоя задача — на основе переданного Markdown-файла с чанками документации по системе вёрстки Typst и пакету modern-g7-32 (оформление по ГОСТ 7.32-2017) сгенерировать около 200 эталонных пользовательских запросов (195–205 шт.) и точно привязать к каждому запросу все релевантные chunk_id.

==================================================
1. СТРУКТУРА ВХОДНЫХ ДАННЫХ И РАБОТА С НИМИ
==================================================
Входной файл содержит всю нарезанную базу знаний в формате:
## <chunk_id>
Документ: <Название страницы> [<категория>] > <Путь по разделам>
Описание страницы: <опциональное описание>
<Содержимое чанка>

Правила работы с чанками:
1. Используй ТОЛЬКО точные 20-символьные идентификаторы `chunk_id`, указанные в заголовках `## <chunk_id>`. Категорически запрещено выдумывать, сокращать или изменять `chunk_id`.
2. Учитывай разбиение длинных страниц на части: одна тема часто продолжается в соседних чанках одного документа или дополняется другим разделом.
3. Полнота разметки `relevant_chunk_ids` критична для честного расчёта Recall@k и MRR:
   - Перед фиксацией ответа проверь весь файл на наличие других чанков, которые тоже дают прямой и достаточный ответ на этот вопрос (или содержат его вторую обязательную часть).
   - Включай в `relevant_chunk_ids` от 1 до 4 чанков в порядке убывания значимости: сначала основной чанк, затем равнозначные или дополняющие.
   - Не добавляй «слегка смежные» чанки, которые лишь косвенно упоминают тему, но не отвечают на сам вопрос.
4. Игнорируй малоинформативные чанки (пустые заглушки, чисто технические фрагменты без самостоятельного смысла), не придумывай по ним искусственные вопросы ради охвата.

==================================================
2. ГЛАВНОЕ ПРАВИЛО: РЕАЛИСТИЧНОСТЬ И ОТСУТСТВИЕ «ПОДГОНКИ»
==================================================
Самая частая ошибка синтетических датасетов — лексическая утечка (lexical leakage), когда вопрос является перефразированным заголовком или копией текста чанка. Такой бенчмарк искусственно завышает метрики. Делай строго наоборот:

1. Иди от задачи пользователя, а не от формулировки чанка:
   - Представь реального пользователя: студента (пишет ВКР, курсовую, НИР, отчёт по практике по ГОСТ) или верстальщика/разработчика на Typst.
   - Сформулируй его жизненную цель, проблему или симптом, а затем определи, какие чанки решают эту задачу.

2. Жёсткий запрет на списывание редких терминов и структуры чанка:
   - НЕ копируй дословно заголовки разделов, описания страниц и длинные фразы из чанка.
   - НЕ подсказывай в вопросе точное имя редкого параметра, функции или опции, если обычный пользователь на этапе поиска его ещё не знает.
     * ПЛОХО (подгонка): «Как использовать параметр force-performers в gost.with?»
     * ХОРОШО (реалистично): «У меня один исполнитель на титульном листе, но блок исполнителей вообще не выводится — как заставить шаблон его показать?»
   - Используй естественную пользовательскую лексику и разговорные синонимы: «титульник», «красная строка», «отточие в содержании», «список литературы», «нумерация с введения», «подпись к рисунку», «таблица рвётся на две страницы», «экспорт в ворд», «шрифт слетел».
   - Названия базовых сущностей (`modern-g7-32`, `table`, `figure`, `#set`, `#show`, сообщения об ошибках) используй только там, где реальный пользователь действительно так спрашивает.

3. Самодостаточность (обязательно для Recall@k / MRR):
   - Каждый вопрос должен быть полностью понятен без контекста диалога и без открытой страницы сайта.
   - Запрещены контекстно-зависимые обрывки вроде «А какие у него параметры?», «Почему это не работает?», «Покажи пример для этого случая».

4. Однозначная проверяемость:
   - На каждый вопрос должен существовать чёткий, фактический ответ в указанных `relevant_chunk_ids`. Не генерируй оффтопик или вопросы, ответа на которые нет в документации.

==================================================
3. БАЛАНС И РАЗНООБРАЗИЕ (ОКОЛО 200 ВОПРОСОВ)
==================================================
Соблюдай реалистичное распределение по тематикам (НЕ распределяй вопросы пропорционально количеству чанков в changelog/symbols):
- `typst-gost` (пакет modern-g7-32, ГОСТ 7.32-2017, титульник, грифы, реферат, исполнители, содержание, приложения, источники, кастомизация отступов/шрифтов): ~40–45% (80–90 вопросов)
- `typst-reference` (синтаксис языка, #set/#show/#let, state/context/counter, таблицы, рисунки, формулы, сноски, ссылки, графика, макет страницы): ~25–30% (50–60 вопросов)
- `typst-guides` и `typst-tutorial` (практические сценарии, структура документа, перенос из LaTeX/Word, шрифты, ошибки и приёмы вёрстки): ~20% (35–45 вопросов)
- `typst-symbols` и `typst-changelog` (символы/эмодзи, изменения между версиями Typst): не более 5–10% (10–15 вопросов), только практически полезные запросы.

Чередуй 6 типов запросов:
1. `howto` (~30%) — практическая задача от цели («Как сделать, чтобы...», «Как убрать/добавить...»).
2. `troubleshooting` (~20%) — описание симптома или проблемы без знания причины («Почему текст не переносится...», «Съехали поля на второй странице...»).
3. `params` (~15%) — поиск настроек, ограничений, значений по умолчанию или форматов («Какие варианты нумерации поддерживаются для...», «Как поменять размер шрифта только в...»).
4. `short_natural` (~15%) — короткие поисковые запросы в стиле Telegram/поисковой строки (3–7 слов: «отступ первой строки абзаца typst», «как вставить скан титульника pdf»).
5. `concept_compare` (~10%) — понимание механики и отличий («Чем отличается X от Y», «Когда нужен context в Typst»).
6. `multi_chunk` (~10%) — составные вопросы, для полного ответа на которые нужны 2–4 разных чанка (например, общая настройка шаблона + оформление конкретного раздела, или базовый `figure` + требования ГОСТ к рисункам).

Распределение по сложности перефразировки (`difficulty`):
- `easy` (~25%): прямой естественный вопрос, допускается упоминание очевидного термина (например, `table`, `outline`, `реферат`).
- `medium` (~45%): перефразировка без копирования заголовка, использование бытовых синонимов или формулировки от цели.
- `hard` (~30%): запрос через симптом/боль пользователя без прямых ключевых слов из чанка, сравнение, либо необходимость объединить 2–4 чанка.

Все вопросы пиши на русском языке (с естественным вкраплением англоязычных идентификаторов Typst/пакетов там, где так пишут реальные пользователи). Исключай дубли и почти одинаковые по смыслу вопросы.

==================================================
4. ВНУТРЕННЯЯ ПРОВЕРКА КАЖДОГО ЭЛЕМЕНТА
==================================================
Перед выводом каждого вопроса мысленно проверь 4 пункта:
1. Звучит ли вопрос так, как его реально напишет человек в чат поддержки или в поиск?
2. Не скопированы ли в вопрос ключевые слова из текста чанка слишком «в лоб»?
3. Все ли `chunk_id` из файла, которые содержат прямой ответ на этот вопрос, добавлены в `relevant_chunk_ids`?
4. Все ли указанные `chunk_id` точно существуют в заголовках `## <chunk_id>` входного файла?

==================================================
5. ФОРМАТ ВЫВОДА
==================================================
Выводи результат СТРОГО в соответствии с JSON Schema:
- Корневой JSON-объект содержит ключ "questions" со списком объектов вопросов.
- Без markdown-разметки (```json ... ```), без вступительного текста и без заключительных комментариев.
- Если генерация выполняется батчами (например, 1–100 и 101–200), соблюдай запрошенный диапазон id и не повторяй уже сгенерированные вопросы.\
"""


def _load_db_settings(sqlite_path: Path) -> dict[str, Any]:
    if not sqlite_path.exists():
        return {}
    try:
        with sqlite3.connect(str(sqlite_path)) as conn:
            rows = conn.execute("SELECT key, value_json FROM gateway_settings").fetchall()
        return {key: json.loads(value_json) for key, value_json in rows}
    except sqlite3.Error:
        return {}


def _download_docs_from_github(
    *,
    repo_slug: str,
    branch: str,
    docs_subdir: str,
    target_dir: Path,
) -> int:
    archive_url = f"https://codeload.github.com/{repo_slug}/tar.gz/refs/heads/{branch}"
    with tempfile.TemporaryDirectory(prefix="export_chunks_") as tmp_root:
        archive_path = Path(tmp_root) / "repo.tar.gz"
        with httpx.stream(
            "GET",
            archive_url,
            timeout=120.0,
            follow_redirects=True,
            headers={"User-Agent": "ai-gateway-chunk-exporter/1.0"},
        ) as resp:
            resp.raise_for_status()
            with archive_path.open("wb") as out_file:
                for chunk in resp.iter_bytes():
                    out_file.write(chunk)

        target_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            count = _extract_subdir(tar, docs_subdir, target_dir)
            if count == 0:
                raise RuntimeError(f"No documentation files found in '{docs_subdir}'")
            return count


def export_chunks_to_markdown(
    *,
    docs_dir: Path,
    output_path: Path,
    docs_base_url: str,
    max_chars: int,
    overlap: int,
) -> tuple[int, int]:
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Docs directory not found: {docs_dir}. Pass --docs-dir or run with --sync-github."
        )

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
        raise RuntimeError(f"No chunks produced from {docs_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(f"## {chunk['chunk_id']}\n{chunk['text'].strip()}\n\n")

    return len(doc_files), len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export chunks to compact Markdown, print system prompt, or print Structured Output JSON Schema",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Print system prompt for golden dataset generation and exit",
    )
    parser.add_argument(
        "--schema",
        "--json-schema",
        "--json-format",
        dest="show_schema",
        action="store_true",
        help="Print OpenAPI/JSON Schema for Structured Output (Google AI Studio / OpenAI response_format) and exit",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Path to downloaded docs directory (default: DOCS_CACHE_DIR or data/docs_content)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown file path (default: data/chunks_for_eval.md)",
    )
    parser.add_argument(
        "--docs-base-url",
        default=None,
        help="Base docs URL used in chunk metadata",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Max characters per chunk",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap between chunks in the same section",
    )
    parser.add_argument(
        "--sync-github",
        action="store_true",
        help="Download docs from GitHub before chunking",
    )
    parser.add_argument("--repo-slug", default=None)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--docs-subdir", default=None)
    args = parser.parse_args()

    if args.prompt:
        print(SYSTEM_PROMPT_EVAL_GENERATION)
        return

    if args.show_schema:
        print(json.dumps(EVAL_JSON_SCHEMA, ensure_ascii=False, indent=2))
        return

    load_dotenv()
    data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
    db_settings = _load_db_settings(
        Path(os.getenv("SQLITE_DB_PATH", str(data_dir / "gateway.db"))).resolve()
    )

    docs_dir = args.docs_dir or Path(
        os.getenv("DOCS_CACHE_DIR", str(data_dir / "docs_content"))
    ).resolve()
    output_path = args.output or (data_dir / "chunks_for_eval.md").resolve()

    docs_base_url = (
        args.docs_base_url
        or str(
            db_settings.get("docs_base_url")
            or os.getenv("DOCS_BASE_URL", "https://typst-gost.ru")
        )
    ).rstrip("/")

    max_chars = (
        args.max_chars
        if args.max_chars is not None
        else int(db_settings.get("chunk_max_chars") or os.getenv("CHUNK_MAX_CHARS", "2500"))
    )

    overlap = (
        args.overlap
        if args.overlap is not None
        else int(
            db_settings.get("chunk_overlap")
            if db_settings.get("chunk_overlap") is not None
            else os.getenv("CHUNK_OVERLAP", "250")
        )
    )

    repo_slug = (
        args.repo_slug
        or str(
            db_settings.get("docs_repo_slug")
            or os.getenv("DOCS_REPO_SLUG", "typst-gost/website")
        )
    )

    branch = (
        args.branch
        or str(db_settings.get("docs_repo_branch") or os.getenv("DOCS_REPO_BRANCH", "main"))
    )

    docs_subdir = (
        args.docs_subdir
        or str(
            db_settings.get("docs_repo_docs_subdir")
            or os.getenv("DOCS_REPO_DOCS_SUBDIR", "content/docs")
        )
    ).strip("/")

    if args.sync_github:
        with tempfile.TemporaryDirectory(prefix="docs_sync_") as tmp_docs:
            tmp_docs_dir = Path(tmp_docs)
            _download_docs_from_github(
                repo_slug=repo_slug,
                branch=branch,
                docs_subdir=docs_subdir,
                target_dir=tmp_docs_dir,
            )
            files_count, chunks_count = export_chunks_to_markdown(
                docs_dir=tmp_docs_dir,
                output_path=output_path,
                docs_base_url=docs_base_url,
                max_chars=max_chars,
                overlap=overlap,
            )
    else:
        files_count, chunks_count = export_chunks_to_markdown(
            docs_dir=docs_dir,
            output_path=output_path,
            docs_base_url=docs_base_url,
            max_chars=max_chars,
            overlap=overlap,
        )

    print(f"Done: files={files_count}, chunks={chunks_count} -> {output_path}")


if __name__ == "__main__":
    main()