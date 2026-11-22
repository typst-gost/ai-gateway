# AI Gateway (Typst & modern-g7-32)

Централизованный FastAPI-сервис для RAG-поиска, классификации запросов (TopicGuard / Decision), стриминга ответов LLM, хранения диалогов в SQLite, индексирования документации из `typst-gost/website` и администрирования моделей/промптов.

## Возможности

- **Публичный API ассистента (без авторизации):**
  - `POST /api/v1/assistant/chat` — чат с RAG и TopicGuard в режимах JSON и SSE (`stream=true`), с профилями под разные сценарии (`website`, `telegram_private`, `telegram_group`, `external`).
  - `POST /api/v1/assistant/fix` — исправление кода Typst по тексту ошибки (профиль `telegram_fix`).
  - `POST /api/v1/assistant/search` — гибридный поиск по документации без вызова LLM.
  - `POST /api/v1/assistant/conversations/clear` — сброс контекстного окна диалога с сохранением истории в БД.
- **Админ-панель API (JWT или статический `ADMIN_API_KEY`, без регистрации):**
  - Раздельная настройка моделей LLM, Decision-моделей, системных промптов, Decision-критериев и порогов для `website`, `telegram_private`, `telegram_group`, `telegram_fix`, `external`.
  - Массовое обновление параметров всех профилей Telegram (`PATCH /api/v1/admin/profiles/telegram/bulk`).
  - Живое тестирование TopicGuard (`POST /api/v1/admin/test-guard`).
  - Просмотр баланса Polza AI (`GET /api/v1/admin/polza/balance`).
  - Получение каталога моделей Polza AI с фильтрацией по типам (`llm`, `decision`, `image`, `embedding`, `audio`, `video`, `tts`, `stt`, `all`).
  - Просмотр, фильтрация, удаление диалогов пользователей и аналитика токенов/расходов из SQLite.
  - Переиндексация базы знаний на лету (`POST /api/v1/admin/rag/reindex`) с загрузкой `content/docs` и `public/docs/compile_assets` напрямую с GitHub и атомарным обновлением LanceDB.

## Быстрый старт локально

```bash
cp .env.example .env
uv sync
uv run ai-gateway-reindex
uv run ai-gateway
```