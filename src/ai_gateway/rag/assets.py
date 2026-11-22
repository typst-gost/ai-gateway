from __future__ import annotations

from pathlib import Path


def list_compile_assets(assets_dir: Path) -> list[str]:
    if not assets_dir.exists() or not assets_dir.is_dir():
        return []
    return [
        path.relative_to(assets_dir).as_posix()
        for path in sorted(assets_dir.rglob("*"))
        if path.is_file() and not any(part.startswith(".") for part in path.parts)
    ]