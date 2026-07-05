"""
client.py
Async clients for MinerU extraction
"""
from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# MinerU client
# ---------------------------------------------------------------------------

class AsyncMinerUClient:
    """Async wrapper around the MinerU CLI for PDF → Markdown conversion."""

    def __init__(
        self,
        command_template: Optional[str] = None,
        debug_markdown_dir: Optional[str] = None,
    ):
        self.command_template = (
            command_template
            or "mineru -p {pdf} -o {out_dir} -b pipeline -m txt -d cpu -f false -t false"
        )
        self.debug_markdown_dir = debug_markdown_dir

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _command_exists(command_template: str) -> bool:
        try:
            first_token = shlex.split(command_template)[0]
        except Exception:
            return False
        return shutil.which(first_token) is not None

    @staticmethod
    def _ensure_cli_on_path(command_template: str) -> None:
        try:
            first_token = shlex.split(command_template)[0]
        except Exception:
            return
        if shutil.which(first_token) is not None:
            return
        venv_bin = Path(sys.executable).resolve().parent
        current_path = os.environ.get("PATH", "")
        if str(venv_bin) not in current_path:
            os.environ["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"

    async def _run_mineru(self, pdf_path: str, out_dir: str) -> None:
        """Run MinerU CLI; raise RuntimeError on failure."""
        def _quote(p: str) -> str:
            if os.name == 'nt':
                return f'"{p}"'
            return shlex.quote(p)

        cmd = self.command_template.format(
            pdf=_quote(pdf_path), out_dir=_quote(out_dir)
        )
        print(f"[MinerU] Running: {cmd}")
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"MinerU failed (exit {proc.returncode}): {err}")

    # ------------------------------------------------------------------
    # Public: PDF → Markdown
    # ------------------------------------------------------------------

    async def extract_markdown(self, pdf_path: str) -> str:
        """Run MinerU on *pdf_path* and return the full Markdown output."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._ensure_cli_on_path(self.command_template)
        if not self._command_exists(self.command_template):
            raise RuntimeError(
                "MinerU CLI not found in PATH. Ensure your venv is active or "
                f"pass a valid command via --mineru_cmd. Template: {self.command_template}"
            )

        with tempfile.TemporaryDirectory(prefix="mineru_out_") as out_dir:
            await self._run_mineru(pdf_path, out_dir)

            md_files = sorted(Path(out_dir).rglob("*.md"))
            if not md_files:
                raise RuntimeError("MinerU completed but no *.md output found")

            best_md = max(md_files, key=lambda p: p.stat().st_size)
            markdown_text = best_md.read_text(encoding="utf-8", errors="ignore")

            if self.debug_markdown_dir:
                self._save_debug_markdown(pdf_path, markdown_text)

            return markdown_text

    def _save_debug_markdown(self, pdf_path: str, markdown_text: str) -> None:
        assert self.debug_markdown_dir
        os.makedirs(self.debug_markdown_dir, exist_ok=True)
        pdf_name = Path(pdf_path).stem

        (Path(self.debug_markdown_dir) / f"{pdf_name}.md").write_text(
            markdown_text, encoding="utf-8"
        )
        refs_block = self.extract_references_block_markdown(markdown_text)
        (Path(self.debug_markdown_dir) / f"{pdf_name}_references.md").write_text(
            refs_block, encoding="utf-8"
        )
        for extra in Path(self.debug_markdown_dir).glob(f"{pdf_name}*"):
            if extra.suffix != ".md":
                try:
                    extra.unlink()
                except Exception:
                    pass