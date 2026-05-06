"""
LlmWorker
QThread that runs LLM correction on the combined OCR output.

Supports any Ollama model (gemma3:12b, mistral-small, etc.) via model_name.
Uses llm_postCorection as backend (pre-cleanup regex + anti-hallucination checks).

Emits:
  log_line(str)
  page_corrected(page_name, corrected_text)   - after each page
  error(str)
"""
import os

from PySide6.QtCore import QThread, Signal


class LlmWorker(QThread):
    log_line       = Signal(str)
    page_corrected = Signal(str, str)   # page_name, corrected_text
    error          = Signal(str)

    def __init__(
        self,
        input_path: str,    # structured_all.txt
        output_path: str,   # structured_all_corrected.txt
        model_name: str,    # e.g. "gemma3:12b" or "mistral-small"
        parent=None,
    ):
        super().__init__(parent)
        self.input_path  = input_path
        self.output_path = output_path
        self.model_name  = model_name

    def run(self):
        try:
            import requests
            import llm_postCorection as backend

            backend.OLLAMA_MODEL = self.model_name

            self.log_line.emit(f"Reading: {self.input_path}")
            with open(self.input_path, "r", encoding="utf-8") as f:
                raw = f.read()

            pages = backend.split_combined(raw)
            total = len(pages)
            self.log_line.emit(f"Model: {self.model_name} | Pages to correct: {total}")

            out_dir = os.path.dirname(self.output_path) or "."
            os.makedirs(out_dir, exist_ok=True)

            session = requests.Session()
            out_chunks: list[str] = []

            for i, (name, content) in enumerate(pages, 1):
                if not content.strip():
                    self.log_line.emit(f"[{i}/{total}] {name}: empty — skipped")
                    out_chunks.append(f"=== {name} ===\n")
                    self.page_corrected.emit(name, "")
                    continue

                self.log_line.emit(f"[{i}/{total}] LLM: {name}…")
                corrected, status = backend.correct_page(session, content, page_id=name)
                self.log_line.emit(f"  -> {status}")
                out_chunks.append(f"=== {name} ===\n{corrected}\n")

                stem      = os.path.splitext(name)[0]
                page_path = os.path.join(out_dir, f"llm_{stem}.txt")
                with open(page_path, "w", encoding="utf-8") as f:
                    f.write(corrected)
                self.log_line.emit(f"  → saved {os.path.basename(page_path)}")

                self.page_corrected.emit(name, corrected)

            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_chunks))
            self.log_line.emit(f"Saved corrected output -> {self.output_path}")

        except Exception as exc:
            self.error.emit(f"LlmWorker: {exc}")
