"""Texterkennung (OCR) für gescannte PDFs ohne Textebene.

Standard unter Windows: die eingebaute Windows-Texterkennung (Windows.Media.Ocr). Sie läuft lokal, braucht
keinen Modell-Download und kann die Sprachen der installierten Windows-Sprachpakete (z. B. de-DE, en-US).

    uv sync --extra ocr        # installiert die WinRT-Anbindung (nur Windows)

Andere Engines (Tesseract, RapidOCR) lassen sich über das Protokoll ``OcrEngine`` ergänzen.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Protocol


class OcrEngine(Protocol):
    name: str

    def recognize(self, png: bytes) -> str:
        """Text eines Bildes (PNG) mit Zeilenumbrüchen."""
        ...


class OcrUnavailableError(RuntimeError):
    pass


class WindowsOcr:
    """Windows.Media.Ocr über pywinrt. Probiert die Sprachen in der angegebenen Reihenfolge."""

    def __init__(self, languages: tuple[str, ...] = ("de-DE", "en-US")) -> None:
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.media.ocr import OcrEngine as WinOcrEngine
        except ImportError as exc:
            raise OcrUnavailableError("Windows-OCR nicht installiert: uv sync --extra ocr") from exc

        engine = None
        for tag in languages:
            lang = Language(tag)
            if WinOcrEngine.is_language_supported(lang):
                engine = WinOcrEngine.try_create_from_language(lang)
                if engine is not None:
                    self.language = tag
                    break
        if engine is None:
            engine = WinOcrEngine.try_create_from_user_profile_languages()
            self.language = "user-profile"
        if engine is None:
            raise OcrUnavailableError("Keine OCR-Sprache verfügbar (Windows-Sprachpaket installieren)")
        self._engine = engine
        self.max_dimension = int(WinOcrEngine.max_image_dimension)
        self.name = f"windows-ocr-{self.language}"

    def recognize(self, png: bytes) -> str:
        return asyncio.run(self._recognize(png))

    async def _recognize(self, png: bytes) -> str:
        from winrt.windows.graphics.imaging import (
            BitmapAlphaMode,
            BitmapDecoder,
            BitmapPixelFormat,
            SoftwareBitmap,
        )
        from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(png)
        await writer.store_async()
        writer.detach_stream()
        stream.seek(0)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        if bitmap.bitmap_pixel_format != BitmapPixelFormat.BGRA8:
            bitmap = SoftwareBitmap.convert(bitmap, BitmapPixelFormat.BGRA8, BitmapAlphaMode.PREMULTIPLIED)
        result = await self._engine.recognize_async(bitmap)
        return "\n".join(line.text for line in result.lines)


def load_ocr(name: str = "auto") -> OcrEngine | None:
    """'auto' → beste verfügbare Engine oder None, 'none' → aus, 'windows' → Windows-OCR (sonst Fehler)."""
    if name == "none":
        return None
    if name in ("auto", "windows"):
        if sys.platform != "win32":
            if name == "windows":
                raise OcrUnavailableError("Windows-OCR gibt es nur unter Windows")
            return None
        try:
            return WindowsOcr()
        except OcrUnavailableError:
            if name == "windows":
                raise
            return None
    raise ValueError(f"Unbekannte OCR-Engine: {name}")
