"""
Leitura e gravação de imagens compatíveis com caminhos Unicode no Windows.

cv2.imread/cv2.imwrite falham em caminhos com acentos (ex.: "Repositórios");
estas funções usam imdecode/imencode sobre bytes para contornar isso.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_u(path: str | Path) -> np.ndarray | None:
    """Equivalente a cv2.imread, mas aceita caminhos com caracteres Unicode."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path: str | Path, image: np.ndarray, quality: int = 95) -> bool:
    """Equivalente a cv2.imwrite, mas aceita caminhos com caracteres Unicode."""
    path = Path(path)
    params = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buffer = cv2.imencode(path.suffix, image, params)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True
