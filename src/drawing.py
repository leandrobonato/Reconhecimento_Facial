"""
Utilitários de desenho: retângulos e etiquetas de nome sobre os rostos.
"""

from __future__ import annotations

import numpy as np
import cv2

from .face_engine import FaceMatch

# Cores em BGR
COLOR_KNOWN = (80, 200, 60)     # verde para pessoas identificadas
COLOR_UNKNOWN = (60, 60, 230)   # vermelho para desconhecidos
COLOR_TEXT = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_matches(
    frame: np.ndarray,
    matches: list[FaceMatch],
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Desenha um retângulo e uma etiqueta com o nome para cada rosto detectado.

    Args:
        frame: imagem BGR (modificada in-place e também retornada).
        matches: lista de FaceMatch vinda de FaceEngine.recognize().
        show_confidence: exibe o percentual de confiança ao lado do nome.
    """
    for match in matches:
        top, right, bottom, left = match.box
        known = match.name != "Desconhecido"
        color = COLOR_KNOWN if known else COLOR_UNKNOWN

        # Espessura proporcional ao tamanho do rosto
        thickness = max(2, (right - left) // 100)
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

        label = match.name
        if show_confidence and known:
            label += f" ({match.confidence:.0%})"

        # Escala da fonte proporcional à largura do rosto
        font_scale = max(0.5, (right - left) / 250)
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, 1)

        # Barra de fundo abaixo do retângulo
        bar_top = bottom
        bar_bottom = bottom + text_h + baseline + 10
        cv2.rectangle(frame, (left, bar_top), (max(right, left + text_w + 12), bar_bottom),
                      color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom + text_h + 5),
                    FONT, font_scale, COLOR_TEXT, 1, cv2.LINE_AA)

    return frame
