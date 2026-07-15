"""
Reconhecimento facial em tempo real pela webcam.

Abre a câmera padrão, identifica pessoas conhecidas ao vivo e desenha
retângulos e nomes sobre o vídeo. Pressione 'q' para sair.

Exemplos:
    python recognize_webcam.py
    python recognize_webcam.py --camera 1 --tolerance 0.5
"""

from __future__ import annotations

import argparse
import sys

import cv2

from src import FaceEngine, draw_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconhecimento facial em tempo real (webcam)."
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Índice da câmera (padrão: 0).")
    parser.add_argument("--known", default="known_faces",
                        help="Diretório com os rostos conhecidos (padrão: known_faces).")
    parser.add_argument("--tolerance", type=float, default=0.6,
                        help="Tolerância do match: menor = mais rígido (padrão: 0.6).")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Fator de redução do frame na detecção (padrão: 0.25).")
    parser.add_argument("--skip", type=int, default=2,
                        help="Processa 1 a cada N frames (padrão: 2).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = FaceEngine(args.known, tolerance=args.tolerance)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        sys.exit(f"Não foi possível abrir a câmera {args.camera}.")

    print("Webcam ativa — pressione 'q' para sair.")
    matches = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % max(1, args.skip) == 0:
            matches = engine.recognize(frame, scale=args.scale)

        draw_matches(frame, matches)
        cv2.imshow("Reconhecimento Facial - Webcam (q para sair)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_index += 1

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
