"""
Reconhecimento facial em fotos.

Detecta rostos em uma imagem (ou em todas as imagens de uma pasta),
identifica pessoas conhecidas e salva uma cópia anotada com retângulos
e nomes.

Exemplos:
    python recognize_image.py examples/photos/foto1.jpg
    python recognize_image.py examples/photos/ --output output/
    python recognize_image.py foto.jpg --tolerance 0.5 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from src import FaceEngine, draw_matches, imread_u, imwrite_u
from src.face_engine import IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconhecimento facial em fotos.")
    parser.add_argument("input", help="Caminho de uma imagem ou de uma pasta de imagens.")
    parser.add_argument("--known", default="known_faces",
                        help="Diretório com os rostos conhecidos (padrão: known_faces).")
    parser.add_argument("--output", default="output",
                        help="Pasta onde salvar as imagens anotadas (padrão: output).")
    parser.add_argument("--tolerance", type=float, default=0.6,
                        help="Tolerância do match: menor = mais rígido (padrão: 0.6).")
    parser.add_argument("--model", choices=["hog", "cnn"], default="hog",
                        help="Modelo de detecção: hog (CPU, rápido) ou cnn (preciso).")
    parser.add_argument("--show", action="store_true",
                        help="Exibe cada imagem anotada em uma janela.")
    return parser.parse_args()


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return [input_path]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Arquivo ou pasta não encontrado: {input_path}")

    engine = FaceEngine(args.known, tolerance=args.tolerance, model=args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in collect_images(input_path):
        frame = imread_u(image_path)
        if frame is None:
            print(f"[AVISO] Não foi possível ler {image_path}, ignorando.")
            continue

        matches = engine.recognize(frame)
        draw_matches(frame, matches)

        names = ", ".join(m.name for m in matches) or "nenhum rosto"
        print(f"{image_path.name}: {len(matches)} rosto(s) -> {names}")

        out_path = output_dir / f"{image_path.stem}_reconhecido{image_path.suffix}"
        imwrite_u(out_path, frame)
        print(f"  Salvo em {out_path}")

        if args.show:
            cv2.imshow("Reconhecimento Facial", frame)
            cv2.waitKey(0)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
