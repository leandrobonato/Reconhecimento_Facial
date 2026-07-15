"""
Reconhecimento facial em vídeos.

Processa um arquivo de vídeo, identifica pessoas conhecidas em cada frame
e gera um novo vídeo anotado com retângulos e nomes.

Exemplos:
    python recognize_video.py examples/videos/video1.mp4
    python recognize_video.py video.mp4 --skip 3 --scale 0.5 --show
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2

from src import FaceEngine, draw_matches


def open_video(path: Path) -> tuple[cv2.VideoCapture, Path | None]:
    """
    Abre um vídeo. Em caminhos com acentos (Windows) o OpenCV pode falhar;
    nesse caso copia o arquivo para uma pasta temporária ASCII antes de abrir.
    Retorna (capture, caminho_temporário_ou_None).
    """
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        return capture, None

    temp_path = Path(tempfile.gettempdir()) / f"rf_in_{path.name}"
    shutil.copy2(path, temp_path)
    return cv2.VideoCapture(str(temp_path)), temp_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconhecimento facial em vídeos.")
    parser.add_argument("input", help="Caminho do arquivo de vídeo.")
    parser.add_argument("--known", default="known_faces",
                        help="Diretório com os rostos conhecidos (padrão: known_faces).")
    parser.add_argument("--output", default=None,
                        help="Arquivo de saída (padrão: output/<nome>_reconhecido.mp4).")
    parser.add_argument("--tolerance", type=float, default=0.6,
                        help="Tolerância do match: menor = mais rígido (padrão: 0.6).")
    parser.add_argument("--model", choices=["hog", "cnn"], default="hog",
                        help="Modelo de detecção: hog (CPU, rápido) ou cnn (preciso).")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Fator de redução do frame na detecção (padrão: 0.5).")
    parser.add_argument("--skip", type=int, default=2,
                        help="Processa 1 a cada N frames e reaproveita o resultado "
                             "nos demais (padrão: 2). Use 1 para processar todos.")
    parser.add_argument("--show", action="store_true",
                        help="Exibe o vídeo em uma janela durante o processamento.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Vídeo não encontrado: {input_path}")

    engine = FaceEngine(args.known, tolerance=args.tolerance, model=args.model)

    capture, temp_input = open_video(input_path)
    if not capture.isOpened():
        sys.exit(f"Não foi possível abrir o vídeo: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("output") / f"{input_path.stem}_reconhecido.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Grava em um caminho temporário ASCII (o OpenCV falha em caminhos com
    # acentos no Windows) e move para o destino final ao terminar.
    temp_output = Path(tempfile.gettempdir()) / f"rf_out_{out_path.name}"
    writer = cv2.VideoWriter(
        str(temp_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    matches = []
    frame_index = 0
    start = time.time()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        # Detecta em 1 a cada N frames; nos demais reaproveita as caixas
        if frame_index % max(1, args.skip) == 0:
            matches = engine.recognize(frame, scale=args.scale)

        draw_matches(frame, matches)
        writer.write(frame)

        if args.show:
            cv2.imshow("Reconhecimento Facial - Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Interrompido pelo usuário.")
                break

        frame_index += 1
        if total and frame_index % 30 == 0:
            pct = 100 * frame_index / total
            print(f"\rProcessando... {pct:5.1f}% ({frame_index}/{total} frames)",
                  end="", flush=True)

    elapsed = time.time() - start
    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    shutil.move(temp_output, out_path)
    if temp_input:
        temp_input.unlink(missing_ok=True)

    print(f"\nConcluído: {frame_index} frames em {elapsed:.1f}s "
          f"({frame_index / max(elapsed, 0.001):.1f} fps)")
    print(f"Vídeo anotado salvo em {out_path}")


if __name__ == "__main__":
    main()
