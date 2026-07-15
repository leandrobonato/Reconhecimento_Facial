"""
Prepara os arquivos de exemplo do projeto:

1. Baixa 10 retratos em domínio público (retratos oficiais do governo dos
   EUA, obtidos via API da Wikipédia) para examples/photos/.
2. Recorta o rosto de algumas pessoas e monta a base known_faces/.
3. Gera 2 vídeos de demonstração (efeito Ken Burns e colagem panorâmica)
   em examples/videos/.

Uso:
    python scripts/prepare_examples.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Importa io_utils diretamente (sem passar pelo pacote src, que depende
# de face_recognition — desnecessário para preparar os exemplos).
sys.path.insert(0, str(ROOT / "src"))

from io_utils import imread_u, imwrite_u  # noqa: E402

PHOTOS_DIR = ROOT / "examples" / "photos"
VIDEOS_DIR = ROOT / "examples" / "videos"
KNOWN_DIR = ROOT / "known_faces"

USER_AGENT = "ReconhecimentoFacialDemo/1.0 (projeto educacional; bonato16@gmail.com)"
REQUEST_DELAY = 2.0  # pausa entre downloads para respeitar o rate limit da API

# Pessoas cujos retratos oficiais estão em domínio público (obras do
# governo dos EUA). O nome da esquerda é o título do artigo na Wikipédia.
PEOPLE = [
    ("Barack_Obama", "barack_obama"),
    ("Joe_Biden", "joe_biden"),
    ("Donald_Trump", "donald_trump"),
    ("George_W._Bush", "george_w_bush"),
    ("Bill_Clinton", "bill_clinton"),
    ("Hillary_Clinton", "hillary_clinton"),
    ("Kamala_Harris", "kamala_harris"),
    ("Michelle_Obama", "michelle_obama"),
    ("Ronald_Reagan", "ronald_reagan"),
    ("Neil_Armstrong", "neil_armstrong"),
]

# Quem entra na base de rostos conhecidos (os demais aparecerão
# como "Desconhecido" na demonstração).
KNOWN_PEOPLE = [
    "barack_obama", "joe_biden", "donald_trump", "george_w_bush",
    "hillary_clinton", "kamala_harris", "neil_armstrong",
]


def fetch_bytes(url: str, retries: int = 4) -> bytes:
    """Baixa uma URL com User-Agent e retry exponencial em caso de 429/erro."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"    tentativa {attempt + 1} falhou ({error}); "
                  f"aguardando {wait}s...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download_photos() -> list[Path]:
    """Baixa o retrato principal de cada pessoa via API da Wikipédia."""
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for article, slug in PEOPLE:
        destination = PHOTOS_DIR / f"{slug}.jpg"
        if destination.exists():
            print(f"[fotos] {destination.name} já existe, pulando.")
            downloaded.append(destination)
            continue

        try:
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{article}"
            data = json.loads(fetch_bytes(summary_url).decode("utf-8"))
            image_url = data["originalimage"]["source"]
            raw = fetch_bytes(image_url)
        except Exception as error:  # noqa: BLE001
            print(f"[fotos] FALHA ao baixar {article}: {error}")
            time.sleep(REQUEST_DELAY)
            continue

        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[fotos] Imagem inválida para {article}, pulando.")
            time.sleep(REQUEST_DELAY)
            continue

        # Redimensiona para no máximo 900px de largura (arquivos menores)
        if image.shape[1] > 900:
            scale = 900 / image.shape[1]
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        imwrite_u(destination, image, quality=90)

        print(f"[fotos] OK: {destination.name}")
        downloaded.append(destination)
        time.sleep(REQUEST_DELAY)

    return downloaded


def build_known_faces() -> None:
    """Recorta o rosto (com margem) das pessoas conhecidas para known_faces/."""
    import face_recognition

    for slug in KNOWN_PEOPLE:
        source = PHOTOS_DIR / f"{slug}.jpg"
        if not source.exists():
            print(f"[known] {source.name} não encontrada, pulando.")
            continue

        person_dir = KNOWN_DIR / slug
        destination = person_dir / f"{slug}_ref.jpg"
        if destination.exists():
            print(f"[known] {destination.name} já existe, pulando.")
            continue

        image = imread_u(source)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            print(f"[known] Nenhum rosto em {source.name}, usando imagem inteira.")
            crop = image
        else:
            # Maior rosto encontrado, com margem ao redor
            top, right, bottom, left = max(
                locations, key=lambda box: (box[1] - box[3]) * (box[2] - box[0])
            )
            margin = int(0.45 * (right - left))
            y0, y1 = max(0, top - margin), min(image.shape[0], bottom + margin)
            x0, x1 = max(0, left - margin), min(image.shape[1], right + margin)
            crop = image[y0:y1, x0:x1]

        person_dir.mkdir(parents=True, exist_ok=True)
        imwrite_u(destination, crop, quality=92)
        print(f"[known] OK: {slug}")


def make_writer(destination: Path, fps: int,
                size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    """Cria um VideoWriter em caminho temporário ASCII (compatível com
    caminhos acentuados no Windows). Retorna (writer, caminho_temporário)."""
    temp_path = Path(tempfile.gettempdir()) / f"rf_gen_{destination.name}"
    writer = cv2.VideoWriter(str(temp_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return writer, temp_path


def ken_burns_video(photos: list[Path], destination: Path,
                    size: tuple[int, int] = (1280, 720), fps: int = 24,
                    seconds_per_photo: float = 4.0) -> None:
    """Vídeo com efeito de zoom suave (Ken Burns) sobre cada foto."""
    width, height = size
    writer, temp_path = make_writer(destination, fps, size)

    for photo in photos:
        image = imread_u(photo)
        if image is None:
            continue
        # Ajusta a foto para cobrir o quadro
        scale = max(width / image.shape[1], height / image.shape[0]) * 1.15
        base = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        bh, bw = base.shape[:2]

        frames = int(seconds_per_photo * fps)
        for i in range(frames):
            t = i / max(frames - 1, 1)
            zoom = 1.0 + 0.12 * t  # zoom-in suave
            zw, zh = int(width * zoom), int(height * zoom)
            zw, zh = min(zw, bw), min(zh, bh)
            x0 = (bw - zw) // 2
            y0 = max(0, (bh - zh) // 3)  # enquadra a parte superior (rosto)
            crop = base[y0:y0 + zh, x0:x0 + zw]
            writer.write(cv2.resize(crop, size))

    writer.release()
    shutil.move(temp_path, destination)
    print(f"[video] OK: {destination.name}")


def collage_pan_video(photos: list[Path], destination: Path,
                      size: tuple[int, int] = (1280, 720), fps: int = 24,
                      seconds: float = 14.0) -> None:
    """Vídeo panorâmico sobre uma colagem horizontal com vários rostos."""
    width, height = size
    tiles = []
    for photo in photos:
        image = imread_u(photo)
        if image is None:
            continue
        scale = height / image.shape[0]
        tiles.append(cv2.resize(image, (int(image.shape[1] * scale), height)))
    collage = np.hstack(tiles)

    writer, temp_path = make_writer(destination, fps, size)
    frames = int(seconds * fps)
    max_x = collage.shape[1] - width
    for i in range(frames):
        t = i / max(frames - 1, 1)
        # Movimento de ida e volta suave ao longo da colagem
        x = int(max_x * 0.5 * (1 - np.cos(2 * np.pi * t))) if max_x > 0 else 0
        x = min(max(0, x), max_x)
        writer.write(np.ascontiguousarray(collage[:, x:x + width]))

    writer.release()
    shutil.move(temp_path, destination)
    print(f"[video] OK: {destination.name}")


def main() -> None:
    photos = download_photos()
    if len(photos) < 4:
        sys.exit("Poucas fotos baixadas — verifique a conexão e tente novamente.")

    build_known_faces()

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    video1 = VIDEOS_DIR / "video1_retratos.mp4"
    video2 = VIDEOS_DIR / "video2_panoramico.mp4"
    if not video1.exists():
        ken_burns_video(photos[:5], video1)
    if not video2.exists():
        collage_pan_video(photos[5:], video2)

    print("\nExemplos prontos em examples/ e known_faces/.")


if __name__ == "__main__":
    main()
