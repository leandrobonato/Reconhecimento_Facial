"""
Núcleo do sistema de reconhecimento facial.

Carrega os rostos conhecidos a partir de um diretório, gera os encodings
(assinaturas faciais de 128 dimensões via dlib/face_recognition) e expõe
métodos para detectar e identificar rostos em qualquer imagem (frame).
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# face_recognition_models ainda usa pkg_resources (deprecado); o aviso não
# afeta o funcionamento e apenas polui a saída.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import face_recognition  # noqa: E402

# Extensões de imagem aceitas ao carregar rostos conhecidos
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FaceMatch:
    """Resultado de um rosto detectado em um frame."""

    top: int
    right: int
    bottom: int
    left: int
    name: str        # "Desconhecido" quando não há correspondência
    distance: float  # distância euclidiana do melhor match (menor = mais parecido)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.top, self.right, self.bottom, self.left

    @property
    def confidence(self) -> float:
        """Converte a distância em um score aproximado de confiança (0 a 1)."""
        return float(max(0.0, min(1.0, 1.0 - self.distance)))


class FaceEngine:
    """
    Motor de reconhecimento facial.

    Uso básico:
        engine = FaceEngine("known_faces")
        matches = engine.recognize(frame_bgr)
    """

    def __init__(
        self,
        known_faces_dir: str | Path = "known_faces",
        tolerance: float = 0.6,
        model: str = "hog",
        upsample: int = 1,
        cache_file: str | Path | None = None,
    ) -> None:
        """
        Args:
            known_faces_dir: diretório com uma subpasta por pessoa
                (known_faces/<nome_da_pessoa>/*.jpg) ou imagens soltas
                nomeadas com o nome da pessoa (known_faces/<nome>.jpg).
            tolerance: distância máxima para considerar um match
                (0.6 é o padrão da biblioteca; menor = mais rígido).
            model: "hog" (rápido, CPU) ou "cnn" (mais preciso, ideal com GPU).
            upsample: quantas vezes ampliar a imagem ao procurar rostos
                (aumente para detectar rostos pequenos, ao custo de velocidade).
            cache_file: caminho opcional de um .pkl para cachear os encodings
                e acelerar inicializações futuras.
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.tolerance = tolerance
        self.model = model
        self.upsample = upsample
        self.cache_file = Path(cache_file) if cache_file else None

        self.known_encodings: list[np.ndarray] = []
        self.known_names: list[str] = []

        self._load_known_faces()

    # ------------------------------------------------------------------ #
    # Carregamento dos rostos conhecidos
    # ------------------------------------------------------------------ #
    def _load_known_faces(self) -> None:
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, "rb") as fh:
                data = pickle.load(fh)
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            print(f"[FaceEngine] {len(self.known_names)} encodings carregados do cache.")
            return

        if not self.known_faces_dir.exists():
            raise FileNotFoundError(
                f"Diretório de rostos conhecidos não encontrado: {self.known_faces_dir}"
            )

        for path in sorted(self.known_faces_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # known_faces/<pessoa>/<foto>.jpg -> nome = pasta
            # known_faces/<pessoa>.jpg        -> nome = arquivo
            if path.parent != self.known_faces_dir:
                name = path.parent.name
            else:
                name = path.stem
            name = name.replace("_", " ").replace("-", " ").title()

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                print(f"[FaceEngine] AVISO: nenhum rosto encontrado em {path.name}, ignorando.")
                continue

            self.known_encodings.append(encodings[0])
            self.known_names.append(name)
            print(f"[FaceEngine] Rosto carregado: {name} ({path.name})")

        if not self.known_encodings:
            print("[FaceEngine] AVISO: nenhum rosto conhecido carregado — "
                  "todos os rostos serão marcados como 'Desconhecido'.")

        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "wb") as fh:
                pickle.dump(
                    {"encodings": self.known_encodings, "names": self.known_names}, fh
                )
            print(f"[FaceEngine] Cache de encodings salvo em {self.cache_file}")

    # ------------------------------------------------------------------ #
    # Detecção + identificação
    # ------------------------------------------------------------------ #
    def recognize(self, frame_bgr: np.ndarray, scale: float = 1.0) -> list[FaceMatch]:
        """
        Detecta e identifica todos os rostos em um frame BGR (formato OpenCV).

        Args:
            frame_bgr: imagem no formato BGR (cv2.imread / VideoCapture).
            scale: fator de redução aplicado antes da detecção para ganhar
                velocidade (ex.: 0.5 processa a imagem na metade do tamanho).
                As coordenadas retornadas são reescaladas para o frame original.

        Returns:
            Lista de FaceMatch com caixa, nome e distância.
        """
        if scale != 1.0:
            small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(
            rgb, number_of_times_to_upsample=self.upsample, model=self.model
        )
        encodings = face_recognition.face_encodings(rgb, locations)

        matches: list[FaceMatch] = []
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            name = "Desconhecido"
            distance = 1.0

            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best = int(np.argmin(distances))
                distance = float(distances[best])
                if distance <= self.tolerance:
                    name = self.known_names[best]

            if scale != 1.0:
                inv = 1.0 / scale
                top, right, bottom, left = (
                    int(top * inv), int(right * inv), int(bottom * inv), int(left * inv)
                )

            matches.append(FaceMatch(top, right, bottom, left, name, distance))

        return matches
