# 🎯 Sistema de Reconhecimento Facial

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4%2B-5C3EE8?logo=opencv&logoColor=white)
![face_recognition](https://img.shields.io/badge/face__recognition-dlib-green)
![License](https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow)

Sistema completo de **detecção e reconhecimento facial** em Python, capaz de identificar pessoas conhecidas em **fotos, vídeos e webcam em tempo real**, desenhando retângulos e etiquetas com nome e nível de confiança sobre cada rosto.

<p align="center">
  <img src="docs/demo_video.gif" alt="Demonstração do reconhecimento facial em vídeo" width="640">
</p>

---

## ✨ Funcionalidades

- 📷 **Detecção de rostos em fotos** — processa uma imagem ou uma pasta inteira de uma vez
- 🧑‍🤝‍🧑 **Identificação de pessoas conhecidas** — compara cada rosto com uma base cadastrada e nomeia quem for reconhecido
- 🎬 **Processamento de vídeos** — gera um novo vídeo anotado, com barra de progresso e otimizações de velocidade
- 📹 **Webcam em tempo real** — reconhecimento ao vivo pela câmera
- 🟩 **Anotações visuais** — retângulo **verde** com nome e confiança para pessoas conhecidas; **vermelho** com "Desconhecido" para não cadastradas
- ⚡ **Otimizações de performance** — redução de escala na detecção, processamento de 1 a cada N frames e cache de encodings
- 🗂️ **Cadastro por pastas** — para adicionar uma pessoa, basta criar uma pasta com fotos dela em `known_faces/`

## 🖼️ Demonstração

| Pessoa conhecida | Pessoa não cadastrada |
|:---:|:---:|
| <img src="docs/barack_obama_reconhecido.jpg" width="320"> | <img src="docs/ronald_reagan_reconhecido.jpg" width="380"> |
| Identificada com nome e confiança | Marcada como "Desconhecido" |

## 🧠 Como funciona

1. **Cadastro** — cada imagem em `known_faces/<nome_da_pessoa>/` é convertida em um *encoding* facial de 128 dimensões usando a rede neural do `dlib` (via `face_recognition`).
2. **Detecção** — em cada foto ou frame, os rostos são localizados com o modelo HOG (rápido, CPU) ou CNN (mais preciso, ideal com GPU).
3. **Identificação** — o encoding de cada rosto detectado é comparado com a base pela **distância euclidiana**; se a menor distância ficar abaixo da tolerância (padrão 0.6), a pessoa é identificada.
4. **Anotação** — o OpenCV desenha os retângulos e etiquetas e salva a imagem/vídeo de saída.

## 📁 Estrutura do projeto

```
Reconhecimento_Facial/
├── src/
│   ├── face_engine.py       # Motor de reconhecimento (encodings, matching)
│   ├── drawing.py           # Desenho de retângulos e etiquetas
│   └── io_utils.py          # Leitura/gravação compatível com caminhos Unicode
├── recognize_image.py       # CLI: reconhecimento em fotos
├── recognize_video.py       # CLI: reconhecimento em vídeos
├── recognize_webcam.py      # CLI: reconhecimento em tempo real (webcam)
├── known_faces/             # Base de pessoas conhecidas (1 pasta por pessoa)
├── examples/
│   ├── photos/              # 10 fotos de exemplo
│   └── videos/              # 2 vídeos de exemplo
├── output/                  # Resultados anotados
├── scripts/
│   └── prepare_examples.py  # Baixa/gera os arquivos de exemplo
└── requirements.txt
```

## 🚀 Instalação

> Requisitos: Python 3.10+ e `pip`.

**Linux / macOS** (requer CMake e um compilador C++ para compilar o `dlib`):

```bash
pip install -r requirements.txt
```

**Windows** (recomendado — usa o `dlib` pré-compilado, sem precisar de CMake/Visual Studio):

```powershell
pip install dlib-bin
pip install "setuptools<81" click pillow numpy opencv-python face-recognition-models
pip install --no-deps face-recognition
```

## 💻 Como usar

### Reconhecimento em fotos

```bash
# Uma foto
python recognize_image.py examples/photos/barack_obama.jpg

# Uma pasta inteira
python recognize_image.py examples/photos --output output/photos

# Mais rígido no matching e exibindo o resultado na tela
python recognize_image.py foto.jpg --tolerance 0.5 --show
```

### Reconhecimento em vídeos

```bash
python recognize_video.py examples/videos/video1_retratos.mp4

# Mais rápido: detecta em 1 a cada 3 frames, com frame reduzido pela metade
python recognize_video.py video.mp4 --skip 3 --scale 0.5

# Acompanhando o processamento em uma janela
python recognize_video.py video.mp4 --show
```

### Webcam em tempo real

```bash
python recognize_webcam.py            # pressione 'q' para sair
python recognize_webcam.py --camera 1 --tolerance 0.5
```

### Principais opções

| Opção | Descrição | Padrão |
|---|---|---|
| `--known` | Diretório da base de rostos conhecidos | `known_faces` |
| `--tolerance` | Distância máxima do match (menor = mais rígido) | `0.6` |
| `--model` | Detector: `hog` (CPU, rápido) ou `cnn` (preciso, GPU) | `hog` |
| `--scale` | Fator de redução do frame na detecção (vídeo/webcam) | `0.5` / `0.25` |
| `--skip` | Detecta em 1 a cada N frames (vídeo/webcam) | `2` |
| `--show` | Exibe o resultado em uma janela | — |

## 👤 Cadastrando novas pessoas

Crie uma pasta com o nome da pessoa dentro de `known_faces/` e coloque uma ou mais fotos nítidas do rosto dela:

```
known_faces/
├── maria_silva/
│   ├── foto1.jpg
│   └── foto2.jpg
└── joao_souza/
    └── perfil.png
```

O nome exibido é derivado do nome da pasta (`maria_silva` → "Maria Silva"). Quanto mais fotos em ângulos diferentes, melhor o reconhecimento.

## 🧪 Arquivos de exemplo

O repositório inclui **10 fotos** (`examples/photos/`) e **2 vídeos** (`examples/videos/`) prontos para teste, além de uma base `known_faces/` com 7 pessoas cadastradas — as demais aparecem como "Desconhecido", demonstrando os dois cenários.

As fotos são **retratos oficiais em domínio público** (obras do governo dos EUA, obtidas via Wikimedia/Wikipédia), e os vídeos foram gerados a partir delas. Para baixar/regenerar tudo do zero:

```bash
python scripts/prepare_examples.py
```

## ⚡ Dicas de performance

- Use `--scale 0.25` e `--skip 3` para vídeos longos ou máquinas modestas.
- O modelo `cnn` é bem mais preciso em rostos de perfil, mas exige GPU (dlib compilado com CUDA) para ser viável.
- Para bases grandes de pessoas, use o cache de encodings (`FaceEngine(..., cache_file="encodings.pkl")`) e evite reprocessar as fotos a cada execução.
- Diminua a `--tolerance` (ex.: `0.5`) para reduzir falsos positivos em ambientes com muitas pessoas parecidas.

## 🛠️ Tecnologias

| Tecnologia | Uso |
|---|---|
| [face_recognition](https://github.com/ageitgey/face_recognition) | Detecção, encodings faciais e matching (baseada em dlib) |
| [OpenCV](https://opencv.org/) | Leitura/gravação de imagens e vídeos, desenho das anotações, webcam |
| [NumPy](https://numpy.org/) | Operações vetoriais sobre os encodings |

## ⚖️ Uso responsável

Reconhecimento facial envolve dados biométricos. Ao usar este projeto em produção, obtenha consentimento das pessoas cadastradas, respeite a legislação aplicável (como a **LGPD** no Brasil) e evite aplicações que possam causar discriminação ou vigilância indevida.

## 📄 Licença

Distribuído sob a licença **MIT** — sinta-se à vontade para usar, estudar e adaptar. As fotos de exemplo são de domínio público (obras do governo dos EUA).

## 📬 Contato

**Leandro Miozzo Bonato**
📧 bonato16@gmail.com

---

<p align="center">⭐ Se este projeto foi útil, deixe uma estrela no repositório!</p>
