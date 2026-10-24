# 🚗📱 DriveSafe Vision — Detecção Embarcada de Uso de Celular na Direção Veicular

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-YOLOv8n-blueviolet)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-MobileNetV2-orange?logo=tensorflow&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%204-C51A4A?logo=raspberrypi&logoColor=white)
![Telegram API](https://img.shields.io/badge/Alerts-Telegram%20Bot%20API-26A5E4?logo=telegram&logoColor=white)

O **DriveSafe Vision** é um sistema embarcado de Visão Computacional e Aprendizado Profundo (*Edge AI*) projetado para detectar, automaticamente e em tempo real, o uso indevido de telefone celular por motoristas durante a condução veicular.

Desenvolvido como Trabalho de Graduação em **Engenharia de Informação** na **Universidade Federal do ABC (UFABC)**, o sistema opera de forma autônoma em um computador de placa única **Raspberry Pi 4 Model B** acoplado a uma câmera dedicada **OV5647 (MIPI CSI)**, utilizando uma arquitetura híbrida em dois estágios (**YOLOv8n + MobileNetV2**) e enviando alertas fotográficos instantâneos para a gestão de frota via **API do Telegram**, sem necessidade de gravação contínua de vídeo.

---

## 🎯 Principais Funcionalidades

- **Arquitetura Híbrida em Dois Estágios (Zero Falsos Positivos):**
  - **1º Estágio (Filtro Espacial — YOLOv8n via ONNX Runtime):** localiza espacialmente aparelhos celulares na cena (classe 67 do conjunto MS COCO), descartando objetos visualmente semelhantes empunhados pelo motorista (garrafas d'água, copos, embalagens, óculos).
  - **2º Estágio (Validador Contextual — MobileNetV2 via TensorFlow Lite):** acionado condicionalmente para analisar a postura do motorista e confirmar o uso efetivo do celular ao volante.
- **Otimização para Computação de Borda (*Edge AI*):**
  - Conversão do classificador para grafo estático **TensorFlow Lite (`float32`)**, proporcionando ganho de velocidade de **~6× em CPU** (de 15,76 FPS para 93,08 FPS em computador) sem qualquer perda de exatidão numérica (`divergência = 0,0000`).
  - Execução do detector de objetos via **ONNX Runtime**, eliminando a sobrecarga de memória dos frameworks de treinamento (PyTorch/TensorFlow) no Raspberry Pi 4.
- **Processamento Digital de Imagens com OpenCV:**
  - Captura padronizada em $640 \times 480$ pixels (menor modo físico nativo *RAW Bayer* de 10 bits do sensor OV5647 a 62,50 FPS);
  - Correção cromática ($\text{BGR} \to \text{RGB}$), reamostragem por interpolação bilinear ($640 \times 640$ e $256 \times 256$) e normalização radiométrica ($[0, 1]$ e $[-1, 1]$);
  - Módulo opcional de remoção/substituição de fundo por segmentação de silhueta humana para ensaios em bancada;
  - Renderização de barra de status dinâmica, caixas delimitadoras e cronômetro de infração.
- **Máquina de Estados Temporal e Alertas via Telegram:**
  - **Persistência mínima de uso ($2,0\text{ s}$):** filtra movimentos rápidos e naturais da condução;
  - **Tolerância a oclusões breves ($2,0\text{ s}$):** evita zerar o cronômetro durante rápidas mudanças de ângulo da mão;
  - **Intervalo de resfriamento (*cooldown* de $30,0\text{ s}$):** evita sobrecarga de notificações repetidas e respeita os limites da API do Telegram;
  - **Envio em memória RAM:** comprime o quadro da evidência em JPEG diretamente em buffer de memória volátil e envia via HTTPS para o Telegram, preservando o cartão microSD e a privacidade do condutor.

---

## 🧠 Arquitetura do Sistema

```mermaid
flowchart LR
    A["Câmera Embarcada\n(640x480 px)"] --> B["Pré-processamento OpenCV\n(BGR->RGB, Resize, Norm.)"]
    B --> C{"1º Estágio: YOLOv8n\n(ONNX Runtime)\nCelular detectado?"}
    C -- Não --> D["Status: SEM CELULAR\n(Condução Segura)"]
    C -- Sim --> E{"2º Estágio: MobileNetV2\n(TFLite float32)\nProbabilidade > 0.05?"}
    E -- Não --> D
    E -- Sim --> F{"Persistência Temporal\nUso contínuo >= 2.0s\ne fora de Cooldown (30s)?"}
    F -- Sim --> G["Compressão JPEG em RAM\n+ Envio via API Telegram"]
```

---

## 📊 Principais Resultados Experimentais

### 1. Comparativo das 8 Arquiteturas de *Deep Learning* (Teste 1 — CPU)
Todas as oito arquiteturas foram treinadas/ajustadas por até 80 épocas sobre o conjunto *Multi-Class Driver Behavior Image Dataset* (5.726 imagens divididas em 70% treino, 20% validação e 10% teste):

| Pos. | Modelo | Paradigma | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | Latência CPU (ms/img) | FPS (CPU) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1º | **RetinaNet** | Detector (1 estágio) | 99,65 | 99,65 | 99,65 | 99,65 | 79,4 | 12,6 |
| 1º | **ResNet50** | Classificador | 99,65 | 99,65 | 99,65 | 99,65 | 103,7 | 9,6 |
| **3º** | **MobileNetV2** ⭐ | **Classificador** | **99,30** | **99,30** | **99,30** | **99,30** | **68,4** | **14,6** |
| 4º | **Faster R-CNN** | Detector (2 estágios) | 98,26 | 99,29 | 97,21 | 98,24 | 251,7 | 3,9 |
| 5º | **AlexNet** | Classificador | 98,08 | 97,26 | 98,95 | 98,10 | 63,3 | 15,8 |
| 6º | **VGG16** | Classificador | 97,91 | 97,58 | 98,26 | 97,92 | 132,8 | 7,5 |
| 6º | **SSD300** | Detector (1 estágio) | 97,91 | 98,26 | 97,58 | 97,92 | 235,3 | 4,2 |
| **8º** | **YOLOv8n** ⭐ | **Detector (1 estágio)** | **84,67** | **79,53** | **93,38** | **85,90** | **51,2** | **19,5** |

> ⭐ **Escolha de Engenharia:** O **YOLOv8n** ($\sim 3,2\text{M}$ parâmetros, rede mais rápida em CPU) foi selecionado como localizador espacial de 1º estágio e a **MobileNetV2** ($\sim 3,4\text{M}$ parâmetros, 99,30% de F1-Score) como validadora contextual de 2º estágio.

### 2. Otimização Keras vs. TensorFlow Lite (32 bits)
| Métrica | Modelo Keras (`.keras`) | Modelo TFLite (`.tflite` float32) | Ganho |
|:---|:---:|:---:|:---:|
| Tempo médio de inferência | 62,89 ms | **9,93 ms** | **6,3× mais rápido** |
| Ciclo total por quadro | 63,72 ms | **10,74 ms** | **-83,1% na latência** |
| Taxa de quadros (FPS) | 15,76 FPS | **93,08 FPS** | **5,9× maior vazão** |
| Divergência numérica | Referência | **0,0000** | **Zero perda de precisão** |

### 3. Validação Cruzada entre Sensores Ópticos (Webcam vs. Câmera CSI Raspberry Pi)
Avaliação sobre **908 imagens reais anotadas manualmente** (`ImgWC` com 457 quadros de webcam e `ImgRP` com 451 quadros da câmera OV5647):

| Conjunto + Formato | Total | VP | VN | FP | FN | Acurácia (%) | Precisão (%) | Recall (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `ImgWC` (Webcam) + Keras / TFLite | 457 | 298 | 138 | **0** | 21 | **95,40%** | **100,00%** | 93,42% |
| `ImgRP` (Raspberry Pi CSI) + Keras / TFLite | 451 | 91 | 350 | **0** | 10 | **97,78%** | **100,00%** | 90,10% |

---

## 🛠️ Requisitos de Hardware e Software

### Hardware Utilizado na Versão Embarcada
- **Placa:** Raspberry Pi 4 Model B (SoC Broadcom BCM2711, CPU Quad-Core ARM Cortex-A72 @ 1,8 GHz, 4 GB RAM LPDDR4)
- **Sistema Operacional:** Raspberry Pi OS 64-bit (Debian 12 *Bookworm*)
- **Câmera:** Módulo Câmera CSI 5 MP — Sensor CMOS OmniVision OV5647 ($f/1,8$, $3,6\text{ mm}$, resolução nativa configurada em $640 \times 480$)
- **Armazenamento:** Cartão microSD de 32 GB

### Principais Dependências de Software
- `Python >= 3.10` (compatível com `Python 3.13` no Raspberry Pi OS)
- `opencv-python`
- `numpy`
- `onnxruntime`
- `ai-edge-litert` / `tflite-runtime` / `tensorflow`
- `requests`
- `picamera2` (para captura via barramento MIPI CSI no Raspberry Pi)

---

## 🚀 Instalação e Configuração

### 1. Clonar o repositório
```bash
git clone https://github.com/SEU-USUARIO/drivesafe-vision.git
cd drivesafe-vision
```

### 2. Instalar as dependências
No **Computador (PC / Notebook)**:
```bash
pip install opencv-python numpy onnxruntime tensorflow requests ultralytics
```

No **Raspberry Pi 4 (Raspberry Pi OS Bookworm)**:
```bash
sudo apt update && sudo apt install -y python3-picamera2 python3-opencv
pip install numpy onnxruntime ai-edge-litert requests
```

### 3. Configurar as variáveis de ambiente do Telegram
Crie um bot pelo [@BotFather](https://t.me/BotFather) no Telegram e exporte seu token e o ID do chat corporativo no terminal (ou no `~/.bashrc`):

```bash
export TELEGRAM_TOKEN="SEU_TOKEN_DO_BOT_AQUI"
export TELEGRAM_CHAT_ID="SEU_CHAT_ID_AQUI"
```

---

## ▶️ Como Executar

### Execução Embarcada no Raspberry Pi 4 (YOLOv8 ONNX + MobileNetV2 TFLite + Câmera CSI + Telegram)
```bash
python3 telegram_RP.py
```

### Execução em Ambiente de Bancada / PC (Webcam + Alertas via Telegram)
```bash
python3 telegram.py
```

> Pressione a tecla **`q`** na janela de visualização para encerrar a sessão de monitoramento (o sistema enviará automaticamente a notificação de encerramento ao Telegram e salvará o resumo estatístico em CSV).

---

## 👩‍💻 Autoria e Orientação

- **Autora:** Ingrid Mara Xavier
- **Orientador:** Prof. Dr. Celso Setsuo Kurashima
- **Coorientador:** Prof. Dr. Ricardo Suyama
- **Instituição:** Universidade Federal do ABC (UFABC) — Centro de Engenharia, Modelagem e Ciências Sociais Aplicadas (CECS) — Engenharia de Informação
- **Ano:** 2026


Link do site: https://ingridmxavier.github.io/TG/
