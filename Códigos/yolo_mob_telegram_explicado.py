
"""
================================================================================
DriveSafe Vision — Estação de Trabalho / Computador (yolo_mob_telegram_explicado.py)
================================================================================

DESCRIÇÃO GERAL DO SISTEMA:
---------------------------
Este script implementa a versão para computador (Desktop/Notebook) do sistema
DriveSafe Vision, permitindo avaliar e comparar o desempenho do pipeline híbrido
tanto ao vivo via webcam (USB/integrada) quanto sobre arquivos de vídeo gravados
(.mp4, .avi, etc.), além de permitir alternar entre os motores de inferência
TensorFlow Lite (`.tflite`) e Keras nativo (`.keras`) para a MobileNetV2.

A arquitetura opera em dois estágios sequenciais acoplados a uma máquina de
estados temporal:
  1. 1º Estágio — Detecção de Objetos (YOLOv8n via biblioteca Ultralytics):
     Localiza aparelhos celulares (classe 67 do dataset MS COCO) no quadro
     original (640x480). Caso nenhum celular seja detectado, o 2º estágio é
     poupado (curto-circuito computacional).
  2. 2º Estágio — Classificação Comportamental (MobileNetV2 em Keras ou TFLite):
     Acionado somente quando o YOLOv8n detecta um celular candidato. Recebe o
     quadro redimensionado para 256x256, convertido para RGB e normalizado via
     `preprocess_input`, estimando a probabilidade de uso ativo pelo motorista.
  3. Validação Temporal, Tolerância de Oclusão e Cooldown:
     - Exige persistência contínua mínima de uso (TEMPO_MINIMO_ALERTA = 2 s) antes
       de confirmar a infração e acionar o alerta remoto.
     - Tolera breves falhas de detecção de até TOLERANCIA_RESET = 2 s sem zerar
       o cronômetro de uso contínuo.
     - Aplica um intervalo mínimo (TEMPO_REPETICAO_ALERTA = 30 s) entre alertas
       sucessivos no Telegram para evitar sobrecarga de notificações.
  4. Notificação Remota e Auditoria:
     - Codifica o quadro da infração em JPEG diretamente na memória RAM e envia
       via requisição HTTP POST para a API REST do Telegram (`sendPhoto`).
     - Salva todos os quadros avaliados com sobreposição gráfica (HUD) em disco
       e exporta dois relatórios CSV (log quadro a quadro e resumo da execução).

PARIDADE COM O RASPBERRY PI:
----------------------------
Para garantir comparação justa de taxa de quadros por segundo (FPS) e de acurácia
com a versão embarcada (`telegram_RP_explicado.py`), este script força a captura
na mesma resolução de 640x480 pixels (via `cap.set` e fallback com `cv2.resize`).

EXEMPLOS DE USO:
----------------
Exemplo 1 — Executar na webcam padrão (índice 0) usando MobileNetV2 TFLite (padrão):
    $ export TELEGRAM_BOT_TOKEN="1234567890:AAH_seu_token_do_botfather_aqui"
    $ export TELEGRAM_CHAT_ID="987654321"
    $ python3 "/home/ingrid/TG/TG III/yolo_mob_telegram_explicado.py"

Exemplo 2 — Executar na webcam padrão (índice 0) usando o modelo Keras (.keras):
    $ python3 "/home/ingrid/TG/TG III/yolo_mob_telegram_explicado.py" --keras

Exemplo 3 — Executar sobre um arquivo de vídeo gravado usando TFLite:
    $ python3 "/home/ingrid/TG/TG III/yolo_mob_telegram_explicado.py" --video /caminho/para/video_teste.mp4

Exemplo 4 — Executar sobre um arquivo de vídeo gravado comparando com o modelo Keras:
    $ python3 "/home/ingrid/TG/TG III/yolo_mob_telegram_explicado.py" --video /caminho/para/video_teste.mp4 --keras

Exemplo 5 — Executar usando uma webcam USB secundária (índice 1):
    $ python3 "/home/ingrid/TG/TG III/yolo_mob_telegram_explicado.py" --video 1

Exemplo 6 — Encerramento seguro:
    Durante a exibição da janela "DriveSafe Vision - YOLO + MobileNet", pressione
    a tecla 'q' para encerrar o loop, liberar a câmera/vídeo e gerar os CSVs.
================================================================================
"""

# ==============================================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# ==============================================================================

import argparse                                                     # Leitura e interpretação de argumentos passados via linha de comando (--video, --keras)
import os                                                           # Manipulação de variáveis de ambiente, caminhos de arquivos e criação de pastas
import time                                                         # Medição de tempo (timestamps de eventos e cronometragem de inferência)
from collections import deque                                       # Fila circular de tamanho fixo usada na suavização de predições (média móvel)
import cv2                                                          # OpenCV: captura de vídeo/webcam, redimensionamento, desenho de HUD e codificação JPEG
import numpy as np                                                  # Operações numéricas e matriciais sobre os arrays de imagem e métricas
import pandas as pd                                                 # Estruturação tabular dos logs e geração dos arquivos CSV de saída
import requests                                                     # Cliente HTTP para envio de mensagens e fotos à API REST do Telegram
import tensorflow as tf                                             # Framework TensorFlow utilizado para carregar e executar o modelo Keras (.keras)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Função oficial de normalização da MobileNetV2 (reescala pixels para [-1, 1])
from ultralytics import YOLO                                        # Biblioteca Ultralytics para carregamento e inferência do detector YOLOv8n (.pt)


# ==============================================================================
# 2. CONFIGURAÇÕES GLOBAIS E HIPERPARÂMETROS DO SISTEMA
# ==============================================================================
# Centraliza as resoluções de imagem, caminhos de modelos e diretórios de saída,
# limiares de decisão, temporizadores de alerta e paleta de cores do HUD.

# --- Dimensões de Imagem e Resolução de Captura ---
IMG_SIZE = (256, 256)                                               # Resolução espacial (largura, altura) esperada na entrada da MobileNetV2
CAMERA_SIZE = (640, 480)                                            # Resolução da captura de vídeo igualada à do Raspberry Pi para comparação justa de FPS

# --- Caminhos dos Modelos e Diretórios de Saída ---
KERAS_PATH = "/home/ingrid/TG/TGII/RP/finetuned/mobilenet_finetuned_rasp.keras"            # Caminho do modelo MobileNetV2 ajustado salvo no formato nativo Keras
TFLITE_PATH = "/home/ingrid/TG/TGII/RP/finetuned/mobilenet_finetuned_rasp_float32.tflite"  # Caminho do mesmo modelo convertido para TensorFlow Lite (float32)
OUTPUT_PHONE = "/home/ingrid/TG/TGII/RP/resultados/phone"                                  # Pasta onde são gravados os quadros classificados como "com celular"
OUTPUT_NO_PHONE = "/home/ingrid/TG/TGII/RP/resultados/no_phone"                            # Pasta onde são gravados os quadros classificados como "sem celular"
CSV_VIDEO = "/home/ingrid/TG/TGII/RP/resultados/resultados_yolo_mob.csv"                   # Caminho do arquivo CSV com o registro detalhado quadro a quadro
CSV_RESUMO = "/home/ingrid/TG/TGII/RP/resultados/resumo_yolo_mob.csv"                      # Caminho do arquivo CSV com o resumo consolidado da execução

# --- Limiares de Decisão e Controle de Inferência ---
THRESHOLD = 0.05                                                    # Limiar de probabilidade da MobileNetV2 acima do qual se confirma "celular em uso"
YOLO_CONF = 0.10                                                    # Confiança mínima exigida na detecção do YOLOv8n para ser aceita como válida
YOLO_CLASS = 67                                                     # ID da classe "cell phone" no dataset MS COCO (utilizado pelo YOLOv8 pré-treinado)
PREDICT_EVERY = 3                                                   # Executa a inferência completa a cada 3 quadros capturados para otimizar o FPS
SMOOTH_WINDOW = 5                                                   # Tamanho da janela da média móvel usada para suavizar as predições da MobileNetV2

# --- Temporizadores da Lógica de Alerta e Persistência ---
TEMPO_MINIMO_ALERTA = 2                                             # Tempo mínimo (segundos) de uso contínuo do celular para disparar o primeiro alerta
TEMPO_REPETICAO_ALERTA = 30                                         # Tempo mínimo (segundos) de espera entre alertas consecutivos no Telegram (cooldown)
TOLERANCIA_RESET = 2                                                # Tempo (segundos) sem detecção tolerado antes de zerar o contador de uso contínuo

# --- Credenciais do Telegram (via Variáveis de Ambiente) ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")           # Token do bot do Telegram lido da variável de ambiente (ou "" se não definido)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")           # ID do chat destinatário no Telegram lido da variável de ambiente

# --- Configurações Visuais do HUD (Head-Up Display) ---
FONT = cv2.FONT_HERSHEY_SIMPLEX                                     # Fonte tipográfica usada no texto sobreposto (overlay) nos quadros de vídeo
COLOR_PHONE = (0, 0, 255)                                           # Cor vermelha em BGR para indicar detecção confirmada de celular
COLOR_NO_PHONE = (0, 200, 0)                                        # Cor verde em BGR para indicar ausência de celular (condução segura)
COLOR_YOLO = (0, 165, 255)                                          # Cor laranja em BGR quando o YOLO detecta objeto, mas a MobileNetV2 não confirma

# --- Criação Automática da Estrutura de Pastas ---
os.makedirs(OUTPUT_PHONE, exist_ok=True)                            # Cria a pasta de saída "phone" caso ela ainda não exista no disco
os.makedirs(OUTPUT_NO_PHONE, exist_ok=True)                         # Cria a pasta de saída "no_phone" caso ela ainda não exista no disco


# ==============================================================================
# 3. VERIFICAÇÃO INICIAL DE CREDENCIAIS DO TELEGRAM
# ==============================================================================

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:                      # Verifica se o token ou o chat_id estão ausentes no ambiente
    print(                                                          # Exibe aviso claro logo na inicialização para evitar falhas silenciosas
        "[TELEGRAM][AVISO] TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não definidos no ambiente. "
        "Alertas NÃO serão enviados até isso ser corrigido."
    )


# ==============================================================================
# 4. MÓDULO DE COMUNICAÇÃO COM A API DO TELEGRAM
# ==============================================================================
# Funções para envio de avisos textuais (`sendMessage`) e evidências visuais
# em formato JPEG codificadas diretamente em memória RAM (`sendPhoto`).

def enviar_telegram_texto(mensagem):
    """
    Envia uma mensagem de texto simples para o chat configurado no Telegram.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:                  # Sem credenciais configuradas, aborta o envio e informa no console
        print("[TELEGRAM] Token/Chat ID ausentes — mensagem NÃO enviada.")  # Aviso de aborto por falta de credenciais
        return                                                      # Encerra a função imediatamente

    try:                                                            # Bloco protegido contra erros de rede ou timeout
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"   # Monta a URL do endpoint sendMessage da API do Telegram
        resp = requests.post(                                       # Envia requisição HTTP POST para a API do Telegram
            url,                                                    # URL do endpoint
            data={                                                  # Dados do formulário HTTP
                "chat_id": TELEGRAM_CHAT_ID,                        # Destinatário da mensagem
                "text": mensagem,                                   # Texto a ser enviado
            },
            timeout=5,                                              # Timeout máximo de 5 segundos para não travar a aplicação
        )
        print(f"[TELEGRAM] Texto → status {resp.status_code}: {resp.text[:200]}")  # Exibe o status HTTP e início da resposta para diagnóstico
    except Exception as e:                                          # Captura qualquer falha de conexão HTTP
        print(f"[TELEGRAM] Erro ao enviar texto: {e}")              # Loga o erro no terminal sem interromper o programa


def enviar_telegram_foto(frame, caption=""):
    """
    Comprime o quadro atual como JPEG em memória RAM e envia como foto anexada
    acompanhada de legenda para o chat configurado no Telegram.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:                  # Verifica se as credenciais estão disponíveis antes de processar a imagem
        print("[TELEGRAM] Token/Chat ID ausentes — alerta NÃO enviado.")    # Avisa no terminal que o alerta não pôde ser enviado
        return                                                      # Retorna sem realizar a requisição

    try:                                                            # Inicia bloco de tentativa de codificação e envio da imagem
        _, buffer = cv2.imencode(".jpg", frame)                     # Codifica o array numpy do frame como JPEG em memória RAM (sem gravar em disco)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"     # Monta a URL do endpoint sendPhoto da API do Telegram
        resp = requests.post(                                       # Executa requisição HTTP POST multipart enviando a foto em bytes
            url,                                                    # Endpoint da API
            data={                                                  # Campos de dados da mensagem
                "chat_id": TELEGRAM_CHAT_ID,                        # ID do destinatário
                "caption": caption,                                 # Legenda contendo o tempo de uso contínuo
            },
            files={                                                 # Payload de arquivo binário
                "photo": ("frame.jpg", buffer.tobytes(), "image/jpeg")      # Tupla (nome do arquivo, bytes JPEG da RAM, tipo MIME)
            },
            timeout=10,                                             # Timeout de 10 segundos adequado para upload de imagem
        )
        if resp.status_code == 200:                                 # Status HTTP 200 indica sucesso no envio
            print("\n[TELEGRAM] Alerta enviado!")                   # Confirma o envio do alerta no console
        else:                                                       # Qualquer outro código HTTP indica falha na API (ex.: token inválido)
            print(f"\n[TELEGRAM] Falha ao enviar. Status {resp.status_code}: {resp.text[:200]}")  # Mostra o erro retornado
    except Exception as e:                                          # Captura erros de rede ou timeout durante o upload
        print(f"[TELEGRAM] Erro: {e}")                              # Exibe a exceção no terminal e mantém o loop rodando


# ==============================================================================
# 5. CARREGAMENTO DOS MODELOS (YOLOv8n, MOBILENETV2 KERAS E TFLITE)
# ==============================================================================
# Carrega previamente na memória os três modelos para permitir tanto a execução
# padrão em TFLite quanto a execução comparativa em Keras (--keras).

print("[INFO] Carregando YOLOv8n...")                               # Informa início do carregamento do detector de objetos
yolo_model = YOLO("yolov8n.pt")                                     # Carrega os pesos pré-treinados do YOLOv8 nano via biblioteca Ultralytics

print("[INFO] Carregando MobileNetV2 Keras...")                     # Informa início do carregamento do classificador em formato Keras
keras_model = tf.keras.models.load_model(KERAS_PATH)                # Carrega o grafo completo da MobileNetV2 fine-tuned (.keras)

print("[INFO] Carregando MobileNetV2 TFLite...")                    # Informa início do carregamento do classificador em formato TFLite
try:                                                                # Tenta importar o runtime TFLite mais moderno (ai_edge_litert)
    from ai_edge_litert.interpreter import Interpreter              # Importa o interpretador LiteRT atualizado
    interpreter = Interpreter(model_path=TFLITE_PATH)               # Instancia o interpretador com o modelo .tflite
except ImportError:                                                 # Caso ai_edge_litert não esteja instalado no ambiente Python:
    import tflite_runtime.interpreter as tflite                     # Usa como fallback a biblioteca tflite_runtime tradicional
    interpreter = tflite.Interpreter(model_path=TFLITE_PATH)        # Instancia o interpretador via tflite_runtime

interpreter.allocate_tensors()                                      # Aloca memória para os tensores de entrada e saída do modelo TFLite
input_details = interpreter.get_input_details()                     # Obtém metadados (índice, shape, tipo) do tensor de entrada TFLite
output_details = interpreter.get_output_details()                   # Obtém metadados do tensor de saída TFLite
print("[INFO] Todos os modelos carregados.\n")                      # Confirma que os modelos estão prontos na memória


# ==============================================================================
# 6. FUNÇÕES DE PRÉ-PROCESSAMENTO, INFERÊNCIA E OVERLAY (HUD)
# ==============================================================================

def preprocess_frame(frame_bgr):
    """
    Pré-processa um quadro BGR capturado pelo OpenCV para alimentar a MobileNetV2
    (compatível tanto com a versão Keras quanto com a versão TFLite):
      1. Redimensiona para 256x256 pixels;
      2. Converte de BGR para RGB;
      3. Aplica `preprocess_input` oficial da MobileNetV2 (escala [-1, +1]);
      4. Expande a dimensão de batch para (1, 256, 256, 3).
    """
    img = cv2.resize(frame_bgr, IMG_SIZE)                           # Redimensiona o quadro para 256x256 pixels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                      # Converte a ordem de canais de BGR (OpenCV) para RGB (MobileNetV2)
    arr = preprocess_input(img.astype(np.float32))                  # Converte para float32 e normaliza os pixels para o intervalo [-1.0, +1.0]
    return np.expand_dims(arr, axis=0)                              # Adiciona a dimensão de batch na posição 0 -> shape (1, 256, 256, 3)


def inferir_keras(arr):
    """
    Executa a inferência usando o modelo MobileNetV2 no backend Keras/TensorFlow.
    """
    return float(keras_model.predict(arr, verbose=0)[0][0])         # Faz a predição sem barra de progresso (verbose=0) e retorna o escalar float


def inferir_tflite(arr):
    """
    Executa a inferência usando o modelo MobileNetV2 no interpretador TFLite.
    """
    interpreter.set_tensor(input_details[0]["index"], arr)          # Copia o array pré-processado para o tensor de entrada do TFLite
    interpreter.invoke()                                            # Dispara a execução da inferência no interpretador TFLite
    return float(interpreter.get_tensor(output_details[0]["index"])[0][0])  # Lê o tensor de saída e retorna a probabilidade escalar float


def detectar_yolo(frame):
    """
    Executa o 1º estágio (YOLOv8n) diretamente sobre o quadro BGR original.
    Verifica se alguma caixa detectada pertence à classe 67 ("cell phone")
    com confiança igual ou superior a YOLO_CONF.
    """
    results = yolo_model(frame, verbose=False)[0]                   # Executa o YOLOv8 no quadro (verbose=False silencia logs) e pega o 1º resultado
    for box in results.boxes:                                       # Percorre todas as caixas delimitadoras (bounding boxes) detectadas no quadro
        if int(box.cls[0]) == YOLO_CLASS and float(box.conf[0]) >= YOLO_CONF:  # Checa se a classe é 67 (celular) e a confiança >= 0.10
            return True, float(box.conf[0])                         # Retorna True e a confiança assim que encontra o primeiro celular válido
    return False, 0.0                                               # Retorna False e confiança 0.0 se nenhum celular for detectado


def overlay(frame, label, pred, color, yolo_conf=0.0, tempo_uso=0.0):
    """
    Desenha o painel de telemetria (HUD) no canto superior esquerdo do quadro,
    incluindo fundo preto para legibilidade, rótulo de estado, probabilidade da
    MobileNetV2, confiança do YOLOv8n e tempo de uso contínuo.
    """
    linha1 = f"{label} | MobileNet: {pred:.2f}"                     # Monta a 1ª linha com o status atual e o score da MobileNetV2
    linha2 = f"YOLO: {yolo_conf:.2f}" if yolo_conf > 0 else "YOLO: ---"          # Monta a 2ª linha com a confiança do YOLO (ou "---")
    linha3 = f"uso continuo: {tempo_uso:.0f}s" if tempo_uso > 0 else ""          # Monta a 3ª linha com o tempo contínuo em segundos (se > 0)
    linhas = [linha1, linha2] + ([linha3] if linha3 else [])        # Concatena as linhas ativas em uma lista

    for i, texto in enumerate(linhas):                              # Itera sobre cada linha a ser desenhada na tela
        (tw, th), _ = cv2.getTextSize(texto, FONT, 0.7, 2)          # Obtém largura (tw) e altura (th) do texto em pixels
        y = 10 + i * (th + 10)                                      # Calcula a posição vertical Y da linha atual com margem de 10 px
        cv2.rectangle(frame, (8, y), (14 + tw, y + th + 4), (0, 0, 0), -1)      # Desenha retângulo preto sólido como fundo do texto
        cv2.putText(frame, texto, (10, y + th), FONT, 0.7, color, 2, cv2.LINE_AA)  # Escreve o texto colorido com anti-aliasing (LINE_AA)

    return frame                                                    # Retorna o quadro anotado


# ==============================================================================
# 7. LOOP PRINCIPAL DE PROCESSAMENTO DE VÍDEO E DECISÃO EM CASCATA
# ==============================================================================

def run_video(video_path, usar_keras=False):
    """
    Executa o fluxo completo de leitura de vídeo/webcam, inferência em dois
    estágios, validação temporal de alertas no Telegram e gravação de CSVs.

    Parâmetros:
      video_path (int | str): Índice numérico da webcam (ex.: 0) ou caminho de arquivo de vídeo.
      usar_keras (bool): Se True, usa MobileNetV2 Keras; se False, usa MobileNetV2 TFLite.
    """
    cap = cv2.VideoCapture(video_path)                              # Inicializa a captura de vídeo do OpenCV (webcam ou arquivo)
    if not cap.isOpened():                                          # Verifica se a fonte de vídeo foi aberta corretamente
        print(f"[ERRO] Não foi possível abrir: {video_path}")       # Informa erro caso o arquivo não exista ou a câmera esteja ocupada
        return                                                      # Encerra a execução da função

    # Solicita ao driver da câmera a mesma resolução utilizada no Raspberry Pi (640x480).
    # Como algumas webcams ignoram cap.set(), o código também garante o tamanho via cv2.resize() no loop.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])               # Solicita largura de captura de 640 pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])              # Solicita altura de captura de 480 pixels

    fps = cap.get(cv2.CAP_PROP_FPS) or 30                           # Lê a taxa de FPS nominal da fonte de vídeo (assume 30 caso retorne 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                  # Lê a largura efetiva reportada pela captura
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                # Lê a altura efetiva reportada pela captura
    modelo = "Keras" if usar_keras else "TFLite"                    # Define string identificadora do backend escolhido para logs e CSV

    print(f"[VÍDEO] {width}x{height} @ {fps:.1f}fps  |  YOLO + {modelo}")        # Exibe resumo da configuração de vídeo e modelo ativo
    print(f"Alerta após {TEMPO_MINIMO_ALERTA}s | Tolerância reset: {TOLERANCIA_RESET}s")  # Exibe parâmetros temporais configurados
    print("Pressione 'q' para sair.\n")                             # Instrução de encerramento para o usuário

    enviar_telegram_texto("🚗 DriveSafe Vision iniciado")           # Envia notificação de início de monitoramento para o Telegram

    # --- Inicialização dos Contadores e Variáveis de Estado ---
    frame_count = 0                                                 # Contador total de quadros lidos da fonte de vídeo
    phone_count = 0                                                 # Contador de quadros classificados como "com celular"
    no_phone_count = 0                                              # Contador de quadros classificados como "sem celular"
    yolo_only_count = 0                                             # Contador de quadros em que o YOLO detectou, mas a MobileNetV2 rejeitou

    pred_buffer = deque(maxlen=SMOOTH_WINDOW)                       # Fila circular (tamanho 5) para cálculo da média móvel das predições
    last_pred = 0.0                                                 # Última probabilidade bruta da MobileNetV2 para exibição contínua no HUD
    last_yolo_conf = 0.0                                            # Última confiança do YOLO para exibição contínua no HUD
    last_label = "SEM CELULAR"                                      # Último rótulo de estado exibido
    last_color = COLOR_NO_PHONE                                     # Última cor de alerta usada no HUD

    tempos = []                                                     # Lista para armazenar a duração (em segundos) de cada ciclo de inferência
    registros = []                                                  # Lista de dicionários com os logs de cada quadro avaliado (para o CSV)

    inicio_mobilenet = None                                         # Timestamp de início do uso contínuo atual (None quando não há uso)
    ultimo_alerta = None                                            # Timestamp do último alerta enviado ao Telegram (controle de cooldown)
    ultimo_sem_celular = None                                       # Timestamp do início de ausência de detecção (controle da tolerância de reset)
    tempo_uso_continuo = 0.0                                        # Tempo acumulado (em segundos) de uso contínuo do celular

    while True:                                                     # Loop principal de leitura de quadros
        ret, frame = cap.read()                                     # Lê o próximo quadro da webcam ou arquivo de vídeo
        if not ret:                                                 # Se `ret` for False, o vídeo chegou ao fim ou a câmera desconectou
            print("Fim do vídeo.")                                  # Informa término do fluxo de vídeo no terminal
            break                                                   # Sai do loop principal

        if (frame.shape[1], frame.shape[0]) != CAMERA_SIZE:         # Verifica se a resolução real do quadro difere de 640x480
            frame = cv2.resize(frame, CAMERA_SIZE)                  # Redimensiona o quadro para 640x480 garantindo paridade com o Raspberry Pi

        agora = time.time()                                         # Captura o timestamp atual (em segundos) para os cálculos temporais

        # ----------------------------------------------------------------------
        # ETAPA DE INFERÊNCIA EM CASCATA (EXECUTADA A CADA `PREDICT_EVERY` FRAMES)
        # ----------------------------------------------------------------------
        if frame_count % PREDICT_EVERY == 0:                        # Executa a inferência apenas quando o índice do frame é múltiplo de 3
            t0 = time.perf_counter()                                # Marca o instante inicial da inferência com alta resolução

            yolo_ok, yolo_conf = detectar_yolo(frame)               # 1º Estágio: roda o YOLOv8n no quadro original

            if yolo_ok:                                             # Se o YOLOv8n detectou um possível celular:
                arr = preprocess_frame(frame)                       # Pré-processa o quadro (256x256, RGB, normalizado em [-1, 1])
                p = inferir_keras(arr) if usar_keras else inferir_tflite(arr)   # 2º Estágio: executa MobileNetV2 (Keras ou TFLite)
            else:                                                   # Se o YOLOv8n NÃO detectou celular:
                p = 0.0                                             # Não executa a MobileNetV2 e define probabilidade zero diretamente

            t1 = time.perf_counter()                                # Marca o instante final da inferência
            tempos.append(t1 - t0)                                  # Registra o tempo total gasto neste ciclo de inferência (segundos)

            pred_buffer.append(p)                                   # Adiciona a probabilidade atual à janela móvel de suavização
            pred_media = float(np.mean(pred_buffer))                # Calcula a média móvel das últimas predições

            # ------------------------------------------------------------------
            # CASO A: USO DE CELULAR CONFIRMADO PELOS DOIS ESTÁGIOS
            # ------------------------------------------------------------------
            if yolo_ok and p > THRESHOLD:                           # Se o YOLO detectou celular E a MobileNetV2 confirmou acima do limiar:
                classe = "phone"                                    # Rótulo da classe para o CSV
                last_label = "CELULAR"                              # Texto exibido no HUD
                last_color = COLOR_PHONE                            # Cor vermelha no HUD
                last_yolo_conf = yolo_conf                          # Atualiza confiança do YOLO para exibição
                phone_count += 1                                    # Incrementa contador de quadros com celular confirmado

                fname = os.path.join(OUTPUT_PHONE, f"frame_{frame_count:05d}.jpg")  # Caminho para salvar o quadro na pasta "phone"
                ultimo_sem_celular = None                           # Reseta o marcador de ausência, pois o celular está presente

                if inicio_mobilenet is None:                        # Caso seja o início de um novo período de uso:
                    inicio_mobilenet = agora                        # Registra o timestamp inicial do uso contínuo

                tempo_uso_continuo = agora - inicio_mobilenet       # Calcula há quantos segundos o celular está sendo utilizado continuamente

                cooldown_ok = (                                     # Verifica se o intervalo mínimo de repetição (30s) já foi cumprido:
                    ultimo_alerta is None                           # Verdadeiro se for o primeiro alerta da sessão, OU
                    or (agora - ultimo_alerta) >= TEMPO_REPETICAO_ALERTA  # se já passaram 30 segundos desde o último envio
                )

                if tempo_uso_continuo >= TEMPO_MINIMO_ALERTA and cooldown_ok:   # Se o uso dura >= 2s E o cooldown está liberado:
                    ultimo_alerta = agora                           # Atualiza o timestamp do último alerta enviado
                    minutos = int(tempo_uso_continuo // 60)         # Calcula a parte inteira em minutos do tempo de uso
                    segundos = int(tempo_uso_continuo % 60)         # Calcula os segundos restantes
                    caption = (                                     # Formata a legenda do alerta para o Telegram
                        "⚠️ ALERTA: USO DE CELULAR AO VOLANTE!\n"
                        f"Tempo de uso contínuo: {minutos}min {segundos}s"
                    )
                    enviar_telegram_foto(frame, caption)            # Envia o quadro atual compactado em memória RAM para o Telegram

            # ------------------------------------------------------------------
            # CASO B: SEM USO DE CELULAR CONFIRMADO
            # ------------------------------------------------------------------
            else:                                                   # Caso o YOLO não tenha detectado celular OU a MobileNetV2 tenha rejeitado
                if yolo_ok and p <= THRESHOLD:                      # Subcaso B1: YOLO detectou candidato, mas a MobileNetV2 não confirmou
                    classe = "no_phone"                             # Classifica como sem celular no CSV
                    last_label = "SEM CELULAR"                      # Rótulo de condução segura
                    last_color = COLOR_YOLO                         # Cor laranja no HUD (indica detecção filtrada pelo 2º estágio)
                    last_yolo_conf = yolo_conf                      # Mantém visível a confiança do YOLO
                    yolo_only_count += 1                            # Contabiliza falso positivo filtrado pela MobileNetV2
                    no_phone_count += 1                             # Contabiliza quadro sem celular confirmado
                else:                                               # Subcaso B2: YOLO não detectou nenhum celular na cena
                    classe = "no_phone"                             # Classifica como sem celular no CSV
                    last_label = "SEM CELULAR"                      # Rótulo de condução segura
                    last_color = COLOR_NO_PHONE                     # Cor verde no HUD
                    last_yolo_conf = 0.0                            # Confiança zero no HUD
                    no_phone_count += 1                             # Contabiliza quadro sem celular

                fname = os.path.join(OUTPUT_NO_PHONE, f"frame_{frame_count:05d}.jpg")  # Caminho para salvar o quadro na pasta "no_phone"

                if ultimo_sem_celular is None:                      # Se este é o primeiro quadro sem detecção após uso:
                    ultimo_sem_celular = agora                      # Marca o início do período de ausência

                tempo_ausencia = agora - ultimo_sem_celular         # Calcula a duração acumulada da ausência de detecção
                if tempo_ausencia >= TOLERANCIA_RESET:              # Se a ausência durar 2 segundos ou mais:
                    inicio_mobilenet = None                         # Reseta o marcador de início de uso contínuo
                    tempo_uso_continuo = 0.0                        # Zera o contador de uso contínuo (mantendo `ultimo_alerta` para respeitar o cooldown)

            last_pred = p                                           # Atualiza a última predição para exibição contínua no HUD

            # --- Gravação do Quadro Anotado e Registro no Log ---
            frame_out = overlay(                                    # Desenha as informações de telemetria em uma cópia do quadro
                frame.copy(),
                last_label,
                last_pred,
                last_color,
                last_yolo_conf,
                tempo_uso_continuo,
            )
            cv2.imwrite(fname, frame_out)                           # Salva o quadro anotado no diretório correspondente ("phone" ou "no_phone")

            registros.append({                                      # Registra todas as variáveis do quadro atual na lista para exportação CSV
                "frame": frame_count,                               # Índice do quadro
                "modelo": f"YOLO+{modelo}",                         # Identificador do pipeline (YOLO+TFLite ou YOLO+Keras)
                "yolo_detectou": yolo_ok,                           # Booleano indicando detecção pelo YOLOv8n
                "yolo_conf": round(yolo_conf, 4),                   # Confiança da detecção do YOLOv8n
                "pred_bruta": round(p, 4),                          # Probabilidade bruta da MobileNetV2
                "pred_media": round(pred_media, 4),                 # Probabilidade suavizada (média móvel)
                "predicao": classe,                                 # Classe atribuída ("phone" ou "no_phone")
                "tempo_uso_s": round(tempo_uso_continuo, 1),        # Tempo de uso contínuo acumulado (segundos)
                "tempo_ms": round((t1 - t0) * 1000, 2),             # Tempo de inferência do ciclo atual (milissegundos)
                "threshold_mob": THRESHOLD,                         # Limiar utilizado na MobileNetV2
                "threshold_yolo": YOLO_CONF,                        # Limiar utilizado no YOLOv8n
            })

            print(                                                  # Imprime status atualizado na mesma linha do terminal (end="\r")
                f"YOLO={yolo_ok}({yolo_conf:.2f}) mob={p:.4f} uso={tempo_uso_continuo:.0f}s → {last_label}",
                end="\r",
            )

        # ----------------------------------------------------------------------
        # EXIBIÇÃO EM TEMPO REAL NA JANELA GRÁFICA (RODA EM TODOS OS QUADROS)
        # ----------------------------------------------------------------------
        frame_display = overlay(                                    # Aplica o HUD sobre o quadro atual para exibição em vídeo fluido
            frame,
            last_label,
            last_pred,
            last_color,
            last_yolo_conf,
            tempo_uso_continuo,
        )
        cv2.imshow("DriveSafe Vision - YOLO + MobileNet", frame_display)  # Exibe o quadro na janela do OpenCV
        frame_count += 1                                            # Incrementa o contador global de quadros

        if cv2.waitKey(1) & 0xFF == ord("q"):                       # Verifica a cada 1 ms se o usuário pressionou a tecla 'q'
            print("\nEncerrado pelo usuário.")                      # Informa encerramento manual no terminal
            break                                                   # Interrompe o loop principal

    # ==========================================================================
    # 8. LIBERAÇÃO DE RECURSOS E EXPORTAÇÃO DOS RELATÓRIOS CSV
    # ==========================================================================

    cap.release()                                                   # Libera o dispositivo de captura de vídeo ou fecha o arquivo
    cv2.destroyAllWindows()                                         # Fecha todas as janelas abertas pelo OpenCV

    pd.DataFrame(registros).to_csv(CSV_VIDEO, index=False)          # Salva o histórico detalhado quadro a quadro em arquivo CSV
    print(f"\nCSV detalhado salvo em: {CSV_VIDEO}")                 # Exibe no console o caminho do CSV detalhado

    enviar_telegram_texto("✅ DriveSafe Vision encerrado")          # Envia notificação de encerramento ao Telegram

    if tempos:                                                      # Se houve ao menos um ciclo de inferência executado:
        tempos_ms = np.array(tempos) * 1000                         # Converte os tempos de inferência de segundos para milissegundos
        tempo_total = float(np.sum(tempos_ms))                      # Calcula o tempo total acumulado em inferências (em ms)
        fps_efetivo = frame_count / (tempo_total / 1000) if tempo_total > 0 else 0  # Calcula o FPS efetivo da execução
    else:                                                           # Caso nenhum quadro tenha passado pela inferência:
        fps_efetivo = 0                                             # Define FPS efetivo como zero para evitar divisão por zero

    print("\n==============================")                       # Cabeçalho do resumo no terminal
    print("RESUMO")                                                 # Título do resumo
    print("==============================")                         # Separador visual
    print(f"Modelo              : YOLO + {modelo}")                 # Mostra qual variante foi executada (TFLite ou Keras)
    print(f"Frames processados  : {frame_count}")                   # Total de quadros lidos
    print(f"Frames com celular  : {phone_count}  ({100 * phone_count / max(1, frame_count):.1f}%)")        # Total e % com celular
    print(f"Frames sem celular  : {no_phone_count}  ({100 * no_phone_count / max(1, frame_count):.1f}%)")  # Total e % sem celular
    print("------------------------------")                         # Linha divisória
    print(f"FPS efetivo         : {fps_efetivo:.1f}")               # Exibe a taxa de quadros por segundo efetiva
    print("==============================")                         # Rodapé do resumo

    pd.DataFrame([{                                                 # Cria um DataFrame de linha única com as métricas consolidadas da sessão
        "modelo": f"YOLO+{modelo}",                                 # Modelo utilizado
        "frames_processados": frame_count,                          # Total de frames processados
        "frames_celular": phone_count,                              # Total de frames com celular confirmado
        "frames_sem_celular": no_phone_count,                       # Total de frames sem celular
        "yolo_sem_confirmacao": yolo_only_count,                    # Falsos positivos do YOLO barrados pela MobileNetV2
        "pct_celular": round(100 * phone_count / max(1, frame_count), 1),        # Percentual de quadros com celular
        "pct_sem_celular": round(100 * no_phone_count / max(1, frame_count), 1), # Percentual de quadros sem celular
        "fps_efetivo": round(fps_efetivo, 1),                       # FPS efetivo alcançado
        "threshold_mobilenet": THRESHOLD,                           # Limiar da MobileNetV2 utilizado
        "threshold_yolo": YOLO_CONF,                                # Limiar do YOLOv8n utilizado
    }]).to_csv(CSV_RESUMO, index=False)                             # Salva o arquivo CSV de resumo
    print(f"CSV resumo salvo em: {CSV_RESUMO}")                     # Confirma o caminho do CSV de resumo salvo


# ==============================================================================
# 9. INTERPRETAÇÃO DE ARGUMENTOS CLI E PONTO DE ENTRADA (ENTRY POINT)
# ==============================================================================

def main():
    """
    Lê os argumentos de linha de comando (`--video` e `--keras`) e inicia
    a execução do pipeline DriveSafe Vision.
    """
    parser = argparse.ArgumentParser(                               # Instancia o analisador de argumentos de linha de comando
        description="DriveSafe Vision (PC) — Detecção de uso de celular ao volante com YOLOv8n + MobileNetV2"
    )
    parser.add_argument(                                            # Define o argumento opcional --video
        "--video",
        type=str,
        default="0",
        help="Índice da webcam (ex.: '0') ou caminho para um arquivo de vídeo (ex.: 'teste.mp4')",
    )
    parser.add_argument(                                            # Define a flag booleana opcional --keras
        "--keras",
        action="store_true",
        help="Utilizar o modelo MobileNetV2 no formato Keras (.keras) em vez de TFLite (.tflite)",
    )
    args = parser.parse_args()                                      # Processa os argumentos informados pelo usuário no terminal

    source = int(args.video) if args.video.isdigit() else args.video  # Converte para inteiro se for índice de webcam ("0" -> 0) ou mantém string para arquivo
    run_video(source, usar_keras=args.keras)                        # Inicia o loop principal com a fonte de vídeo e o backend selecionados


if __name__ == "__main__":                                          # Garante que main() só seja chamada quando o script for executado diretamente
    main()                                                          # Executa a função principal
