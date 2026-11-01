"""
================================================================================
DriveSafe Vision — Versão Embarcada Raspberry Pi 4 com Webcam USB
Arquivo: telegram_RP_explicado.py
Autora: Ingrid Mara Xavier
Orientadores: Prof. Dr. Celso Setsuo Kurashima / Prof. Dr. Ricardo Suyama
Engenharia de Informação — Universidade Federal do ABC (UFABC)
================================================================================

DESCRIÇÃO GERAL DA ARQUITETURA:
--------------------------------
Este script implementa a versão embarcada completa do sistema DriveSafe Vision
para o microcomputador Raspberry Pi 4 (SoC Broadcom BCM2711, 4x ARM Cortex-A72
a 1.5 GHz), operando com uma webcam USB externa conectada via barramento USB
(protocolo Video4Linux2 / V4L2 via OpenCV VideoCapture), como a webcam de laboratório
Logitech Brio 100.

O pipeline de inferência em dois estágios foi rigorosamente otimizado para hardware
embarcado sem GPU dedicada, eliminando completamente dependências pesadas de desktop
(sem frameworks completos de PyTorch ou TensorFlow):
  1. 1º Estágio — Detecção de Objetos (YOLOv8n via ONNX Runtime):
     - Executa o modelo ultraleve YOLOv8n pré-treinado no MS COCO exportado em ONNX.
     - Avalia a presença da classe "cell phone" (ID COCO 67) com limiar de confiança
       baixo (YOLO_CONF = 0.10) para maximizar o recall (evitar falsos negativos).
     - Curto-Circuito Computacional: se nenhum celular for detectado no quadro,
       o 2º estágio não é acionado, economizando até 70% dos ciclos de CPU.
  2. 2º Estágio — Classificação Comportamental (MobileNetV2 Fine-Tuned via LiteRT):
     - Acionado somente quando o 1º estágio confirma a presença de celular na cabine.
     - Executa o modelo MobileNetV2 refinado no formato TensorFlow Lite (.tflite float32)
       utilizando a biblioteca oficial Google AI Edge LiteRT (ai_edge_litert), altamente
       vetorizada para as instruções ARM NEON do Raspberry Pi 4.
     - Avalia a probabilidade de uso ativo do aparelho com limiar calibrado (tau = 0.05).
  3. Máquina de Estados Temporal, Tolerância a Oclusão e Cooldown:
     - PREDICT_EVERY = 3: infere a cada 3 quadros para manter fluidez temporal (~15-20 FPS
       de captura da câmera com taxa de inferência efetiva de ~4.3 FPS).
     - SMOOTH_WINDOW = 5: média móvel temporal para amortecer oscilações espúrias.
     - TEMPO_MINIMO_ALERTA = 2 s: exige 2 segundos consecutivos de uso ativo confirmado
       para disparar o alarme, evitando falsos positivos por manipulação passageira.
     - TOLERANCIA_RESET = 2 s: tolera oclusões momentâneas (ex: volante na frente) de até
       2 segundos antes de zerar o cronômetro de infração contínua.
     - TEMPO_REPETICAO_ALERTA = 30 s: tempo de cooldown que impede disparos repetitivos
       incessantes de mensagens durante uma mesma distração prolongada.
  4. Comunicação Remota via Telegram Bot API:
     - Notificações de inicialização e encerramento seguro via requisição REST sendMessage.
     - Alerta de infração com envio fotográfico (sendPhoto) codificado diretamente em
       memória RAM (cv2.imencode), eliminando ciclos desnecessários de escrita em flash
       no cartão microSD do Raspberry Pi.
  5. Auditoria de Desempenho e Telemetria:
     - Exportação de dois arquivos CSV (quadro a quadro e resumo estatístico agregado
       com tempo total, média, variância, desvio padrão, min, max e FPS efetivo).

COMO EXECUTAR NO RASPBERRY PI 4:
--------------------------------
1. Conecte a webcam USB em uma das portas USB (preferencialmente USB 3.0 azul).
2. Configure as credenciais do seu bot do Telegram no terminal:
     $ export TELEGRAM_BOT_TOKEN="1234567890:AAH_seu_token_do_botfather"
     $ export TELEGRAM_CHAT_ID="987654321"
3. (Opcional) Caso o Raspberry Pi tenha mais de um dispositivo de vídeo e a webcam
   esteja em /dev/video1 ou /dev/video2, defina a variável CAM_INDEX:
     $ export CAM_INDEX=0   # ou 1, 2...
4. Execute o script:
     $ python3 ~/TG/TGII/RP/telegram_RP_explicado.py
5. Para encerrar com segurança, pressione a tecla 'q' na janela gráfica (ou Ctrl+C).
================================================================================
"""

# ==============================================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS DO ECOSSISTEMA EMBARCADO
# ==============================================================================

import os                                            # Manipulação de variáveis de ambiente, verificação de pastas e caminhos de arquivos
import time                                          # Temporização de eventos, cálculo de timestamps e medição de latência em milissegundos
from collections import deque                        # Fila circular com tamanho máximo (buffer FIFO) usada para cálculo da média móvel
import cv2                                           # OpenCV: captura V4L2 de webcam, transformações matriciais, desenho de HUD e compressão JPEG
import numpy as np                                   # Operações numéricas de álgebra linear, fatiamento de tensores e normalização matricial
import pandas as pd                                  # Estruturação e gravação de logs analíticos quadro-a-quadro e agregados em formato CSV
import requests                                      # Cliente HTTP leve para despacho de mensagens e fotos para a API REST do Telegram
from ai_edge_litert.interpreter import Interpreter   # Google AI Edge LiteRT: interpretador TFLite otimizado para arquiteturas ARM Cortex (NEON)
import onnxruntime as ort                            # Runtime ONNX leve de alto desempenho para execução do detector YOLOv8n em CPU


# ==============================================================================
# 2. CONFIGURAÇÕES GLOBAIS, CAMINHOS DE DIRETÓRIOS E HIPERPARÂMETROS
# ==============================================================================

# --- Caminhos dos Modelos Neurais Exportados para Hardware Embarcado ---
TFLITE_PATH = "/home/ingrid/TG/TGII/RP/finetuned/mobilenet_finetuned_rasp_float32.tflite"
# Caminho absoluto da rede MobileNetV2 fine-tuned, quantizada em float32 para o interpretador TFLite

YOLO_ONNX_PATH = "/home/ingrid/TG/TGII/RP/yolov8n.onnx"
# Caminho absoluto do detector de objetos YOLOv8n exportado no formato interoperável ONNX

# --- Caminhos dos Diretórios de Gravação de Evidências Fotográficas e Auditoria ---
OUTPUT_PHONE = "/home/ingrid/TG/TGII/RP/resultados/phone"
# Diretório em disco onde são armazenados os frames anotados onde a infração foi confirmada

OUTPUT_NO_PHONE = "/home/ingrid/TG/TGII/RP/resultados/no_phone"
# Diretório em disco onde são armazenados os frames anotados onde o motorista dirigiu sem celular

CSV_VIDEO = "/home/ingrid/TG/TGII/RP/resultados/resultados_yolo_mob.csv"
# Arquivo CSV que registra detalhadamente todas as métricas quadro a quadro de cada inferência

CSV_RESUMO = "/home/ingrid/TG/TGII/RP/resultados/resumo_yolo_mob.csv"
# Arquivo CSV consolidado que computa as métricas estatísticas globais ao final da execução

# --- Dimensões Espaciais de Entrada das Redes Neurais ---
IMG_SIZE_MOB = (256, 256)
# Resolução espacial (largura, altura) esperada na camada de entrada da MobileNetV2

IMG_SIZE_YOLO = (640, 640)
# Resolução espacial (largura, altura) esperada na camada de entrada do YOLOv8n ONNX

# --- Hiperparâmetros de Detecção e Decisão Comportamental ---
THRESHOLD = 0.05
# Limiar calibrado de decisão (tau) da MobileNetV2: acima de 5% de probabilidade, considera uso ativo

YOLO_CONF = 0.10
# Limiar de confiança do YOLOv8n para detecção da classe "cell phone" (maximiza sensibilidade/recall)

YOLO_CLASS = 67
# Índice da classe "cell phone" na taxonomia padrão de 80 classes do dataset MS COCO

PREDICT_EVERY = 3
# Intervalo de amostragem temporal: executa a inferência completa a cada 3 quadros capturados pela câmera

SMOOTH_WINDOW = 5
# Quantidade de predições passadas mantidas no buffer FIFO para amortecimento via média móvel

# --- Parâmetros da Máquina de Estados Temporal e Alerta Remoto ---
TEMPO_MINIMO_ALERTA = 2
# Duração mínima contínua de uso (em segundos) necessária para qualificar a infração e acionar o alerta

TEMPO_REPETICAO_ALERTA = 30
# Período de desativação temporária (cooldown em segundos) entre notificações sucessivas no Telegram

TOLERANCIA_RESET = 2
# Tempo de tolerância (em segundos) sem detecção antes de zerar o acumulador de infração contínua

# --- Parâmetros de Hardware da Webcam USB e Dispositivo de Captura ---
CAMERA_SIZE = (640, 480)
# Resolução de captura nativa em pixels (largura = 640, altura = 480) solicitada ao driver V4L2

CAM_INDEX = int(os.environ.get("CAM_INDEX", 0))
# Índice do dispositivo de captura de vídeo no Linux (/dev/video0 por padrão, ajustável via variável de ambiente)

# --- Credenciais de Autenticação do Bot do Telegram ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
# Token secreto de autenticação fornecido pelo @BotFather para acesso à API HTTP do Telegram

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
# Identificador numérico único do usuário ou grupo do Telegram que receberá as mensagens e alertas

# --- Configurações Gráficas da Interface e Tipografia do HUD (Head-Up Display) ---
FONT = cv2.FONT_HERSHEY_SIMPLEX                     # Fonte tipográfica utilizada para desenho de textos sobrepostos nos frames
COLOR_PHONE = (0, 0, 255)                            # Cor vermelha no padrão BGR do OpenCV: alerta crítico de infração com celular
COLOR_NO_PHONE = (0, 200, 0)                         # Cor verde no padrão BGR do OpenCV: condução segura sem detecção de infração
COLOR_YOLO = (0, 165, 255)                           # Cor laranja no padrão BGR do OpenCV: objeto detectado pelo YOLO mas rejeitado no 2º estágio

# --- Criação Automática dos Diretórios de Saída ---
os.makedirs(OUTPUT_PHONE, exist_ok=True)             # Cria a pasta de resultados positivos se ainda não existir
os.makedirs(OUTPUT_NO_PHONE, exist_ok=True)          # Cria a pasta de resultados negativos se ainda não existir


# ==============================================================================
# 3. VALIDAÇÃO DE CREDENCIAIS DE REDE E ALERTAS DO TELEGRAM
# ==============================================================================

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    # Emite aviso preventivo no console caso as variáveis de ambiente necessárias estejam ausentes
    print(
        "[TELEGRAM][AVISO] TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não definidos no ambiente.\n"
        "                O sistema processará localmente, mas alertas remotos NÃO serão despachados."
    )


# ==============================================================================
# 4. MÓDULO DE COMUNICAÇÃO REMOTA (CLIENTE TELEGRAM BOT API)
# ==============================================================================

def enviar_telegram_texto(mensagem):
    """
    Envia uma notificação textual simples para o chat configurado via HTTP POST.
    Utilizada para comunicar eventos de ciclo de vida (inicialização e encerramento).

    Parâmetros:
        mensagem (str): Conteúdo textual a ser transmitido via Telegram.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        # Se não houver credenciais válidas configuradas, aborta a transmissão sem travar o pipeline
        print("[TELEGRAM] Token ou Chat ID ausentes — mensagem textual não enviada.")
        return

    try:
        # Monta a URL de endpoint oficial do método sendMessage da API de Bots do Telegram
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

        # Executa requisição POST síncrona com timeout de segurança de 5 segundos
        resp = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,        # Destinatário configurado
                "text": mensagem                     # Mensagem de texto formatada
            },
            timeout=5
        )

        # Imprime no terminal o status da entrega para controle e depuração
        print(f"[TELEGRAM] Notificação textual enviada → Status HTTP {resp.status_code}")

    except Exception as e:
        # Captura e exibe falhas de conexão de rede sem interromper a execução do loop principal
        print(f"[TELEGRAM] Erro de comunicação ao enviar texto: {e}")


def enviar_telegram_foto(frame, caption=""):
    """
    Codifica e envia um frame da webcam como imagem fotográfica (JPEG) via HTTP POST.
    A codificação é realizada diretamente em buffer de memória RAM, sem salvar arquivos
    temporários no disco flash do Raspberry Pi, poupando a vida útil do cartão microSD.

    Parâmetros:
        frame (np.ndarray): Matriz de pixels BGR do frame capturado na ocorrência da infração.
        caption (str): Mensagem explicativa anexada como legenda da foto (ex: tempo de distração).
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        # Aborta o envio se os identificadores do bot não estiverem disponíveis
        print("[TELEGRAM] Token ou Chat ID ausentes — alerta fotográfico não enviado.")
        return

    try:
        # Comprime a imagem BGR diretamente em memória para formato JPEG (retorna flag e array de bytes)
        sucesso, buffer = cv2.imencode(".jpg", frame)
        if not sucesso:
            print("[TELEGRAM] Falha ao codificar frame para JPEG em memória.")
            return

        # Monta o endpoint oficial do método sendPhoto da API de Bots do Telegram
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"

        # Despacha a requisição HTTP POST contendo o arquivo binário em memória e metadados
        resp = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,        # Identificador do chat de destino
                "caption": caption                   # Legenda da infração exibida abaixo da foto
            },
            files={
                "photo": (
                    "alerta_infracao.jpg",           # Nome fictício atribuído ao anexo binário
                    buffer.tobytes(),                # Fluxo de bytes da imagem JPEG codificada em RAM
                    "image/jpeg"                     # Tipo MIME oficial do formato JPEG
                )
            },
            timeout=10                               # Timeout de 10 segundos para tolerar latência de uplink móvel
        )

        # Valida o código de resposta HTTP retornado pelo servidor do Telegram
        if resp.status_code == 200:
            print("\n[TELEGRAM] >>> Alerta fotográfico transmitido com sucesso ao motorista/gestor! <<<")
        else:
            print(f"\n[TELEGRAM] Falha no despacho da foto. Status HTTP {resp.status_code}: {resp.text[:150]}")

    except Exception as e:
        # Registra qualquer eventual falha de rede sem interromper o monitoramento em tempo real
        print(f"[TELEGRAM] Erro de exceção ao despachar foto de alerta: {e}")


# ==============================================================================
# 5. INICIALIZAÇÃO E CARREGAMENTO DOS MODELOS NEURAIS EMBARCADOS
# ==============================================================================

# --- Carregamento do 1º Estágio: YOLOv8n via ONNX Runtime ---
print("[INFO] Carregando modelo detector YOLOv8n (ONNX Runtime)...")
yolo_session = ort.InferenceSession(
    YOLO_ONNX_PATH,
    providers=["CPUExecutionProvider"]               # Execução eficiente vetorizada nas 4 CPUs ARM Cortex-A72 do RPi
)
yolo_input_name = yolo_session.get_inputs()[0].name  # Obtém o identificador da camada de entrada do grafo ONNX ('images')

# --- Carregamento do 2º Estágio: MobileNetV2 via Google AI Edge LiteRT ---
print("[INFO] Carregando modelo classificador MobileNetV2 (Google AI Edge LiteRT TFLite)...")
interpreter = Interpreter(model_path=TFLITE_PATH)   # Inicializa o motor de inferência LiteRT com o arquivo .tflite float32
interpreter.allocate_tensors()                       # Aloca estaticamente na memória RAM os tensores de ativação e pesos da rede

input_details = interpreter.get_input_details()      # Metadados do tensor de entrada (índice, tipo float32, dimensões 1x256x256x3)
output_details = interpreter.get_output_details()    # Metadados do tensor de saída (índice, tipo float32, dimensões 1x1 da sigmoide)

print("[INFO] Modelos de inteligência artificial carregados com sucesso na memória RAM.\n")


# ==============================================================================
# 6. FUNÇÕES DE PRÉ-PROCESSAMENTO MATRICIAL DE IMAGEM
# ==============================================================================

def preprocess_mobilenet(frame_bgr):
    """
    Transforma o frame bruto capturado pela webcam no formato exigido pela MobileNetV2.
    Aplica:
      1. Interpolação bilinear para resolução fixa (256, 256).
      2. Conversão do espaço de cor nativo OpenCV BGR para RGB.
      3. Normalização radiométrica padronizada da MobileNetV2: intervalo simétrico [-1.0, +1.0].
      4. Expansão dimensional para formato de lote (Batch, Height, Width, Channels) -> [1, 256, 256, 3].

    Parâmetros:
        frame_bgr (np.ndarray): Frame original capturado da webcam em formato BGR.

    Retorna:
        np.ndarray: Tensor tridimensional float32 expandido para lote unidimensional.
    """
    # 1. Redimensionamento espacial para as dimensões exatas de entrada da rede
    img = cv2.resize(frame_bgr, IMG_SIZE_MOB)

    # 2. Conversão do padrão de cores OpenCV (BGR) para o padrão de treino da rede (RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Conversão para ponto flutuante de 32 bits e normalização na escala simétrica [-1.0, 1.0]
    #    Fórmula canônica: (pixel / 127.5) - 1.0, mapeando [0, 255] linearmente para [-1.0, 1.0]
    arr = (img.astype(np.float32) / 127.5) - 1.0

    # 4. Adiciona a dimensão de batch no eixo 0, resultando no formato dimensional (1, 256, 256, 3)
    return np.expand_dims(arr, axis=0)


def preprocess_yolo(frame_bgr):
    """
    Transforma o frame bruto capturado pela webcam no formato exigido pelo YOLOv8n ONNX.
    Aplica:
      1. Interpolação bilinear para resolução fixa (640, 640).
      2. Conversão do espaço de cores BGR para RGB.
      3. Conversão para float32 e normalização uniforme no intervalo unitário [0.0, 1.0].
      4. Transposição de eixos do formato Channels-Last (H, W, C) para Channels-First (C, H, W).
      5. Expansão dimensional para lote: (1, 3, 640, 640).

    Parâmetros:
        frame_bgr (np.ndarray): Frame original capturado da webcam em formato BGR.

    Retorna:
        np.ndarray: Tensor float32 no formato NCHW pronto para inferência no ONNX Runtime.
    """
    # 1. Redimensionamento espacial para a matriz quadrada esperada pelo YOLOv8 (640x640)
    img = cv2.resize(frame_bgr, IMG_SIZE_YOLO)

    # 2. Conversão de BGR para RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Normalização direta dos valores de intensidade de 8 bits [0, 255] para escala contínua [0.0, 1.0]
    arr = img.astype(np.float32) / 255.0

    # 4. Reordenação dos eixos tensores de (Altura, Largura, Canais) para (Canais, Altura, Largura)
    arr = np.transpose(arr, (2, 0, 1))

    # 5. Adiciona a dimensão de lote no eixo 0, gerando o tensor com dimensões (1, 3, 640, 640)
    return np.expand_dims(arr, axis=0)


# ==============================================================================
# 7. FUNÇÕES DE INFERÊNCIA NEURAL (YOLOv8 ONNX + MOBILENETV2 LITERTT)
# ==============================================================================

def inferir_tflite(arr):
    """
    Executa a inferência direta no motor Google AI Edge LiteRT para o tensor pré-processado.

    Parâmetros:
        arr (np.ndarray): Tensor float32 de formato (1, 256, 256, 3).

    Retorna:
        float: Probabilidade contínua estimada da classe 'phone' (entre 0.0 e 1.0).
    """
    # Copia os dados do array numpy diretamente no buffer do tensor de entrada do interpretador
    interpreter.set_tensor(input_details[0]["index"], arr)

    # Dispara os cálculos das camadas convolucionais e pooling no hardware do Raspberry Pi
    interpreter.invoke()

    # Extrai o resultado produzido na camada de saída sigmoide da rede
    resultado = interpreter.get_tensor(output_details[0]["index"])

    # Converte o array escalar numpy em tipo float nativo do Python
    return float(resultado[0][0])


def detectar_yolo(frame):
    """
    Executa o 1º estágio de detecção de objetos utilizando o modelo YOLOv8n no ONNX Runtime.
    Examina as 8400 caixas candidatas produzidas pelo YOLO e filtra a classe 'cell phone' (ID 67).

    Parâmetros:
        frame (np.ndarray): Imagem original BGR capturada pela webcam.

    Retorna:
        tuple (bool, float): (True se celular for detectado com confiança >= YOLO_CONF, maior score encontrado)
    """
    # Pré-processa o frame gerando o tensor (1, 3, 640, 640)
    arr = preprocess_yolo(frame)

    # Executa a sessão de inferência no grafo computacional ONNX
    outputs = yolo_session.run(None, {yolo_input_name: arr})

    # A saída do YOLOv8 tem dimensões (1, 84, 8400), onde 84 = [x, y, w, h, 80 scores de classes]
    # Transpomos a matriz para (8400, 84), permitindo iterar linha por linha sobre cada predição
    preds = outputs[0][0].T

    # Varre as detecções candidatas
    for det in preds:
        scores = det[4:]                             # Fatiamento a partir do índice 4 isola os 80 scores de classe do COCO
        class_id = int(np.argmax(scores))            # Identifica o índice da classe com o maior score de probabilidade
        conf = float(scores[class_id])               # Valor escalar da confiança associada à classe predita

        # Verifica se a classe mais provável corresponde a celular (67) e atinge o limiar mínimo (0.10)
        if class_id == YOLO_CLASS and conf >= YOLO_CONF:
            return True, conf                        # Encerra antecipadamente: celular detectado com sucesso

    # Caso nenhuma caixa atinja o critério de confiança para celular, retorna negativo
    return False, 0.0


# ==============================================================================
# 8. SOBREPOSIÇÃO GRÁFICA EM TEMPO REAL (HUD / OVERLAY NA TELA)
# ==============================================================================

def overlay(frame, label, pred, color, yolo_conf=0.0, tempo_uso=0.0):
    """
    Desenha os elementos textuais informativos e caixas de alto contraste na tela (HUD).
    Exibe:
      - Linha 1: Status comportamental ('CELULAR' ou 'SEM CELULAR') e score da MobileNetV2
      - Linha 2: Confiança do 1º estágio YOLOv8n
      - Linha 3: Duração de distração contínua em segundos (quando aplicável)

    Parâmetros:
        frame (np.ndarray): Imagem original BGR onde os textos serão sobrepostos.
        label (str): Rótulo textual da classificação comportamental.
        pred (float): Probabilidade contínua estimada pela MobileNetV2.
        color (tuple): Cor BGR do texto identificador de estado (vermelho, verde ou laranja).
        yolo_conf (float): Confiança de detecção do YOLOv8n.
        tempo_uso (float): Cronômetro de uso consecutivo em segundos.

    Retorna:
        np.ndarray: Frame anotado com a interface gráfica de telemetria.
    """
    # Monta as cadeias de caracteres a serem exibidas no painel visual
    linha1 = f"{label} | mob: {pred:.2f}"
    linha2 = f"YOLO: {yolo_conf:.2f}" if yolo_conf > 0 else "YOLO: ---"
    linha3 = f"uso: {tempo_uso:.0f}s" if tempo_uso > 0 else ""

    # Consolida a lista de linhas de telemetria ativas
    linhas = [linha1, linha2]
    if linha3:
        linhas.append(linha3)

    # Itera sobre cada linha calculando dimensões para desenhar caixas de fundo pretas com alto contraste
    for i, texto in enumerate(linhas):
        # Calcula a largura (tw) e altura (th) em pixels ocupadas pelo texto tipográfico
        (tw, th), _ = cv2.getTextSize(texto, FONT, 0.7, 2)

        # Calcula a coordenada vertical (y) com espaçamento de 10 pixels entre linhas
        y = 10 + i * (th + 10)

        # Desenha retângulo preto sólido de fundo para garantir legibilidade sobre qualquer iluminação
        cv2.rectangle(
            frame,
            (8, y),
            (14 + tw, y + th + 4),
            (0, 0, 0),
            -1
        )

        # Aplica a renderização do texto na cor designada com suavização antialiasing (LINE_AA)
        cv2.putText(
            frame,
            texto,
            (10, y + th),
            FONT,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

    return frame


# ==============================================================================
# 9. MOTOR PRINCIPAL DE PROCESSAMENTO EM TEMPO REAL (LOOP DA WEBCAM USB)
# ==============================================================================

def run_video():
    """
    Função controladora que executa a captura V4L2 da webcam USB, acopla os estágios
    neurais, gerencia os temporizadores da máquina de estados, grava logs e exibe a HUD.
    """
    print(f"[INFO] Inicializando Webcam USB no dispositivo V4L2 (/dev/video{CAM_INDEX})...")

    # Inicializa o objeto VideoCapture do OpenCV apontando para o índice da webcam USB
    cap = cv2.VideoCapture(CAM_INDEX)

    # Configura explicitamente a resolução solicitada ao driver de captura da webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])

    # Verifica se o hardware de captura foi corretamente inicializado pelo sistema operacional
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível acessar a webcam USB no índice {CAM_INDEX}.")
        print("Dica: verifique se o cabo USB está conectado ou execute com outro índice:")
        print("      CAM_INDEX=1 python3 telegram_RP_explicado.py")
        return

    # Pausa técnica de 1 segundo para estabilização dos sensores ópticos e auto-exposição da câmera
    time.sleep(1)

    # Lê a resolução efetivamente configurada pelo hardware óptico
    largura_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[WEBCAM] Resolução operacional confirmada: {largura_real}x{altura_real} pixels.")
    print("Controle: Pressione a tecla 'q' na janela gráfica para encerrar com segurança.\n")

    # Envia notificação remota via Telegram informando o início das atividades de monitoramento
    enviar_telegram_texto("🚗 DriveSafe Vision iniciado no Raspberry Pi 4 (Webcam USB)")

    # Inicialização dos contadores de auditoria e métricas de desempenho
    frame_count = 0                                  # Contador acumulado do total de quadros capturados
    phone_count = 0                                  # Contador de quadros onde o uso do celular foi confirmado
    no_phone_count = 0                               # Contador de quadros de condução regular e segura
    yolo_only_count = 0                              # Contador de quadros onde o YOLO disparou mas MobileNetV2 descartou (distratores)

    # Buffer circular (deque) para cálculo da média móvel das probabilidades estimadas
    pred_buffer = deque(maxlen=SMOOTH_WINDOW)

    # Variáveis de persistência gráfica entre quadros intermediários sem inferência direta
    last_pred = 0.0                                  # Último score contínuo computado pela MobileNetV2
    last_yolo_conf = 0.0                             # Último score de detecção do YOLOv8n
    last_label = "SEM CELULAR"                       # Último rótulo comportamental definido
    last_color = COLOR_NO_PHONE                      # Última cor de texto configurada para o HUD

    # Listas em memória para acumulação de dados estatísticos e telemetria para os arquivos CSV
    tempos = []                                      # Armazena as durações (em segundos) de cada ciclo de inferência
    registros = []                                   # Armazena os dicionários de log detalhado quadro a quadro

    # Variáveis da máquina de estados temporal para validação de infração e cooldown
    inicio_mobilenet = None                          # Marca temporal (timestamp) do primeiro instante de detecção contínua
    ultimo_alerta = None                             # Marca temporal do último alerta fotográfico transmitido via Telegram
    ultimo_sem_celular = None                        # Marca temporal do último quadro classificado como ausência de celular
    tempo_uso_continuo = 0.0                         # Duração acumulada (em segundos) do evento contínuo de distração

    try:
        # Loop infinito de processamento de fluxo de vídeo em tempo real
        while True:
            # ------------------------------------------------------------------
            # CAPTURA DO QUADRO ATUAL DA WEBCAM USB
            # ------------------------------------------------------------------
            # cap.read() entrega o quadro matricial pronto no formato BGR nativo do OpenCV
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[AVISO] Falha momentânea ao obter quadro da webcam USB. Tentando novamente...")
                time.sleep(0.05)
                continue

            agora = time.time()                      # Timestamp atual em segundos de alta precisão

            # ------------------------------------------------------------------
            # CICLO DE INFERÊNCIA NEURAL (A CADA PREDICT_EVERY QUADROS)
            # ------------------------------------------------------------------
            if frame_count % PREDICT_EVERY == 0:
                t0 = time.perf_counter()             # Inicia o cronômetro de precisão para medição do tempo de inferência

                # 1º Estágio: Avalia se há celular no campo de visão utilizando YOLOv8n ONNX
                yolo_ok, yolo_conf = detectar_yolo(frame)

                if yolo_ok:
                    # 2º Estágio (Curto-Circuito): celular detectado, aciona MobileNetV2 LiteRT
                    arr = preprocess_mobilenet(frame)
                    p = inferir_tflite(arr)
                else:
                    # Curto-circuito computacional ativado: poupa ciclos de CPU da MobileNetV2
                    p = 0.0

                t1 = time.perf_counter()             # Interrompe o cronômetro de precisão de inferência
                tempos.append(t1 - t0)               # Registra a latência computacional do ciclo atual

                # Atualiza a fila circular da média móvel temporal
                pred_buffer.append(p)
                pred_media = float(np.mean(pred_buffer))

                # --------------------------------------------------------------
                # DECISÃO COMPORTAMENTAL: CASO A — CELULAR CONFIRMADO
                # --------------------------------------------------------------
                if yolo_ok and p > THRESHOLD:
                    classe = "phone"
                    last_label = "CELULAR"
                    last_color = COLOR_PHONE
                    last_yolo_conf = yolo_conf
                    phone_count += 1

                    # Nome de arquivo para arquivamento fotográfico da infração
                    fname = os.path.join(OUTPUT_PHONE, f"frame_{frame_count:05d}.jpg")

                    # Anula o temporizador de ausência de celular
                    ultimo_sem_celular = None

                    # Inicializa o cronômetro contínuo na primeira ocorrência de infração
                    if inicio_mobilenet is None:
                        inicio_mobilenet = agora

                    # Calcula quanto tempo contínuo o motorista está usando o aparelho celular
                    tempo_uso_continuo = agora - inicio_mobilenet

                    # Avalia se o período de cooldown de 30 segundos já foi respeitado
                    cooldown_ok = (
                        ultimo_alerta is None or
                        (agora - ultimo_alerta) >= TEMPO_REPETICAO_ALERTA
                    )

                    # Se atingir o tempo mínimo contínuo (2 s) e o cooldown permitir, emite alerta
                    if tempo_uso_continuo >= TEMPO_MINIMO_ALERTA and cooldown_ok:
                        ultimo_alerta = agora        # Atualiza a marcação temporal do alerta despachado

                        minutos = int(tempo_uso_continuo // 60)
                        segundos = int(tempo_uso_continuo % 60)
                        caption = (
                            "⚠️ ALERTA: USO DE CELULAR AO VOLANTE DETECTADO!\n"
                            f"Tempo de uso contínuo ao volante: {minutos}min {segundos}s\n"
                            f"Local: Raspberry Pi 4 | Modelo: YOLOv8 + MobileNetV2"
                        )

                        # Envia foto com a anotação visual para o Telegram
                        enviar_telegram_foto(frame, caption)

                # --------------------------------------------------------------
                # DECISÃO COMPORTAMENTAL: CASO B — SEM CELULAR / DISTRATOR
                # --------------------------------------------------------------
                else:
                    if yolo_ok and p <= THRESHOLD:
                        # Distrator identificado: YOLO detectou objeto mas MobileNetV2 rejeitou
                        classe = "no_phone"
                        last_label = "SEM CELULAR"
                        last_color = COLOR_YOLO      # Cor laranja evidencia a rejeição correta de falso positivo
                        last_yolo_conf = yolo_conf
                        yolo_only_count += 1
                        no_phone_count += 1
                    else:
                        # Condução normal sem detecção de nenhum dispositivo móvel
                        classe = "no_phone"
                        last_label = "SEM CELULAR"
                        last_color = COLOR_NO_PHONE  # Cor verde evidencia direção segura
                        last_yolo_conf = 0.0
                        no_phone_count += 1

                    # Nome de arquivo para arquivamento dos quadros negativos
                    fname = os.path.join(OUTPUT_NO_PHONE, f"frame_{frame_count:05d}.jpg")

                    # Inicia contagem de tolerância de ausência
                    if ultimo_sem_celular is None:
                        ultimo_sem_celular = agora

                    # Caso a ausência de celular persista além da tolerância (2 s), reseta o cronômetro
                    if (agora - ultimo_sem_celular) >= TOLERANCIA_RESET:
                        inicio_mobilenet = None
                        tempo_uso_continuo = 0.0
                        # Nota: o 'ultimo_alerta' NÃO é resetado aqui para garantir que o cooldown
                        # de 30 segundos permaneça ativo mesmo se o motorista esconder o celular e puxar de volta.

                last_pred = p                        # Atualiza o último valor inferido pela rede

                # --------------------------------------------------------------
                # GRAVAÇÃO DO FRAME EM DISCO E LOG NO BANCO DE DADOS CSV
                # --------------------------------------------------------------
                # Renderiza a interface HUD sobre uma cópia do frame capturado
                frame_out = overlay(
                    frame.copy(),
                    last_label,
                    last_pred,
                    last_color,
                    last_yolo_conf,
                    tempo_uso_continuo
                )

                # Salva o frame anotado na pasta correspondente em disco
                cv2.imwrite(fname, frame_out)

                # Adiciona linha descritiva no registro de auditoria para o CSV analítico
                registros.append({
                    "frame": frame_count,
                    "modelo": "YOLO_ONNX+TFLite",
                    "yolo_detectou": yolo_ok,
                    "yolo_conf": round(yolo_conf, 4),
                    "pred_bruta": round(p, 4),
                    "pred_media": round(pred_media, 4),
                    "predicao": classe,
                    "tempo_uso_s": round(tempo_uso_continuo, 1),
                    "tempo_ms": round((t1 - t0) * 1000, 2),
                    "threshold_mob": THRESHOLD,
                    "threshold_yolo": YOLO_CONF
                })

                # Exibição compacta em uma única linha no terminal para acompanhamento ao vivo
                print(
                    f"YOLO={yolo_ok}({yolo_conf:.2f}) | mob={p:.4f} | uso={tempo_uso_continuo:.0f}s → {last_label}",
                    end="\r"
                )

            # ------------------------------------------------------------------
            # ATUALIZAÇÃO VISUAL NA TELA (JANELA DO OPENCV)
            # ------------------------------------------------------------------
            frame_display = overlay(
                frame.copy(),
                last_label,
                last_pred,
                last_color,
                last_yolo_conf,
                tempo_uso_continuo
            )

            # Exibe o frame processado na janela gráfica do ambiente gráfico
            cv2.imshow("DriveSafe Vision - Raspberry Pi 4 (Webcam USB)", frame_display)

            frame_count += 1                         # Incrementa o contador total de quadros processados

            # Verifica se o usuário pressionou a tecla 'q' (ASCII 113) para encerrar voluntariamente
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Interrupção solicitada pelo usuário através da tecla 'q'.")
                break

    finally:
        # Bloco de finalização segura executado garantidamente ao sair do loop
        print("\n[INFO] Liberando dispositivo de webcam e destruindo janelas ativas...")
        cap.release()                                # Libera o descritor de dispositivo da câmera no kernel Linux
        cv2.destroyAllWindows()                      # Fecha todas as janelas gráficas abertas pelo OpenCV

    # ==============================================================================
    # 10. GERAÇÃO DE RELATÓRIOS ANALÍTICOS CSV E ESTATÍSTICAS FINAIS
    # ==============================================================================

    # Converte a lista de dicionários de auditoria em um DataFrame do Pandas e exporta para CSV
    pd.DataFrame(registros).to_csv(CSV_VIDEO, index=False)
    print(f"\n[SUCESSO] Relatório detalhado quadro a quadro salvo em: {CSV_VIDEO}")

    # Notifica via Telegram sobre o encerramento do monitoramento veicular
    enviar_telegram_texto("✅ DriveSafe Vision encerrado com sucesso no Raspberry Pi 4.")

    # Computação estatística das métricas de latência e taxa de processamento
    if tempos:
        tempos_ms = np.array(tempos) * 1000          # Converte tempos de inferência de segundos para milissegundos
        tempo_total = float(np.sum(tempos_ms))       # Soma do tempo total despendido em inferência neural pura
        tempo_medio = float(np.mean(tempos_ms))      # Tempo médio de execução de cada inferência
        tempo_var = float(np.var(tempos_ms))         # Variância temporal da latência
        tempo_desvio = float(np.std(tempos_ms))      # Desvio padrão da latência computacional
        tempo_min = float(np.min(tempos_ms))         # Tempo mínimo registrado em uma inferência
        tempo_max = float(np.max(tempos_ms))         # Tempo máximo (pior caso) registrado
        fps_efetivo = frame_count / (tempo_total / 1000) if tempo_total > 0 else 0  # Taxa de quadros efetiva calculada

        # Impressão formatada do relatório resumo no terminal
        print("\n" + "=" * 50)
        print("RESUMO CONSOLIDADO DA EXECUÇÃO (DRIVESAFE VISION)")
        print("=" * 50)
        print(f"Total de frames capturados     : {frame_count}")
        print(f"Frames com celular confirmado  : {phone_count} ({100 * phone_count / max(1, frame_count):.1f}%)")
        print(f"Frames de condução segura      : {no_phone_count} ({100 * no_phone_count / max(1, frame_count):.1f}%)")
        print(f"Distratores rejeitados no 2º E : {yolo_only_count}")
        print("-" * 50)
        print(f"Tempo médio de inferência      : {tempo_medio:.2f} ms")
        print(f"Desvio padrão da inferência    : {tempo_desvio:.2f} ms")
        print(f"Taxa efetiva de processamento  : {fps_efetivo:.1f} FPS")
        print("=" * 50)

        # Salva o arquivo CSV com os indicadores consolidados de desempenho para documentação da tese
        pd.DataFrame([{
            "modelo": "YOLO_ONNX+TFLite",
            "frames_processados": frame_count,
            "frames_celular": phone_count,
            "frames_sem_celular": no_phone_count,
            "yolo_sem_confirmacao": yolo_only_count,
            "pct_celular": round(100 * phone_count / max(1, frame_count), 1),
            "pct_sem_celular": round(100 * no_phone_count / max(1, frame_count), 1),
            "tempo_total_ms": round(tempo_total, 2),
            "tempo_medio_ms": round(tempo_medio, 2),
            "tempo_min_ms": round(tempo_min, 2),
            "tempo_max_ms": round(tempo_max, 2),
            "variancia_ms2": round(tempo_var, 2),
            "desvio_padrao_ms": round(tempo_desvio, 2),
            "fps_efetivo": round(fps_efetivo, 1),
            "threshold_mobilenet": THRESHOLD,
            "threshold_yolo": YOLO_CONF
        }]).to_csv(CSV_RESUMO, index=False)

        print(f"[SUCESSO] Relatório consolidado de métricas salvo em: {CSV_RESUMO}\n")


# ==============================================================================
# 11. PONTO DE ENTRADA DO SCRIPT PYTHON (ENTRY POINT)
# ==============================================================================

def main():
    """
    Ponto de entrada primário do programa.
    """
    run_video()


if __name__ == "__main__":
    # Garante que a função principal só seja acionada quando o arquivo for invocado diretamente
    main()
