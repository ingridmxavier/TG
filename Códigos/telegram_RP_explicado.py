
"""
================================================================================
DriveSafe Vision — Monitoramento Embarcado no Raspberry Pi 4 (telegram_RP_explicado.py)
================================================================================

DESCRIÇÃO GERAL DO SISTEMA:
---------------------------
Este script implementa a versão embarcada (Edge AI) do sistema DriveSafe Vision,
projetada especificamente para rodar em tempo real no computador de placa única
Raspberry Pi 4 Model B acoplado à câmera oficial Raspberry Pi Camera Module (CSI).

O pipeline opera em uma arquitetura híbrida de dois estágios em cascata:
  1. 1º Estágio — Detecção de Objetos (YOLOv8n em formato ONNX via ONNX Runtime):
     Inspeciona o quadro completo (640x640) em busca da classe 67 do dataset COCO
     ("cell phone"). Opera como filtro espacial rápido: se nenhum celular for
     detectado no quadro, o 2º estágio nem é acionado, economizando CPU.
  2. 2º Estágio — Classificação Comportamental (MobileNetV2 em formato TFLite float32):
     Acionado apenas quando o YOLOv8n encontra um celular candidato. Avalia se o
     objeto detectado configura efetivamente uso ativo pelo condutor, reduzindo
     falsos positivos.
  3. Validação Temporal, Tolerância de Oclusão e Cooldown:
     - Exige persistência contínua mínima de uso (TEMPO_MINIMO_ALERTA = 2 s) antes
       de confirmar a infração.
     - Tolera falhas breves de detecção de até TOLERANCIA_RESET = 2 s sem zerar
       o cronômetro de uso contínuo.
     - Aplica um intervalo mínimo (TEMPO_REPETICAO_ALERTA = 30 s) entre alertas
       sucessivos no Telegram para evitar spam.
  4. Notificação Remota e Auditoria:
     - Codifica a evidência visual diretamente na memória RAM (JPEG via OpenCV) e
       envia de forma automática para um bot no Telegram via API HTTP REST.
     - Salva os quadros anotados (HUD) nas pastas de saída e gera dois arquivos CSV
       (log detalhado quadro a quadro e resumo estatístico de desempenho/latência).

DEPENDÊNCIAS LEVES (SEM PYTORCH E SEM TENSORFLOW COMPLETO):
-----------------------------------------------------------
Para maximizar a taxa de quadros por segundo (FPS) e economizar memória RAM na
arquitetura ARM64 (AArch64) do Raspberry Pi, este script NÃO carrega o PyTorch
nem o TensorFlow completo, utilizando exclusivamente:
  - onnxruntime     (inferência otimizada em CPU para o YOLOv8n .onnx)
  - ai_edge_litert  (interpretador leve TFLite para a MobileNetV2 .tflite)
  - picamera2       (interface nativa libcamera para captura CSI de baixa latência)
  - opencv-python   (pré-processamento matricial, anotação HUD e compressão JPEG)
  - numpy, pandas, requests

EXEMPLOS DE USO:
----------------
Exemplo 1 — Execução padrão com alertas ativos no Telegram:
    $ export TELEGRAM_BOT_TOKEN="1234567890:AAH_seu_token_do_botfather_aqui"
    $ export TELEGRAM_CHAT_ID="987654321"
    $ python3 "/home/ingrid/TG/TG III/telegram_RP_explicado.py"

Exemplo 2 — Passando as variáveis de ambiente na mesma linha do comando:
    $ TELEGRAM_BOT_TOKEN="seu_token" TELEGRAM_CHAT_ID="seu_chat_id" \
      python3 "/home/ingrid/TG/TG III/telegram_RP_explicado.py"

Exemplo 3 — Execução offline (apenas teste local de câmera, HUD e geração de CSVs,
sem envio de mensagens para o Telegram):
    $ python3 "/home/ingrid/TG/TG III/telegram_RP_explicado.py"
    (O sistema exibirá um aviso inicial informando que o token/chat_id não foram
     definidos e continuará operando normalmente, salvando os frames e planilhas CSV.)

Exemplo 4 — Encerramento seguro da execução:
    Com a janela de vídeo "DriveSafe Vision" em foco, pressione a tecla 'q'.
    O bloco `finally` desligará a câmera CSI com segurança, fechará as janelas
    gráficas, enviará a notificação de encerramento ao Telegram e salvará os CSVs.
================================================================================
"""

# ==============================================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# ==============================================================================

import os                                            # Acesso a variáveis de ambiente e manipulação de caminhos/pastas do sistema operacional
import time                                          # Medição de tempo (timestamps de eventos, cronometragem de inferência e pausas)
from collections import deque                        # Fila circular de tamanho fixo para suavizar predições (média móvel temporal)
import cv2                                           # OpenCV: redimensionamento, conversão de cor, desenho de HUD (overlay) e codificação JPEG
import numpy as np                                   # Operações matemáticas e matriciais vetorizadas sobre os tensores de imagem
import onnxruntime as ort                            # Motor de inferência ONNX Runtime para executar o detector YOLOv8n em CPU
import pandas as pd                                  # Estruturação dos logs de execução e exportação para arquivos CSV
import requests                                      # Cliente HTTP para envio de mensagens e fotos à API REST do Telegram
from ai_edge_litert.interpreter import Interpreter   # Interpretador LiteRT (TFLite) otimizado para dispositivos de borda (MobileNetV2)
from picamera2 import Picamera2                      # Biblioteca oficial baseada em libcamera para controle da câmera CSI no Raspberry Pi


# ==============================================================================
# 2. CONFIGURAÇÕES GLOBAIS E HIPERPARÂMETROS DO SISTEMA
# ==============================================================================
# Nesta seção ficam centralizados todos os caminhos de arquivos, dimensões de
# entrada das redes neurais, limiares de decisão, parâmetros da máquina de
# estados temporal e cores da interface visual (HUD).

# --- Caminhos dos Modelos e Diretórios de Saída ---
TFLITE_PATH = "/home/ingrid/TG/TGII/RP/finetuned/mobilenet_finetuned_rasp_float32.tflite"  # Caminho do modelo MobileNetV2 ajustado (fine-tuned) exportado em TFLite float32
YOLO_ONNX_PATH = "/home/ingrid/TG/TGII/RP/yolov8n.onnx"                                    # Caminho do modelo detector YOLOv8 nano exportado para o formato aberto ONNX
OUTPUT_PHONE = "/home/ingrid/TG/TGII/RP/resultados/phone"                                  # Diretório onde serão salvos os quadros classificados como "com celular"
OUTPUT_NO_PHONE = "/home/ingrid/TG/TGII/RP/resultados/no_phone"                            # Diretório onde serão salvos os quadros classificados como "sem celular"
CSV_VIDEO = "/home/ingrid/TG/TGII/RP/resultados/resultados_yolo_mob.csv"                   # Caminho do arquivo CSV que armazenará o histórico detalhado quadro a quadro
CSV_RESUMO = "/home/ingrid/TG/TGII/RP/resultados/resumo_yolo_mob.csv"                      # Caminho do arquivo CSV que armazenará as métricas consolidadas da sessão

# --- Dimensões de Imagem e Resolução de Captura ---
IMG_SIZE_MOB = (256, 256)                            # Resolução espacial (largura, altura) esperada na entrada da MobileNetV2
IMG_SIZE_YOLO = (640, 640)                           # Resolução espacial (largura, altura) esperada na entrada do YOLOv8n ONNX
CAMERA_SIZE = (640, 480)                             # Resolução (largura, altura) configurada para captura bruta na câmera CSI (proporção 4:3)

# --- Limiares de Decisão e Controle de Inferência ---
THRESHOLD = 0.05                                     # Limiar de probabilidade da MobileNetV2 acima do qual o uso do celular é confirmado
YOLO_CONF = 0.10                                     # Confiança mínima exigida para que uma detecção do YOLOv8n seja aceita como candidata válida
YOLO_CLASS = 67                                      # Identificador numérico da classe "cell phone" no conjunto de dados MS COCO (0 a 79)
PREDICT_EVERY = 3                                    # Executa a inferência completa a cada 3 quadros capturados (frame skipping) para elevar o FPS
SMOOTH_WINDOW = 5                                    # Quantidade de inferências recentes mantidas no buffer para cálculo da média móvel

# --- Temporizadores da Lógica de Alerta e Persistência ---
TEMPO_MINIMO_ALERTA = 2                              # Tempo mínimo (em segundos) de detecção ininterrupta necessário para disparar o alerta
TEMPO_REPETICAO_ALERTA = 30                          # Tempo de espera ou "cooldown" (em segundos) entre dois envios consecutivos ao Telegram
TOLERANCIA_RESET = 2                                 # Tempo máximo (em segundos) sem detecção tolerado antes de zerar a contagem de uso contínuo

# --- Credenciais do Telegram (via Variáveis de Ambiente) ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")    # Lê o token do bot a partir do ambiente do sistema (retorna "" caso não exista)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")    # Lê o identificador do chat destinatário a partir do ambiente do sistema

# --- Configurações Visuais do HUD (Head-Up Display) ---
FONT = cv2.FONT_HERSHEY_SIMPLEX                      # Fonte tipográfica padrão do OpenCV utilizada para desenhar textos sobre o vídeo
COLOR_PHONE = (0, 0, 255)                            # Cor vermelha no padrão BGR (Blue=0, Green=0, Red=255) para estado de infração confirmada
COLOR_NO_PHONE = (0, 200, 0)                         # Cor verde no padrão BGR (Blue=0, Green=200, Red=0) para estado de condução segura
COLOR_YOLO = (0, 165, 255)                           # Cor laranja no padrão BGR (Blue=0, Green=165, Red=255) quando apenas o YOLO detecta

# --- Criação Automática da Estrutura de Pastas ---
os.makedirs(OUTPUT_PHONE, exist_ok=True)             # Garante que a pasta de saída para quadros "com celular" exista no disco (não dá erro se já existir)
os.makedirs(OUTPUT_NO_PHONE, exist_ok=True)          # Garante que a pasta de saída para quadros "sem celular" exista no disco


# ==============================================================================
# 3. VERIFICAÇÃO INICIAL DE CREDENCIAIS DO TELEGRAM
# ==============================================================================
# Verifica logo na partida do script se o usuário exportou as variáveis de
# ambiente necessárias. Caso contrário, emite um alerta claro no terminal,
# mas permite que o sistema continue rodando localmente.

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:       # Checa se o token do bot ou o ID do chat estão vazios
    print(                                           # Imprime aviso preventivo no console para orientar o operador
        "[TELEGRAM][AVISO] TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não definidos no ambiente. "
        "Alertas NÃO serão enviados até isso ser corrigido."
    )


# ==============================================================================
# 4. MÓDULO DE COMUNICAÇÃO COM A API DO TELEGRAM
# ==============================================================================
# Funções responsáveis por encapsular as chamadas HTTP POST para a API REST do
# Telegram (`sendMessage` para avisos de status e `sendPhoto` para envio de
# evidências visuais compactadas em memória RAM).

def enviar_telegram_texto(mensagem):
    """
    Envia uma mensagem de texto simples para o chat configurado no Telegram.
    Utilizada para notificar o início e o encerramento das sessões de monitoramento.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:   # Aborta imediatamente caso as credenciais não tenham sido configuradas
        print("[TELEGRAM] Token/Chat ID ausentes — mensagem NÃO enviada.")  # Informa no terminal o motivo do não envio
        return                                       # Retorna sem fazer requisição de rede

    try:                                             # Inicia bloco protegido contra falhas de conexão ou indisponibilidade de rede
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"   # Monta a URL do endpoint oficial sendMessage da API do Telegram
        resp = requests.post(                        # Dispara requisição HTTP POST síncrona para o servidor do Telegram
            url,                                     # Endereço da API montado acima
            data={                                   # Dicionário de dados do formulário HTTP (payload)
                "chat_id": TELEGRAM_CHAT_ID,         # Identificador do chat ou grupo que receberá a mensagem
                "text": mensagem,                    # Conteúdo textual da notificação
            },
            timeout=5,                               # Tempo limite de 5 segundos para evitar que falhas de internet travem o sistema
        )
        print(f"[TELEGRAM] Texto → status {resp.status_code}: {resp.text[:200]}")  # Exibe código HTTP de retorno e resumo da resposta
    except Exception as e:                           # Captura qualquer exceção de rede (DNS, timeout, queda de Wi-Fi/4G)
        print(f"[TELEGRAM] Erro ao enviar texto: {e}")  # Registra o erro no terminal sem interromper a execução principal


def enviar_telegram_foto(frame, caption=""):
    """
    Codifica o quadro de vídeo atual em formato JPEG diretamente na memória RAM
    e envia como foto anexada com legenda explicativa para o Telegram.
    Evita escrita e leitura desnecessárias no cartão microSD para o envio.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:   # Verifica se as credenciais do Telegram estão presentes antes de codificar a imagem
        print("[TELEGRAM] Token/Chat ID ausentes — alerta NÃO enviado.")    # Avisa no terminal caso faltem credenciais
        return                                       # Sai da função sem enviar

    try:                                             # Inicia bloco de tratamento de exceções para codificação e envio HTTP
        _, buffer = cv2.imencode(".jpg", frame)      # Comprime a matriz numpy do frame para o formato JPEG diretamente em um buffer na RAM
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"     # Monta a URL do endpoint oficial sendPhoto da API do Telegram
        resp = requests.post(                        # Envia requisição HTTP POST multipart/form-data contendo a imagem em bytes
            url,                                     # URL do endpoint sendPhoto
            data={                                   # Metadados textuais da requisição
                "chat_id": TELEGRAM_CHAT_ID,         # Destinatário do alerta
                "caption": caption,                  # Legenda da foto contendo o aviso de infração e o tempo de uso contínuo
            },
            files={                                  # Arquivo binário anexado à requisição HTTP
                "photo": (                           # Chave exigida pela API do Telegram para upload de imagens
                    "frame.jpg",                     # Nome virtual atribuído ao arquivo enviado
                    buffer.tobytes(),                # Converte o buffer numpy compactado para sequência bruta de bytes
                    "image/jpeg",                    # Cabeçalho MIME indicando que o conteúdo é uma imagem JPEG
                )
            },
            timeout=10,                              # Timeout de 10 segundos (maior que o de texto devido ao upload da imagem)
        )
        if resp.status_code == 200:                  # Código HTTP 200 confirma que o servidor do Telegram recebeu e processou a foto
            print("\n[TELEGRAM] Alerta enviado!")    # Confirma sucesso no terminal
        else:                                        # Caso a API retorne erro (ex.: 400 Bad Request, 401 Unauthorized)
            print(f"\n[TELEGRAM] Falha ao enviar. Status {resp.status_code}: {resp.text[:200]}")  # Exibe detalhes do erro HTTP
    except Exception as e:                           # Captura falhas de conexão durante o upload da foto
        print(f"[TELEGRAM] Erro: {e}")               # Imprime o erro sem derrubar o loop de monitoramento


# ==============================================================================
# 5. INICIALIZAÇÃO DOS MOTORES DE INFERÊNCIA (ONNX RUNTIME E TFLITE)
# ==============================================================================
# Os modelos são carregados uma única vez na memória RAM antes de iniciar o loop
# de captura de vídeo, evitando sobrecarga de I/O durante a operação em tempo real.

print("[INFO] Carregando YOLOv8 ONNX...")            # Informa no console o início do carregamento do detector de 1º estágio
yolo_session = ort.InferenceSession(                 # Instancia a sessão de inferência do ONNX Runtime
    YOLO_ONNX_PATH,                                  # Caminho do arquivo .onnx do YOLOv8n
    providers=["CPUExecutionProvider"],              # Define execução explícita na CPU ARM do Raspberry Pi
)
yolo_input_name = yolo_session.get_inputs()[0].name  # Recupera dinamicamente o nome do nó de entrada do grafo ONNX (ex.: "images")

print("[INFO] Carregando MobileNetV2 TFLite...")     # Informa no console o início do carregamento do classificador de 2º estágio
interpreter = Interpreter(model_path=TFLITE_PATH)    # Inicializa o interpretador LiteRT (TFLite) com o arquivo .tflite
interpreter.allocate_tensors()                       # Reserva na memória RAM os blocos necessários para os tensores de entrada e saída
input_details = interpreter.get_input_details()      # Obtém dicionário com índice, shape e dtype do tensor de entrada da MobileNetV2
output_details = interpreter.get_output_details()    # Obtém dicionário com índice, shape e dtype do tensor de saída da MobileNetV2
print("[INFO] Modelos carregados.\n")                # Confirma que ambos os motores de inferência estão prontos


# ==============================================================================
# 6. FUNÇÕES DE PRÉ-PROCESSAMENTO DE IMAGEM (OPENCV + NUMPY)
# ==============================================================================
# Adequam o quadro capturado pela câmera (formato BGR uint8) para o formato
# matemático exato exigido por cada rede neural (dimensões, ordem de canais
# RGB, escala de normalização em ponto flutuante e dimensão de batch).

def preprocess_mobilenet(frame_bgr):
    """
    Prepara o quadro BGR para a entrada da MobileNetV2 TFLite:
      1. Redimensiona para 256x256 pixels;
      2. Converte o espaço de cores de BGR para RGB;
      3. Normaliza os valores de pixel de [0, 255] para [-1.0, +1.0];
      4. Adiciona a dimensão de lote (batch), resultando no shape (1, 256, 256, 3).
    """
    img = cv2.resize(frame_bgr, IMG_SIZE_MOB)        # Redimensiona a imagem para 256x256 (largura x altura exigida pela MobileNetV2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # Converte os canais de cor do padrão OpenCV (BGR) para RGB
    arr = (img.astype(np.float32) / 127.5) - 1.0     # Converte para float32 e aplica normalização simétrica no intervalo [-1, 1]
    return np.expand_dims(arr, axis=0)               # Insere o eixo de batch na posição 0 -> shape final: (1, 256, 256, 3)


def preprocess_yolo(frame_bgr):
    """
    Prepara o quadro BGR para a entrada do YOLOv8n ONNX:
      1. Redimensiona para 640x640 pixels;
      2. Converte o espaço de cores de BGR para RGB;
      3. Normaliza os valores de pixel de [0, 255] para [0.0, 1.0];
      4. Transpõe os eixos do formato HWC (Altura, Largura, Canais) para CHW;
      5. Adiciona a dimensão de lote (batch), resultando no shape (1, 3, 640, 640).
    """
    img = cv2.resize(frame_bgr, IMG_SIZE_YOLO)       # Redimensiona o quadro para 640x640 (resolução padrão do YOLOv8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # Converte os canais de cor de BGR para RGB
    arr = img.astype(np.float32) / 255.0             # Converte para float32 e reescala os pixels para o intervalo [0.0, 1.0]
    arr = np.transpose(arr, (2, 0, 1))               # Reordena os eixos de (H, W, C) para (C, H, W), padrão exigido pelo ONNX Runtime
    return np.expand_dims(arr, axis=0)               # Insere o eixo de batch na posição 0 -> shape final: (1, 3, 640, 640)


# ==============================================================================
# 7. FUNÇÕES DE INFERÊNCIA (1º E 2º ESTÁGIOS)
# ==============================================================================

def inferir_tflite(arr):
    """
    Executa a passagem direta (forward pass) na MobileNetV2 TFLite e retorna
    a probabilidade escalar (float entre 0.0 e 1.0) de presença de celular em uso.
    """
    interpreter.set_tensor(input_details[0]["index"], arr)          # Copia o tensor pré-processado para o buffer de entrada do TFLite
    interpreter.invoke()                                            # Executa o grafo computacional da MobileNetV2 na CPU
    resultado = interpreter.get_tensor(output_details[0]["index"])  # Extrai a matriz de saída gerada pela camada final (Sigmoid)
    return float(resultado[0][0])                                   # Converte o elemento escalar [0][0] para float nativo do Python


def detectar_yolo(frame):
    """
    Executa a inferência do YOLOv8n via ONNX Runtime sobre o quadro atual.
    Inspeciona as 8400 caixas candidatas retornadas pelo modelo em busca da
    classe 67 ("cell phone") com confiança mínima YOLO_CONF.

    Retorna:
      (True, confianca) se encontrar ao menos um celular válido;
      (False, 0.0) caso contrário.
    """
    arr = preprocess_yolo(frame)                                    # Pré-processa o frame para o tensor (1, 3, 640, 640) esperado pelo ONNX
    outputs = yolo_session.run(None, {yolo_input_name: arr})        # Executa a inferência no ONNX Runtime e coleta todas as saídas do grafo
    preds = outputs[0][0].T                                         # Remove o eixo de batch e transpõe de (84, 8400) para (8400, 84)

    for det in preds:                                               # Itera sobre cada uma das 8400 predições candidatas do YOLOv8
        scores = det[4:]                                            # Os 4 primeiros valores são (x, y, w, h); do índice 4 em diante estão os scores das 80 classes COCO
        class_id = int(np.argmax(scores))                           # Identifica o índice da classe que obteve a maior pontuação nesta caixa
        conf = float(scores[class_id])                              # Extrai o valor de confiança correspondente à classe vencedora

        if class_id == YOLO_CLASS and conf >= YOLO_CONF:            # Verifica se a classe é "cell phone" (67) e supera o limiar mínimo (0.10)
            return True, conf                                       # Retorna imediatamente confirmando a detecção no 1º estágio

    return False, 0.0                                               # Retorna falso caso nenhuma caixa atenda aos critérios de celular


# ==============================================================================
# 8. RENDERIZAÇÃO DA INTERFACE VISUAL (HUD / OVERLAY)
# ==============================================================================

def overlay(frame, label, pred, color, yolo_conf=0.0, tempo_uso=0.0):
    """
    Desenha sobre o quadro de vídeo uma caixa de fundo preta semitransparente/sólida
    e escreve os indicadores de telemetria em tempo real:
      - Linha 1: Status da classificação (CELULAR / SEM CELULAR) + score da MobileNetV2;
      - Linha 2: Confiança do detector YOLOv8n;
      - Linha 3: Tempo acumulado de uso contínuo (exibido apenas quando > 0 s).
    """
    linha1 = f"{label} | mob: {pred:.2f}"                           # Formata a 1ª linha com o rótulo de estado e a probabilidade da MobileNetV2
    linha2 = f"YOLO: {yolo_conf:.2f}" if yolo_conf > 0 else "YOLO: ---"  # Formata a 2ª linha com a confiança do YOLO ou "---" se inativo
    linha3 = f"uso: {tempo_uso:.0f}s" if tempo_uso > 0 else ""      # Formata a 3ª linha com o contador de segundos de uso contínuo (se houver)

    linhas = [linha1, linha2]                                       # Inicia a lista de textos que serão renderizados no canto superior esquerdo
    if linha3:                                                      # Verifica se há tempo de uso acumulado para exibir
        linhas.append(linha3)                                       # Inclui a 3ª linha na lista de renderização

    for i, texto in enumerate(linhas):                              # Percorre cada linha de texto com seu respectivo índice vertical (i)
        (tw, th), _ = cv2.getTextSize(texto, FONT, 0.7, 2)          # Calcula a largura (tw) e altura (th) em pixels que o texto ocupará
        y = 10 + i * (th + 10)                                      # Calcula a coordenada vertical Y da linha atual com espaçamento de 10 px
        cv2.rectangle(frame, (8, y), (14 + tw, y + th + 4), (0, 0, 0), -1)  # Desenha retângulo preto preenchido (-1) atrás do texto para alto contraste
        cv2.putText(frame, texto, (10, y + th), FONT, 0.7, color, 2, cv2.LINE_AA)  # Escreve o texto colorido com suavização de bordas (LINE_AA)

    return frame                                                    # Devolve a matriz de imagem anotada pronta para exibição ou gravação


# ==============================================================================
# 9. LOOP PRINCIPAL DE CAPTURA, DECISÃO EM CASCATA E AUDITORIA
# ==============================================================================

def run_video():
    """
    Gerencia todo o ciclo de vida da execução embarcada:
      1. Inicializa e aquece a câmera CSI via Picamera2;
      2. Envia mensagem de partida ao Telegram;
      3. Executa o laço contínuo de captura e inferência intercalada (PREDICT_EVERY);
      4. Controla os cronômetros de persistência temporal, tolerância de oclusão e cooldown;
      5. Salva evidências visuais e exporta os relatórios CSV ao encerrar.
    """
    print("[INFO] Iniciando câmera CSI...")                         # Loga no terminal o início da configuração do hardware de câmera
    picam2 = Picamera2()                                            # Instancia o controlador principal da câmera CSI (libcamera)

    # Nota técnica sobre o Picamera2/libcamera: a especificação "RGB888" na API
    # DRM/libcamera armazena os bytes em memória na ordem (B, G, R), que é
    # diretamente compatível com o padrão nativo BGR esperado pelo OpenCV.
    config = picam2.create_preview_configuration(                   # Cria configuração otimizada para fluxo contínuo de vídeo (preview)
        main={"size": CAMERA_SIZE, "format": "RGB888"}              # Define resolução 640x480 e entrega de pixels compatível com OpenCV
    )
    picam2.configure(config)                                        # Aplica a configuração de resolução e formato ao sensor CSI
    picam2.start()                                                  # Inicia o fluxo de captura contínua de quadros
    time.sleep(2)                                                   # Aguarda 2 segundos para estabilização automática de ganho, exposição e balanço de branco

    print(f"[CÂMERA] {CAMERA_SIZE[0]}x{CAMERA_SIZE[1]}")            # Confirma no terminal a resolução ativa de captura
    print("Pressione 'q' para sair.\n")                             # Orienta o usuário sobre a tecla de encerramento
    enviar_telegram_texto("🚗 DriveSafe Vision iniciado")           # Notifica no Telegram que o monitoramento veicular entrou em operação

    # --- Inicialização dos Contadores e Variáveis de Estado ---
    frame_count = 0                                                 # Total de quadros capturados desde o início da sessão
    phone_count = 0                                                 # Total de quadros avaliados em que o uso de celular foi confirmado
    no_phone_count = 0                                              # Total de quadros avaliados classificados como condução sem celular
    yolo_only_count = 0                                             # Total de quadros onde o YOLO detectou candidato, mas a MobileNetV2 rejeitou

    pred_buffer = deque(maxlen=SMOOTH_WINDOW)                       # Buffer circular para armazenar as últimas 5 probabilidades da MobileNetV2
    last_pred = 0.0                                                 # Última probabilidade inferida pela MobileNetV2 (mantida no HUD entre inferências)
    last_yolo_conf = 0.0                                            # Última confiança obtida pelo YOLOv8n (mantida no HUD entre inferências)
    last_label = "SEM CELULAR"                                      # Rótulo textual vigente exibido na tela
    last_color = COLOR_NO_PHONE                                     # Cor vigente do HUD (inicia em verde = sem celular)

    tempos = []                                                     # Lista que acumula a duração (em segundos) de cada ciclo de inferência
    registros = []                                                  # Lista de dicionários contendo os dados de cada quadro avaliado para o CSV

    inicio_mobilenet = None                                         # Marca o timestamp de início do uso contínuo atual (None = sem uso ativo)
    ultimo_alerta = None                                            # Marca o timestamp do último alerta enviado ao Telegram (controle de cooldown)
    ultimo_sem_celular = None                                       # Marca o timestamp de quando o celular sumiu pela última vez (tolerância de oclusão)
    tempo_uso_continuo = 0.0                                        # Duração acumulada (em segundos) do evento de uso contínuo em andamento

    try:                                                            # Bloco try garante que a câmera será liberada no bloco finally mesmo em caso de erro
        while True:                                                 # Laço principal de monitoramento em tempo real
            frame = picam2.capture_array()                          # Captura o quadro atual diretamente do buffer da câmera CSI como array numpy
            agora = time.time()                                     # Registra o instante atual (epoch em segundos) para cálculos temporais

            # ------------------------------------------------------------------
            # ETAPA DE INFERÊNCIA EM CASCATA (EXECUTADA A CADA `PREDICT_EVERY` FRAMES)
            # ------------------------------------------------------------------
            if frame_count % PREDICT_EVERY == 0:                    # Verifica se o quadro atual é múltiplo de PREDICT_EVERY (ex.: 0, 3, 6, 9...)
                t0 = time.perf_counter()                            # Aciona cronômetro de alta precisão antes de iniciar os modelos

                yolo_ok, yolo_conf = detectar_yolo(frame)           # 1º Estágio: executa o YOLOv8n ONNX para buscar celular na cena

                if yolo_ok:                                         # Se o YOLOv8n encontrou um celular candidato:
                    arr = preprocess_mobilenet(frame)               # Pré-processa o quadro para o formato 256x256 RGB normalizado em [-1, 1]
                    p = inferir_tflite(arr)                         # 2º Estágio: executa a MobileNetV2 TFLite para validar o uso do celular
                else:                                               # Se o YOLOv8n NÃO encontrou celular na cena:
                    p = 0.0                                         # Poupa a CPU de rodar a MobileNetV2 e atribui probabilidade zero diretamente

                t1 = time.perf_counter()                            # Para o cronômetro de alta precisão após concluir a inferência
                tempos.append(t1 - t0)                              # Armazena o tempo gasto neste ciclo de inferência (em segundos)

                pred_buffer.append(p)                               # Insere a probabilidade atual na fila circular de suavização
                pred_media = float(np.mean(pred_buffer))            # Calcula a média móvel das últimas predições armazenadas

                # --------------------------------------------------------------
                # CASO A: USO DE CELULAR CONFIRMADO PELOS DOIS ESTÁGIOS
                # --------------------------------------------------------------
                if yolo_ok and p > THRESHOLD:                       # Condição dupla: YOLO detectou objeto E MobileNetV2 superou o limiar
                    classe = "phone"                                # Define classe para gravação no log CSV
                    last_label = "CELULAR"                          # Atualiza texto do HUD para alerta de celular
                    last_color = COLOR_PHONE                        # Atualiza cor do HUD para vermelho
                    last_yolo_conf = yolo_conf                      # Atualiza confiança do YOLO exibida no HUD
                    phone_count += 1                                # Incrementa contador estatístico de quadros positivos

                    fname = os.path.join(OUTPUT_PHONE, f"frame_{frame_count:05d}.jpg")  # Define caminho de salvamento na pasta "phone"
                    ultimo_sem_celular = None                       # Zera o marcador de ausência, pois o celular está visível neste instante

                    if inicio_mobilenet is None:                    # Se este é o primeiro quadro positivo de uma nova sequência de uso:
                        inicio_mobilenet = agora                    # Registra o timestamp exato em que o uso contínuo começou

                    tempo_uso_continuo = agora - inicio_mobilenet   # Atualiza o tempo total (em segundos) que o motorista está usando o aparelho

                    cooldown_ok = (                                 # Avalia se o sistema já está autorizado a enviar um novo alerta no Telegram:
                        ultimo_alerta is None                       # Verdadeiro se nenhum alerta foi enviado ainda nesta sessão, OU
                        or (agora - ultimo_alerta) >= TEMPO_REPETICAO_ALERTA  # se já transcorreram pelo menos 30s desde o último envio
                    )

                    if tempo_uso_continuo >= TEMPO_MINIMO_ALERTA and cooldown_ok:  # Se atingiu 2s de persistência E o cooldown permite:
                        ultimo_alerta = agora                       # Atualiza o timestamp do último alerta disparado
                        minutos = int(tempo_uso_continuo // 60)     # Converte o tempo contínuo acumulado para minutos inteiros
                        segundos = int(tempo_uso_continuo % 60)     # Calcula os segundos restantes
                        caption = (                                 # Monta a legenda informativa que acompanhará a foto no Telegram
                            "⚠️ ALERTA: USO DE CELULAR AO VOLANTE!\n"
                            f"Tempo de uso contínuo: {minutos}min {segundos}s"
                        )
                        enviar_telegram_foto(frame, caption)        # Comprime o quadro em RAM e envia evidência fotográfica ao Telegram

                # --------------------------------------------------------------
                # CASO B: SEM USO DE CELULAR CONFIRMADO
                # --------------------------------------------------------------
                else:                                               # Ocorre se o YOLO não detectou nada OU se a MobileNetV2 rejeitou a detecção
                    if yolo_ok and p <= THRESHOLD:                  # Subcaso B1: YOLO detectou candidato, mas MobileNetV2 classificou como não-uso
                        classe = "no_phone"                         # Registra no CSV como sem celular confirmado
                        last_label = "SEM CELULAR"                  # Mantém rótulo de segurança no HUD
                        last_color = COLOR_YOLO                     # Altera cor do HUD para laranja (indica filtragem ativa do 2º estágio)
                        last_yolo_conf = yolo_conf                  # Exibe a confiança que o YOLO teve no objeto descartado
                        yolo_only_count += 1                        # Incrementa contador de falsos positivos filtrados pela MobileNetV2
                        no_phone_count += 1                         # Incrementa contador geral de quadros sem infração
                    else:                                           # Subcaso B2: Nem o YOLO detectou celular na cena
                        classe = "no_phone"                         # Registra no CSV como sem celular
                        last_label = "SEM CELULAR"                  # Rótulo de condução segura
                        last_color = COLOR_NO_PHONE                 # Cor verde no HUD
                        last_yolo_conf = 0.0                        # Zera confiança do YOLO no HUD
                        no_phone_count += 1                         # Incrementa contador de quadros sem celular

                    fname = os.path.join(OUTPUT_NO_PHONE, f"frame_{frame_count:05d}.jpg")  # Define caminho de salvamento na pasta "no_phone"

                    if ultimo_sem_celular is None:                  # Se o celular acabou de desaparecer neste quadro:
                        ultimo_sem_celular = agora                  # Inicia a contagem de tempo de ausência/oclusão

                    if (agora - ultimo_sem_celular) >= TOLERANCIA_RESET:  # Se a ausência já superou a tolerância de 2 segundos:
                        inicio_mobilenet = None                     # Reseta o marcador de início de uso contínuo
                        tempo_uso_continuo = 0.0                    # Zera o cronômetro de uso contínuo (preservando `ultimo_alerta` para manter o cooldown)

                last_pred = p                                       # Armazena a predição atual para continuar exibindo nos quadros intermediários

                # --- Gravação do Quadro Anotado e Registro de Log ---
                frame_out = overlay(                                # Cria cópia do quadro com o HUD desenhado para salvar como evidência em disco
                    frame.copy(),
                    last_label,
                    last_pred,
                    last_color,
                    last_yolo_conf,
                    tempo_uso_continuo,
                )
                cv2.imwrite(fname, frame_out)                       # Grava o arquivo JPEG anotado na pasta correspondente (phone ou no_phone)

                registros.append({                                  # Anexa dicionário com as métricas completas deste quadro para o CSV detalhado
                    "frame": frame_count,                           # Número sequencial do quadro
                    "modelo": "YOLO_ONNX+TFLite",                   # Identificação da arquitetura em execução
                    "yolo_detectou": yolo_ok,                       # Booleano indicando se o 1º estágio foi ativado
                    "yolo_conf": round(yolo_conf, 4),               # Confiança do YOLOv8n arredondada em 4 casas decimais
                    "pred_bruta": round(p, 4),                      # Probabilidade instantânea da MobileNetV2
                    "pred_media": round(pred_media, 4),             # Probabilidade suavizada pela janela móvel
                    "predicao": classe,                             # Decisão final ("phone" ou "no_phone")
                    "tempo_uso_s": round(tempo_uso_continuo, 1),    # Tempo de uso contínuo acumulado até este instante (em segundos)
                    "tempo_ms": round((t1 - t0) * 1000, 2),         # Latência de inferência do pipeline neste quadro (em milissegundos)
                    "threshold_mob": THRESHOLD,                     # Limiar configurado para a MobileNetV2
                    "threshold_yolo": YOLO_CONF,                    # Limiar configurado para o YOLOv8n
                })

                print(                                              # Atualiza linha de status dinâmica no terminal (sobrescreve com \r)
                    f"YOLO={yolo_ok}({yolo_conf:.2f}) mob={p:.4f} uso={tempo_uso_continuo:.0f}s → {last_label}",
                    end="\r",
                )

            # ------------------------------------------------------------------
            # EXIBIÇÃO EM TEMPO REAL NA TELA (RODA EM TODOS OS QUADROS)
            # ------------------------------------------------------------------
            frame_display = overlay(                                # Desenha o HUD usando os estados mais recentes (mesmo nos quadros sem inferência)
                frame.copy(),
                last_label,
                last_pred,
                last_color,
                last_yolo_conf,
                tempo_uso_continuo,
            )
            cv2.imshow("DriveSafe Vision", frame_display)           # Atualiza a janela gráfica com o quadro anotado ao vivo
            frame_count += 1                                        # Avança o contador global de quadros capturados

            if cv2.waitKey(1) & 0xFF == ord("q"):                   # Aguarda 1 ms por evento de teclado e verifica se a tecla 'q' foi pressionada
                print("\nEncerrado pelo usuário.")                  # Informa o encerramento voluntário no console
                break                                               # Interrompe o laço `while True`

    finally:                                                        # Bloco executado obrigatoriamente ao sair do loop (normalmente ou por erro)
        print("\n[INFO] Encerrando câmera...")                      # Informa liberação do hardware
        picam2.stop()                                               # Interrompe o stream da câmera CSI com segurança
        cv2.destroyAllWindows()                                     # Fecha todas as janelas gráficas abertas pelo OpenCV

    # ==========================================================================
    # 10. EXPORTAÇÃO DE RELATÓRIOS CSV E CONSOLIDAÇÃO DE MÉTRICAS ESTATÍSTICAS
    # ==========================================================================

    pd.DataFrame(registros).to_csv(CSV_VIDEO, index=False)          # Converte a lista de registros em tabela Pandas e salva o CSV frame a frame
    print(f"\nCSV salvo em: {CSV_VIDEO}")                           # Confirma o caminho do arquivo detalhado gerado
    enviar_telegram_texto("✅ DriveSafe Vision encerrado")          # Envia aviso final ao Telegram informando término do monitoramento

    if tempos:                                                      # Verifica se houve ao menos uma inferência realizada para calcular estatísticas
        tempos_ms = np.array(tempos) * 1000                         # Converte o vetor de tempos de segundos para milissegundos
        tempo_total = float(np.sum(tempos_ms))                      # Soma o tempo total dedicado exclusivamente às inferências (em ms)
        tempo_medio = float(np.mean(tempos_ms))                     # Calcula a latência média por inferência (em ms)
        tempo_var = float(np.var(tempos_ms))                        # Calcula a variância da latência de inferência (em ms²)
        tempo_desvio = float(np.std(tempos_ms))                     # Calcula o desvio padrão da latência (em ms)
        tempo_min = float(np.min(tempos_ms))                        # Identifica a menor latência registrada na sessão (em ms)
        tempo_max = float(np.max(tempos_ms))                        # Identifica a maior latência registrada na sessão (em ms)
        fps_efetivo = frame_count / (tempo_total / 1000) if tempo_total > 0 else 0  # Calcula a taxa de quadros efetiva (FPS) do sistema

        print("\n==============================")                   # Imprime cabeçalho visual do resumo no terminal
        print("RESUMO")                                             # Título do bloco de métricas
        print("==============================")                     # Linha separadora
        print(f"Frames processados  : {frame_count}")               # Exibe total de frames lidos da câmera
        print(f"Frames com celular  : {phone_count} ({100 * phone_count / max(1, frame_count):.1f}%)")        # Total e percentual com celular
        print(f"Frames sem celular  : {no_phone_count} ({100 * no_phone_count / max(1, frame_count):.1f}%)")  # Total e percentual sem celular
        print("------------------------------")                     # Linha divisória interna
        print(f"FPS efetivo         : {fps_efetivo:.1f}")           # Exibe velocidade média de processamento em quadros por segundo
        print("==============================")                     # Rodapé do resumo

        pd.DataFrame([{                                             # Constrói tabela de linha única com o resumo estatístico completo da sessão
            "modelo": "YOLO_ONNX+TFLite",                           # Nome do pipeline avaliado
            "frames_processados": frame_count,                      # Total de quadros capturados
            "frames_celular": phone_count,                          # Total de quadros com detecção confirmada
            "frames_sem_celular": no_phone_count,                   # Total de quadros sem detecção confirmada
            "yolo_sem_confirmacao": yolo_only_count,                # Total de ativações do YOLO rejeitadas pela MobileNetV2
            "pct_celular": round(100 * phone_count / max(1, frame_count), 1),        # Porcentagem de quadros com celular
            "pct_sem_celular": round(100 * no_phone_count / max(1, frame_count), 1), # Porcentagem de quadros sem celular
            "tempo_total_ms": round(tempo_total, 2),                # Soma do tempo de inferência (ms)
            "tempo_medio_ms": round(tempo_medio, 2),                # Latência média por ciclo de inferência (ms)
            "tempo_min_ms": round(tempo_min, 2),                    # Latência mínima (ms)
            "tempo_max_ms": round(tempo_max, 2),                    # Latência máxima (ms)
            "variancia_ms2": round(tempo_var, 2),                   # Variância temporal (ms²)
            "desvio_padrao_ms": round(tempo_desvio, 2),             # Desvio padrão temporal (ms)
            "fps_efetivo": round(fps_efetivo, 1),                   # Taxa de FPS efetivo alcançada
            "threshold_mobilenet": THRESHOLD,                       # Limiar utilizado na MobileNetV2
            "threshold_yolo": YOLO_CONF,                            # Limiar utilizado no YOLOv8n
        }]).to_csv(CSV_RESUMO, index=False)                         # Exporta o resumo consolidado para arquivo CSV
        print(f"CSV resumo salvo em: {CSV_RESUMO}")                 # Confirma gravação do CSV resumo no console


# ==============================================================================
# 11. PONTO DE ENTRADA DO SCRIPT (ENTRY POINT)
# ==============================================================================

def main():
    """Função principal de inicialização do programa."""
    run_video()                                                     # Dispara o pipeline principal de captura e detecção


if __name__ == "__main__":                                          # Verifica se este arquivo está sendo executado diretamente pelo Python
    main()                                                          # Invoca a função principal
