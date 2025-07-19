import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Classificador de Gestos em Libras",
    page_icon="🤟",
    layout="centered"
)

# --- FUNÇÕES E CONFIGURAÇÕES ---

# Usamos o cache do Streamlit para carregar o modelo apenas uma vez, otimizando a performance
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo Keras treinado."""
    try:
        # Caminho para o modelo salvo
        caminho_modelo = 'meu_modelo_gestos.keras'
        modelo = tf.keras.models.load_model(caminho_modelo)
        return modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Carrega o modelo ao iniciar o app
modelo = carregar_modelo()

# =====================================================================================
# ATENÇÃO: COLE AQUI A LISTA `class_names` EXATA QUE FOI GERADA NO SEU NOTEBOOK COLAB
# A ORDEM É FUNDAMENTAL PARA O RESULTADO CORRETO!
# =====================================================================================
class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 
    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing'
]
# Se sua lista for diferente (ex: sem o 'nothing'), ajuste-a para que corresponda!


def preprocessar_imagem(imagem):
    """Prepara a imagem para ser enviada ao modelo."""
    # O modelo foi treinado com imagens 128x128
    img = imagem.resize((128, 128))
    # Converte a imagem para um array numpy
    img_array = np.array(img)
    # Adiciona uma dimensão para criar o "lote" (batch) de uma única imagem
    img_array = np.expand_dims(img_array, axis=0)
    # IMPORTANTE: Não dividimos por 255.0 aqui, pois o modelo já possui uma camada de Rescaling.
    return img_array


# --- INTERFACE DA APLICAÇÃO ---

st.title("🤖 Classificador de Gestos em Libras")
st.markdown("Faça o upload de uma imagem e veja a mágica acontecer!")

# Área para upload de imagem
arquivo_carregado = st.file_uploader("Escolha uma imagem de um gesto...", type=["jpg", "jpeg", "png"])

# Verifica se o modelo foi carregado e se um arquivo foi enviado
if modelo is not None and arquivo_carregado is not None:
    # Mostra a imagem na tela
    imagem = Image.open(arquivo_carregado).convert("RGB")
    st.image(imagem, caption="Imagem carregada.", use_column_width=True)

    st.markdown("---")
    st.write("### 🧠 Analisando a imagem...")

    # Prepara a imagem e faz a predição
    imagem_processada = preprocessar_imagem(imagem)
    predicao = modelo.predict(imagem_processada)[0] # Pega o primeiro (e único) resultado do lote

    # Pega os 5 melhores resultados
    top_indices = predicao.argsort()[-5:][::-1] # Índices das 5 maiores probabilidades
    top_classes = [class_names[i] for i in top_indices]
    top_confidences = [predicao[i] * 100 for i in top_indices]

    # Mostra o resultado principal
    st.success(f"**Principal Palpite:** **{top_classes[0]}** ({top_confidences[0]:.2f}%)")

    # Mostra um gráfico com as 5 principais previsões
    st.write("#### Detalhes das Previsões:")
    
    # Cria um DataFrame do pandas para o gráfico
    df_predicoes = pd.DataFrame({
        'Gesto': top_classes,
        'Confiança (%)': top_confidences
    })
    
    st.dataframe(
        df_predicoes,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(df_predicoes.set_index('Gesto'))

elif modelo is None:
    st.error("O modelo não pôde ser carregado. Verifique o console para erros.")
else:
    st.info("Aguardando o upload de uma imagem.")