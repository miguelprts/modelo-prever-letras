# app.py (VERSÃO CORRIGIDA)

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
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo Keras treinado."""
    try:
        caminho_modelo = 'meu_modelo_gestos.keras'
        modelo = tf.keras.models.load_model(caminho_modelo)
        return modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

modelo = carregar_modelo()

# Cole aqui a sua lista de classes gerada no Colab
class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing'
]

def preprocessar_imagem(imagem):
    """Prepara a imagem para ser enviada ao modelo."""
    img = imagem.resize((128, 128))
    
    # --- MUDANÇA PRINCIPAL AQUI ---
    # Convertendo para array e GARANTINDO o tipo de dado float32
    img_array = np.array(img, dtype=np.float32) 
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- INTERFACE DA APLICAÇÃO ---
st.title("🤖 Classificador de Gestos em Libras")
st.markdown("Faça o upload de uma imagem e veja a mágica acontecer!")

arquivo_carregado = st.file_uploader("Escolha uma imagem de um gesto...", type=["jpg", "jpeg", "png"])

if modelo is not None and arquivo_carregado is not None:
    imagem = Image.open(arquivo_carregado).convert("RGB")
    
    # --- PEQUENA CORREÇÃO NO PARÂMETRO DO STREAMLIT ---
    st.image(imagem, caption="Imagem carregada.", use_container_width=True)

    st.markdown("---")
    st.write("### 🧠 Analisando a imagem...")

    imagem_processada = preprocessar_imagem(imagem)
    predicao = modelo.predict(imagem_processada)[0]

    top_indices = predicao.argsort()[-5:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_confidences = [predicao[i] * 100 for i in top_indices]

    st.success(f"**Principal Palpite:** **{top_classes[0]}** ({top_confidences[0]:.2f}%)")

    st.write("#### Detalhes das Previsões:")
    
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