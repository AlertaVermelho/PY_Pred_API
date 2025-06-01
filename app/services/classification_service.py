import numpy as np
from scipy.sparse import hstack, csr_matrix

from app.utils.text_processor import preprocessar_texto, contar_palavras_lexico
from app.ml.lexicons import LEXICOS_SEVERIDADE
from app.ml.model_loader import ml_models

ORDEM_LEXICOS_SEVERIDADE = ['critica', 'alta', 'media', 'baixa']

def obter_predicoes_classificacao(texto_original: str) -> dict:
    """
    Processa o texto de um alerta e retorna as classificações de tipo e severidade.
    """
    if not ml_models.get('vetorizador_tipo') or \
       not ml_models.get('modelo_tipo') or \
       not ml_models.get('vetorizador_severidade') or \
       not ml_models.get('modelo_severidade'):
        raise RuntimeError("Modelos de classificação ou vetorizadores não foram carregados corretamente.")

    texto_processado = preprocessar_texto(texto_original)

    vetorizador_tipo = ml_models['vetorizador_tipo']
    modelo_tipo = ml_models['modelo_tipo']
    
    features_tfidf_tipo = vetorizador_tipo.transform([texto_processado])
    predicao_tipo_array = modelo_tipo.predict(features_tfidf_tipo)
    tipo_predito = predicao_tipo_array[0]

    vetorizador_severidade = ml_models['vetorizador_severidade']
    modelo_severidade = ml_models['modelo_severidade']

    features_tfidf_severidade = vetorizador_severidade.transform([texto_processado])
    
    features_lexico_severidade_lista = []
    for nome_lex in ORDEM_LEXICOS_SEVERIDADE:
        lista_palavras = LEXICOS_SEVERIDADE[nome_lex] 
        features_lexico_severidade_lista.append(
            contar_palavras_lexico(texto_processado, lista_palavras)
        )
    
    features_lexico_severidade_array = np.array([features_lexico_severidade_lista]) 
    
    features_combinadas_severidade = hstack([
        features_tfidf_severidade, 
        csr_matrix(features_lexico_severidade_array)
    ])
    
    predicao_severidade_array = modelo_severidade.predict(features_combinadas_severidade)
    severidade_predita = predicao_severidade_array[0]
    
    return {
        "alertId": None,
        "classifiedType": tipo_predito,
        "classifiedSeverity": severidade_predita
    }
