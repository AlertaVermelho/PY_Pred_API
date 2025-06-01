import joblib
import os
from app.core.config import settings

MODELS_PATH = settings.MODELS_DIR

ml_models = {}

def load_all_models():
    """Carrega todos os modelos e vetorizadores na inicialização."""
    if not ml_models: # Carregar apenas uma vez
        print(f"Carregando modelos e vetorizadores da pasta: {MODELS_PATH}...")
        try:
            ml_models['vetorizador_tipo'] = joblib.load(os.path.join(MODELS_PATH, settings.VETORIZADOR_TIPO_FILENAME))
            ml_models['modelo_tipo'] = joblib.load(os.path.join(MODELS_PATH, settings.MODELO_TIPO_FILENAME))
            ml_models['vetorizador_severidade'] = joblib.load(os.path.join(MODELS_PATH, settings.VETORIZADOR_SEVERIDADE_FILENAME))
            ml_models['modelo_severidade'] = joblib.load(os.path.join(MODELS_PATH, settings.MODELO_SEVERIDADE_FILENAME))
            print("Modelos e vetorizadores carregados com sucesso.")
        except FileNotFoundError as e:
            print(f"Erro CRÍTICO: Arquivo de modelo não encontrado. Verifique os caminhos e nomes de arquivo nas configurações e na pasta '{MODELS_PATH}'. Detalhes: {e}")
        except Exception as e:
            print(f"Erro CRÍTICO ao carregar modelos: {e}")
    return ml_models

load_all_models()
