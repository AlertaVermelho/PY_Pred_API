from pydantic_settings import BaseSettings
from pydantic import Field
import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Settings(BaseSettings):
    APP_NAME: str = "Red Alert - API de Classificação de Desastres Urbanos"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Esta API classifica o tipo e a severidade de relatos de desastres urbanos."

    MODELS_DIR: str = os.path.join(PROJECT_ROOT_DIR, "saved_models")

    VETORIZADOR_TIPO_FILENAME: str = "vetorizador_tfidf_TIPO_final.joblib"
    MODELO_TIPO_FILENAME: str = "modelo_svc_TIPO_final.joblib"
    VETORIZADOR_SEVERIDADE_FILENAME: str = "vetorizador_tfidf_SEVERIDADE_final.joblib"
    MODELO_SEVERIDADE_FILENAME: str = "modelo_svc_SEVERIDADE_tuned_final.joblib"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        pass

settings = Settings()

if __name__ == "__main__":
    print(f"Raiz do Projeto (detectada): {PROJECT_ROOT_DIR}")
    print(f"Pasta de Modelos (configurada): {settings.MODELS_DIR}")
    print(f"Caminho completo para vetorizador_tipo: {os.path.join(settings.MODELS_DIR, settings.VETORIZADOR_TIPO_FILENAME)}")
