from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.endpoints import predict_endpoints
from app.ml.model_loader import load_all_models, ml_models
from app.core.config import settings

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    print("Aplicativo iniciando... Carregando modelos de ML.")
    load_all_models()
    if not all(ml_models.values()):
        print("ALERTA: Um ou mais modelos/vetorizadores não foram carregados corretamente!")
    else:
        print("Modelos carregados. Aplicativo pronto.")
    yield
    print("Aplicativo encerrando.")

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.include_router(predict_endpoints.router)

@app.get("/")
async def root():
    return {"message": f"Bem-vindo à {settings.APP_NAME}"}
