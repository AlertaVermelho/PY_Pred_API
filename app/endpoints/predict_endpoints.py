from fastapi import APIRouter, HTTPException, Body
from typing import Annotated

from app.models_schemas.schemas import RelatoInputSchema, PredicaoOutputSchema, ErrorDetail
from app.services.classification_service import obter_predicoes_classificacao

router = APIRouter(
    prefix="/ia",
    tags=["IA Classification"]
)

@router.post(
    "/classify_text",
    response_model=PredicaoOutputSchema,
    summary="Classifica a severidade e o tipo de um texto de alerta",
    responses={
        200: {"description": "Classificação bem-sucedida"},
        400: {"model": ErrorDetail, "description": "Requisição inválida (ex: campo 'text' ausente ou vazio). O Pydantic geralmente retorna 422 para falhas de validação."},
        500: {"model": ErrorDetail, "description": "Erro interno no servidor durante a classificação."}
    }
)
async def classify_text_endpoint(relato_input: Annotated[RelatoInputSchema, Body(
                                description="ID do alerta e o texto a ser classificado.",
                                examples=[
                                    {
                                        "alertId": 12345,
                                        "text": "Grande deslizamento de terra bloqueou a via principal no Morro da Esperança. Risco de novas ocorrências."
                                    },
                                    {
                                        "alertId": 12346,
                                        "text": "Alagamento na Marginal Tietê causa lentidão."
                                    }
                                ]
                            )]):
    """
    Recebe um texto de alerta (com seu ID) e retorna a classificação de severidade e tipo.
    """
    try:
        resultados_predicao = obter_predicoes_classificacao(texto_original=relato_input.text)

        return PredicaoOutputSchema(
            alertId=relato_input.alertId,
            classifiedSeverity=resultados_predicao["classifiedSeverity"],
            classifiedType=resultados_predicao["classifiedType"]
        )
    except RuntimeError as e:
        print(f"Erro de Runtime (modelos não carregados?): {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "Service Unavailable", "message": str(e)}
        )
    except Exception as e:
        print(f"Erro inesperado durante a classificação: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": "Erro ao processar a classificação do texto."}
        )
