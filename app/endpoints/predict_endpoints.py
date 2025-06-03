from fastapi import APIRouter, HTTPException, Body
from typing import Annotated

from app.models_schemas.schemas import RelatoInputSchema, PredicaoOutputSchema, ErrorDetail
from app.services.classification_service import obter_predicoes_classificacao
from app.models_schemas.schemas import ClusteringInput, ClusteringResponse, HotspotSummaryOutput, ClusteringResultItem, ErrorDetail
from app.services.clustering_service import realizar_clustering_alertas

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
    

@router.post(
    "/cluster_alerts",
    response_model=ClusteringResponse,
    summary="Agrupa alertas geograficamente para identificar hotspots",
    responses={
        200: {"description": "Clustering realizado com sucesso."},
        400: {"model": ErrorDetail, "description": "A lista 'alertsToCluster' é obrigatória e seus itens devem conter os campos requeridos."},
        500: {"model": ErrorDetail, "description": "Falha no processo de clustering."}
    }
)
async def cluster_alerts_endpoint_v2(payload: ClusteringInput):
    """
    Recebe uma lista de alertas (com suas localizações e classificações de IA)
    e retorna os hotspots identificados e a atribuição de cluster para cada alerta.
    """
    if not payload.alertsToCluster:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Bad Request",
                "message": "A lista 'alertsToCluster' é obrigatória e não pode ser vazia."
            }
        )
    
    for alerta in payload.alertsToCluster:
        if not all([
            hasattr(alerta, 'alertId'), hasattr(alerta, 'latitude'), hasattr(alerta, 'longitude'),
            hasattr(alerta, 'severityIA'), hasattr(alerta, 'typeIA')
        ]):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Bad Request",
                    "message": "Itens em 'alertsToCluster' devem conter 'alertId', 'latitude', 'longitude', 'severityIA', e 'typeIA'."
                }
            )

    try:
        clustering_results_list, hotspot_summaries_list_of_dicts = realizar_clustering_alertas(payload.alertsToCluster)        
        hotspot_summaries_obj_list = [HotspotSummaryOutput(**data) for data in hotspot_summaries_list_of_dicts]

        return ClusteringResponse(
            clusteringResults=[ClusteringResultItem(**item) for item in clustering_results_list],
            hotspotSummaries=hotspot_summaries_obj_list
        )
    
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail={"error": "Bad Request", "message": str(ve)}
        )
    
    except Exception as e:
        print(f"Erro crítico durante o clustering: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": "Falha no processo de clustering."}
        )
