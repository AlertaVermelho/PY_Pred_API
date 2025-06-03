from pydantic import BaseModel, Field
from typing import List, Optional

class RelatoInputSchema(BaseModel):
    alertId: int
    text: str = Field(..., min_length=1, description="O texto da descrição do alerta. Obrigatório e não pode ser vazio.")

class PredicaoOutputSchema(BaseModel):
    alertId: int
    classifiedSeverity: str
    classifiedType: str

class AlertItemForClustering(BaseModel):
    alertId: int
    latitude: float
    longitude: float
    severityIA: str
    typeIA: str
    timestampReporte: Optional[str] = None 

class ClusteringInput(BaseModel):
    alertsToCluster: List[AlertItemForClustering] = Field(..., min_items=1)

class ClusteringResultItem(BaseModel):
    alertId: int
    clusterLabel: int

class HotspotSummaryOutput(BaseModel):
    clusterLabel: int
    centroidLat: float
    centroidLon: float
    pointCount: int
    dominantSeverity: str
    dominantType: str
    alertIdsInCluster: List[int]
    estimatedRadiusKm: Optional[float] = None
    publicSummary: Optional[str] = None
    lastActivityTimestamp: Optional[str] = None

class ClusteringResponse(BaseModel):
    clusteringResults: List[ClusteringResultItem]
    hotspotSummaries: List[HotspotSummaryOutput]

class ErrorDetail(BaseModel):
    error: str
    message: str