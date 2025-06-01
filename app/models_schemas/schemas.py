from pydantic import BaseModel, Field
from typing import Optional

class RelatoInputSchema(BaseModel):
    alertId: int
    text: str = Field(..., min_length=1, description="O texto da descrição do alerta. Obrigatório e não pode ser vazio.")

class PredicaoOutputSchema(BaseModel):
    alertId: int
    classifiedSeverity: str
    classifiedType: str

class ErrorDetail(BaseModel):
    error: str
    message: Optional[str]
