import pandas as pd
from typing import Tuple, Optional

def calcular_distribuicao_percentual(series: pd.Series) -> pd.Series:
    if series.empty: return pd.Series(dtype=float)
    return series.value_counts(normalize=True)

def obter_item_predominante(series: pd.Series, limiar_percentual: float) -> Tuple[Optional[str], float]:
    if series.empty: return None, 0.0
    distribuicao = calcular_distribuicao_percentual(series)
    if not distribuicao.empty:
        item_mais_comum = distribuicao.index[0]
        percentual_mais_comum = distribuicao.iloc[0]
        if percentual_mais_comum >= limiar_percentual:
            return str(item_mais_comum), float(percentual_mais_comum)
    return None, 0.0