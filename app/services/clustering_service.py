import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from collections import Counter
from typing import List, Dict, Tuple
from datetime import datetime, timezone

from app.models_schemas.schemas import AlertItemForClustering
from app.utils.cluster_utils import calcular_distribuicao_percentual, obter_item_predominante

R_TERRA_KM = 6371.0088
DBSCAN_EPS_KM = 0.200
DBSCAN_MIN_SAMPLES = 4
REFINEMENT_MIN_ALERTS_FOR_HOTSPOT = 3
REFINEMENT_PORC_SEVERIDADE_PREDOMINANTE = 0.50
REFINEMENT_PORC_ALTA_EM_MEDIA = 0.25
DEFAULT_HOTSPOT_RADIUS_KM_SINGLE_POINT = 0.05


def refinar_e_caracterizar_hotspots_para_api(
    df_alertas_com_cluster_ids: pd.DataFrame,
    config_refinamento: dict
) -> List[Dict]:
    
    hotspots_formatados = []
    ids_clusters_unicos = df_alertas_com_cluster_ids['cluster_id_dbscan'].unique()
    hotspot_output_id_counter = 0

    min_alertas_para_hotspot_refinado = config_refinamento.get("MIN_ALERTAS_PARA_HOTSPOT_REFINADO", 3)
    porc_severidade_predominante = config_refinamento.get("PORCENTAGEM_PARA_SEVERIDADE_PREDOMINANTE", 0.50)
    porc_alta_em_media = config_refinamento.get("PORCENTAGEM_ALTA_EM_MEDIA_PARA_HOTSPOT", 0.25)

    for id_cluster_dbscan_bruto in ids_clusters_unicos:
        if id_cluster_dbscan_bruto == -1: continue

        alertas_do_geocluster_bruto = df_alertas_com_cluster_ids[df_alertas_com_cluster_ids['cluster_id_dbscan'] == id_cluster_dbscan_bruto]
        
        if len(alertas_do_geocluster_bruto) < min_alertas_para_hotspot_refinado: continue

        tipos_presentes_no_geocluster = alertas_do_geocluster_bruto['typeIA'].unique()

        for tipo_candidato_hotspot in tipos_presentes_no_geocluster:
            alertas_deste_tipo_especifico = alertas_do_geocluster_bruto[alertas_do_geocluster_bruto['typeIA'] == tipo_candidato_hotspot]

            if len(alertas_deste_tipo_especifico) < min_alertas_para_hotspot_refinado: continue
            
            dist_sev_tipo = calcular_distribuicao_percentual(alertas_deste_tipo_especifico['severityIA'])
            sev_pred_tipo_tupla = obter_item_predominante(alertas_deste_tipo_especifico['severityIA'], porc_severidade_predominante)
            sev_pred_tipo = sev_pred_tipo_tupla[0] if sev_pred_tipo_tupla else "N/A"

            forma_hotspot = False
            dominant_severity_final = sev_pred_tipo

            if tipo_candidato_hotspot == 'OUTRO_PERIGO':
                if sev_pred_tipo == 'CRITICA': forma_hotspot = True
            elif tipo_candidato_hotspot in ['ALAGAMENTO', 'RISCO_DESLIZAMENTO', 'DESLIZAMENTO_OCORRIDO']:
                if sev_pred_tipo in ['CRITICA', 'ALTA']: forma_hotspot = True
                elif sev_pred_tipo == 'MEDIA' and dist_sev_tipo.get('ALTA', 0.0) >= porc_alta_em_media:
                    forma_hotspot = True
                    dominant_severity_final = 'MEDIA'

            if not forma_hotspot: continue

            hotspot_output_id_counter += 1
            centroid_lat = alertas_deste_tipo_especifico['latitude'].mean()
            centroid_lon = alertas_deste_tipo_especifico['longitude'].mean()
            
            radius_km = DEFAULT_HOTSPOT_RADIUS_KM_SINGLE_POINT
            if len(alertas_deste_tipo_especifico) > 1:
                points_rad = np.radians(alertas_deste_tipo_especifico[['latitude', 'longitude']].values)
                centroid_rad = np.radians([[centroid_lat, centroid_lon]])
                distances_km = haversine_distances(points_rad, centroid_rad) * R_TERRA_KM
                if distances_km.size > 0: radius_km = np.max(distances_km)
            
            summary = f"Hotspot de {tipo_candidato_hotspot.replace('_', ' ').title()} com severidade {dominant_severity_final}. {len(alertas_deste_tipo_especifico)} alertas."
            
            last_ts = None
            if 'timestampReporte' in alertas_deste_tipo_especifico.columns and not alertas_deste_tipo_especifico['timestampReporte'].dropna().empty:
                try:
                    last_ts = alertas_deste_tipo_especifico['timestampReporte'].dropna().max()
                except Exception:
                    try:
                        timestamps_dt = pd.to_datetime(alertas_deste_tipo_especifico['timestampReporte'].dropna(), errors='coerce')
                        if not timestamps_dt.empty:
                             last_ts = timestamps_dt.max().isoformat() + "Z"
                    except Exception:
                        last_ts = datetime.now(timezone.utc).isoformat()
            else:
                 last_ts = datetime.now(timezone.utc).isoformat()


            hotspots_formatados.append({
                "clusterLabel": hotspot_output_id_counter,
                "centroidLat": float(centroid_lat),
                "centroidLon": float(centroid_lon),
                "pointCount": len(alertas_deste_tipo_especifico),
                "dominantType": str(tipo_candidato_hotspot),
                "dominantSeverity": str(dominant_severity_final),
                "alertIdsInCluster": sorted(alertas_deste_tipo_especifico['alertId'].tolist()),
                "estimatedRadiusKm": float(radius_km) if radius_km is not None else None,
                "publicSummary": summary,
                "lastActivityTimestamp": str(last_ts) if last_ts is not None else None,
                "distribuicaoTipos": dict(Counter(alertas_deste_tipo_especifico['typeIA'])),
                "distribuicaoSeveridades": dict(Counter(alertas_deste_tipo_especifico['severityIA']))
            })
            
    return hotspots_formatados


def realizar_clustering_alertas(alertas_para_clusterizar: List[AlertItemForClustering]) -> Tuple[List[Dict], List[Dict]]:
    """
    Função principal do serviço de clustering.
    Retorna os resultados de clustering e os sumários dos hotspots.
    """
    if not alertas_para_clusterizar:
        return [], []

    df_alertas = pd.DataFrame([alerta.model_dump() for alerta in alertas_para_clusterizar])

    if df_alertas.empty or not all(col in df_alertas.columns for col in ['latitude', 'longitude', 'alertId', 'typeIA', 'severityIA']):
        raise ValueError("Dados de alerta inválidos ou faltando campos obrigatórios para clustering.")

    coordenadas_rad = np.radians(df_alertas[['latitude', 'longitude']].values)
    eps_rad = DBSCAN_EPS_KM / R_TERRA_KM

    db = DBSCAN(eps=eps_rad, 
                min_samples=DBSCAN_MIN_SAMPLES, 
                metric='haversine', 
                algorithm='ball_tree'
               ).fit(coordenadas_rad)
    
    df_alertas['cluster_id_dbscan'] = db.labels_

    clustering_results = []
    for idx, row in df_alertas.iterrows():
        clustering_results.append({
            "alertId": int(row['alertId']),
            "clusterLabel": int(row['cluster_id_dbscan']) 
        })
    
    config_refinamento = {
        "MIN_ALERTAS_PARA_HOTSPOT_REFINADO": REFINEMENT_MIN_ALERTS_FOR_HOTSPOT,
        "PORCENTAGEM_PARA_SEVERIDADE_PREDOMINANTE": REFINEMENT_PORC_SEVERIDADE_PREDOMINANTE,
        "PORCENTAGEM_ALTA_EM_MEDIA_PARA_HOTSPOT": REFINEMENT_PORC_ALTA_EM_MEDIA
    }

    hotspot_summaries_dicts = refinar_e_caracterizar_hotspots_para_api(df_alertas, config_refinamento)
        
    return clustering_results, hotspot_summaries_dicts
