# RedAlert - API de Inteligência Artificial para Desastres Urbanos

## Introdução

O projeto **RedAlert** consiste em uma API Python desenvolvida com FastAPI, projetada para fornecer inteligência artificial como serviço, focando na análise e processamento de alertas de desastres urbanos. Suas funcionalidades principais incluem a classificação de texto de alertas para determinar seu **tipo** e **severidade**, e o agrupamento (clustering) geográfico desses alertas para identificar **hotspots** de eventos. Esta API é projetada para ser consumida por outros microsserviços (primariamente a API Java) dentro de um ecossistema maior de gerenciamento de alertas.

O desenvolvimento e a avaliação dos modelos de Machine Learning foram realizados utilizando notebooks Jupyter (disponíveis na pasta `/notebooks`), os modelos treinados e algorítimos são servidos através desta API.

---

## Funcionalidades Principais

A API oferece os seguintes endpoints principais:

1.  **Classificação de Texto de Alertas:**
    * **Endpoint:** `POST /ia/classify_text`
    * **Descrição:** Recebe um `alertId` e o `text` (descrição textual) de um alerta. Utiliza modelos de Machine Learning (LinearSVC com features TF-IDF e léxicos customizados) para classificar o alerta quanto ao seu **tipo** (ex: `ALAGAMENTO`, `RISCO_DESLIZAMENTO`) e **severidade** (ex: `BAIXA`, `MEDIA`, `ALTA`, `CRITICA`).
    * **Objetivo:** Fornecer uma avaliação rápida e automatizada da natureza e do impacto potencial de um novo alerta reportado.

2.  **Clustering de Alertas para Identificação de Hotspots:**
    * **Endpoint:** `POST /ia/cluster_alerts`
    * **Descrição:** Recebe uma lista de alertas (cada um contendo `alertId`, `latitude`, `longitude`, e as classificações `typeIA` e `severityIA` já fornecidas pela IA). Aplica o algoritmo DBSCAN para agrupar geograficamente os alertas e, em seguida, utiliza uma lógica de refinamento para caracterizar cada cluster significativo (hotspot).
    * **Retorno:**
        * `clusteringResults`: Uma lista mapeando cada `alertId` de entrada ao seu `clusterLabel` (onde -1 indica ruído).
        * `hotspotSummaries`: Uma lista de sumários para cada hotspot identificado, incluindo seu `clusterLabel`, centroide (`centroidLat`, `centroidLon`), contagem de alertas (`pointCount`), tipo (`dominantType`) e severidade (`dominantSeverity`) predominantes, e a lista de `alertIdsInCluster`. Campos opcionais como `estimatedRadiusKm`, `publicSummary`, e `lastActivityTimestamp` também são fornecidos.
    * **Objetivo:** Identificar áreas de concentração de alertas (hotspots) em tempo real (ou quase real, baseado na janela de tempo dos alertas enviados), permitindo uma visualização e resposta mais eficaz a múltiplas ocorrências.

---

## Tecnologias Utilizadas

* **Python 3.x**
* **FastAPI:** Framework web para construção da API.
* **Uvicorn:** Servidor ASGI para rodar a aplicação FastAPI.
* **Scikit-learn:** Para os modelos de Machine Learning (`LinearSVC`, `TfidfVectorizer`, `DBSCAN`) e métricas.
* **Joblib:** Para serialização e carregamento dos modelos treinados.
* **NLTK (Natural Language Toolkit):** Para pré-processamento de texto (tokenização, remoção de stopwords, stemming com RSLPStemmer).
* **Pandas:** Para manipulação de dados, especialmente na etapa de processamento dos alertas para clustering.
* **NumPy:** Suporte fundamental para operações numéricas.
* **SciPy:** Utilizada implicitamente pelo Scikit-learn para operações com matrizes esparsas.
* **Pydantic (com Pydantic-Settings):** Para validação de dados (schemas de request/response) e gerenciamento de configurações.
* **Shapely (nos notebooks):** Para manipulação de geometrias na fase de criação de dados sintéticos.
* **Folium (nos notebooks):** Para visualização de dados geoespaciais e resultados de clustering.

---

## Configuração e Execução do Ambiente

Siga os passos abaixo para configurar e rodar a API localmente: (a containerização dom Dockerfile foi um requisito opcional para a entrega do projeto e não impacta na execução normal dessa API, por isso a inicialização via Dockerfile não será explicitada nesse documento)

1.  **Clone o Repositório:**
    ```bash
    git clone [https://github.com/AlertaVermelho/PY_Pred_API]
    cd PY_PRED_API
    ```

2.  **Crie e Ative um Ambiente Virtual Python:**
    É altamente recomendado usar um ambiente virtual.
    ```bash
    python -m venv .venv
    ```
    Para ativar:
    * Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
        (Se encontrar erro de política de execução, execute no PowerShell como admin: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` ou para a sessão atual: `Set-ExecutionPolicy RemoteSigned -Scope Process`)
    * Linux/macOS: `source .venv/bin/activate`

3.  **Instale as Dependências:**
    Com o ambiente virtual ativado, instale os pacotes listados no `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixe os Recursos do NLTK:**
    O código da API tentará baixar os recursos `stopwords` e `rslp` (para o stemmer) automaticamente na primeira execução, caso não os encontre. Se preferir fazer manualmente, execute em um interpretador Python:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('rslp')
    ```

5.  **Modelos Treinados:**
    Certifique-se de que os arquivos de modelo `.joblib` serializados (`vetorizador_tfidf_TIPO_final.joblib`, `modelo_svc_TIPO_final.joblib`, `vetorizador_tfidf_SEVERIDADE_final.joblib`, `modelo_svc_SEVERIDADE_tuned_final.joblib`) estão presentes na pasta `/saved_models` na raiz do projeto.

6.  **Rodando a API:**
    Com o ambiente virtual ativado e a partir da raiz do projeto:
    ```bash
    uvicorn app.main:app --reload
    ```
    A API estará disponível em `http://127.0.0.1:8000`.
    * Documentação interativa (Swagger UI): `http://127.0.0.1:8000/docs`
    * Documentação alternativa (ReDoc): `http://127.0.0.1:8000/redoc`

---

## Correlação com os Notebooks de Desenvolvimento (Pasta `/notebooks`)

Os modelos de Machine Learning e a lógica de processamento desta API foram desenvolvidos e testados usando uma série de notebooks Jupyter, localizados na pasta `/notebooks` na raiz deste projeto.

* **`01_Synthetic_Data_Generation.ipynb` (ou similar):** Detalha o processo de criação do dataset sintético inicial, incluindo a geração de frases, atribuição de tipos/severidades e, crucialmente, a geração de coordenadas geográficas para simular alertas e hotspots para os testes de clustering.
* **`02_Evaluate_Risk_Classifier.ipynb` (ou similar):** Contém todo o desenvolvimento dos modelos de classificação de **Tipo** e **Severidade**. Inclui as etapas de pré-processamento de texto, vetorização (TF-IDF), treinamento e avaliação de diferentes algoritmos (Naive Bayes, Regressão Logística, LinearSVC), o desenvolvimento e refinamento dos léxicos para o modelo de severidade, e o tuning de hiperparâmetros. Os modelos finais salvos em `/saved_models` são o resultado deste notebook.
* **`03_Alerts_Clustering.ipynb` (ou similar):** Descreve a exploração e implementação do algoritmo de clustering DBSCAN. Detalha a preparação dos dados sintéticos (gerados no notebook 01 e classificados com os modelos do notebook 02), a experimentação com os parâmetros `eps` e `min_samples` do DBSCAN, a visualização dos clusters em mapas, e o desenvolvimento da lógica de sumarização e refinamento de hotspots que é utilizada pela API.

Estes notebooks servem como uma documentação viva do processo de P&D por trás da inteligência desta API e podem ser consultados para entender em detalhes as decisões tomadas e os resultados intermediários.
