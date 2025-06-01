# Projeto de API para Classificação de Desastres Urbanos

## Funcionalidades

* Endpoint `/predict/`: Recebe um relato de desastre e retorna a classificação de TIPO e SEVERIDADE.

## Configuração do Ambiente

1.  Clone o repositório:
    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd seu_projeto_fastapi
    ```
2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Linux/macOS
    # .venv\Scripts\activate   # No Windows
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
4.  (Se necessário) Baixe os recursos do NLTK (o código da API deve tentar fazer isso automaticamente na primeira execução, mas pode ser feito manualmente):
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('rslp')
    ```
## Como Rodar a API

Com o ambiente virtual ativado e a partir da raiz do projeto:
```bash
uvicorn app.main:app --reload