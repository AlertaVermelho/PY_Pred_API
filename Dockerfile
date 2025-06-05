FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app_service

RUN groupadd -r appuser && useradd --no-log-init -r -g appuser -d ${APP_HOME} -s /sbin/nologin -c "Docker image user" appuser

RUN mkdir -p ${APP_HOME}
WORKDIR ${APP_HOME} # Define o diretório de trabalho padrão dentro do container

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords -d /usr/local/share/nltk_data && \
    python -m nltk.downloader rslp -d /usr/local/share/nltk_data

ENV NLTK_DATA /usr/local/share/nltk_data

COPY ./app ${APP_HOME}/app
COPY ./saved_models ${APP_HOME}/saved_models

RUN chown -R appuser:appuser ${APP_HOME}

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]