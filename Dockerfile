FROM python:3.7-slim-buster

WORKDIR /app

# Install Requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python3 -m spacy download en_core_web_sm \
    && python3 -m spacy download es_core_news_sm \
    && python3 -m spacy download de_core_news_sm

COPY . .
ENTRYPOINT [ "python3", "-u", "worker.py" ]

# Img Name: registry.hive-discover.tech/vectorizer:0.2