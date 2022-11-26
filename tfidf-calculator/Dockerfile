FROM python:3.7-slim-buster

WORKDIR /app

RUN apt-get update \
    && apt-get install -y gcc python3-dev build-essential \
    && apt-get clean

# Install Requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
ENTRYPOINT [ "python3", "-u" ]

# Run command "enter_wordvecs.py" or "calculate_idf_scores.py"
# registry.hive-discover.tech/tfidf-calculator:0.1