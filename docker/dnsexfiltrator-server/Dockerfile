FROM python:latest

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && pip install -r requirements.txt

CMD ["/bin/bash"]