FROM python:3.6-stretch
LABEL maintainer="Max Erenberg"

WORKDIR app
COPY . .

RUN pip install -r requirements.txt 
ENV FLASK_ENV=development

CMD ["python", "tensorflow_detection.py"]
