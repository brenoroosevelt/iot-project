FROM python:3.10-slim

WORKDIR /app

COPY treino.py .

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib xgboost

CMD ["python", "treino.py"]

