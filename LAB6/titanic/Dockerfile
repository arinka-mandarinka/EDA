FROM python:3.11-slim

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Подготовка данных и обучение
RUN python eda.py
RUN python model.py

EXPOSE 5000

CMD ["python", "app.py"]
