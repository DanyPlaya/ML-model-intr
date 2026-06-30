# Итоговый проект по дисциплине «Внедрение моделей машинного обучения»

Проект демонстрирует production-like внедрение ML-сервиса для прогнозирования дефолта по кредитным картам на датасете [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).

## Что реализовано

- Flask API: `GET /health`, `GET /metadata`, `POST /predict`.
- Обучение моделей на реальном Kaggle/UCI CSV.
- Сохранение моделей в pickle: `models/model_v1.pkl`, `models/model_v2.pkl`.
- A/B-выбор модели через явное поле `model_version` или 50/50 hash split по `customer_id`.
- JSON Lines логирование запросов в `logs/api.jsonl` и stdout контейнера.
- Dockerfile, `docker-compose.yml`, тесты API и отдельный план A/B-теста.

## Структура проекта

```text
.
├── app/
│   ├── __init__.py
│   ├── api.py
│   └── model_handler.py
├── data/
│   └── raw/
│       └── UCI_Credit_Card.csv     
├── docker/
│   └── Dockerfile
├── logs/
├── models/
│   ├── train_model.py
│   ├── model_v1.pkl
│   └── model_v2.pkl
├── scripts/
│   └── download_data.py
├── tests/
│   └── test_api.py
├── ab_test_plan.md
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Быстрый запуск локально

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m flask --app app.api run --host 0.0.0.0 --port 5000
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m flask --app app.api run --host 0.0.0.0 --port 5000
```

## Датасет и обучение

Датасет скачивается с Kaggle в `data/raw/UCI_Credit_Card.csv`. CSV и ZIP исключены из Git, поэтому чистый клон воспроизводится одной командой:

```bash
python scripts/download_data.py
```

Ручное скачивание через публичный Kaggle endpoint:

```powershell
Invoke-WebRequest `
  -Uri "https://www.kaggle.com/api/v1/datasets/download/uciml/default-of-credit-card-clients-dataset" `
  -OutFile "data\raw\default-of-credit-card-clients-dataset.zip"
Expand-Archive -LiteralPath "data\raw\default-of-credit-card-clients-dataset.zip" -DestinationPath "data\raw" -Force
```

Переобучение моделей:

```bash
python models/train_model.py --csv data/raw/UCI_Credit_Card.csv --model models/model_v1.pkl --algorithm random_forest
python models/train_model.py --csv data/raw/UCI_Credit_Card.csv --model models/model_v2.pkl --algorithm gradient_boosting
```

Текущие метрики на holdout 20%:

| Модель | Алгоритм | F1 | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| v1 | RandomForestClassifier | 0.5422 | 0.5065 | 0.5833 | 0.7734 |
| v2 | GradientBoostingClassifier | 0.4688 | 0.6634 | 0.3625 | 0.7784 |

`v1` лучше по F1 и recall, `v2` точнее по precision и немного выше по ROC-AUC. Это делает сравнение осмысленным для A/B-плана.

## Формат API

`GET /health`:

```json
{
  "status": "ok",
  "message": "Service is running",
  "model_versions": ["v1", "v2"]
}
```

`GET /metadata` возвращает список признаков, целевую колонку и доступные модели.

`POST /predict` принимает JSON с 23 признаками. Дополнительно можно передать:

- `model_version`: `"v1"` или `"v2"` для явного выбора модели.
- `customer_id`: стабильный идентификатор клиента для hash split 50/50, если `model_version` не указан.

Пример запроса:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"client-42","LIMIT_BAL":200000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":34,"PAY_0":0,"PAY_2":0,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":0,"BILL_AMT1":12000,"BILL_AMT2":11000,"BILL_AMT3":10500,"BILL_AMT4":9800,"BILL_AMT5":9000,"BILL_AMT6":8500,"PAY_AMT1":3000,"PAY_AMT2":2500,"PAY_AMT3":2000,"PAY_AMT4":2000,"PAY_AMT5":2000,"PAY_AMT6":2000}'
```

Пример ответа:

```json
{
  "assignment": "ab_hash",
  "model_alias": "v1",
  "model_artifact": "model_v1",
  "model_version": "v1",
  "prediction": 0,
  "probability": 0.184739,
  "probability_default": 0.184739,
  "request_id": "2e5d3fd4-81d4-4d59-b694-ae4127ca2b03"
}
```

`probability_default` — основной документированный ключ вероятности. `probability` оставлен как совместимый короткий псевдоним. `model_version` всегда содержит публичную версию `v1`/`v2`, а `model_artifact` — имя обученного артефакта.

Явное тестирование `v2`:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-Model-Version: v2" \
  -d '{"LIMIT_BAL":200000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":34,"PAY_0":0,"PAY_2":0,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":0,"BILL_AMT1":12000,"BILL_AMT2":11000,"BILL_AMT3":10500,"BILL_AMT4":9800,"BILL_AMT5":9000,"BILL_AMT6":8500,"PAY_AMT1":3000,"PAY_AMT2":2500,"PAY_AMT3":2000,"PAY_AMT4":2000,"PAY_AMT5":2000,"PAY_AMT6":2000}'
```

## Docker

```bash
docker build -f docker/Dockerfile -t credit-card-default-api:latest .
docker run --rm -p 5000:5000 credit-card-default-api:latest
```

Контейнер запускает Gunicorn с двумя воркерами и встроенным `HEALTHCHECK` по `/health`.

Через Compose:

```bash
docker compose up --build
```

Публичный образ пока не опубликован: для сдачи его нужно собрать в среде с Docker, выполнить `docker login`, затем:

```text
docker tag credit-card-default-api:latest <dockerhub-user>/credit-card-default-api:latest
docker push <dockerhub-user>/credit-card-default-api:latest
```

## Архитектура

Для учебного проекта выбран монолитный сервис: REST API, валидация входа, выбор версии модели и инференс находятся в одном Flask-приложении. Это проще запускать, тестировать и контейнеризировать. Микросервисный подход был бы оправдан при независимых командах, разных SLA, отдельной batch-обработке или высокой нагрузке на отдельные части системы.

RabbitMQ или другой брокер очередей можно добавить для асинхронных batch-предсказаний, повторной обработки ошибок, доставки результатов в downstream-системы и передачи событий в контур мониторинга. В текущем минимальном сервисе синхронный REST API достаточен.

Логи пишутся в JSON Lines: `logs/api.jsonl` и stdout. Для каждого ответа сохраняются `request_id`, версия и артефакт модели, способ назначения A/B-группы, прогноз, вероятность и задержка. Сырые финансовые признаки намеренно не логируются. В production события могут собираться Docker logging driver, Filebeat или Fluent Bit и отправляться в ELK/OpenSearch.

## MLOps-концепты

- DVC: версионирование датасетов и связь версии данных с версией модели.
- MLflow: трекинг экспериментов, параметров, метрик и артефактов моделей.
- ONNX-ML: перенос sklearn-пайплайна в независимый формат инференса через `skl2onnx` и запуск через `onnxruntime`. Для конвертации нужны пакеты `skl2onnx`, `onnx` и вход `FloatTensorType([None, 23])`:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

onnx_model = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, 23]))])
with open("models/model_v1.onnx", "wb") as file:
    file.write(onnx_model.SerializeToString())
```
- Gunicorn/uWSGI + NGINX: WSGI-сервер запускает Python-приложение, NGINX принимает внешние соединения, проксирует запросы, завершает TLS и обслуживает статические ресурсы.

## Бизнес-метрики

Помимо F1, precision и recall, заказчику важны:

- Ожидаемые финансовые потери: `PD * EAD * LGD`, где `PD` берётся из вероятности дефолта модели.
- Доля одобренных заявок при фиксированном уровне риска: выбирается порог вероятности дефолта, затем сравнивается доля клиентов ниже порога.

## Тесты

```bash
pytest
```

Тесты проверяют health-check, контракт признаков, обе версии модели, стабильность A/B-назначения, успешный прогноз, неверный HTTP-метод, невалидный JSON, типы и пропущенные признаки.
