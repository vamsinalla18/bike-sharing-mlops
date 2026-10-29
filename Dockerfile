FROM python:3.11-slim

WORKDIR /app

COPY requirements.docker.txt .
# The API loads this artifact at import time. Fail the build if training/DVC
# artifacts have not been restored instead of shipping an unusable image.
COPY models/RandomForest.joblib /app/models/RandomForest.joblib
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
