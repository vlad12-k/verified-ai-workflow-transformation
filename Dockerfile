FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE alembic.ini ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install ".[platform]"

USER 65532:65532

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "vait.platform.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
