# --------- builder stage (install and build wheels) ----------
FROM python:3.11-slim AS builder
WORKDIR /build

# Install build dependencies needed for many binary Python packages (onnxruntime, pyarrow, etc.)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        pkg-config \
        libgomp1 \
        libstdc++6 \
        libcurl4-openssl-dev \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Upgrade packaging tools (helps pip find wheels)
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first for layer caching
COPY src/requirements.txt /build/requirements.txt

# Install all Python requirements into the builder image
RUN pip install --no-cache-dir -r /build/requirements.txt

# --------- runtime stage (minimal runtime) ----------
FROM python:3.11-slim
WORKDIR /app

# Install minimal runtime libs required by binary wheels
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application source (this will put local src/ into /app/src/)
COPY src ./src

# Build-time args (optional) and exported env vars for runtime
ARG GROQ_API_KEY
ARG UNSTRUCTURED_API_KEY
ARG QDRANT_URL
ARG QDRANT_API_KEY
ARG OPENWEATHERMAP_API_KEY
ARG LANGSMITH_API_KEY
ARG LANGSMITH_TRACING
ARG LANGSMITH_ENDPOINT
ARG NOMIC_API

ENV GROQ_API_KEY=${GROQ_API_KEY}
ENV UNSTRUCTURED_API_KEY=${UNSTRUCTURED_API_KEY}
ENV QDRANT_URL=${QDRANT_URL}
ENV QDRANT_API_KEY=${QDRANT_API_KEY}
ENV OPENWEATHERMAP_API_KEY=${OPENWEATHERMAP_API_KEY}
ENV LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
ENV LANGSMITH_TRACING=${LANGSMITH_TRACING}
ENV LANGSMITH_ENDPOINT=${LANGSMITH_ENDPOINT}
ENV NOMIC_API=${NOMIC_API}
EXPOSE 8051

CMD ["streamlit", "run", "./src/streamlit_app.py", "--server.port=8051", "--server.address=0.0.0.0"]