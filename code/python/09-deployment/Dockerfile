# ============================================================
# Dockerfile — AI Agent Service
# Multi-stage build for a lean, secure production image.
# ============================================================

# ── Stage 1: dependency builder ──────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first — maximise layer caching.
# If requirements.txt doesn't change, this layer is reused on every build.
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="AI Agent Service" \
      org.opencontainers.image.description="Production FastAPI service for AI agents" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.source="https://github.com/your-org/ai-agents-in-action"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # curl is used by the HEALTHCHECK below
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Create a non-root user and group for security.
# Running as root inside a container is a common security mistake.
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/sh --create-home appuser

# Copy application source code
COPY --chown=appuser:appgroup . .

# Switch to non-root user before the final CMD
USER appuser

# Health check — calls the /health endpoint every 30 seconds.
# The container is marked "unhealthy" if it fails 3 times in a row.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use exec form so the process receives OS signals directly (enables graceful shutdown).
# --workers 1 is correct for asyncio-based apps; scale via k8s replicas instead.
CMD ["python", "-m", "uvicorn", "agent_service:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-graceful-shutdown", "30"]
