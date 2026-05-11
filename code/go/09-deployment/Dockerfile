# ============================================================
# Dockerfile — AI Agent Service (Go)
# Multi-stage build: compile with full Go toolchain, run in minimal Alpine.
# ============================================================

# ── Stage 1: builder ──────────────────────────────────────────
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /build

# Copy go.mod and go.sum first — cache the dependency download layer.
COPY go.mod go.sum ./
RUN go mod download

# Copy all source files and compile a static binary.
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o /agent_service ./...


# ── Stage 2: runtime ─────────────────────────────────────────
FROM alpine:3.19 AS runtime

LABEL org.opencontainers.image.title="AI Agent Service (Go)" \
      org.opencontainers.image.description="Production Go HTTP service for AI agents" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.source="https://github.com/your-org/ai-agents-in-action"

# curl for health checks; ca-certificates for TLS to external APIs
RUN apk add --no-cache curl ca-certificates

# Create non-root user
RUN addgroup --gid 1001 appgroup \
    && adduser --uid 1001 --ingroup appgroup --shell /bin/sh --disabled-password appuser

WORKDIR /app

COPY --from=builder /agent_service /app/agent_service

RUN chown appuser:appgroup /app/agent_service

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

ENTRYPOINT ["/app/agent_service"]
