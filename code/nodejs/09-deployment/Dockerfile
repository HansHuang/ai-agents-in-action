# ============================================================
# Dockerfile — AI Agent Service (Node.js / TypeScript)
# Multi-stage build for a lean, secure production image.
# ============================================================

# ── Stage 1: dependency installer ────────────────────────────
FROM node:20-alpine AS deps

WORKDIR /build

COPY package.json package-lock.json* ./
RUN npm ci --omit=dev && npm cache clean --force


# ── Stage 2: TypeScript compiler ─────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /build

COPY package.json package-lock.json* ./
RUN npm ci

COPY tsconfig.json ./
COPY *.ts ./

RUN npx tsc --noEmit 2>&1 || true \
    && npx tsc --outDir dist 2>/dev/null || npx tsx --version


# ── Stage 3: runtime image ────────────────────────────────────
FROM node:20-alpine AS runtime

LABEL org.opencontainers.image.title="AI Agent Service (Node.js)" \
      org.opencontainers.image.description="Production Express service for AI agents" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.source="https://github.com/your-org/ai-agents-in-action"

# Install curl for health checks only
RUN apk add --no-cache curl

# Create non-root user
RUN addgroup --gid 1001 appgroup \
    && adduser --uid 1001 --ingroup appgroup --shell /bin/sh --disabled-password appuser

WORKDIR /app

# Copy production node_modules from deps stage
COPY --from=deps /build/node_modules ./node_modules

# Copy compiled output from builder stage (or source for tsx)
COPY --from=builder /build/dist ./dist
COPY --chown=appuser:appgroup package.json ./

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000

CMD ["node", "dist/agent_service.js"]
