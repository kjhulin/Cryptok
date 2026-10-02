# Production image for `cryptok serve` (see docs/DEPLOY-AWS.md).
#   docker build -t cryptok .
#   docker run --rm -p 8077:8077 -e CRYPTOK_AUTH='user:a-long-random-password' \
#       --read-only --cap-drop ALL --memory 1g cryptok --allowed-host cryptok.example.com
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
COPY corpus corpus
COPY data data
RUN cargo build --release --locked \
 && ./target/release/cryptok train --corpus corpus --out /src/cryptok.cklm

FROM debian:bookworm-slim
# Tesseract does the OCR server-side, so browsers never need to reach a CDN (and the page's
# content-security policy can stay strict).
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin cryptok
WORKDIR /app
COPY --from=build /src/target/release/cryptok /src/cryptok.cklm ./
# Public-domain texts only. Do NOT copy bench/private (copyrighted keys) into this image:
# the known-text search would reveal their contents to users.
COPY corpus corpus
COPY data data
USER 10001:10001
# Memory: glibc's per-thread arenas and lazy trimming otherwise keep freed search buffers
# resident long after a job ends. Two arenas, and large blocks (search buffers) go straight to
# mmap so they are returned to the OS when freed.
ENV MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TRIM_THRESHOLD_=131072
EXPOSE 8077
# Authentication is mandatory off-localhost: pass CRYPTOK_AUTH=user:password at run time
# (from AWS Secrets Manager / SSM Parameter Store, never baked into the image).
ENTRYPOINT ["/app/cryptok", "serve", "--host", "0.0.0.0", "--port", "8077", "--no-open", "--model", "/app/cryptok.cklm", "--sources", "/app/corpus"]
