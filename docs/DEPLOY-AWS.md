# Deploying `cryptok serve` on AWS

`cryptok serve` is a small dependency-free HTTP server. It speaks **plain HTTP**, so put it
behind a TLS-terminating AWS load balancer and never expose its port directly. Read
[SECURITY.md](../SECURITY.md) first for what the server does and does not defend against.

## Recommended layout

```
Internet ──HTTPS 443──> ALB (ACM certificate, WAF) ──HTTP 8077──> cryptok (private subnet)
```

* **ALB** with an ACM certificate, an HTTP→HTTPS redirect on port 80, and TLS policy
  `ELBSecurityPolicy-TLS13-1-2-2021-06` or newer.
* **Security groups:** the instance/task accepts port 8077 *only from the ALB's security group*.
  No SSH from the internet (use SSM Session Manager). Nothing else inbound.
* **AWS WAF** on the ALB with a rate-based rule (for example 300 requests per 5 minutes per
  IP) and the AWS managed *Common Rule Set*. The server has global limits but no per-client
  rate limiting, because behind a load balancer every client shares the ALB's address.
* **Target group health check:** `GET /healthz` on port 8077, expecting `200` (it needs no
  credentials and reveals nothing).
* **Compute:** ECS Fargate or one EC2 instance with at least 2 vCPU and **2 GB RAM** (the model
  takes about 350 MB and each running search adds more; two concurrent searches peaked
  near 400 MB in testing, with headroom for OCR). Pick more CPU for faster solves.
* **Logs:** send stdout/stderr to CloudWatch Logs. The server logs one line per request
  (client address, method, path; never the query string, which holds the ciphertext) and every
  authentication failure.

## Credentials

The server refuses to listen on a non-loopback address without credentials. Store
`user:password` (at least 12 characters, ideally 24+ random) in **AWS Secrets Manager** or SSM
Parameter Store and inject it as the `CRYPTOK_AUTH` environment variable (ECS `secrets:`
field). Do not put it in the image, the task definition's plain `environment`, or a command
line (visible in `ps`). Rotate by updating the secret and restarting the task.

HTTP Basic is simple and stops drive-by use, but it is one shared login. For per-user access
put the ALB's built-in **OIDC/Cognito authentication** action in front and then start the
server with `--insecure-no-auth` (only safe when the security group guarantees that nothing
but the ALB can reach the port).

## Docker / ECS

```
docker build -t cryptok .
aws ecr ...                      # push to your registry
```

Task definition essentials:

* `command`/entrypoint args: `--allowed-host cryptok.example.com` (the public host name; the
  ALB passes it through, and anything else gets `421`).
* `secrets`: `CRYPTOK_AUTH` from Secrets Manager.
* `readonlyRootFilesystem: true`. The app writes no files (uploaded photos are piped to Tesseract
  in memory and never stored), so no writable volume is needed. Run as the image's non-root user,
  drop all Linux capabilities, `memory: 2048`, `cpu: 2048` or more.
* No task role permissions beyond logging.

## EC2 + systemd (alternative)

```
[Service]
User=cryptok
EnvironmentFile=/etc/cryptok/auth.env        # CRYPTOK_AUTH=user:password, mode 0600, root-owned
ExecStart=/opt/cryptok/cryptok serve --host 0.0.0.0 --port 8077 --no-open \
          --model /opt/cryptok/cryptok.cklm --sources /opt/cryptok/corpus --allowed-host cryptok.example.com
Restart=on-failure
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
PrivateDevices=false        # the server reads /dev/urandom for nonces
MemoryMax=2G
CapabilityBoundingSet=
RestrictAddressFamilies=AF_INET AF_INET6
SystemCallFilter=@system-service
```

Install `tesseract-ocr` on the instance so image upload is handled server-side.

## Limits (public defaults)

Anything non-loopback uses these defaults; override with the flags shown in `cryptok --help`.

| Limit | Default | Flag |
|---|---|---|
| Concurrent connections | 64 | `--max-conns` |
| Concurrent searches / OCR jobs (others get `503`) | 2 | `--max-jobs` |
| Wall-clock time for a streamed search | 120 s | `--job-timeout` |
| Cipher length | 2,000 letters | `--max-letters` |
| Running-key beam | 100,000 | `--max-beam` |
| Candidate keywords | 500 | `--max-keywords` |
| Image upload | 25 MB, 12,000 px a side, 50 megapixels, 40 s of OCR; held in memory only, never stored | fixed |

## Checklist before going live

- [ ] ALB serves HTTPS only; port 8077 reachable only from the ALB.
- [ ] `CRYPTOK_AUTH` comes from Secrets Manager and is long and random.
- [ ] `--allowed-host` is set to the real host name.
- [ ] `tesseract-ocr` installed (the Dockerfile does this).
- [ ] `bench/private` is **not** in the image or on the host: its copyrighted key texts would be
      readable through the known-text search.
- [ ] WAF rate rule attached; CloudWatch alarms on 5xx and on authentication failures.
- [ ] Photos are not retained: ALB/WAF/CDN logging must not capture request bodies (ALB access logs record the URL but not the body).
- [ ] Ciphertext is sent in URLs (`GET /api/rkc?cipher=...`), so ALB access logs, if enabled,
      would record it. Leave ALB access logging off or restrict who can read that bucket.
