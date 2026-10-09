# Security

## Reporting

Please report vulnerabilities privately (GitHub *Security advisories* on this repository)
rather than in a public issue.

## Threat model

`cryptok` is a code-breaking toolbox. Run on a laptop it is a local tool; deployed on a server
(`cryptok serve --host 0.0.0.0`, see [docs/DEPLOY-AWS.md](docs/DEPLOY-AWS.md)) it becomes an
internet-facing service that runs expensive computations and parses untrusted images. The
points below were audited for that second case.

Out of scope: TLS (terminate it at the load balancer), per-user accounts, per-client rate
limiting (use AWS WAF), and protecting ciphertext in transit through proxies that log URLs.

## Audit summary

Scope: `crates/cli` (HTTP server, OCR, UI), `crates/core` (solvers), CI workflows, build
files. The project has **no third-party Rust dependencies**, which removes supply-chain risk
from crates (the `Cargo.lock` lists only workspace crates).

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | High | No authentication: anyone reaching the port could run CPU-heavy solvers and upload images. | **Fixed.** HTTP Basic (constant-time compare, 300 ms delay on failure). `serve` refuses a non-loopback `--host` without `CRYPTOK_AUTH` unless `--insecure-no-auth` is given. `/healthz` is the only open path. |
| 2 | High | Denial of service: unlimited threads, no read timeouts (slowloris), unbounded request line/headers, 25 MB body allocated up front, unbounded solver parameters (beam, period, keyword lists, cipher length). | **Fixed.** Connection cap, 10 s read / 30 s write timeouts, 60 s body deadline, 16 KB request line / 8 KB header / 64 header limits, body read incrementally, concurrency cap with `503`, 120 s wall-clock cancellation of streamed searches, and per-endpoint input caps (see the limits table in the deployment guide). |
| 3 | High | Image upload: decompression bombs (a tiny file declaring billions of pixels), an unbounded external `tesseract` run, temp files in a shared directory. | **Fixed.** Only PNG/JPEG/GIF/BMP/WebP accepted; dimensions read from the header and capped (12,000 px a side, 16 MP); TIFF refused; Tesseract killed after 40 s with capped output and `OMP_THREAD_LIMIT=1`; the image is piped to Tesseract's standard input and **never written to disk** (see *Photo privacy* below); Tesseract's errors stay in the server log. |
| 4 | Medium | DNS rebinding / Host spoofing against the local server (a web page could script `127.0.0.1:8077`). | **Fixed.** Host header allow-list (loopback names by default, `--allowed-host` for deployments); the upload endpoint also needs a custom header that cross-site pages cannot send without a CORS preflight, which is never granted. |
| 5 | Medium | HTTP request smuggling surface behind a proxy. | **Fixed.** `Transfer-Encoding` refused, conflicting `Content-Length` refused, `Connection: close` always, only `GET` and one `POST` route accepted. |
| 6 | Medium | No security headers; inline script/style without CSP. | **Fixed.** Per-response nonce CSP (`default-src 'none'`, no inline script without the nonce, `frame-ancestors 'none'`), `nosniff`, `X-Frame-Options`, `Referrer-Policy: no-referrer`, COOP/CORP, `Permissions-Policy` (camera for self only), HSTS, `no-store`. JSON responses carry a locked-down CSP. The CDN allowances for browser-side OCR appear in the policy only when the server has no Tesseract. |
| 7 | Medium | DOM XSS risk: server values (file names, error text, solver output containing user characters) were inserted with `innerHTML`, some with incomplete escaping. | **Fixed.** All such values go through `esc()`; the camera menu is built with DOM APIs. Verified in a browser: a payload with `<img onerror>` and `<script>` renders as text, and the CSP blocks injected inline script. |
| 8 | Medium | Memory safety: `DenseNgram::window` used `get_unchecked` on an index computed from caller-supplied letters. All current callers pass scrubbed letters, but an out-of-range byte would read out of bounds. | **Fixed.** Now bounds-checked. The remaining `unsafe` (`known.rs`, `map.rs`) indexes with values masked or sized by construction. |
| 9 | Low | Information disclosure: `/api/info` returned the server's model path; OCR errors could include temp paths. | **Fixed.** File name only; internal detail is logged, not returned. |
| 10 | Low | No request log or auth-failure log. | **Fixed.** One line per request (client address, method, path, never the query string) and per failed login, to stderr. |
| 11 | Low | CI token permissions defaulted to broad; release workflow had `contents: write` for every job. | **Fixed.** Read-only by default; only the publish job writes. Dependabot added for Actions, Cargo and Docker. |
| 12 | Info | Ciphertext travels in `GET` query strings (needed for Server-Sent Events), so any proxy or ALB access log would record it. | **Accepted, documented.** Keep ALB access logs off or access-controlled. Request-line limit makes very long URLs fail early. |
| 13 | Info | Basic auth is a single shared credential and does not rate-limit by client. | **Accepted, documented.** Use ALB OIDC/Cognito for per-user access and WAF for rate limiting. |
| 14 | Info | GitHub Actions are pinned to major tags, not commit SHAs. | **Open.** Pin to SHAs (Dependabot will keep them current) before relying on the release pipeline. |
| 15 | Info | `bench/private` is added to the known-text sources automatically when it exists. A deployment that includes it would expose those copyrighted texts. | **Documented.** The Dockerfile copies only `corpus` and `data`; the checklist says not to ship it. |

## Photo privacy

Uploaded photos are not retained anywhere the project controls:

* **App server:** the image is read into memory, piped to Tesseract's stdin, and the buffer is
  overwritten (zeroed) when the request ends, including when an upload is cut short. No file is
  created: a `strace` of the server and its Tesseract child during real uploads showed no file
  opened for writing. Only the recognised text is returned; the image is not echoed back, cached,
  or logged (the request log holds client, method and path only).
* **nginx:** by default nginx writes any request body larger than its buffer to a temporary file
  (`/var/lib/nginx/body/...`, deleted afterwards). `deploy/nginx/cryptok-locations.conf` sets
  `proxy_request_buffering off` on the upload route so the photo streams straight to the app.
  Verified with `strace`: with buffering on nginx created `/var/lib/nginx/body/0000000002` for an
  11 MB upload; with it off, nothing was written.
* **Browser:** the preview is an in-memory object URL, dropped when replaced, when the user presses
  *Remove picture*, and when the page closes. Responses are `Cache-Control: no-store`. Without
  server-side Tesseract the picture never leaves the browser at all.
* **Host level:** the sample systemd unit sets `LimitCORE=0` and `MemorySwapMax=0` so a crash
  dump or swap cannot hold a photo. Other layers you add (an ALB or CDN with request logging or
  body capture, a WAF with body inspection, endpoint backups) are outside this program: keep them
  from storing request bodies.

## What was tested

Unit tests cover authentication, the constant-time compare, Host allow-list (including a
rebinding-style host), slot accounting, request-parser limits over a real socket, the CSP, image
header parsing and decompression-bomb refusal. A live server was also driven with `curl`, raw
sockets and headless Chromium: auth failures, bad Host, oversize line/headers/body, chunked and
conflicting-length requests, connection cap, concurrent-job `503`, the wall-clock cutoff, input
length caps, an OCR bomb, the strict CSP with the page fully working, and the XSS payload above.

Not tested: the `Dockerfile` (no container build was possible in the audit environment), the
browser-side OCR fallback under its CSP (the CDN was unreachable), behaviour on an actual AWS
account, and sustained load.
