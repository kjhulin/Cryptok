# Running behind nginx at a sub-path

This is an example of serving the web UI at `https://example.com/cryptok/` from an existing
nginx site. Replace `example.com` with your host name; the prefix can be anything as long as
you change it consistently in `cryptok-locations.conf`.

```
browser ──HTTPS──> nginx (example.com)
                     └─ /cryptok/*  ──> cryptok 127.0.0.1:8077
```

nginx terminates TLS, rate-limits per client, and forwards to `cryptok serve`, which listens
only on loopback. The files are in `deploy/`:

| File | Install as |
|---|---|
| `deploy/nginx/cryptok-http.conf` | `/etc/nginx/conf.d/cryptok-http.conf` (http context: upstream, rate-limit zones) |
| `deploy/nginx/cryptok-proxy.inc` | `/etc/nginx/snippets/cryptok-proxy.inc` (shared proxy headers) |
| `deploy/nginx/cryptok-locations.conf` | `/etc/nginx/snippets/cryptok-locations.conf`, then `include` it inside your existing `server {}` |
| `deploy/systemd/cryptok.service` | `/etc/systemd/system/` |

## 1. The app

```
sudo useradd --system --home /opt/cryptok --shell /usr/sbin/nologin cryptok
# copy the binary, cryptok.cklm (cryptok train), corpus/ and data/ to /opt/cryptok
sudo apt install tesseract-ocr          # server-side OCR (keeps the page's CSP strict)
```

The unit starts `cryptok serve --host 127.0.0.1 --public --lean --allowed-host example.com`.
`--lean` sizes the server for a 512 MB machine (about 180 MB idle, about 300 MB at peak in testing: one search at a time, known texts reloaded per search). `--public` keeps the loopback bind but applies the conservative limits (2,000 letters, two
concurrent searches, 120 s per search, request log) and requires `--allowed-host`.

## 2. nginx

Add `include /etc/nginx/snippets/cryptok-locations.conf;` inside the server
block that already listens on 443 with your certificate, then:

```
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl enable --now cryptok
```

What the locations do:

* `/cryptok` redirects to `/cryptok/`. The page uses relative URLs, so it works under the prefix;
  `proxy_pass ...:8077/` strips `/cryptok/` before the request reaches the app.
* `/cryptok/api/rkc` and `/cryptok/api/known` are Server-Sent Events: `proxy_buffering off`, so
  progress appears live, and `proxy_read_timeout 180s` outlasts the app's own 120 s limit.
* `/cryptok/api/ocr` is the only route allowed a request body (26 MB); everything else is capped at 1 KB. `proxy_request_buffering off` streams the photo to the app so nginx never writes it to a temporary file (by default nginx spills bodies over its buffer to `/var/lib/nginx/body`).
* Rate limits per client address: page 10 r/s, searches 30 r/min (burst 10), OCR 6 r/min.
  Adjust in `cryptok-http.conf`.
* Whatever `X-Forwarded-For` a browser sends is replaced with the real client address, which
  the app writes to its request log.

## What was tested

The locations were run on nginx 1.24 against the real app (`--public`): redirect from
`/cryptok`, forged `X-Forwarded-For` headers being overwritten, wrong `Host` refused, SSE
progress arriving incrementally through the proxy, a 30 MB upload rejected by nginx while a
normal image OCRs, the 1 KB body limit on page routes, `429` after the rate limit, and the whole
UI (solve, crib, transposition, OCR upload) working in headless Chromium at `/cryptok/`.

Not tested here: TLS and a real certificate.

## Notes

* The server also sends `Strict-Transport-Security`; if your server block adds its own, drop one.
* Ciphertext is carried in query strings. Do not log `$request_uri` or `$args` in nginx access
  logs for `/cryptok/api/`: use a `log_format` without them (or `access_log off;` in those locations).
