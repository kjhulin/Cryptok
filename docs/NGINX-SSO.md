# Running behind nginx at `cryptok.space/solver/` with Google sign-in

```
browser ──HTTPS──> nginx (cryptok.space)
                     ├─ /oauth2/*      ──> oauth2-proxy  127.0.0.1:4180  ──> Google
                     └─ /solver/*      ──(auth_request: signed in?)──> cryptok 127.0.0.1:8077
```

nginx terminates TLS, asks oauth2-proxy on every request whether the visitor has signed in with
Google, rate-limits per client, and forwards to `cryptok serve`, which listens only on loopback.
The files are in `deploy/`:

| File | Install as |
|---|---|
| `deploy/nginx/cryptok-http.conf` | `/etc/nginx/conf.d/cryptok-http.conf` (http context: upstream, rate-limit zones) |
| `deploy/nginx/cryptok-proxy.inc` | `/etc/nginx/snippets/cryptok-proxy.inc` (shared proxy headers; contains the secret) |
| `deploy/nginx/cryptok-locations.conf` | `/etc/nginx/snippets/cryptok-locations.conf`, then `include` it inside your existing `server {}` for cryptok.space |
| `deploy/oauth2-proxy/oauth2-proxy.cfg` | `/etc/oauth2-proxy/oauth2-proxy.cfg` |
| `deploy/systemd/cryptok.service`, `oauth2-proxy.service` | `/etc/systemd/system/` |

## 1. Google OAuth client

1. Google Cloud Console → *APIs & Services* → **OAuth consent screen**. Choose *Internal* if
   everyone is in one Google Workspace, otherwise *External* (add test users while it is in
   "Testing", or publish it). Scopes: `openid`, `email`, `profile`.
2. **Credentials → Create credentials → OAuth client ID → Web application.**
   *Authorised redirect URI:* `https://cryptok.space/oauth2/callback` (exactly).
3. Put the client ID in `oauth2-proxy.cfg`; the secret goes in
   `/etc/oauth2-proxy/secrets.env` (mode `0600`) together with a cookie secret:

   ```
   OAUTH2_PROXY_CLIENT_SECRET=...
   OAUTH2_PROXY_COOKIE_SECRET=<openssl rand -base64 32 | tr -- '+/' '-_'>
   ```
4. List who may sign in, one Google address per line, in `/etc/oauth2-proxy/allowed-emails.txt`
   (or switch the config to `email_domains = ["your-workspace.com"]`). **Never** set
   `email_domains = ["*"]`: that lets any Google account in.

Install oauth2-proxy from its [releases](https://github.com/oauth2-proxy/oauth2-proxy/releases)
(or your distribution's package) to `/usr/local/bin`, and create the `oauth2-proxy` user.

## 2. The app

```
sudo useradd --system --home /opt/cryptok --shell /usr/sbin/nologin cryptok
# copy the binary, cryptok.cklm (cryptok train), corpus/ and data/ to /opt/cryptok
sudo apt install tesseract-ocr          # server-side OCR (keeps the page's CSP strict)
```

Choose a long random secret that only nginx and the app share:

```
SECRET=$(openssl rand -base64 36 | tr -d '=+/')
echo "CRYPTOK_AUTH=nginx:$SECRET" | sudo tee /etc/cryptok/auth.env && sudo chmod 600 /etc/cryptok/auth.env
printf 'nginx:%s' "$SECRET" | base64 -w0        # paste into cryptok-proxy.inc (Authorization line)
```

The unit starts `cryptok serve --host 127.0.0.1 --public --lean --allowed-host cryptok.space`.
`--lean` sizes the server for a 512 MB machine (about 180 MB idle, about 300 MB at peak in testing: one search at a time, known texts reloaded per search). `--public` keeps the loopback bind but applies the conservative limits (2,000 letters, two
concurrent searches, 120 s per search, request log) and requires `--allowed-host`. The app also
refuses any request that lacks the shared secret, so another process on the machine (or a
server-side request forgery in a neighbouring site) cannot use it without nginx.

## 3. nginx

Add `include /etc/nginx/snippets/cryptok-locations.conf;` inside the `cryptok.space` server
block that already listens on 443 with your certificate, then:

```
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl enable --now cryptok oauth2-proxy
```

What the locations do:

* `/solver` redirects to `/solver/`. The page uses relative URLs, so it works under the prefix;
  `proxy_pass ...:8077/` strips `/solver/` before the request reaches the app.
* Navigations without a session are redirected to Google (`/oauth2/start`). API calls get a plain
  `401` instead, because a redirect to Google would break `fetch` and `EventSource`. If a session
  expires mid-visit, reload the page.
* `/solver/api/rkc` and `/solver/api/known` are Server-Sent Events: `proxy_buffering off`, so
  progress appears live, and `proxy_read_timeout 180s` outlasts the app's own 120 s limit.
* `/solver/api/ocr` is the only route allowed a request body (26 MB); everything else is capped at 1 KB. `proxy_request_buffering off` streams the photo to the app so nginx never writes it to a temporary file (by default nginx spills bodies over its buffer to `/var/lib/nginx/body`). The `/oauth2/auth` location also needs `client_max_body_size`: nginx applies the sub-request's limit to the original upload, so without it any photo over 1 MB fails with a 500.
* Rate limits per client address: page 10 r/s, searches 30 r/min (burst 10), OCR 6 r/min.
  Adjust in `cryptok-http.conf`.
* Whatever `Authorization`, `X-Forwarded-For` or `X-Auth-Request-Email` a browser sends is
  replaced. The signed-in user's address reaches the app only for its request log
  (`... GET /api/crib user=alice@example.com`).

## What was tested

The locations were run on nginx 1.24 against the real app (`--public`) with a stand-in for
oauth2-proxy (the real oauth2-proxy and Google were not available): redirect from `/solver`,
sign-in redirect for the page versus `401` for the API, forged `Authorization`,
`X-Forwarded-For` and `X-Auth-Request-Email` headers being overwritten, wrong `Host` refused,
direct access without the shared secret refused, SSE progress arriving incrementally through the
proxy, a 30 MB upload rejected by nginx while a normal image and an 11.7 MB one OCR (testing found and fixed the 1 MB sub-request limit above), `strace` showing no file written by the app, Tesseract or (with buffering off) nginx, the 1 KB body limit on page
routes, `429` after the rate limit, and the whole UI (solve, crib, transposition, OCR upload)
working in headless Chromium at `/solver/`.

Not tested here: the Google consent flow itself, oauth2-proxy's cookie handling, TLS, and a
real certificate. Do a sign-in with an allowed and a non-allowed Google account after deploying,
and confirm that `https://cryptok.space/solver/api/info` returns `401` in a private window.

## Notes

* `/oauth2/` is shared by every site on this server name. If other apps on `cryptok.space`
  also use oauth2-proxy they can share it; otherwise set `proxy_prefix` in the config and
  adjust the location names.
* The server also sends `Strict-Transport-Security`; if your server block adds its own, drop one.
* Ciphertext is carried in query strings. Do not log `$request_uri` or `$args` in nginx access
  logs for `/solver/api/`: use a `log_format` without them (or `access_log off;` in those locations).
