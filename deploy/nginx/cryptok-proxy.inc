# Shared proxy settings. Save as /etc/nginx/snippets/cryptok-proxy.inc
proxy_pass_request_headers on;
proxy_http_version 1.1;
proxy_set_header Connection        "";
proxy_set_header Host              $host;                 # the app only accepts --allowed-host
proxy_set_header X-Real-IP         $remote_addr;
proxy_set_header X-Forwarded-For   $remote_addr;          # overwrite: never trust a client-sent chain
proxy_set_header X-Forwarded-Proto $scheme;

# Who signed in (from oauth2-proxy), for the app's request log. Always overwritten, so a
# client cannot forge it.
proxy_set_header X-Auth-Request-Email $cryptok_user;

# Second lock between nginx and the app (defense in depth): the app is started with
# CRYPTOK_AUTH=nginx:<long secret>, and only nginx knows it. Replace the value with
#   printf 'nginx:<long secret>' | base64
# Whatever Authorization header a browser sent is discarded.
proxy_set_header Authorization "Basic REPLACE_WITH_BASE64_OF_nginx:SECRET";

proxy_connect_timeout 5s;
proxy_send_timeout    60s;
# proxy_read_timeout is set in each location (searches may legitimately run for minutes).
