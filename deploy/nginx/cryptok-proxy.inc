# Shared proxy settings. Save as /etc/nginx/snippets/cryptok-proxy.inc
proxy_pass_request_headers on;
proxy_http_version 1.1;
proxy_set_header Connection        "";
proxy_set_header Host              $host;                 # the app only accepts --allowed-host
proxy_set_header X-Real-IP         $remote_addr;
proxy_set_header X-Forwarded-For   $remote_addr;          # overwrite: never trust a client-sent chain
proxy_set_header X-Forwarded-Proto $scheme;

proxy_connect_timeout 5s;
proxy_send_timeout    60s;
# proxy_read_timeout is set in each location (searches may legitimately run for minutes).
