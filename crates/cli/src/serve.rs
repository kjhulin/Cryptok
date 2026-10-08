//! `cryptok serve` — a small web server for the browser UI.
//!
//! Dependency-free: std `TcpListener`, one thread per connection. Everything is GET except
//! `POST /api/ocr` (an image upload). Long-running solvers stream progress with
//! Server-Sent Events; closing the page (or pressing Stop) drops the connection, which
//! cancels the search.
//!
//! The server is meant to sit behind a TLS-terminating reverse proxy when exposed to a
//! network (see docs/DEPLOY-AWS.md). Hardening in this file: HTTP Basic authentication,
//! Host-header allow-list (DNS-rebinding protection), request-size and time limits, caps on
//! connections and concurrent solver jobs, per-request input limits, a wall-clock job
//! deadline, a content-security policy with a per-response nonce, and security headers.

use cryptok_core::classic::{self, Alphabet};
use cryptok_core::known::{self, KnownOptions, Source};
use cryptok_core::lm::{DenseNgram, LangModel};
use cryptok_core::rkc::{self, RkcOptions, RkcSolution, StepInfo};
use cryptok_core::text::{scrub, unscrub};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const INDEX_HTML: &str = include_str!("../ui/index.html");

/// Limits and access control. [`Config::for_host`] picks defaults: generous on loopback,
/// conservative when the server is reachable from a network.
#[derive(Clone)]
pub struct Config {
    /// Reachable by untrusted clients (non-loopback bind, or `--public` behind a proxy):
    /// conservative limits and request logging.
    pub public: bool,
    pub host: String,
    pub port: u16,
    /// `(user, password)` for HTTP Basic authentication.
    pub auth: Option<(String, String)>,
    /// Accepted `Host` header values (lower case, no port). Empty = accept any.
    pub allowed_hosts: Vec<String>,
    pub max_conns: usize,
    /// Concurrent solver / OCR jobs; further requests get 503 until one finishes.
    pub max_jobs: usize,
    /// Wall-clock limit for a cancellable job (0 = none).
    pub job_timeout: Duration,
    /// Longest ciphertext accepted by any endpoint, in letters.
    pub max_letters: usize,
    pub max_beam: usize,
    pub max_keywords: usize,
    /// Private directory for OCR uploads.
    pub tmp_dir: PathBuf,
}

pub fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]")
}

impl Config {
    /// `public` forces the conservative profile even on a loopback bind: use it behind a
    /// reverse proxy on the same machine.
    pub fn for_host(host: &str, port: u16, public: bool) -> Self {
        let local = is_loopback(host) && !public;
        Config {
            public: !local,
            host: host.to_string(),
            port,
            auth: None,
            allowed_hosts: if local { vec!["localhost".into(), "127.0.0.1".into(), "[::1]".into(), "::1".into()] } else { vec![] },
            max_conns: if local { 256 } else { 64 },
            max_jobs: if local { 8 } else { 2 },
            job_timeout: if local { Duration::ZERO } else { Duration::from_secs(120) },
            max_letters: if local { 100_000 } else { 2_000 },
            max_beam: if local { 2_000_000 } else { 100_000 },
            max_keywords: if local { 20_000 } else { 500 },
            tmp_dir: std::env::temp_dir(),
        }
    }
}

pub struct ServerState {
    pub lm: LangModel,
    pub quad: DenseNgram,
    pub sources: Vec<Source>,
    pub model_path: String,
    /// Word model, its trie and the word-score weight (None = character model only).
    pub words: Option<(cryptok_core::words::WordModel, cryptok_core::words::WordTrie, f32)>,
    /// The `tesseract` program is installed (server-side OCR available).
    pub ocr: bool,
    pub cfg: Config,
    conns: Arc<AtomicUsize>,
    jobs: Arc<AtomicUsize>,
}

impl ServerState {
    pub fn new(
        lm: LangModel,
        quad: DenseNgram,
        sources: Vec<Source>,
        model_path: String,
        words: Option<(cryptok_core::words::WordModel, cryptok_core::words::WordTrie, f32)>,
        ocr: bool,
        cfg: Config,
    ) -> Self {
        ServerState { lm, quad, sources, model_path, words, ocr, cfg, conns: Arc::new(AtomicUsize::new(0)), jobs: Arc::new(AtomicUsize::new(0)) }
    }
}

/// Counts an in-flight connection or job; released on drop (also when a handler panics).
struct Slot(Arc<AtomicUsize>);
impl Slot {
    fn take(counter: &Arc<AtomicUsize>, max: usize) -> Option<Slot> {
        let mut cur = counter.load(Ordering::Acquire);
        loop {
            if cur >= max {
                return None;
            }
            match counter.compare_exchange(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(Slot(Arc::clone(counter))),
                Err(c) => cur = c,
            }
        }
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn run(state: ServerState, open: bool) -> Result<(), String> {
    let (host, port) = (state.cfg.host.clone(), state.cfg.port);
    let listener = TcpListener::bind((host.as_str(), port)).map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    let local = !state.cfg.public;
    let url = format!("http://{}:{port}/", if is_loopback(&host) { "127.0.0.1" } else { host.as_str() });
    println!("Cryptok Code Cracker is running at {url}  (Ctrl-C to stop)");
    if !local {
        eprintln!(
            "note: public mode on {host}. This server speaks plain HTTP: put a TLS-terminating proxy (nginx, an AWS load balancer) in front of it. \
             Authentication here: {}. Host allow-list: {}.",
            if state.cfg.auth.is_some() { "HTTP Basic (enabled)" } else { "none (the proxy must authenticate)" },
            if state.cfg.allowed_hosts.is_empty() { "any".to_string() } else { state.cfg.allowed_hosts.join(", ") }
        );
    }
    if open {
        open_browser(&url);
    }
    let state = Arc::new(state);
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
        let Some(slot) = Slot::take(&state.conns, state.cfg.max_conns) else {
            let _ = respond_with(&stream, "503 Service Unavailable", "text/plain", &["Retry-After: 5"], "too many connections");
            continue;
        };
        let st = Arc::clone(&state);
        std::thread::spawn(move || {
            let _slot = slot;
            let _ = handle(stream, &st);
        });
    }
    Ok(())
}

fn open_browser(url: &str) {
    let r = if cfg!(target_os = "windows") {
        std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if r.is_err() {
        println!("Open {url} in your browser.");
    }
}

// ------------------------------------------------------------------ HTTP plumbing

/// Largest image upload accepted (a 12-megapixel phone photo is about 5 MB).
const MAX_UPLOAD: usize = 25 * 1024 * 1024;
const MAX_REQUEST_LINE: usize = 16 * 1024;
const MAX_HEADER_LINE: usize = 8 * 1024;
const MAX_HEADERS: usize = 64;
/// Total time allowed to receive a request body.
const BODY_DEADLINE: Duration = Duration::from_secs(60);

struct Request {
    method: String,
    path: String,
    query: Vec<(String, String)>,
    /// `X-Cryptok` request header present. A cross-site page cannot send it without a CORS
    /// preflight, which this server never grants, so it guards the POST endpoint.
    custom_header: bool,
    host: String,
    authorization: String,
    forwarded_for: String,
    /// `X-Auth-Request-Email` set by a trusted SSO proxy; used only for the request log.
    user: String,
    body: Vec<u8>,
}

impl Request {
    fn q(&self, k: &str) -> String {
        self.query.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone()).unwrap_or_default()
    }
    fn num(&self, k: &str, d: usize) -> usize {
        self.q(k).parse().unwrap_or(d)
    }
}

/// A request we refuse before routing: `(status, message)`.
type Refusal = (&'static str, &'static str);

/// Read one line of at most `limit` bytes. `Ok(None)` = connection closed.
fn read_line_limited(r: &mut impl BufRead, limit: usize, too_long: Refusal) -> Result<Option<String>, Refusal> {
    let mut buf = Vec::new();
    let n = r.by_ref().take(limit as u64 + 1).read_until(b'\n', &mut buf).map_err(|_| ("408 Request Timeout", "timed out"))?;
    if n == 0 {
        return Ok(None);
    }
    if buf.len() > limit {
        return Err(too_long);
    }
    Ok(Some(String::from_utf8_lossy(&buf).to_string()))
}

fn read_request(stream: &TcpStream) -> Result<Option<Request>, Refusal> {
    let mut r = BufReader::new(stream);
    let Some(line) = read_line_limited(&mut r, MAX_REQUEST_LINE, ("414 URI Too Long", "request line too long"))? else { return Ok(None) };
    let (mut content_length, mut custom_header) = (None::<usize>, false);
    let (mut host, mut authorization, mut forwarded_for, mut user) = (String::new(), String::new(), String::new(), String::new());
    for count in 0.. {
        let Some(h) = read_line_limited(&mut r, MAX_HEADER_LINE, ("431 Request Header Fields Too Large", "header too long"))? else { break };
        if h == "\r\n" || h == "\n" {
            break;
        }
        if count >= MAX_HEADERS {
            return Err(("431 Request Header Fields Too Large", "too many headers"));
        }
        let Some((k, v)) = h.split_once(':') else { return Err(("400 Bad Request", "malformed header")) };
        let v = v.trim();
        match k.trim().to_ascii_lowercase().as_str() {
            "content-length" => {
                let n: usize = v.parse().map_err(|_| ("400 Bad Request", "bad Content-Length"))?;
                if content_length.is_some_and(|c| c != n) {
                    return Err(("400 Bad Request", "conflicting Content-Length"));
                }
                content_length = Some(n);
            }
            // We never decode chunked bodies; refusing them closes the request-smuggling door.
            "transfer-encoding" => return Err(("501 Not Implemented", "Transfer-Encoding is not supported")),
            "x-cryptok" => custom_header = true,
            "host" => host = v.to_string(),
            "authorization" => authorization = v.to_string(),
            "x-forwarded-for" => forwarded_for = v.to_string(),
            "x-auth-request-email" => user = v.to_string(),
            _ => {}
        }
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or(("400 Bad Request", "empty request"))?.to_string();
    let target = parts.next().ok_or(("400 Bad Request", "no target"))?;
    let (path, qs) = target.split_once('?').unwrap_or((target, ""));
    let query = qs
        .split('&')
        .filter(|s| !s.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (url_decode(k), url_decode(v))
        })
        .collect();
    let mut body = Vec::new();
    match method.as_str() {
        "GET" => {}
        "POST" => {
            let n = content_length.ok_or(("411 Length Required", "Content-Length required"))?;
            if n > MAX_UPLOAD {
                return Err(("413 Payload Too Large", "upload too large (limit 25 MB)"));
            }
            // Grow as bytes arrive instead of trusting the declared length.
            let deadline = Instant::now() + BODY_DEADLINE;
            let mut chunk = vec![0u8; 64 * 1024];
            while body.len() < n {
                if Instant::now() > deadline {
                    return Err(("408 Request Timeout", "upload too slow"));
                }
                let want = chunk.len().min(n - body.len());
                match r.read(&mut chunk[..want]) {
                    Ok(0) | Err(_) => return Err(("400 Bad Request", "incomplete upload")),
                    Ok(k) => body.extend_from_slice(&chunk[..k]),
                }
            }
        }
        _ => return Err(("405 Method Not Allowed", "GET only")),
    }
    Ok(Some(Request { method, path: path.to_string(), query, custom_header, host, authorization, forwarded_for, user, body }))
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Headers sent with every response.
const SECURITY_HEADERS: &str = "X-Content-Type-Options: nosniff\r\n\
X-Frame-Options: DENY\r\n\
Referrer-Policy: no-referrer\r\n\
Cross-Origin-Opener-Policy: same-origin\r\n\
Cross-Origin-Resource-Policy: same-origin\r\n\
Permissions-Policy: camera=(self), microphone=(), geolocation=(), payment=(), usb=()\r\n\
Strict-Transport-Security: max-age=31536000\r\n\
Cache-Control: no-store\r\n";

fn respond_with(mut s: &TcpStream, status: &str, ctype: &str, extra: &[&str], body: &str) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n{SECURITY_HEADERS}", body.len());
    // JSON and plain responses must never be rendered or framed.
    if !ctype.starts_with("text/html") {
        head.push_str("Content-Security-Policy: default-src 'none'; frame-ancestors 'none'\r\n");
    }
    for e in extra {
        head.push_str(e);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes())?;
    s.write_all(body.as_bytes())
}

fn respond(s: &TcpStream, status: &str, ctype: &str, body: &str) -> std::io::Result<()> {
    respond_with(s, status, ctype, &[], body)
}

fn refuse(s: &TcpStream, (status, msg): Refusal) -> std::io::Result<()> {
    respond(s, status, "text/plain", msg)
}

fn random_hex(n: usize) -> String {
    let mut b = vec![0u8; n];
    let ok = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).is_ok();
    if !ok {
        // Non-Unix fallback: std's randomly keyed hasher plus the clock.
        use std::hash::{BuildHasher, Hasher};
        for (i, x) in b.iter_mut().enumerate() {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
            h.write_usize(i);
            *x = h.finish() as u8;
        }
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn csp(nonce: &str, browser_ocr: bool) -> String {
    // Tesseract.js (browser OCR fallback) loads its script, worker and language data from CDNs.
    let (script_extra, connect_extra, worker) = if browser_ocr {
        (" https://cdn.jsdelivr.net 'wasm-unsafe-eval'", " https://cdn.jsdelivr.net https://tessdata.projectnaptha.com", "worker-src blob:; ")
    } else {
        ("", "", "")
    };
    format!(
        "default-src 'none'; script-src 'nonce-{nonce}'{script_extra}; style-src 'nonce-{nonce}'; style-src-attr 'unsafe-inline'; \
         img-src 'self' blob: data:; connect-src 'self'{connect_extra}; media-src blob:; {worker}\
         base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
}

fn respond_index(s: &TcpStream, st: &ServerState) -> std::io::Result<()> {
    let nonce = random_hex(16);
    let html = INDEX_HTML.replacen("<script>", &format!("<script nonce=\"{nonce}\">"), 1).replacen("<style>", &format!("<style nonce=\"{nonce}\">"), 1);
    let policy = format!("Content-Security-Policy: {}", csp(&nonce, !st.ocr));
    respond_with(s, "200 OK", "text/html; charset=utf-8", &[&policy], &html)
}

// ------------------------------------------------------------------ authentication

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes().filter(|&c| c != b'=') {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Compare without leaking where the first difference is.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = (a.len() ^ b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

fn authorised(cfg: &Config, req: &Request) -> bool {
    let Some((user, pass)) = &cfg.auth else { return true };
    let Some(b64) = req.authorization.strip_prefix("Basic ").or_else(|| req.authorization.strip_prefix("basic ")) else { return false };
    let Some(cred) = base64_decode(b64.trim()) else { return false };
    ct_eq(&cred, format!("{user}:{pass}").as_bytes())
}

fn host_allowed(cfg: &Config, req: &Request) -> bool {
    if cfg.allowed_hosts.is_empty() {
        return true;
    }
    let h = req.host.to_ascii_lowercase();
    // Strip the port: "[::1]:8077" -> "[::1]", "example.com:443" -> "example.com".
    let name = if h.starts_with('[') { h.split_once("]").map(|(a, _)| format!("{a}]")).unwrap_or(h.clone()) } else { h.split(':').next().unwrap_or("").to_string() };
    cfg.allowed_hosts.iter().any(|a| *a == name)
}

fn client_ip(stream: &TcpStream, req: &Request) -> String {
    let peer = stream.peer_addr().map(|a| a.ip().to_string()).unwrap_or_default();
    // Behind a proxy the last X-Forwarded-For entry is the one the proxy itself appended.
    match req.forwarded_for.rsplit(',').next().map(str::trim).filter(|s| !s.is_empty()) {
        Some(f) => format!("{f} (via {peer})"),
        None => peer,
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn num(x: f32) -> String {
    if x.is_finite() {
        format!("{x:.4}")
    } else {
        "null".into()
    }
}



/// Server-Sent Events writer shared between threads. A failed write marks the
/// client as gone, which cancels the running search.
struct Sse {
    stream: Mutex<TcpStream>,
    gone: AtomicBool,
}

impl Sse {
    fn start(stream: TcpStream) -> std::io::Result<Self> {
        let mut s = stream;
        write!(
            s,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n{SECURITY_HEADERS}Content-Security-Policy: default-src 'none'; frame-ancestors 'none'\r\n\r\n"
        )?;
        Ok(Sse { stream: Mutex::new(s), gone: AtomicBool::new(false) })
    }
    fn send(&self, event: &str, data: &str) {
        if self.gone.load(Ordering::Relaxed) {
            return;
        }
        let Ok(mut s) = self.stream.lock() else { return };
        if write!(s, "event: {event}\ndata: {data}\n\n").and_then(|_| s.flush()).is_err() {
            self.gone.store(true, Ordering::Relaxed);
        }
    }
}

/// Run `work` while a watchdog cancels the job (`sse.gone`) after the configured deadline.
fn with_deadline<T>(st: &ServerState, sse: &Sse, work: impl FnOnce() -> T) -> T {
    let limit = st.cfg.job_timeout;
    if limit.is_zero() {
        return work();
    }
    let done = AtomicBool::new(false);
    std::thread::scope(|sc| {
        sc.spawn(|| {
            let start = Instant::now();
            while !done.load(Ordering::Relaxed) {
                if start.elapsed() > limit {
                    // Tell the client first: `send` is a no-op once `gone` is set.
                    sse.send("failed", &json_str("Stopped: this server limits a search to its time budget. Try a smaller beam or a shorter cipher."));
                    sse.gone.store(true, Ordering::Relaxed);
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        let r = work();
        done.store(true, Ordering::Relaxed);
        r
    })
}

fn busy(stream: &TcpStream) -> std::io::Result<()> {
    respond_with(stream, "503 Service Unavailable", "application/json", &["Retry-After: 5"], "{\"error\":\"The server is busy with other searches. Try again in a few seconds.\"}")
}

fn too_long(stream: &TcpStream, max: usize) -> std::io::Result<()> {
    respond(stream, "413 Payload Too Large", "application/json", &format!("{{\"error\":\"This server accepts at most {max} cipher letters.\"}}"))
}


// ------------------------------------------------------------------ routes

/// `POST /api/ocr?psm=6&digits=0` with the raw image bytes as the body.
fn api_ocr(stream: &TcpStream, st: &ServerState, req: &Request) -> std::io::Result<()> {
    let fail = |status: &str, msg: &str| respond(stream, status, "application/json", &format!("{{\"error\":{}}}", json_str(msg)));
    if req.method != "POST" {
        return fail("405 Method Not Allowed", "send the image with POST");
    }
    if !req.custom_header {
        return fail("403 Forbidden", "missing X-Cryptok header");
    }
    if !st.ocr {
        return fail("501 Not Implemented", "tesseract is not installed on the server");
    }
    let Some(_job) = Slot::take(&st.jobs, st.cfg.max_jobs) else { return busy(stream) };
    let opt = crate::ocr::OcrOptions {
        psm: req.num("psm", 6).clamp(0, 13) as u32,
        allow_digits: req.q("digits") == "1",
        lang: "eng".into(),
    };
    match crate::ocr::ocr_bytes(&req.body, &opt, &st.cfg.tmp_dir) {
        Ok(r) => respond(
            stream,
            "200 OK",
            "application/json",
            &format!(
                "{{\"text\":{},\"raw\":{},\"letters\":{},\"dropped\":{},\"engine\":\"tesseract\"}}",
                json_str(&r.cleaned.text),
                json_str(&r.raw),
                r.cleaned.letters,
                r.cleaned.dropped
            ),
        ),
        Err(e) => {
            // `e.public` is safe to show; internal detail stays in the server log.
            if let Some(d) = &e.detail {
                eprintln!("ocr error: {d}");
            }
            fail("422 Unprocessable Entity", &e.public)
        }
    }
}

fn handle(stream: TcpStream, st: &ServerState) -> std::io::Result<()> {
    let req = match read_request(&stream) {
        Ok(Some(r)) => r,
        Ok(None) => return Ok(()),
        Err(e) => return refuse(&stream, e),
    };
    // Load-balancer health check: no Host or credentials, reveals nothing.
    if req.path == "/healthz" {
        return respond(&stream, "200 OK", "text/plain", "ok");
    }
    if !host_allowed(&st.cfg, &req) {
        return respond(&stream, "421 Misdirected Request", "text/plain", "unrecognised Host header");
    }
    if !authorised(&st.cfg, &req) {
        eprintln!("auth failure from {}", client_ip(&stream, &req));
        std::thread::sleep(Duration::from_millis(300)); // slow down guessing
        return respond_with(&stream, "401 Unauthorized", "text/plain", &["WWW-Authenticate: Basic realm=\"cryptok\", charset=\"UTF-8\""], "authentication required");
    }
    if st.cfg.public {
        // Request log. The query string is left out: it carries the ciphertext. `user` is the
        // identity a trusted proxy's SSO layer vouches for (X-Auth-Request-Email), if any.
        let user = if req.user.is_empty() { String::new() } else { format!(" user={}", req.user.chars().filter(|c| c.is_ascii_graphic()).take(80).collect::<String>()) };
        eprintln!("{} {} {}{user}", client_ip(&stream, &req), req.method, req.path);
    }
    if req.method == "POST" && req.path != "/api/ocr" {
        return respond(&stream, "405 Method Not Allowed", "text/plain", "POST is only used for /api/ocr");
    }
    let max_letters = st.cfg.max_letters;
    match req.path.as_str() {
        "/api/ocr" => api_ocr(&stream, st, &req),
        "/favicon.ico" => respond(&stream, "204 No Content", "text/plain", ""),
        "/" | "/index.html" => respond_index(&stream, st),
        "/api/info" => {
            let letters: usize = st.sources.iter().map(|s| s.letters.len()).sum();
            let names: Vec<String> = st.sources.iter().map(|s| json_str(&s.name)).collect();
            // Only the file name of the model, never a server path.
            let model = std::path::Path::new(&st.model_path).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let body = format!(
                "{{\"order\":{},\"model\":{},\"sources\":[{}],\"source_letters\":{},\"threads\":{},\"ocr\":{},\"max_letters\":{}}}",
                st.lm.order(),
                json_str(&model),
                names.join(","),
                letters,
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
                st.ocr,
                max_letters
            );
            respond(&stream, "200 OK", "application/json", &body)
        }
        "/api/rkc" => api_rkc(stream, st, &req),
        "/api/known" => api_known(stream, st, &req),
        "/api/crib" => {
            let cipher = scrub(&req.q("cipher"));
            if cipher.len() > max_letters {
                return too_long(&stream, max_letters);
            }
            let word = scrub(&req.q("word"));
            if word.len() > 64 {
                return respond(&stream, "400 Bad Request", "application/json", "{\"error\":\"crib too long\"}");
            }
            let hits = rkc::crib_search(&st.lm, &cipher, &word);
            let items: Vec<String> = hits
                .iter()
                .take(req.num("results", 12).clamp(1, 50))
                .map(|h| format!("{{\"pos\":{},\"other\":{},\"score\":{}}}", h.pos, json_str(&unscrub(&h.other)), num(h.score)))
                .collect();
            respond(&stream, "200 OK", "application/json", &format!("[{}]", items.join(",")))
        }
        "/api/vigenere" => {
            let text = req.q("text");
            let cipher = scrub(&text);
            if cipher.len() > max_letters {
                return too_long(&stream, max_letters);
            }
            let alphabets: Vec<String> = req.q("alphabets").split(',').take(20).map(|s| s.trim().chars().take(40).collect()).collect();
            let words: Vec<String> = req
                .q("keywords")
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|w| !w.is_empty() && w.len() <= 40 && w.chars().all(|c| c.is_ascii_alphabetic()))
                .map(String::from)
                .collect();
            if words.len() > st.cfg.max_keywords {
                return respond(&stream, "413 Payload Too Large", "application/json", &format!("{{\"error\":\"At most {} keywords are accepted.\"}}", st.cfg.max_keywords));
            }
            let Some(_job) = Slot::take(&st.jobs, st.cfg.max_jobs) else { return busy(&stream) };
            let max_period = req.num("max_period", 20).clamp(1, 40);
            let t = Instant::now();
            let q = st.lm.dense(classic::climb_ngram_size(&st.lm, cipher.len()));
            let mut all = Vec::new();
            for kw in &alphabets {
                let a = Alphabet::from_keyword(kw);
                all.extend(classic::solve_vigenere_with(&st.lm, &q, &cipher, &a, max_period, 30).into_iter().take(3));
            }
            if !words.is_empty() {
                all.extend(classic::solve_vigenere_keyword_search(&st.lm, &cipher, &words, max_period, 5).1);
            }
            let pen = 26f32.ln();
            all.sort_by(|x, y| (y.score - y.period as f32 * pen).total_cmp(&(x.score - x.period as f32 * pen)));
            let items: Vec<String> = all
                .iter()
                .take(5)
                .map(|s| {
                    format!(
                        "{{\"period\":{},\"key\":{},\"alphabet\":{},\"plain\":{},\"per_letter\":{}}}",
                        s.period,
                        json_str(&unscrub(&s.key)),
                        json_str(&s.alphabet),
                        json_str(&relayout(&text, &s.plain)),
                        num(s.per_letter())
                    )
                })
                .collect();
            respond(
                &stream,
                "200 OK",
                "application/json",
                &format!("{{\"secs\":{:.2},\"results\":[{}]}}", t.elapsed().as_secs_f64(), items.join(",")),
            )
        }
        "/api/transpose" => {
            use cryptok_core::transpo;
            let text = req.q("text");
            let letters = scrub(&text).len();
            if letters < 8 {
                return respond(&stream, "200 OK", "application/json", "{\"secs\":0,\"results\":[]}");
            }
            if letters > max_letters {
                return too_long(&stream, max_letters);
            }
            let Some(_job) = Slot::take(&st.jobs, st.cfg.max_jobs) else { return busy(&stream) };
            let t = Instant::now();
            let q = st.lm.dense(classic::climb_ngram_size(&st.lm, letters));
            let mut all = transpo::solve_route(&st.lm, &q, &text, 60, 3);
            all.extend(transpo::solve_columnar(&st.lm, &q, &text, 2, req.num("max_cols", 12).clamp(2, 12), 8, 3));
            all.sort_by(|x, y| y.per_letter.total_cmp(&x.per_letter));
            let items: Vec<String> = all
                .iter()
                .take(5)
                .map(|s| format!("{{\"method\":{},\"text\":{},\"per_letter\":{}}}", json_str(&s.describe()), json_str(&s.text), num(s.per_letter)))
                .collect();
            respond(&stream, "200 OK", "application/json", &format!("{{\"secs\":{:.2},\"results\":[{}]}}", t.elapsed().as_secs_f64(), items.join(",")))
        }
        _ => respond(&stream, "404 Not Found", "text/plain", "not found"),
    }
}

fn relayout(original: &str, letters: &[u8]) -> String {
    let mut it = letters.iter();
    original
        .chars()
        .map(|ch| if ch.is_ascii_alphabetic() { it.next().map(|&l| (b'A' + l) as char).unwrap_or(ch) } else { ch })
        .collect()
}

fn parse_hint(h: &str, n: usize) -> Vec<Option<u8>> {
    let mut v: Vec<Option<u8>> = h
        .bytes()
        .map(|c| match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a'),
            _ => None,
        })
        .collect();
    v.resize(n, None);
    v.truncate(n);
    v
}

/// `1` for every letter that starts a word in the best segmentation, `0` otherwise.
fn breaks(wm: &cryptok_core::words::WordModel, letters: &[u8]) -> String {
    let mut out = String::with_capacity(letters.len());
    for l in wm.segment(letters).lengths {
        out.push('1');
        out.extend(std::iter::repeat('0').take(l - 1));
    }
    out
}

fn sols_json(s: &[RkcSolution], words: Option<&cryptok_core::words::WordModel>) -> String {
    let items: Vec<String> = s
        .iter()
        .map(|x| {
            let wb = words.map_or(String::new(), |w| format!(",\"ab\":{},\"bb\":{}", json_str(&breaks(w, &x.key)), json_str(&breaks(w, &x.plain))));
            format!("{{\"a\":{},\"b\":{},\"per_letter\":{}{}}}", json_str(&unscrub(&x.key)), json_str(&unscrub(&x.plain)), num(x.per_letter()), wb)
        })
        .collect();
    format!("[{}]", items.join(","))
}


fn api_rkc(stream: TcpStream, st: &ServerState, req: &Request) -> std::io::Result<()> {
    let cipher = scrub(&req.q("cipher"));
    let sse = Sse::start(stream)?;
    if cipher.is_empty() {
        sse.send("failed", &json_str("Enter some cipher letters first."));
        return Ok(());
    }
    if cipher.len() > st.cfg.max_letters {
        sse.send("failed", &json_str(&format!("This server accepts at most {} cipher letters.", st.cfg.max_letters)));
        return Ok(());
    }
    let Some(_job) = Slot::take(&st.jobs, st.cfg.max_jobs) else {
        sse.send("failed", &json_str("The server is busy with other searches. Try again in a few seconds."));
        return Ok(());
    };
    let opts = RkcOptions {
        beam: req.num("beam", 20_000).clamp(1, st.cfg.max_beam),
        results: req.num("results", 12).clamp(1, 100),
        key_hints: parse_hint(&req.q("key"), cipher.len()),
        plain_hints: parse_hint(&req.q("plain"), cipher.len()),
        threads: 0,
        word_weight: st.words.as_ref().map_or(0.0, |w| w.2),
        ..Default::default()
    };
    let wm = st.words.as_ref().map(|w| &w.0);
    let t = Instant::now();
    let mut last = Instant::now();
    let mut cb = |s: &StepInfo| {
        if last.elapsed().as_millis() >= 120 || s.step == s.total {
            last = Instant::now();
            sse.send("progress", &format!("{{\"step\":{},\"total\":{},\"best\":{}}}", s.step, s.total, sols_json(s.best, wm)));
        }
    };
    let sols = with_deadline(st, &sse, || rkc::solve_words(&st.lm, st.words.as_ref().map(|w| &w.1), &cipher, &opts, Some(&mut cb), Some(&sse.gone)));
    if !sse.gone.load(Ordering::Relaxed) {
        sse.send("done", &format!("{{\"secs\":{:.2},\"results\":{}}}", t.elapsed().as_secs_f64(), sols_json(&sols, wm)));
    }
    Ok(())
}

fn api_known(stream: TcpStream, st: &ServerState, req: &Request) -> std::io::Result<()> {
    let cipher = scrub(&req.q("cipher"));
    let sse = Sse::start(stream)?;
    if cipher.len() < 8 {
        sse.send("failed", &json_str("Known-text search needs at least 8 cipher letters."));
        return Ok(());
    }
    if cipher.len() > st.cfg.max_letters {
        sse.send("failed", &json_str(&format!("This server accepts at most {} cipher letters.", st.cfg.max_letters)));
        return Ok(());
    }
    let Some(_job) = Slot::take(&st.jobs, st.cfg.max_jobs) else {
        sse.send("failed", &json_str("The server is busy with other searches. Try again in a few seconds."));
        return Ok(());
    };
    let opts = KnownOptions { window: req.num("window", 24).clamp(4, 64), results: req.num("results", 12).clamp(1, 50), ..Default::default() };
    let t = Instant::now();
    let last_pct = AtomicUsize::new(0);
    let prog = |d: usize, tot: usize| {
        let pct = d * 100 / tot.max(1);
        if pct > last_pct.load(Ordering::Relaxed) {
            last_pct.store(pct, Ordering::Relaxed);
            sse.send("progress", &format!("{{\"pct\":{pct}}}"));
        }
    };
    let hits = with_deadline(st, &sse, || known::search(&st.lm, &st.quad, &cipher, &st.sources, &opts, Some(&prog), Some(&sse.gone)));
    if sse.gone.load(Ordering::Relaxed) {
        return Ok(());
    }
    let items: Vec<String> = hits
        .iter()
        .map(|h| {
            format!(
                "{{\"source\":{},\"reference\":{},\"offset\":{},\"start\":{},\"end\":{},\"key\":{},\"other\":{},\"window\":{},\"score\":{},\"coverage\":{}}}",
                json_str(&h.source),
                json_str(&h.reference),
                h.offset,
                h.start,
                h.end,
                json_str(&unscrub(&h.key)),
                json_str(&unscrub(&h.plain)),
                num(h.window_score),
                num(h.score),
                num(h.coverage)
            )
        })
        .collect();
    sse.send("done", &format!("{{\"secs\":{:.2},\"results\":[{}]}}", t.elapsed().as_secs_f64(), items.join(",")));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn req(host: &str, auth: &str) -> Request {
        Request { method: "GET".into(), path: "/".into(), query: vec![], custom_header: false, host: host.into(), authorization: auth.into(), forwarded_for: String::new(), user: String::new(), body: vec![] }
    }

    #[test]
    fn base64_and_constant_time_compare() {
        assert_eq!(base64_decode("YWxpY2U6c2VjcmV0").unwrap(), b"alice:secret");
        assert_eq!(base64_decode("YWI=").unwrap(), b"ab");
        assert!(base64_decode("not base64!").is_none());
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"abcd"));
        assert!(!ct_eq(b"", b"x"));
    }

    #[test]
    fn basic_auth() {
        let mut cfg = Config::for_host("0.0.0.0", 1, false);
        assert!(authorised(&cfg, &req("h", ""))); // auth disabled
        cfg.auth = Some(("alice".into(), "secret-password".into()));
        let good = format!("Basic {}", "YWxpY2U6c2VjcmV0LXBhc3N3b3Jk");
        assert!(authorised(&cfg, &req("h", &good)));
        assert!(!authorised(&cfg, &req("h", "")));
        assert!(!authorised(&cfg, &req("h", "Basic YWxpY2U6d3Jvbmc=")));
        assert!(!authorised(&cfg, &req("h", "Bearer YWxpY2U6c2VjcmV0LXBhc3N3b3Jk")));
    }

    #[test]
    fn host_allow_list() {
        let local = Config::for_host("127.0.0.1", 1, false);
        for ok in ["localhost", "localhost:8077", "127.0.0.1:8077", "[::1]:8077", "LOCALHOST"] {
            assert!(host_allowed(&local, &req(ok, "")), "{ok}");
        }
        // DNS rebinding: attacker's name resolving to 127.0.0.1.
        for bad in ["evil.example", "evil.example:8077", "127.0.0.1.evil.example", ""] {
            assert!(!host_allowed(&local, &req(bad, "")), "{bad}");
        }
        let mut public = Config::for_host("0.0.0.0", 1, false);
        assert!(host_allowed(&public, &req("anything", ""))); // not configured: any
        public.allowed_hosts = vec!["demo.example.com".into()];
        assert!(host_allowed(&public, &req("demo.example.com:443", "")));
        assert!(!host_allowed(&public, &req("other.example.com", "")));
    }

    #[test]
    fn slots_are_limited_and_released() {
        let c = Arc::new(AtomicUsize::new(0));
        let a = Slot::take(&c, 2).unwrap();
        let b = Slot::take(&c, 2).unwrap();
        assert!(Slot::take(&c, 2).is_none());
        drop(a);
        let _c3 = Slot::take(&c, 2).unwrap();
        drop(b);
        assert_eq!(c.load(Ordering::SeqCst), 1);
    }

    /// Feed raw bytes through `read_request` over a real socket.
    fn parse(bytes: Vec<u8>) -> Result<Option<Request>, Refusal> {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = std::thread::spawn(move || {
            let mut c = TcpStream::connect(addr).unwrap();
            let _ = c.write_all(&bytes);
            std::thread::sleep(Duration::from_millis(300));
        });
        let (s, _) = l.accept().unwrap();
        s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let r = read_request(&s);
        t.join().unwrap();
        r
    }

    #[test]
    fn request_limits() {
        let ok = parse(b"GET /a?x=1 HTTP/1.1\r\nHost: h\r\nX-Cryptok: 1\r\n\r\n".to_vec()).unwrap().unwrap();
        assert_eq!((ok.path.as_str(), ok.q("x").as_str(), ok.custom_header), ("/a", "1", true));
        let long = format!("GET /{} HTTP/1.1\r\n\r\n", "A".repeat(20_000));
        assert_eq!(parse(long.into_bytes()).err().unwrap().0, "414 URI Too Long");
        let many: String = (0..80).map(|i| format!("X{i}: 1\r\n")).collect();
        assert_eq!(parse(format!("GET / HTTP/1.1\r\n{many}\r\n").into_bytes()).err().unwrap().0, "431 Request Header Fields Too Large");
        assert_eq!(parse(b"POST /api/ocr HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec()).err().unwrap().0, "501 Not Implemented");
        assert_eq!(parse(b"POST /api/ocr HTTP/1.1\r\n\r\n".to_vec()).err().unwrap().0, "411 Length Required");
        assert_eq!(parse(b"POST /api/ocr HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n".to_vec()).err().unwrap().0, "413 Payload Too Large");
        assert_eq!(parse(b"POST /x HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab".to_vec()).err().unwrap().0, "400 Bad Request");
        assert_eq!(parse(b"DELETE / HTTP/1.1\r\n\r\n".to_vec()).err().unwrap().0, "405 Method Not Allowed");
        let post = parse(b"POST /api/ocr HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello".to_vec()).unwrap().unwrap();
        assert_eq!(post.body, b"hello");
    }

    #[test]
    fn csp_is_strict_when_server_ocr_is_available() {
        let strict = csp("abc", false);
        assert!(strict.contains("script-src 'nonce-abc'") && !strict.contains("jsdelivr") && !strict.contains("unsafe-inline'; style"));
        assert!(csp("abc", true).contains("cdn.jsdelivr.net"));
        assert!(strict.contains("default-src 'none'") && strict.contains("frame-ancestors 'none'"));
    }
}
