//! `cryptok serve` — a small local web server for the browser UI.
//!
//! Dependency-free: std `TcpListener`, one thread per connection. Everything is GET except
//! `POST /api/ocr` (an image upload).
//! Long-running solvers stream progress with Server-Sent Events; closing the page
//! (or pressing Stop) drops the connection, which cancels the search.

use cryptok_core::classic::{self, Alphabet};
use cryptok_core::known::{self, KnownOptions, Source};
use cryptok_core::lm::{DenseNgram, LangModel};
use cryptok_core::rkc::{self, RkcOptions, RkcSolution, StepInfo};
use cryptok_core::text::{scrub, unscrub};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const INDEX_HTML: &str = include_str!("../ui/index.html");

pub struct ServerState {
    pub lm: LangModel,
    pub quad: DenseNgram,
    pub sources: Vec<Source>,
    pub model_path: String,
    /// Word model, its trie and the word-score weight (None = character model only).
    pub words: Option<(cryptok_core::words::WordModel, cryptok_core::words::WordTrie, f32)>,
    /// The `tesseract` program is installed (server-side OCR available).
    pub ocr: bool,
}

pub fn run(state: ServerState, host: &str, port: u16, open: bool) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    let local = host == "127.0.0.1" || host == "localhost" || host == "::1";
    if !local {
        eprintln!(
            "warning: listening on {host}: anyone who can reach this port can use the solvers and the OCR upload. \
             Only do this on a network you trust. Browsers also withhold webcam access on plain http except on localhost; \
             the phone 'Take photo' button still works."
        );
    }
    let url = format!("http://{}:{port}/", if local { "127.0.0.1" } else { host });
    println!("Cryptok Code Cracker is running at {url}  (Ctrl-C to stop)");
    if open {
        open_browser(&url);
    }
    let state = Arc::new(state);
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let st = Arc::clone(&state);
        std::thread::spawn(move || {
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

struct Request {
    method: String,
    path: String,
    query: Vec<(String, String)>,
    /// `X-Cryptok` request header present. A cross-site page cannot send it without a CORS
    /// preflight, which this server never grants, so it guards the POST endpoint.
    custom_header: bool,
    body: Vec<u8>,
    /// The request could not be read (bad length, oversized body).
    error: Option<&'static str>,
}

impl Request {
    fn q(&self, k: &str) -> String {
        self.query.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone()).unwrap_or_default()
    }
    fn num(&self, k: &str, d: usize) -> usize {
        self.q(k).parse().unwrap_or(d)
    }
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

fn read_request(stream: &TcpStream) -> Option<Request> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    r.read_line(&mut line).ok()?;
    let (mut content_length, mut custom_header) = (0usize, false);
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).ok()? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => content_length = v.trim().parse().unwrap_or(usize::MAX),
                "x-cryptok" => custom_header = true,
                _ => {}
            }
        }
    }
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?;
    let (path, qs) = target.split_once('?').unwrap_or((target, ""));
    let query = qs
        .split('&')
        .filter(|s| !s.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (url_decode(k), url_decode(v))
        })
        .collect();
    let mut req = Request { method: method.clone(), path: path.to_string(), query, custom_header, body: vec![], error: None };
    if method == "POST" {
        if content_length > MAX_UPLOAD {
            req.error = Some("upload too large (limit 25 MB)");
        } else {
            let mut body = vec![0u8; content_length];
            if std::io::Read::read_exact(&mut r, &mut body).is_err() {
                req.error = Some("incomplete upload");
            }
            req.body = body;
        }
    } else if method != "GET" {
        req.path = "!method".into();
    }
    Some(req)
}

fn respond(mut s: &TcpStream, status: &str, ctype: &str, body: &str) -> std::io::Result<()> {
    write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
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
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n")?;
        Ok(Sse { stream: Mutex::new(s), gone: AtomicBool::new(false) })
    }
    fn send(&self, event: &str, data: &str) {
        if self.gone.load(Ordering::Relaxed) {
            return;
        }
        let mut s = self.stream.lock().unwrap();
        if write!(s, "event: {event}\ndata: {data}\n\n").and_then(|_| s.flush()).is_err() {
            self.gone.store(true, Ordering::Relaxed);
        }
    }
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
    if let Some(e) = req.error {
        return fail("400 Bad Request", e);
    }
    if !st.ocr {
        return fail("501 Not Implemented", "tesseract is not installed on the server");
    }
    let opt = crate::ocr::OcrOptions {
        psm: req.num("psm", 6).clamp(0, 13) as u32,
        allow_digits: req.q("digits") == "1",
        lang: "eng".into(),
    };
    match crate::ocr::ocr_bytes(&req.body, &opt) {
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
        Err(e) => fail("422 Unprocessable Entity", &e),
    }
}

fn handle(stream: TcpStream, st: &ServerState) -> std::io::Result<()> {
    let Some(req) = read_request(&stream) else { return Ok(()) };
    if req.method == "POST" && req.path != "/api/ocr" {
        return respond(&stream, "405 Method Not Allowed", "text/plain", "POST is only used for /api/ocr");
    }
    match req.path.as_str() {
        "/api/ocr" => api_ocr(&stream, st, &req),
        "/" | "/index.html" => respond(&stream, "200 OK", "text/html; charset=utf-8", INDEX_HTML),
        "/api/info" => {
            let letters: usize = st.sources.iter().map(|s| s.letters.len()).sum();
            let names: Vec<String> = st.sources.iter().map(|s| json_str(&s.name)).collect();
            let body = format!(
                "{{\"order\":{},\"model\":{},\"sources\":[{}],\"source_letters\":{},\"threads\":{},\"ocr\":{}}}",
                st.lm.order(),
                json_str(&st.model_path),
                names.join(","),
                letters,
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
                st.ocr
            );
            respond(&stream, "200 OK", "application/json", &body)
        }
        "/api/rkc" => api_rkc(stream, st, &req),
        "/api/known" => api_known(stream, st, &req),
        "/api/crib" => {
            let cipher = scrub(&req.q("cipher"));
            let word = scrub(&req.q("word"));
            let hits = rkc::crib_search(&st.lm, &cipher, &word);
            let items: Vec<String> = hits
                .iter()
                .take(req.num("results", 12))
                .map(|h| format!("{{\"pos\":{},\"other\":{},\"score\":{}}}", h.pos, json_str(&unscrub(&h.other)), num(h.score)))
                .collect();
            respond(&stream, "200 OK", "application/json", &format!("[{}]", items.join(",")))
        }
        "/api/vigenere" => {
            let text = req.q("text");
            let cipher = scrub(&text);
            let t = Instant::now();
            let q = st.lm.dense(classic::climb_ngram_size(&st.lm, cipher.len()));
            let mut all = Vec::new();
            for kw in req.q("alphabets").split(',') {
                let a = Alphabet::from_keyword(kw.trim());
                all.extend(classic::solve_vigenere_with(&st.lm, &q, &cipher, &a, req.num("max_period", 20), 30).into_iter().take(3));
            }
            let words: Vec<String> = req
                .q("keywords")
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|w| !w.is_empty() && w.chars().all(|c| c.is_ascii_alphabetic()))
                .map(String::from)
                .collect();
            if !words.is_empty() {
                all.extend(classic::solve_vigenere_keyword_search(&st.lm, &cipher, &words, req.num("max_period", 20), 5).1);
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
            let t = Instant::now();
            let q = st.lm.dense(classic::climb_ngram_size(&st.lm, letters));
            let mut all = transpo::solve_route(&st.lm, &q, &text, 60, 3);
            all.extend(transpo::solve_columnar(&st.lm, &q, &text, 2, req.num("max_cols", 12), 8, 3));
            all.sort_by(|x, y| y.per_letter.total_cmp(&x.per_letter));
            let items: Vec<String> = all
                .iter()
                .take(5)
                .map(|s| format!("{{\"method\":{},\"text\":{},\"per_letter\":{}}}", json_str(&s.describe()), json_str(&s.text), num(s.per_letter)))
                .collect();
            respond(&stream, "200 OK", "application/json", &format!("{{\"secs\":{:.2},\"results\":[{}]}}", t.elapsed().as_secs_f64(), items.join(",")))
        }
        "!method" => respond(&stream, "405 Method Not Allowed", "text/plain", "GET only"),
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
    let opts = RkcOptions {
        beam: req.num("beam", 20_000).clamp(1, 2_000_000),
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
    let sols = rkc::solve_words(&st.lm, st.words.as_ref().map(|w| &w.1), &cipher, &opts, Some(&mut cb), Some(&sse.gone));
    sse.send("done", &format!("{{\"secs\":{:.2},\"results\":{}}}", t.elapsed().as_secs_f64(), sols_json(&sols, wm)));
    Ok(())
}

fn api_known(stream: TcpStream, st: &ServerState, req: &Request) -> std::io::Result<()> {
    let cipher = scrub(&req.q("cipher"));
    let sse = Sse::start(stream)?;
    if cipher.len() < 8 {
        sse.send("failed", &json_str("Known-text search needs at least 8 cipher letters."));
        return Ok(());
    }
    let opts = KnownOptions { window: req.num("window", 24), results: req.num("results", 12), ..Default::default() };
    let t = Instant::now();
    let last_pct = AtomicUsize::new(0);
    let prog = |d: usize, tot: usize| {
        let pct = d * 100 / tot.max(1);
        if pct > last_pct.load(Ordering::Relaxed) {
            last_pct.store(pct, Ordering::Relaxed);
            sse.send("progress", &format!("{{\"pct\":{pct}}}"));
        }
    };
    let hits = known::search(&st.lm, &st.quad, &cipher, &st.sources, &opts, Some(&prog), Some(&sse.gone));
    let items: Vec<String> = hits
        .iter()
        .map(|h| {
            format!(
                "{{\"source\":{},\"offset\":{},\"start\":{},\"end\":{},\"key\":{},\"other\":{},\"window\":{},\"score\":{},\"coverage\":{}}}",
                json_str(&h.source),
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
