//! `cryptok serve` — a small local web server for the browser UI.
//!
//! Dependency-free: std `TcpListener`, one thread per connection, GET requests only.
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
}

pub fn run(state: ServerState, port: u16, open: bool) -> Result<(), String> {
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("cannot listen on port {port}: {e}"))?;
    let url = format!("http://127.0.0.1:{port}/");
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

struct Request {
    path: String,
    query: Vec<(String, String)>,
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
    // Drain headers.
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).ok()? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
    }
    let mut parts = line.split_whitespace();
    if parts.next()? != "GET" {
        return Some(Request { path: "!method".into(), query: vec![] });
    }
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
    Some(Request { path: path.to_string(), query })
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

fn handle(stream: TcpStream, st: &ServerState) -> std::io::Result<()> {
    let Some(req) = read_request(&stream) else { return Ok(()) };
    match req.path.as_str() {
        "/" | "/index.html" => respond(&stream, "200 OK", "text/html; charset=utf-8", INDEX_HTML),
        "/api/info" => {
            let letters: usize = st.sources.iter().map(|s| s.letters.len()).sum();
            let names: Vec<String> = st.sources.iter().map(|s| json_str(&s.name)).collect();
            let body = format!(
                "{{\"order\":{},\"model\":{},\"sources\":[{}],\"source_letters\":{},\"threads\":{}}}",
                st.lm.order(),
                json_str(&st.model_path),
                names.join(","),
                letters,
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
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

fn sols_json(s: &[RkcSolution]) -> String {
    let items: Vec<String> = s
        .iter()
        .map(|x| format!("{{\"a\":{},\"b\":{},\"per_letter\":{}}}", json_str(&unscrub(&x.key)), json_str(&unscrub(&x.plain)), num(x.per_letter())))
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
        beam: req.num("beam", 100_000).clamp(1, 2_000_000),
        results: req.num("results", 12).clamp(1, 100),
        key_hints: parse_hint(&req.q("key"), cipher.len()),
        plain_hints: parse_hint(&req.q("plain"), cipher.len()),
        threads: 0,
    };
    let t = Instant::now();
    let mut last = Instant::now();
    let mut cb = |s: &StepInfo| {
        if last.elapsed().as_millis() >= 120 || s.step == s.total {
            last = Instant::now();
            sse.send("progress", &format!("{{\"step\":{},\"total\":{},\"best\":{}}}", s.step, s.total, sols_json(s.best)));
        }
    };
    let sols = rkc::solve(&st.lm, &cipher, &opts, Some(&mut cb), Some(&sse.gone));
    sse.send("done", &format!("{{\"secs\":{:.2},\"results\":{}}}", t.elapsed().as_secs_f64(), sols_json(&sols)));
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
