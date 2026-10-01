//! `cryptok` — command-line interface for Cryptok Code Cracker 2.0.

use cryptok_core::lm::LangModel;
use cryptok_core::rkc::{self, RkcOptions, StepInfo};
use cryptok_core::text::{enc, scrub, strip_gutenberg, unscrub};
mod serve;

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
Cryptok Code Cracker 2.0

USAGE:
  cryptok serve  [--model FILE] [--sources DIR_OR_FILE,...] [--port 8077] [--no-open]
                 (web UI in your browser)
  cryptok train  [--corpus DIR] [--order N] [--out FILE] [--exclude a.txt,b.txt]
  cryptok eval   [--model FILE] [--corpus DIR] --files a.txt,b.txt
                 (held-out cross-entropy in bits per letter; lower is better)
  cryptok score  [--model FILE] TEXT...
  cryptok rkc    [--model FILE] [--beam N] [--results N] [--threads N]
                 [--key-hint HINT] [--plain-hint HINT] [--quiet] CIPHER...
  cryptok crib   [--model FILE] [--results N] --word WORD CIPHER...
  cryptok known  [--model FILE] [--sources DIR_OR_FILE,...] [--window N] [--results N] CIPHER...
                 (slide known texts along the cipher as candidate running keys)
  cryptok vigenere [--model FILE] [--alphabet KW1,KW2,...] [--alphabet-file FILE]
                 [--max-period N] [--restarts N] [--results N] TEXT...
                 (--alphabet-file: one candidate alphabet keyword per line; each is tried)
  cryptok bench gen [--corpus DIR] --holdout a.txt,b.txt [--out FILE] [--seed N] [--per-length N]
  cryptok bench run [--model FILE] [--cases FILE] [--beam N] [--threads N]

HINTS are aligned with the cipher's letters; use '_' (or '?' or '.') for unknown positions,
e.g. --plain-hint '____THE_____'. Non-letter characters in CIPHER are ignored.

Defaults: --corpus corpus  --order 6  --model/--out cryptok.cklm  --beam 100000  --results 10
";

struct Args {
    flags: HashMap<String, String>,
    switches: Vec<String>,
    pos: Vec<String>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    const SWITCHES: &[&str] = &["quiet", "help", "no-open"];
    let mut a = Args { flags: HashMap::new(), switches: vec![], pos: vec![] };
    let mut i = 0;
    while i < args.len() {
        let s = &args[i];
        if let Some(name) = s.strip_prefix("--") {
            if let Some((k, v)) = name.split_once('=') {
                a.flags.insert(k.to_string(), v.to_string());
            } else if SWITCHES.contains(&name) {
                a.switches.push(name.to_string());
            } else {
                let v = args.get(i + 1).ok_or(format!("missing value for --{name}"))?;
                a.flags.insert(name.to_string(), v.clone());
                i += 1;
            }
        } else {
            a.pos.push(s.clone());
        }
        i += 1;
    }
    Ok(a)
}

impl Args {
    fn get(&self, k: &str, d: &str) -> String {
        self.flags.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    fn num(&self, k: &str, d: usize) -> Result<usize, String> {
        match self.flags.get(k) {
            None => Ok(d),
            Some(v) => v.replace('_', "").parse().map_err(|_| format!("--{k} expects a number, got '{v}'")),
        }
    }
    fn has(&self, k: &str) -> bool {
        self.switches.iter().any(|s| s == k)
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "--help" || argv[0] == "-h" || argv[0] == "help" {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let res = match argv[0].as_str() {
        "train" => parse(&argv[1..]).and_then(|a| cmd_train(&a)),
        "eval" => parse(&argv[1..]).and_then(|a| cmd_eval(&a)),
        "score" => parse(&argv[1..]).and_then(|a| cmd_score(&a)),
        "rkc" => parse(&argv[1..]).and_then(|a| cmd_rkc(&a)),
        "serve" => parse(&argv[1..]).and_then(|a| cmd_serve(&a)),
        "crib" => parse(&argv[1..]).and_then(|a| cmd_crib(&a)),
        "known" => parse(&argv[1..]).and_then(|a| cmd_known(&a)),
        "vigenere" | "vig" => parse(&argv[1..]).and_then(|a| cmd_vigenere(&a)),
        "bench" => match argv.get(1).map(String::as_str) {
            Some("gen") => parse(&argv[2..]).and_then(|a| cmd_bench_gen(&a)),
            Some("run") => parse(&argv[2..]).and_then(|a| cmd_bench_run(&a)),
            _ => Err("usage: cryptok bench gen|run ...".into()),
        },
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn load_model(a: &Args) -> Result<LangModel, String> {
    let p = PathBuf::from(a.get("model", "cryptok.cklm"));
    let t = Instant::now();
    let lm = LangModel::load(&p).map_err(|e| format!("cannot load model {}: {e} (run `cryptok train` first)", p.display()))?;
    eprintln!("loaded model {} (order {}) in {:.2}s", p.display(), lm.order(), t.elapsed().as_secs_f64());
    Ok(lm)
}

fn cmd_train(a: &Args) -> Result<(), String> {
    let corpus = PathBuf::from(a.get("corpus", "corpus"));
    let order = a.num("order", 6)?;
    let out = PathBuf::from(a.get("out", "cryptok.cklm"));
    let exclude: Vec<String> = a.get("exclude", "").split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let t = Instant::now();
    let (lm, st) = LangModel::train_dir(&corpus, order, &exclude).map_err(|e| e.to_string())?;
    let tt = t.elapsed().as_secs_f64();
    lm.save(&out).map_err(|e| e.to_string())?;
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!("trained order-{order} model on {} letters in {tt:.1}s", st.letters);
    println!("contexts per level: {:?}", st.contexts_per_level);
    println!("discounts: {:?}", st.discounts.iter().map(|d| (d * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    println!("wrote {} ({:.1} MB)", out.display(), size as f64 / 1e6);
    Ok(())
}

fn cmd_eval(a: &Args) -> Result<(), String> {
    let lm = load_model(a)?;
    let corpus = PathBuf::from(a.get("corpus", "corpus"));
    let mut total = 0f64;
    let mut letters = 0usize;
    for f in a.get("files", "").split(',').filter(|s| !s.is_empty()) {
        let b = std::fs::read(corpus.join(f)).map_err(|e| format!("{f}: {e}"))?;
        let l = scrub(strip_gutenberg(&String::from_utf8_lossy(&b)));
        // Score in 1,000-letter chunks (the model starts each chunk without context).
        for ch in l.chunks(1000) {
            total += lm.score(ch) as f64;
            letters += ch.len();
        }
    }
    if letters == 0 {
        return Err("no letters to evaluate (use --files)".into());
    }
    println!("{letters} letters, {:.4} bits/letter", -total / letters as f64 / std::f64::consts::LN_2);
    Ok(())
}

fn cmd_score(a: &Args) -> Result<(), String> {
    let lm = load_model(a)?;
    let text = a.pos.join(" ");
    let l = scrub(&text);
    println!("letters={} total={:.3} per_letter={:.4}", l.len(), lm.score(&l), lm.score_per_letter(&l));
    Ok(())
}

fn parse_hint(h: &str, n: usize) -> Vec<Option<u8>> {
    let mut v: Vec<Option<u8>> = h
        .bytes()
        .filter_map(|c| match c {
            b'A'..=b'Z' => Some(Some(c - b'A')),
            b'a'..=b'z' => Some(Some(c - b'a')),
            b'_' | b'?' | b'.' | b'*' => Some(None),
            _ => None,
        })
        .collect();
    v.resize(n, None);
    v
}

fn cmd_rkc(a: &Args) -> Result<(), String> {
    let cipher = scrub(&a.pos.join(""));
    if cipher.is_empty() {
        return Err("no cipher letters given".into());
    }
    let lm = load_model(a)?;
    let opts = RkcOptions {
        beam: a.num("beam", 100_000)?,
        results: a.num("results", 10)?,
        threads: a.num("threads", 0)?,
        key_hints: parse_hint(&a.get("key-hint", ""), cipher.len()),
        plain_hints: parse_hint(&a.get("plain-hint", ""), cipher.len()),
    };
    let quiet = a.has("quiet");
    let t = Instant::now();
    let mut cb = |s: &StepInfo| {
        if !quiet {
            eprint!("\r  step {:>4}/{}  beam {:>7}  ", s.step, s.total, s.beam_size);
            let _ = std::io::stderr().flush();
        }
    };
    let sols = rkc::solve(&lm, &cipher, &opts, Some(&mut cb), None);
    if !quiet {
        eprintln!();
    }
    println!("solved {} letters, beam {} in {:.2}s", cipher.len(), opts.beam, t.elapsed().as_secs_f64());
    for (i, s) in sols.iter().enumerate() {
        println!("[{:>2}] {:8.3}/letter  total {:9.2}", i + 1, s.per_letter(), s.score);
        println!("     A: {}", unscrub(&s.key));
        println!("     B: {}", unscrub(&s.plain));
    }
    Ok(())
}

/// Put solved letters back into the original text layout (keeps '?', spaces, punctuation).
fn relayout(original: &str, letters: &[u8]) -> String {
    let mut it = letters.iter();
    original
        .chars()
        .map(|ch| if ch.is_ascii_alphabetic() { it.next().map(|&l| (b'A' + l) as char).unwrap_or(ch) } else { ch })
        .collect()
}

fn cmd_vigenere(a: &Args) -> Result<(), String> {
    let original = a.pos.join("\n");
    let cipher = scrub(&original);
    if cipher.is_empty() {
        return Err("no cipher letters given".into());
    }
    let lm = load_model(a)?;
    let max_period = a.num("max-period", 20)?;
    let restarts = a.num("restarts", 30)?;
    let results = a.num("results", 3)?;
    let alphabets: Vec<String> = a.get("alphabet", "").split(',').map(|s| s.trim().to_string()).collect();
    let t = Instant::now();
    let mut all = Vec::new();
    let q = lm.dense(cryptok_core::classic::climb_ngram_size(&lm, cipher.len()));
    for kw in &alphabets {
        let alpha = cryptok_core::classic::Alphabet::from_keyword(kw);
        all.extend(cryptok_core::classic::solve_vigenere_with(&lm, &q, &cipher, &alpha, max_period, restarts).into_iter().take(results));
    }
    if let Some(f) = a.flags.get("alphabet-file") {
        let words: Vec<String> = std::fs::read_to_string(f)
            .map_err(|e| format!("{f}: {e}"))?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_alphabetic()))
            .collect();
        let t2 = Instant::now();
        let (ranked, sols) = cryptok_core::classic::solve_vigenere_keyword_search(&lm, &cipher, &words, max_period, 5);
        eprintln!(
            "ranked {} keywords in {:.1}s; best alphabets: {}",
            words.len(),
            t2.elapsed().as_secs_f64(),
            ranked.iter().map(|h| h.keyword.as_str()).collect::<Vec<_>>().join(", ")
        );
        all.extend(sols);
    }
    let pen = 26f32.ln();
    all.sort_by(|x, y| (y.score - y.period as f32 * pen).total_cmp(&(x.score - x.period as f32 * pen)));
    println!("searched periods 1..={max_period} in {:.2}s", t.elapsed().as_secs_f64());
    for (i, s) in all.iter().take(results).enumerate() {
        println!("[{}] period {:>2}  key {:<20} alphabet {}  {:.3}/letter", i + 1, s.period, unscrub(&s.key), s.alphabet, s.per_letter());
        println!("    {}", relayout(&original, &s.plain).replace('\n', "\n    "));
    }
    Ok(())
}

fn cmd_serve(a: &Args) -> Result<(), String> {
    let lm = load_model(a)?;
    let default_sources = if Path::new("bench/private").is_dir() { "corpus,bench/private" } else { "corpus" };
    let paths: Vec<PathBuf> = a.get("sources", default_sources).split(',').filter(|s| !s.is_empty()).map(PathBuf::from).collect();
    let sources = cryptok_core::known::load_sources(&paths).map_err(|e| e.to_string())?;
    eprintln!("loaded {} known-text sources", sources.len());
    let quad = lm.dense(4.min(lm.order() + 1));
    let port = a.num("port", 8077)? as u16;
    let state = serve::ServerState { lm, quad, sources, model_path: a.get("model", "cryptok.cklm") };
    serve::run(state, port, !a.has("no-open"))
}

fn cmd_crib(a: &Args) -> Result<(), String> {
    let cipher = scrub(&a.pos.join(""));
    let word = scrub(&a.get("word", ""));
    if cipher.is_empty() || word.is_empty() {
        return Err("usage: cryptok crib --word WORD CIPHER".into());
    }
    let lm = load_model(a)?;
    for h in rkc::crib_search(&lm, &cipher, &word).iter().take(a.num("results", 15)?) {
        println!("{:>4}  {:7.3}/letter  {}", h.pos, h.score, unscrub(&h.other));
    }
    Ok(())
}

fn cmd_known(a: &Args) -> Result<(), String> {
    use cryptok_core::known::{self, KnownOptions};
    let cipher = scrub(&a.pos.join(""));
    if cipher.is_empty() {
        return Err("no cipher letters given".into());
    }
    let lm = load_model(a)?;
    let paths: Vec<PathBuf> = a.get("sources", "corpus").split(',').filter(|s| !s.is_empty()).map(PathBuf::from).collect();
    let t = Instant::now();
    let sources = known::load_sources(&paths).map_err(|e| e.to_string())?;
    let letters: usize = sources.iter().map(|s| s.letters.len()).sum();
    eprintln!("loaded {} sources ({:.1}M letters) in {:.2}s", sources.len(), letters as f64 / 1e6, t.elapsed().as_secs_f64());
    let quad = lm.dense(4.min(lm.order() + 1));
    let opts = KnownOptions { window: a.num("window", 24)?, results: a.num("results", 10)?, threads: a.num("threads", 0)?, ..Default::default() };
    let t = Instant::now();
    let quiet = a.has("quiet");
    let prog = |d: usize, tot: usize| {
        if !quiet {
            eprint!("\r  {:>5.1}%", d as f64 * 100.0 / tot.max(1) as f64);
        }
    };
    let hits = known::search(&lm, &quad, &cipher, &sources, &opts, Some(&prog), None);
    if !quiet {
        eprintln!();
    }
    println!("searched {} alignments in {:.2}s", letters, t.elapsed().as_secs_f64());
    for (i, h) in hits.iter().enumerate() {
        let pad = |v: &[u8]| format!("{}{}{}", ".".repeat(h.start), unscrub(v), ".".repeat(cipher.len() - h.end));
        println!(
            "[{:>2}] {} @ {}  best-window {:.3}/letter  overall {:.3}/letter  English coverage {:.0}%",
            i + 1, h.source, h.offset, h.window_score, h.score, h.coverage * 100.0
        );
        println!("     key:   {}", pad(&h.key));
        println!("     other: {}", pad(&h.plain));
    }
    Ok(())
}

// ---------------------------------------------------------------- benchmark

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const BENCH_LENGTHS: &[usize] = &[40, 60, 100, 150, 300];

fn cmd_bench_gen(a: &Args) -> Result<(), String> {
    let corpus = PathBuf::from(a.get("corpus", "corpus"));
    let holdout: Vec<String> = a.get("holdout", "").split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    if holdout.len() < 2 {
        return Err("--holdout needs at least two files (key and plaintext come from different books)".into());
    }
    let out = PathBuf::from(a.get("out", "bench/cases.tsv"));
    let per = a.num("per-length", 4)?;
    let mut rng = Lcg(a.num("seed", 2026)? as u64);
    let texts: Vec<Vec<u8>> = holdout
        .iter()
        .map(|h| {
            let s = std::fs::read(corpus.join(h)).map_err(|e| format!("{h}: {e}"))?;
            Ok(scrub(strip_gutenberg(&String::from_utf8_lossy(&s))))
        })
        .collect::<Result<_, String>>()?;
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let mut f = std::fs::File::create(&out).map_err(|e| e.to_string())?;
    writeln!(f, "# id\tlength\tcipher\tkey\tplain  (generated from holdout: {})", holdout.join(",")).unwrap();
    let mut id = 0;
    for &len in BENCH_LENGTHS {
        for _ in 0..per {
            let bk = rng.below(texts.len());
            let mut bp = rng.below(texts.len() - 1);
            if bp >= bk {
                bp += 1;
            }
            let ko = rng.below(texts[bk].len() - len);
            let po = rng.below(texts[bp].len() - len);
            let key = &texts[bk][ko..ko + len];
            let plain = &texts[bp][po..po + len];
            let c: Vec<u8> = plain.iter().zip(key).map(|(&p, &k)| enc(p, k)).collect();
            id += 1;
            writeln!(f, "{id}\t{len}\t{}\t{}\t{}", unscrub(&c), unscrub(key), unscrub(plain)).unwrap();
        }
    }
    println!("wrote {id} cases to {}", out.display());
    Ok(())
}

fn read_cases(p: &Path) -> Result<Vec<(String, Vec<u8>, Vec<u8>, Vec<u8>)>, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(s.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f.len() >= 5).then(|| (f[0].to_string(), scrub(f[2]), scrub(f[3]), scrub(f[4])))
        })
        .collect())
}

fn cmd_bench_run(a: &Args) -> Result<(), String> {
    let lm = load_model(a)?;
    let cases = read_cases(&PathBuf::from(a.get("cases", "bench/cases.tsv")))?;
    let opts = RkcOptions { beam: a.num("beam", 100_000)?, results: 1, threads: a.num("threads", 0)?, ..Default::default() };
    println!("{:>4} {:>5} {:>8} {:>8}", "id", "len", "acc%", "secs");
    let mut by_len: Vec<(usize, Vec<(f64, f64)>)> = vec![];
    let total = Instant::now();
    for (id, c, k, p) in &cases {
        let t = Instant::now();
        let s = rkc::solve(&lm, c, &opts, None, None);
        let secs = t.elapsed().as_secs_f64();
        let acc = s.first().map(|s| rkc::pair_accuracy(s, k, p)).unwrap_or(0.0);
        println!("{:>4} {:>5} {:>8.1} {:>8.2}", id, c.len(), acc * 100.0, secs);
        match by_len.iter_mut().find(|(l, _)| *l == c.len()) {
            Some((_, v)) => v.push((acc, secs)),
            None => by_len.push((c.len(), vec![(acc, secs)])),
        }
    }
    println!("\nbeam {}  —  summary by length", opts.beam);
    println!("{:>5} {:>6} {:>10} {:>10}", "len", "cases", "mean acc%", "mean secs");
    for (l, v) in &by_len {
        let n = v.len() as f64;
        println!(
            "{:>5} {:>6} {:>10.1} {:>10.2}",
            l,
            v.len(),
            v.iter().map(|x| x.0).sum::<f64>() / n * 100.0,
            v.iter().map(|x| x.1).sum::<f64>() / n
        );
    }
    let all: Vec<f64> = by_len.iter().flat_map(|(_, v)| v.iter().map(|x| x.0)).collect();
    let overall = all.iter().sum::<f64>() / all.len().max(1) as f64 * 100.0;
    println!("overall mean acc {overall:.1}%");
    println!("total {:.1}s", total.elapsed().as_secs_f64());
    let min = a.num("min-acc", 0)? as f64;
    if overall < min {
        return Err(format!("accuracy regression: {overall:.1}% < required {min:.1}%"));
    }
    Ok(())
}
