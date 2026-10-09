//! `cryptok` — command-line interface for Cryptok Code Cracker 2.0.

use cryptok_core::words::WordModel;
use cryptok_core::lm::LangModel;
use cryptok_core::rkc::{self, RkcOptions, StepInfo};
use cryptok_core::text::{enc, scrub, strip_gutenberg, unscrub};
mod ocr;
mod serve;

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
Cryptok Code Cracker 2.0

USAGE:
  cryptok serve  [--model FILE] [--sources DIR_OR_FILE,...] [--port 8077] [--host 127.0.0.1] [--no-open]
                 (web UI in your browser. Any --host other than localhost requires CRYPTOK_AUTH=user:password
                  and sets conservative limits; see docs/DEPLOY-AWS.md. Tuning: --allowed-host a.com,b.com
                  --max-conns --max-jobs --job-timeout SECS --max-letters --max-beam --max-keywords
                  --job-memory-mb MB (traceback memory per running-key search; the beam narrows to fit).
                  --lean: for a ~512 MB machine: one search at a time, beam <= 50,000, cipher <= 1,500 letters,
                  known texts reloaded per search instead of held in memory.
                  --public: keep --host 127.0.0.1 but use the conservative limits, for a reverse proxy
                  on the same machine that does the authentication; requires --allowed-host. See docs/NGINX-SSO.md)
  cryptok train  [--corpus DIR[,DIR...]] [--order N] [--prune N] [--out FILE] [--exclude a.txt,b.txt]
                 (--prune N drops order-N n-grams seen fewer than N times: a smaller, lighter model)
  cryptok eval   [--model FILE] [--corpus DIR[,DIR...]] --files a.txt,b.txt
                 (held-out cross-entropy in bits per letter; lower is better)
  cryptok score  [--model FILE] TEXT...
  cryptok rkc    [--model FILE] [--beam N] [--results N] [--threads N]
                 [--key-hint HINT] [--plain-hint HINT] [--quiet] CIPHER...
  cryptok crib   [--model FILE] [--results N] --word WORD CIPHER...
  cryptok transpose [--model FILE] [--max-width N] [--max-cols N] [--results N] [--double] TEXT...
                 (route transpositions, one or two steps, and keyed columnar;
                  --double also tries double columnar, slower, --max-cols is capped at 8)
  cryptok known  [--model FILE] [--sources DIR_OR_FILE,...] [--window N] [--results N] CIPHER...
                 (slide known texts along the cipher as candidate running keys)
  cryptok vigenere [--model FILE] [--alphabet KW1,KW2,...] [--alphabet-file FILE]
                 [--max-period N] [--restarts N] [--results N] TEXT...
                 (--alphabet-file: one candidate alphabet keyword per line; each is tried)
  cryptok analyze TEXT...
                 (statistics: IC, periods, and which solver below to try)
  cryptok subst  [--model FILE] [--restarts N] TEXT...
                 (Caesar / Atbash / Affine, then general monoalphabetic substitution)
  cryptok periodic [--model FILE] [--mode MODE] [--plain-alphabet KW] [--cipher-alphabet KW]
                 [--max-period N] [--restarts N] [--results N] TEXT...
                 (MODE: vigenere, beaufort, variant-beaufort, porta, gronsfeld, quagmire;
                  quagmire I/II/III/IV come from the alphabet keywords: I = plain only,
                  II = cipher only, III = same for both, IV = different)
  cryptok autokey [--model FILE] [--max-primer N] [--restarts N] [--results N] TEXT...
                 (Vigenère autokey, plaintext and ciphertext variants)
  cryptok rail   [--model FILE] [--max-rails N] [--results N] TEXT...
  cryptok playfair [--model FILE] [--iters N] [--restarts N] TEXT...
  cryptok bifid  [--model FILE] [--periods 0,5,6,...] [--iters N] [--restarts N] TEXT...
                 (period 0 = whole message)
  cryptok hill   [--model FILE] [--results N] TEXT...
                 (2x2 Hill cipher, all keys)
  cryptok chain  [--model FILE] --steps STEP1,STEP2,... [--beam N] [--max-period N] [--max-cols N]
                 [--max-width N] [--max-rails N] [--restarts N] [--results N] TEXT...
                 (several layers, listed outermost first = the order you undo them, e.g.
                  --steps rail,subst for a rail fence applied over a substitution)
                 (STEP: affine, subst, vigenere, beaufort, variant-beaufort, porta, gronsfeld,
                  rail, route, columnar; autokey, hill, playfair, bifid only as the last step)
  cryptok decode --kind KIND TEXT...
                 (KIND: morse, a1z26, baconian, baconian26, polybius, binary, hex)
  cryptok ocr    [--psm 6] [--digits] [--lang eng] [--raw] IMAGE
                 (read cipher text from a photo or scan with Tesseract; prints the cleaned text,
                  e.g. cryptok analyze $(cryptok ocr page.jpg); --psm 7 = a single line,
                  11 = scattered text. Needs the `tesseract` program installed.)
  cryptok contest run [--model FILE] [--cases bench/contests.tsv] [--only id,id,...] [--exhaustive]
                 [--rkc-beam N] [--word-weight W] [--pass 0.9] [--out results.tsv] [--verbose]
                 (run every solver automatically on contest ciphers with known answers and
                  report which ones fall; see bench/contests.tsv)
  cryptok bench gen [--corpus DIR[,DIR...]] --holdout a.txt,b.txt [--out FILE] [--seed N] [--per-length N]
  cryptok bench run [--model FILE] [--cases FILE] [--beam N] [--threads N]

HINTS are aligned with the cipher's letters; use '_' (or '?' or '.') for unknown positions,
e.g. --plain-hint '____THE_____'. Non-letter characters in CIPHER are ignored.

Defaults: --corpus corpus[,corpus-extra]  --order 6  --model/--out cryptok.cklm  --beam 100000  --results 10
";

struct Args {
    flags: HashMap<String, String>,
    switches: Vec<String>,
    pos: Vec<String>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    const SWITCHES: &[&str] = &["quiet", "help", "no-open", "double", "exhaustive", "verbose", "digits", "raw", "insecure-no-auth", "public", "unigram", "trigram", "lean"];
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
        "transpose" | "trans" => parse(&argv[1..]).and_then(|a| cmd_transpose(&a)),
        "known" => parse(&argv[1..]).and_then(|a| cmd_known(&a)),
        "vigenere" | "vig" => parse(&argv[1..]).and_then(|a| cmd_vigenere(&a)),
        "analyze" => parse(&argv[1..]).and_then(|a| cmd_analyze(&a)),
        "subst" => parse(&argv[1..]).and_then(|a| cmd_subst(&a)),
        "periodic" => parse(&argv[1..]).and_then(|a| cmd_periodic(&a)),
        "autokey" => parse(&argv[1..]).and_then(|a| cmd_autokey(&a)),
        "rail" => parse(&argv[1..]).and_then(|a| cmd_rail(&a)),
        "playfair" => parse(&argv[1..]).and_then(|a| cmd_playfair(&a)),
        "bifid" => parse(&argv[1..]).and_then(|a| cmd_bifid(&a)),
        "hill" => parse(&argv[1..]).and_then(|a| cmd_hill(&a)),
        "chain" => parse(&argv[1..]).and_then(|a| cmd_chain(&a)),
        "decode" => parse(&argv[1..]).and_then(|a| cmd_decode(&a)),
        "ocr" => parse(&argv[1..]).and_then(|a| cmd_ocr(&a)),
        "contest" => match argv.get(1).map(String::as_str) {
            Some("run") => parse(&argv[2..]).and_then(|a| cmd_contest_run(&a)),
            _ => Err("usage: cryptok contest run [--cases FILE] ...".into()),
        },
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

/// Corpus directories: `--corpus a,b` (default `corpus`, plus `corpus-extra` when it exists).
fn corpus_dirs(a: &Args) -> Vec<PathBuf> {
    let default = if Path::new("corpus-extra").is_dir() { "corpus,corpus-extra" } else { "corpus" };
    a.get("corpus", default).split(',').filter(|s| !s.is_empty()).map(PathBuf::from).collect()
}

/// Read a corpus file by name from the first corpus directory that has it.
fn read_corpus_file(dirs: &[PathBuf], name: &str) -> Result<Vec<u8>, String> {
    for d in dirs {
        if let Ok(b) = std::fs::read(d.join(name)) {
            return Ok(b);
        }
    }
    Err(format!("{name}: not found in any corpus directory"))
}

fn load_model(a: &Args) -> Result<LangModel, String> {
    let p = PathBuf::from(a.get("model", "cryptok.cklm"));
    let t = Instant::now();
    let lm = LangModel::load(&p).map_err(|e| format!("cannot load model {}: {e} (run `cryptok train` first)", p.display()))?;
    eprintln!("loaded model {} (order {}) in {:.2}s", p.display(), lm.order(), t.elapsed().as_secs_f64());
    Ok(lm)
}

fn cmd_train(a: &Args) -> Result<(), String> {
    let corpus = corpus_dirs(a);
    let order = a.num("order", 6)?;
    let out = PathBuf::from(a.get("out", "cryptok.cklm"));
    let exclude: Vec<String> = a.get("exclude", "").split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let t = Instant::now();
    let (lm, st) = LangModel::train_dir_pruned(&corpus, order, &exclude, a.num("prune", 1)? as u32).map_err(|e| e.to_string())?;
    let words_path = out.with_extension("words");
    let wm = WordModel::from_corpus(&corpus, &exclude, 3, a.has("trigram")).map_err(|e| e.to_string())?;
    wm.save(&words_path).map_err(|e| format!("{}: {e}", words_path.display()))?;
    eprintln!("wrote {} ({} words)", words_path.display(), wm.vocab_size());
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
    let corpus = corpus_dirs(a);
    let mut total = 0f64;
    let mut letters = 0usize;
    for f in a.get("files", "").split(',').filter(|s| !s.is_empty()) {
        let b = read_corpus_file(&corpus, f)?;
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
    let words = word_setup(a, "")?;
    let opts = RkcOptions {
        beam: a.num("beam", if words.is_some() { 20_000 } else { 100_000 })?,
        results: a.num("results", 10)?,
        threads: a.num("threads", 0)?,
        key_hints: parse_hint(&a.get("key-hint", ""), cipher.len()),
        plain_hints: parse_hint(&a.get("plain-hint", ""), cipher.len()),
        word_weight: words.as_ref().map_or(0.0, |w| w.weight),
        ..Default::default()
    };
    let quiet = a.has("quiet");
    let t = Instant::now();
    let mut cb = |s: &StepInfo| {
        if !quiet {
            eprint!("\r  step {:>4}/{}  beam {:>7}  ", s.step, s.total, s.beam_size);
            let _ = std::io::stderr().flush();
        }
    };
    let sols = rkc::solve_words(&lm, words.as_ref().map(|w| &w.trie), &cipher, &opts, Some(&mut cb), None);
    if !quiet {
        eprintln!();
    }
    println!("solved {} letters, beam {} in {:.2}s", cipher.len(), opts.beam, t.elapsed().as_secs_f64());
    for (i, s) in sols.iter().enumerate() {
        println!("[{:>2}] {:8.3}/letter  total {:9.2}", i + 1, s.per_letter(), s.score);
        match &words {
            Some(w) => {
                println!("     A: {}", w.model.segment(&s.key).render(&s.key));
                println!("     B: {}", w.model.segment(&s.plain).render(&s.plain));
            }
            None => {
                println!("     A: {}", unscrub(&s.key));
                println!("     B: {}", unscrub(&s.plain));
            }
        }
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

/// Letters of the positional text, plus the original for re-layout; errors if empty.
fn cipher_input(a: &Args) -> Result<(String, Vec<u8>), String> {
    let original = a.pos.join("\n");
    let cipher = scrub(&original);
    if cipher.is_empty() {
        return Err("no cipher letters given".into());
    }
    Ok((original, cipher))
}

fn cmd_analyze(a: &Args) -> Result<(), String> {
    let (_, cipher) = cipher_input(a)?;
    let r = cryptok_core::analyze::analyze(&a.pos.join(" "));
    println!("{} letters, {} distinct{}", r.letters, r.distinct, if r.has_j { "" } else { ", no J" });
    println!("index of coincidence {:.4}  (English 0.066, random 0.038)", r.ic);
    println!("E/T/A/O/I/N share    {:.1}%  (English ~52%)", r.etaoin * 100.0);
    println!("best periods         {}", r.periods.iter().map(|(p, v)| format!("{p} ({v:.2})")).collect::<Vec<_>>().join(", "));
    println!("doubled digraphs     {}", r.doubled_digraphs);
    let _ = cipher;
    println!("\nTry, in order:");
    for (cmd, why) in &r.suggestions {
        if cmd.is_empty() {
            println!("  - {why}");
        } else {
            println!("  cryptok {cmd:<9} {why}");
        }
    }
    Ok(())
}

fn cmd_subst(a: &Args) -> Result<(), String> {
    use cryptok_core::subst;
    let (original, cipher) = cipher_input(a)?;
    let lm = load_model(a)?;
    let q = lm.dense(cryptok_core::classic::climb_ngram_size(&lm, cipher.len()));
    println!("-- Caesar / Atbash / Affine");
    for c in subst::solve_affine_family(&lm, &cipher, 3) {
        println!("{} ({:.3}/letter)\n    {}", c.description, c.per_letter, relayout(&original, &c.plain));
    }
    println!("-- General substitution");
    let t = Instant::now();
    let c = subst::solve_substitution(&lm, &q, &cipher, a.num("restarts", 200)?, 1);
    eprintln!("substitution climb {:.2}s", t.elapsed().as_secs_f64());
    println!("{} ({:.3}/letter)\n    {}", c.description, c.per_letter, relayout(&original, &c.plain));
    Ok(())
}

fn cmd_periodic(a: &Args) -> Result<(), String> {
    use cryptok_core::classic::Alphabet;
    use cryptok_core::periodic::{solve_periodic, Mode};
    let (original, cipher) = cipher_input(a)?;
    let mode = match a.get("mode", "vigenere").as_str() {
        "vigenere" => Mode::Vigenere,
        "beaufort" => Mode::Beaufort,
        "variant-beaufort" | "variant" => Mode::VariantBeaufort,
        "porta" => Mode::Porta,
        "gronsfeld" => Mode::Gronsfeld,
        "quagmire" => Mode::Quagmire {
            plain: Alphabet::from_keyword(&a.get("plain-alphabet", "")),
            cipher: Alphabet::from_keyword(&a.get("cipher-alphabet", "")),
        },
        m => return Err(format!("unknown mode '{m}'")),
    };
    let lm = load_model(a)?;
    let q = lm.dense(cryptok_core::classic::climb_ngram_size(&lm, cipher.len()));
    let t = Instant::now();
    let sols = solve_periodic(&lm, &q, &cipher, &mode, a.num("max-period", 20)?, a.num("restarts", 20)?);
    eprintln!("searched in {:.2}s", t.elapsed().as_secs_f64());
    for (i, s) in sols.iter().take(a.num("results", 3)?).enumerate() {
        println!("[{}] {} period {:>2}  key {:<20} {:.3}/letter", i + 1, s.mode, s.period, s.key, s.per_letter());
        println!("    {}", relayout(&original, &s.plain).replace('\n', "\n    "));
    }
    Ok(())
}

fn cmd_autokey(a: &Args) -> Result<(), String> {
    use cryptok_core::periodic::{solve_autokey, Autokey};
    let (original, cipher) = cipher_input(a)?;
    let lm = load_model(a)?;
    let q = lm.dense(cryptok_core::classic::climb_ngram_size(&lm, cipher.len()));
    let t = Instant::now();
    let sols = solve_autokey(&lm, &q, &cipher, a.num("max-primer", 12)?, a.num("restarts", 10)?);
    eprintln!("searched in {:.2}s", t.elapsed().as_secs_f64());
    for (i, s) in sols.iter().take(a.num("results", 3)?).enumerate() {
        let kind = if s.kind == Autokey::Plaintext { "plaintext autokey" } else { "ciphertext autokey" };
        println!("[{}] {kind}, primer {}  {:.3}/letter", i + 1, unscrub(&s.primer), s.score / s.plain.len().max(1) as f32);
        println!("    {}", relayout(&original, &s.plain).replace('\n', "\n    "));
    }
    Ok(())
}

fn cmd_rail(a: &Args) -> Result<(), String> {
    let text = a.pos.join("");
    if scrub(&text).len() < 8 {
        return Err("need at least 8 letters".into());
    }
    let lm = load_model(a)?;
    for (i, s) in cryptok_core::transpo::solve_rail_fence(&lm, &text, a.num("max-rails", 20)?, a.num("results", 3)?).iter().enumerate() {
        println!("[{}] {:.3}/letter  {}\n    {}", i + 1, s.per_letter, s.describe(), s.text);
    }
    Ok(())
}

fn print_square_solution(original: &str, s: &cryptok_core::polygraphic::SquareSolution, label: &str) {
    println!("{label}  key square {}  {:.3}/letter", cryptok_core::polygraphic::square_string(&s.square), s.score / s.plain.len().max(1) as f32);
    println!("    {}", relayout(original, &s.plain).replace('\n', "\n    "));
}

fn cmd_playfair(a: &Args) -> Result<(), String> {
    let (original, cipher) = cipher_input(a)?;
    let lm = load_model(a)?;
    let q = lm.dense(cryptok_core::classic::climb_ngram_size(&lm, cipher.len()));
    let t = Instant::now();
    let s = cryptok_core::polygraphic::solve_playfair(&lm, &q, &cipher, a.num("iters", 500_000)?, a.num("restarts", 32)?, 1);
    eprintln!("annealed in {:.2}s", t.elapsed().as_secs_f64());
    print_square_solution(&original, &s, "Playfair");
    Ok(())
}

fn cmd_bifid(a: &Args) -> Result<(), String> {
    let (original, cipher) = cipher_input(a)?;
    let lm = load_model(a)?;
    let q = lm.dense(cryptok_core::classic::climb_ngram_size(&lm, cipher.len()));
    let periods: Vec<usize> = a.get("periods", "0,3,4,5,6,7,8,9,10,11,12").split(',').filter_map(|p| p.trim().parse().ok()).collect();
    let t = Instant::now();
    let sols = cryptok_core::polygraphic::solve_bifid(&lm, &q, &cipher, &periods, a.num("iters", 100_000)?, a.num("restarts", 4)?, 1);
    eprintln!("annealed in {:.2}s", t.elapsed().as_secs_f64());
    for s in sols.iter().take(3) {
        print_square_solution(&original, s, &format!("Bifid period {}", s.period));
    }
    Ok(())
}

fn cmd_hill(a: &Args) -> Result<(), String> {
    let (original, mut cipher) = cipher_input(a)?;
    if cipher.len() % 2 == 1 {
        cipher.pop();
    }
    let lm = load_model(a)?;
    let q = lm.dense(cryptok_core::classic::climb_ngram_size(&lm, cipher.len()));
    let t = Instant::now();
    let sols = cryptok_core::polygraphic::solve_hill2(&lm, &q, &cipher, a.num("results", 3)?);
    eprintln!("searched 157,248 keys in {:.2}s", t.elapsed().as_secs_f64());
    for (i, s) in sols.iter().enumerate() {
        let m = s.matrix;
        println!("[{}] Hill 2x2 key [{} {}; {} {}]  {:.3}/letter", i + 1, m[0], m[1], m[2], m[3], s.score / s.plain.len().max(1) as f32);
        println!("    {}", relayout(&original, &s.plain).replace('\n', "\n    "));
    }
    Ok(())
}

fn cmd_chain(a: &Args) -> Result<(), String> {
    use cryptok_core::chain::{self, ChainOptions};
    let (original, cipher) = cipher_input(a)?;
    let steps: Vec<chain::Step> = a
        .get("steps", "")
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| chain::parse_step(s).ok_or(format!("unknown step '{s}'; steps are: {}", chain::STEP_NAMES)))
        .collect::<Result<_, _>>()?;
    let d = ChainOptions::default();
    let opt = ChainOptions {
        beam: a.num("beam", d.beam)?,
        max_period: a.num("max-period", d.max_period)?,
        restarts: a.num("restarts", d.restarts)?,
        max_width: a.num("max-width", d.max_width)?,
        max_cols: a.num("max-cols", d.max_cols)?,
        max_rails: a.num("max-rails", d.max_rails)?,
    };
    chain::plan(&steps)?; // fail before loading the model
    let lm = load_model(a)?;
    let t = Instant::now();
    let res = chain::solve_chain(&lm, &cipher, &steps, &opt)?;
    eprintln!("solved chain in {:.2}s", t.elapsed().as_secs_f64());
    for (i, r) in res.iter().take(a.num("results", 3)?).enumerate() {
        println!("[{}] {:.3}/letter", i + 1, r.per_letter);
        for (n, p) in r.path.iter().enumerate() {
            println!("    step {}: {p}", n + 1);
        }
        println!("    {}", relayout(&original, &r.plain).replace('\n', "\n    "));
    }
    Ok(())
}

fn cmd_contest_run(a: &Args) -> Result<(), String> {
    use cryptok_core::auto::{accuracy, auto_solve, AutoOptions};
    let cases = PathBuf::from(a.get("cases", "bench/contests.tsv"));
    let raw = std::fs::read_to_string(&cases).map_err(|e| format!("{}: {e}", cases.display()))?;
    let base = cases.parent().and_then(Path::parent).unwrap_or(Path::new("."));
    let only: Vec<String> = a.get("only", "").split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let pass: f64 = a.get("pass", "0.9").parse().map_err(|_| "--pass expects a number")?;
    let lm = load_model(a)?;
    let words = word_setup(a, "")?.map(|w| (std::sync::Arc::new(w.trie), w.weight));
    let mut rows = Vec::new();
    let t_all = Instant::now();
    println!("{:<20} {:<11} {:<10} {:<10} {:>6} {:>7}  {}", "id", "contest", "type", "found", "acc", "secs", "result");
    for line in raw.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 {
            return Err(format!("bad row (need >= 5 tab-separated fields): {line}"));
        }
        let (id, contest, typ, cipher_field, expected) = (f[0], f[1], f[2], f[3], f[4]);
        if !only.is_empty() && !only.iter().any(|o| o == id) {
            continue;
        }
        let hints = f.get(5).copied().unwrap_or("-");
        let text = match cipher_field.strip_prefix('@') {
            Some(p) => std::fs::read_to_string(base.join(p)).map_err(|e| format!("{id}: {p}: {e}"))?,
            None => cipher_field.to_string(),
        };
        let mut opt = AutoOptions { exhaustive: a.has("exhaustive"), rkc_beam: a.num("rkc-beam", 20_000)?, words: words.clone(), ..Default::default() };
        for h in hints.split(';') {
            if let Some(v) = h.strip_prefix("alphabet=") {
                opt.alphabet_keywords = v.split(',').map(String::from).collect();
            }
        }
        let t = Instant::now();
        let attempts = auto_solve(&lm, &text, &opt);
        let secs = t.elapsed().as_secs_f64();
        let exp = scrub(expected);
        // Judge the winner, but also note if a losing attempt would have been right.
        let (found, acc) = match attempts.first() {
            Some(b) => (b.solver.clone(), accuracy(b, &exp)),
            None => ("-".into(), 0.0),
        };
        let oracle = attempts.iter().map(|x| (accuracy(x, &exp), x.solver.clone())).max_by(|x, y| x.0.total_cmp(&y.0));
        let verdict = if acc >= pass {
            "SOLVED".to_string()
        } else if acc >= 0.5 {
            "partial".to_string()
        } else {
            match oracle {
                Some((o, s)) if o >= pass => format!("missed (ranked below {s})"),
                _ => "failed".to_string(),
            }
        };
        println!("{id:<20} {contest:<11} {typ:<10} {found:<10} {:>5.0}% {secs:>7.1}  {verdict}", acc * 100.0);
        if a.has("verbose") {
            for x in &attempts {
                println!("    {:<16} adj {:>7.3}  plain {:>7.3}  acc {:>4.0}%  {:>5.1}s  {}", x.solver, x.adjusted(), x.per_letter, accuracy(x, &exp) * 100.0, x.secs, x.detail.chars().take(60).collect::<String>());
            }
        }
        rows.push((id.to_string(), contest.to_string(), typ.to_string(), found, acc, secs, verdict));
    }
    println!("\nsummary by contest ({} cases, {:.0}s):", rows.len(), t_all.elapsed().as_secs_f64());
    let mut contests: Vec<&str> = rows.iter().map(|r| r.1.as_str()).collect();
    contests.dedup();
    contests.sort();
    contests.dedup();
    for c in contests {
        let of: Vec<_> = rows.iter().filter(|r| r.1 == c).collect();
        let solved = of.iter().filter(|r| r.6 == "SOLVED").count();
        let partial = of.iter().filter(|r| r.6 == "partial").count();
        println!("  {c:<12} solved {solved}/{}  partial {partial}", of.len());
    }
    if let Some(out) = a.flags.get("out") {
        let mut s = String::from("id\tcontest\ttype\tfound\taccuracy\tsecs\tresult\n");
        for r in &rows {
            s += &format!("{}\t{}\t{}\t{}\t{:.3}\t{:.1}\t{}\n", r.0, r.1, r.2, r.3, r.4, r.5, r.6);
        }
        std::fs::write(out, s).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn cmd_ocr(a: &Args) -> Result<(), String> {
    let path = a.pos.first().ok_or("usage: cryptok ocr [--psm N] [--digits] IMAGE")?;
    let opt = ocr::OcrOptions { psm: a.num("psm", 6)? as u32, allow_digits: a.has("digits"), lang: a.get("lang", "eng") };
    let r = ocr::ocr_file(Path::new(path), &opt).map_err(|e| e.to_string())?;
    if a.has("raw") {
        println!("{}", r.raw.trim_end());
        return Ok(());
    }
    eprintln!("{} letters read, {} stray characters dropped. Check the text against the image: one wrong letter breaks key and crib alignment.", r.cleaned.letters, r.cleaned.dropped);
    println!("{}", r.cleaned.text);
    Ok(())
}

fn cmd_decode(a: &Args) -> Result<(), String> {
    let kind = a.get("kind", "");
    let out = cryptok_core::decode::decode(&kind, &a.pos.join(" ")).ok_or(format!("--kind must be one of: {}", cryptok_core::decode::KINDS))?;
    println!("{out}");
    Ok(())
}

fn cmd_serve(a: &Args) -> Result<(), String> {
    let lm = load_model(a)?;
    let default_sources = if Path::new("bench/private").is_dir() { "corpus,bench/private" } else { "corpus" };
    let paths: Vec<PathBuf> = a.get("sources", default_sources).split(',').filter(|s| !s.is_empty()).map(PathBuf::from).collect();
    let sources = cryptok_core::known::load_sources(&paths).map_err(|e| e.to_string())?;
    eprintln!("loaded {} known-text sources{}", sources.len(), if a.has("lean") { " (reloaded per search to save memory)" } else { "" });
    let sources = serve::SourceStore::new(sources, paths, a.has("lean"));
    let quad = lm.dense(4.min(lm.order() + 1));
    let host = a.get("host", "127.0.0.1");
    let port = a.num("port", 8077)? as u16;
    let words = word_setup(a, "")?.map(|w| (w.model, w.trie, w.weight));
    let ocr = ocr::tesseract_available();
    eprintln!("{}", if ocr { "OCR: tesseract found (image upload uses it)" } else { "OCR: tesseract not installed; the browser will use Tesseract.js instead (needs internet)" });

    let public = a.has("public");
    let mut cfg = serve::Config::for_host(&host, port, public);
    if a.has("lean") {
        cfg = cfg.lean();
    }
    // Credentials: --auth user:password, or better the CRYPTOK_AUTH environment variable
    // (command lines are visible to other users in `ps`).
    let auth = a.flags.get("auth").cloned().or_else(|| std::env::var("CRYPTOK_AUTH").ok().filter(|s| !s.is_empty()));
    if let Some(cred) = auth {
        let (u, p) = cred.split_once(':').ok_or("--auth / CRYPTOK_AUTH must look like user:password")?;
        if u.is_empty() || p.len() < 12 {
            return Err("the password must be at least 12 characters (set CRYPTOK_AUTH=user:password)".into());
        }
        cfg.auth = Some((u.to_string(), p.to_string()));
    }
    if public && !a.flags.contains_key("allowed-host") {
        return Err("--public needs --allowed-host your.domain (requests with any other Host header are refused)".into());
    }
    if cfg.public && !public && cfg.auth.is_none() && !a.has("insecure-no-auth") {
        return Err(format!(
            "refusing to listen on {host} without authentication: anyone who can reach the port could use the server. \
             Set CRYPTOK_AUTH=user:password (see docs/DEPLOY-AWS.md), or pass --insecure-no-auth if something in front of the server already authenticates."
        ));
    }
    if let Some(h) = a.flags.get("allowed-host") {
        cfg.allowed_hosts = h.split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect();
    }
    cfg.max_conns = a.num("max-conns", cfg.max_conns)?;
    cfg.max_jobs = a.num("max-jobs", cfg.max_jobs)?.max(1);
    cfg.job_timeout = std::time::Duration::from_secs(a.num("job-timeout", cfg.job_timeout.as_secs() as usize)? as u64);
    cfg.max_letters = a.num("max-letters", cfg.max_letters)?;
    cfg.max_beam = a.num("max-beam", cfg.max_beam)?;
    cfg.max_keywords = a.num("max-keywords", cfg.max_keywords)?;
    cfg.job_memory_mb = a.num("job-memory-mb", cfg.job_memory_mb)?;

    let state = serve::ServerState::new(lm, quad, sources, a.get("model", "cryptok.cklm"), words, ocr, cfg);
    serve::run(state, !a.has("no-open"))
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

fn cmd_transpose(a: &Args) -> Result<(), String> {
    use cryptok_core::transpo;
    let text = a.pos.join("");
    let letters = scrub(&text).len();
    if letters < 8 {
        return Err("need at least 8 letters".into());
    }
    let lm = load_model(a)?;
    let q = lm.dense(cryptok_core::classic::climb_ngram_size(&lm, letters));
    let results = a.num("results", 3)?;
    let t = Instant::now();
    let mut all = transpo::solve_route(&lm, &q, &text, a.num("max-width", 60)?, results);
    eprintln!("route search {:.2}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    all.extend(transpo::solve_columnar(&lm, &q, &text, 2, a.num("max-cols", 12)?, a.num("restarts", 8)?, results));
    eprintln!("columnar search {:.2}s", t.elapsed().as_secs_f64());
    if a.has("double") {
        let t = Instant::now();
        let max_cols = a.num("max-cols", 8)?.min(8);
        all.extend(transpo::solve_double_columnar(&lm, &q, &text, 2, max_cols, a.num("iters", 30_000)?, a.num("restarts", 4)?, results));
        eprintln!("double columnar search {:.2}s", t.elapsed().as_secs_f64());
    }
    all.sort_by(|x, y| y.per_letter.total_cmp(&x.per_letter));
    for (i, s) in all.iter().take(results).enumerate() {
        println!("[{}] {:.3}/letter  {}", i + 1, s.per_letter, s.describe());
        println!("    {}", s.text);
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
        let reference = if h.reference.is_empty() { String::new() } else { format!(" ({})", h.reference) };
        println!(
            "[{:>2}] {}{} @ {}  best-window {:.3}/letter  overall {:.3}/letter  English coverage {:.0}%",
            i + 1, h.source, reference, h.offset, h.window_score, h.score, h.coverage * 100.0
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
    let corpus = corpus_dirs(a);
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
            let s = read_corpus_file(&corpus, h)?;
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
    let words = word_setup(a, "1342.txt,2701.txt,84.txt")?;
    let opts = RkcOptions {
        beam: a.num("beam", if words.is_some() { 20_000 } else { 100_000 })?,
        results: 1,
        threads: a.num("threads", 0)?,
        word_weight: words.as_ref().map_or(0.0, |w| w.weight),
        merge_len: a.num("merge-len", 0)?,
        ..Default::default()
    };
    let trie = words.as_ref().map(|w| &w.trie);
    if a.num("diag", 0)? > 0 {
        return cmd_bench_diag(a, &lm, &cases);
    }
    println!("{:>4} {:>5} {:>8} {:>8}", "id", "len", "acc%", "secs");
    let mut by_len: Vec<(usize, Vec<(f64, f64)>)> = vec![];
    let total = Instant::now();
    for (id, c, k, p) in &cases {
        let t = Instant::now();
        let s = rkc::solve_words(&lm, trie, c, &opts, None, None);
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

/// Diagnostic: does the truth outscore the solver's answer under the char / word models?
fn cmd_bench_diag(a: &Args, lm: &LangModel, cases: &[(String, Vec<u8>, Vec<u8>, Vec<u8>)]) -> Result<(), String> {
    let corpus = corpus_dirs(a);
    let excl: Vec<String> = a.get("word-exclude", "1342.txt,2701.txt,84.txt").split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let wm = WordModel::from_corpus(&corpus, &excl, a.num("min-count", 3)? as u32, a.has("trigram")).map_err(|e| e.to_string())?;
    let opts = RkcOptions { beam: a.num("beam", 10_000)?, results: 1, threads: a.num("threads", 0)?, ..Default::default() };
    println!("{:>4} {:>5} {:>6} {:>9} {:>9} {:>9}", "id", "len", "acc%", "d_char", "d_word", "d_all(w=1)");
    let (mut cwin, mut wwin, mut awin, mut n) = (0, 0, 0, 0);
    for (id, c, k, p) in cases {
        let Some(sol) = rkc::solve(lm, c, &opts, None, None).into_iter().next() else { continue };
        let ch = |k: &[u8], p: &[u8]| lm.score(k) + lm.score(p);
        let wd = |k: &[u8], p: &[u8]| wm.segment(k).score + wm.segment(p).score;
        let d_char = ch(k, p) - ch(&sol.key, &sol.plain);
        let d_word = wd(k, p) - wd(&sol.key, &sol.plain);
        println!("{:>4} {:>5} {:>6.1} {:>9.1} {:>9.1} {:>9.1}", id, c.len(), rkc::pair_accuracy(&sol, k, p) * 100.0, d_char, d_word, d_char + d_word);
        n += 1;
        cwin += (d_char > 0.0) as usize;
        wwin += (d_word > 0.0) as usize;
        awin += (d_char + d_word > 0.0) as usize;
    }
    println!("truth scores higher than solver output: char {cwin}/{n}, word {wwin}/{n}, combined {awin}/{n}");
    Ok(())
}

/// Word model + trie for word-aware solving. Weight 0 (or a missing corpus) disables it.
struct Words {
    model: WordModel,
    trie: cryptok_core::words::WordTrie,
    weight: f32,
}

fn num_f32(a: &Args, k: &str, d: &str) -> Result<f32, String> {
    a.get(k, d).parse().map_err(|_| format!("--{k} expects a number"))
}

/// `default_exclude` lists corpus files left out of the word model (the benchmark holds books out).
fn word_setup(a: &Args, default_exclude: &str) -> Result<Option<Words>, String> {
    let weight = num_f32(a, "word-weight", "0.4")?;
    if weight <= 0.0 {
        return Ok(None);
    }
    let t = Instant::now();
    let sibling = PathBuf::from(a.get("model", "cryptok.cklm")).with_extension("words");
    let model = if !a.flags.contains_key("corpus") && !a.flags.contains_key("word-exclude") && sibling.is_file() {
        WordModel::load(&sibling).map_err(|e| format!("{}: {e}", sibling.display()))?
    } else {
        let corpus = corpus_dirs(a);
        if !corpus.iter().all(|d| d.is_dir()) {
            eprintln!("note: no word list ({}) and no corpus directory, solving without the word model", sibling.display());
            return Ok(None);
        }
        let excl: Vec<String> = a.get("word-exclude", default_exclude).split(',').filter(|s| !s.is_empty()).map(String::from).collect();
        WordModel::from_corpus(&corpus, &excl, a.num("min-count", 3)? as u32, a.has("trigram")).map_err(|e| e.to_string())?
    };
    let model = if a.has("unigram") {
        model.without_bigrams()
    } else if a.has("trigram") {
        model
    } else {
        model.without_trigrams()
    };
    let trie = model.trie().with_oov(num_f32(a, "oov-base", "-4")?, num_f32(a, "oov-per", "-3.5")?);
    eprintln!("word model: {} words, {} pairs, {} triples in {:.2}s (weight {weight})", model.vocab_size(), model.bigram_count(), model.trigram_count(), t.elapsed().as_secs_f64());
    Ok(Some(Words { model, trie, weight }))
}
