//! Regression tests on real puzzles. They need a trained model at the workspace root
//! (`cryptok train`); without it they are skipped.

use cryptok_core::classic::{solve_vigenere, solve_vigenere_keyword_search, Alphabet};
use cryptok_core::known::{self, KnownOptions};
use cryptok_core::lm::LangModel;
use cryptok_core::text::{scrub, unscrub};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model() -> Option<LangModel> {
    let p = root().join("cryptok.cklm");
    if !p.exists() {
        eprintln!("skipping: no model at {}", p.display());
        return None;
    }
    Some(LangModel::load(&p).unwrap())
}

#[test]
fn kryptos_k1_and_k2() {
    let Some(lm) = model() else { return };
    let a = Alphabet::from_keyword("KRYPTOS");
    for (file, key, start) in [("k1.txt", "PALIMPSEST", "BETWEENSUBTLESHADING"), ("k2.txt", "ABSCISSA", "ITWASTOTALLYINVISIBLE")] {
        let c = scrub(&std::fs::read_to_string(root().join("bench/kryptos").join(file)).unwrap());
        let best = &solve_vigenere(&lm, &c, &a, 20, 30)[0];
        assert_eq!(unscrub(&best.key), key);
        assert!(unscrub(&best.plain).starts_with(start));
    }
}

#[test]
fn dc23_known_source() {
    let Some(lm) = model() else { return };
    let sources = known::load_sources(&[root().join("corpus")]).unwrap();
    let quad = lm.dense(4);
    let c = scrub("BVFBHGHXAWJEKEDMDZAPRMWGNMTVIRPWIKHGIPUU");
    let hits = known::search(&lm, &quad, &c, &sources, &KnownOptions::default(), None, None);
    assert_eq!(hits[0].source, "11.txt");
    assert_eq!(unscrub(&hits[0].plain), "FORTUNATEISTHEREDSHIRTENGINEERWHOLIVEPHR");
}

#[test]
fn kryptos_k3_route() {
    use cryptok_core::transpo::solve_route;
    let Some(lm) = model() else { return };
    let t = std::fs::read_to_string(root().join("bench/kryptos/k3.txt")).unwrap();
    let q = lm.dense(4);
    let best = &solve_route(&lm, &q, &t, 60, 1)[0];
    assert!(best.text.starts_with("SLOWLYDESPARATLYSLOWLY"), "{}", best.text);
}

#[test]
fn keyed_columnar() {
    use cryptok_core::text::strip_gutenberg;
    use cryptok_core::transpo::solve_columnar;
    let Some(lm) = model() else { return };
    let book = std::fs::read_to_string(root().join("corpus/84.txt")).unwrap();
    let plain: String = unscrub(&scrub(strip_gutenberg(&book)))[5000..5250].to_string();
    let p: Vec<char> = plain.chars().collect();
    let order = [3usize, 6, 0, 4, 1, 5, 2];
    let mut c = String::new();
    for &col in &order {
        c.extend(p.iter().skip(col).step_by(order.len()));
    }
    let q = lm.dense(4);
    let best = &solve_columnar(&lm, &q, &c, 2, 10, 8, 1)[0];
    assert_eq!(best.text, plain);
}

#[test]
fn kryptos_k2_unknown_alphabet() {
    let Some(lm) = model() else { return };
    // KRYPTOS hidden among ~300 dictionary words.
    let mut words: Vec<String> = std::fs::read_to_string(root().join("data/words.txt"))
        .unwrap()
        .lines()
        .filter(|w| (5..=10).contains(&w.len()) && w.chars().all(|c| c.is_ascii_alphabetic()))
        .step_by(1000)
        .map(String::from)
        .collect();
    words.insert(words.len() / 2, "KRYPTOS".into());
    let c = scrub(&std::fs::read_to_string(root().join("bench/kryptos/k2.txt")).unwrap());
    let (ranked, sols) = solve_vigenere_keyword_search(&lm, &c, &words, 20, 3);
    assert_eq!(ranked[0].keyword, "KRYPTOS");
    assert_eq!(unscrub(&sols[0].key), "ABSCISSA");
}
