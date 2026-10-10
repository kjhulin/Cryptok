//! Small, dependency-free hash containers keyed by packed n-grams (`u64`).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Fast multiplicative hasher for integer keys (FxHash-style).
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // A multiply only carries information upwards, so the low bits (which pick the bucket)
        // would depend on the low bits of the key alone: packed n-grams that differ only in
        // their high part, such as word pairs ending in the same word, would all collide.
        // Rotating brings the well-mixed high bits down.
        self.0.rotate_left(26)
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[inline]
fn mix(k: u64) -> u64 {
    // splitmix64 finaliser — good dispersion for sequential keys.
    let mut z = k.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const EMPTY: u64 = u64::MAX;

/// Read-mostly open-addressing map `u64 -> u32` (linear probing, power-of-two capacity).
#[derive(Clone)]
pub struct U64Map {
    keys: Vec<u64>,
    vals: Vec<u32>,
    mask: usize,
    len: usize,
}

impl U64Map {
    pub fn with_capacity(n: usize) -> Self {
        let cap = (n.max(4) * 2).next_power_of_two();
        U64Map { keys: vec![EMPTY; cap], vals: vec![0; cap], mask: cap - 1, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert; returns the previous value if the key existed.
    pub fn insert(&mut self, k: u64, v: u32) -> Option<u32> {
        debug_assert!(k != EMPTY);
        if (self.len + 1) * 2 > self.keys.len() {
            self.grow();
        }
        let mut i = mix(k) as usize & self.mask;
        loop {
            let cur = self.keys[i];
            if cur == EMPTY {
                self.keys[i] = k;
                self.vals[i] = v;
                self.len += 1;
                return None;
            }
            if cur == k {
                return Some(std::mem::replace(&mut self.vals[i], v));
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Insert only if absent. Returns `true` if inserted.
    #[inline]
    pub fn insert_if_absent(&mut self, k: u64, v: u32) -> bool {
        if (self.len + 1) * 2 > self.keys.len() {
            self.grow();
        }
        let mut i = mix(k) as usize & self.mask;
        loop {
            let cur = self.keys[i];
            if cur == EMPTY {
                self.keys[i] = k;
                self.vals[i] = v;
                self.len += 1;
                return true;
            }
            if cur == k {
                return false;
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline]
    pub fn get(&self, k: u64) -> Option<u32> {
        let mut i = mix(k) as usize & self.mask;
        loop {
            let cur = unsafe { *self.keys.get_unchecked(i) };
            if cur == k {
                return Some(unsafe { *self.vals.get_unchecked(i) });
            }
            if cur == EMPTY {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Visit every (key, value) pair in unspecified order.
    pub fn for_each(&self, mut f: impl FnMut(u64, u32)) {
        for (&k, &v) in self.keys.iter().zip(&self.vals) {
            if k != EMPTY {
                f(k, v);
            }
        }
    }

    pub fn clear(&mut self) {
        if self.len > 0 {
            self.keys.fill(EMPTY);
            self.len = 0;
        }
    }

    fn grow(&mut self) {
        let old_k = std::mem::take(&mut self.keys);
        let old_v = std::mem::take(&mut self.vals);
        let cap = old_k.len() * 2;
        self.keys = vec![EMPTY; cap];
        self.vals = vec![0; cap];
        self.mask = cap - 1;
        self.len = 0;
        for (k, v) in old_k.into_iter().zip(old_v) {
            if k != EMPTY {
                self.insert(k, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keys that differ only in their high half (word pairs sharing a second word) must still
    /// spread over the buckets, which are picked by the low bits of the hash.
    #[test]
    fn fx_hash_spreads_keys_that_differ_in_high_bits() {
        use std::hash::BuildHasher;
        let build = BuildHasherDefault::<FxHasher>::default();
        let buckets: std::collections::HashSet<u64> = (0..2000u64).map(|v| build.hash_one((v << 32) | 7) & 0xFFFF).collect();
        assert!(buckets.len() > 1800, "only {} distinct buckets for 2000 keys", buckets.len());
    }

    #[test]
    fn map_basic() {
        let mut m = U64Map::with_capacity(2);
        for i in 0..10_000u64 {
            assert!(m.insert_if_absent(i * 7, i as u32));
        }
        for i in 0..10_000u64 {
            assert_eq!(m.get(i * 7), Some(i as u32));
            assert_eq!(m.get(i * 7 + 1), None);
        }
        assert_eq!(m.len(), 10_000);
    }
}
