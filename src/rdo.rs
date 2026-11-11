//! Rate-distortion optimization of already-encoded GPU blocks.
//!
//! The GPU format is fixed-rate, so "rate" here means bits *after* a downstream LZ compressor
//! (Zstd, Deflate, Kraken, GDeflate...). For each block, in storage order, we try alternative
//! encodings that copy bytes from recently emitted blocks — whole blocks, endpoint fields or
//! index fields, with the other field re-optimized — and keep the one minimizing
//!
//! ```text
//! J = smooth_scale(block) * SSE / 16  +  lambda * estimated_LZ_bits
//! ```
//!
//! The rate estimate is a greedy LZ parse against a sliding window ([`LzModel`]). Smooth
//! blocks get a distortion multiplier because gradients band visibly under RDO, and a hard
//! cap on error growth keeps any single block from getting much worse than its base encoding.

use crate::bc1;
use crate::bc4;
use crate::bc7;
use crate::format::Format;
use crate::ranker::bc4::{Rank1, RankSrc1};
use crate::ranker::bc5::{RankBlock, RankSrc};
use crate::ranker::bc7::{RankBlock7, RankSrc7};
use crate::surface::{Block, block_sse};
use rayon::prelude::*;

/// RDO tuning knobs.
#[derive(Clone, Copy, Debug)]
pub struct RdoParams {
    /// Rate weight. 0 disables RDO. Typical range 0.25 – 8.
    pub lambda: f32,
    /// LZ lookback window in bytes for the rate model. `None` picks a per-format default
    /// ([`RdoParams::window_for`]).
    pub window: Option<usize>,
    /// How many previous blocks to try copying from (capped by the window). `None` picks a
    /// per-format default ([`RdoParams::max_back_for`]).
    pub max_back: Option<usize>,
    /// Partial-span candidates are only tried for the nearest this many previous blocks; farther
    /// ones get only whole-block and whole-field copies. `None` picks a per-format default
    /// ([`RdoParams::span_back_for`]).
    pub span_back: Option<usize>,
    /// After the search, try replacing one field (endpoints or indices, per BC4 half for BC5)
    /// of the chosen block with that field of each of this many previous blocks. `None` picks a
    /// per-format default ([`RdoParams::refine_back_for`]); 0 disables it.
    pub refine_back: Option<usize>,
    /// Beyond `max_back`, try the endpoints of earlier blocks up to this far back (capped by the
    /// window), re-indexed: each distinct pair once, at its nearest position, and only when a
    /// cheap lower bound on its error leaves it a chance. `None` picks a per-format default
    /// ([`RdoParams::far_back_for`]); 0 disables it. BC1/BC4/BC5 only.
    pub far_back: Option<usize>,
    /// Channel standard deviation at which a block gets half of `smooth_scale`'s extra weight
    /// (see [`smooth_factor`]).
    pub smooth_std: f32,
    /// Distortion multiplier applied to perfectly flat blocks (falling towards 1 as they get busier).
    pub smooth_scale: f32,
    /// Reject candidates whose RMS error exceeds this multiple of the base encoding's.
    pub max_rms_ratio: f32,
    /// ... unless it is within this RMS per channel: blocks the base encodes almost exactly would
    /// otherwise have almost no room to move.
    pub min_rms: f32,
    /// Bytes of blocks per independently optimized chunk (enables parallelism; matches are not
    /// searched across chunk boundaries). Sized in bytes so the window-sized stretch at the start
    /// of each chunk that cannot see the previous chunk is the same fraction for 8- and 16-byte
    /// formats. Kept independent of the thread count so output does not depend on the machine.
    /// `None` picks a per-format default ([`RdoParams::chunk_bytes_for`]).
    pub chunk_bytes: Option<usize>,
    /// Hash-chain depth for the match finder.
    pub chain_depth: usize,
    /// After optimizing a chunk, compress it with real Zstd at this level and revert the
    /// chunk to its base encoding if RDO did not make it smaller. `None` skips the check.
    pub verify_zstd_level: Option<i32>,
    /// BC7: refit endpoints with the entropy-reduced quantization (set when the base encode
    /// used [`crate::format::BlockOpts::reduce_entropy`]).
    pub reduce_entropy: bool,
    /// Localized search (BC4, BC5, BC7): beyond `max_back`, the format's learned ranker
    /// ([`crate::ranker`]) picks this many earlier blocks of the chunk for the full search. `None` picks a per-format default ([`RdoParams::local_k_for`]); 0 disables it.
    pub local_k: Option<usize>,
    /// Blocks per row of the surface (blocks are in raster order), or 0 when unknown. The
    /// localized search also tries the blocks around the one directly above.
    pub row: usize,
}

impl Default for RdoParams {
    fn default() -> Self {
        Self {
            lambda: 0.0,
            window: None,
            // Wide search for whole-block and whole-field copies, partial spans only nearby:
            // far spans rarely win (BC1/BC5 within ~1% of trying them everywhere).
            max_back: None,
            span_back: None,
            refine_back: None,
            far_back: None,
            // 4x falling as k²/(k²+std²) with k = 12 (was 5x ramping linearly to 1 at std 14): the
            // linear ramp left busy-but-gradual blocks unprotected and over-weighted the flattest.
            // Kodak-24 BC7 vs Oodle Texture at λ 10-40 went from +1.8..+3.1% to -0.6..-1.5% size at
            // equal RMSE while staying ahead at equal SSIM; BC1/BC4 gained at equal RMSE too.
            smooth_std: 12.0,
            smooth_scale: 4.0,
            max_rms_ratio: 10.0,
            // 3 rather than 1.5: −5% BD-rate for BC4/BC5 on Kodak (smooth single-channel blocks
            // were stuck at their base encoding), BC1/BC7 within ±0.1%.
            min_rms: 3.0,
            chunk_bytes: None,
            chain_depth: 16,
            verify_zstd_level: Some(19),
            reduce_entropy: false,
            local_k: None,
            row: 0,
        }
    }
}

impl RdoParams {
    pub fn with_lambda(lambda: f32) -> Self {
        Self {
            lambda,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.lambda > 0.0
    }

    /// Largest block SSE a candidate may have, given the base encoding's.
    fn error_limit(&self, base_err: u64, channels: f64) -> u64 {
        let ratio2 = (self.max_rms_ratio as f64).powi(2);
        let floor = 16.0 * channels * (self.min_rms as f64).powi(2);
        (base_err as f64 * ratio2 + floor) as u64
    }

    /// The rate model's window: by default 16 KiB for BC7, 8 KiB for BC1, 3 KiB for BC5, 2 KiB
    /// for BC4. BC7 gains most from distant history (Oodle-matched sizes 2-5% smaller than with
    /// 2 KiB windows and 16 KiB chunks) and has the time to spare; BC7 and BC1 need it for the
    /// far searches ([`RdoParams::far_back_for`]). A wider window alone changes little; it is the
    /// search depth it allows (`max_back`, capped at window / block size) that counts. BC5 at
    /// 3 KiB could search 192 blocks deep (Kodak-24 BD-rate −2.8% vs 2 KiB for 1.36× the time);
    /// it searches 128 deep plus the localized search's picks instead.
    pub fn window_for(&self, format: Format) -> usize {
        self.window.unwrap_or(match format {
            Format::Bc7 => 16 * 1024,
            Format::Bc1 => 8 * 1024,
            Format::Bc5 => 3072,
            Format::Bc4 => 2048,
        })
    }

    /// Exhaustive search depth: by default 128 for BC5 and BC4 with the localized search (the
    /// ranker's picks beyond it are worth more than blocks 129-192: BC5 Kodak-8 BD-rate −0.9% vs
    /// 192 deep without it, held-out Poly Haven 1k textures −0.4%, same time; BC4 held-out Poly
    /// Haven −0.71% vs 256 deep without it, same time), 32 for BC7 with it (as good as 64 on
    /// held-out Poly Haven textures), 256 otherwise.
    pub fn max_back_for(&self, format: Format) -> usize {
        // BC7 16 rather than 32 (the localized search finds the rest): Kodak-24 BD-rate -0.00%.
        self.max_back.unwrap_or(match self.local_k_for(format) {
            0 => 256,
            _ if format == Format::Bc7 => 16,
            _ => 128,
        })
    }

    /// Partial-span distance: 8 for BC1/BC4, off for BC5 and BC7, where spans almost never win
    /// (Kodak-24 BD-rate +0.00% and +0.02% without them, for 11% and 7% less time).
    pub fn span_back_for(&self, format: Format) -> usize {
        self.span_back.unwrap_or(match format {
            Format::Bc5 | Format::Bc7 => 0,
            _ => 8,
        })
    }

    /// Localized-search picks: by default 16 for BC5, 8 for BC7, 32 for BC4 (each with its own
    /// ranker), off for BC1.
    pub fn local_k_for(&self, format: Format) -> usize {
        // BC7 8 rather than 16: Kodak-24 BD-rate -0.01%, 8% less time.
        self.local_k.unwrap_or(match format {
            Format::Bc7 => 8,
            Format::Bc5 => 16,
            Format::Bc4 => 32,
            _ => 0,
        })
    }

    /// Far-endpoint search depth: by default the whole 8 KiB window for BC1 (1024 blocks; Kodak-24
    /// BD-rate −0.55% for 1.09× the time, −0.59% at 2048 blocks for 1.16×) and the whole 16 KiB
    /// window for BC7 (1024 blocks: past `max_back`, whole-block copies and endpoint re-indexes
    /// only; Kodak-8 BD-rate −1.46% for 1.25× the time, −0.85% at 512 blocks for 1.07×, −1.67% at
    /// 2048), off for BC4 (−0.38% for 1.4× the time) and BC5 (no time to spare).
    pub fn far_back_for(&self, format: Format) -> usize {
        self.far_back.unwrap_or(match format {
            Format::Bc1 | Format::Bc7 => 1024,
            _ => 0,
        })
    }

    /// Field-swap refinement depth: by default 256 for BC1/BC4/BC7 (Kodak-24 BD-rate −1.0% /
    /// −0.4% / −0.1%; for BC7 it also trims 0.3-0.9 points off the equal-RMSE gap to Oodle at
    /// λ 10-40, for 1.07× the time), off for BC5 (−0.2% at depth 32, −0.3% at 256, but BC5 RDO
    /// has no time to spare against bc7enc_rdo).
    pub fn refine_back_for(&self, format: Format) -> usize {
        self.refine_back.unwrap_or(match format {
            Format::Bc1 | Format::Bc4 | Format::Bc7 => 256,
            Format::Bc5 => 0,
        })
    }

    /// Bytes per independently optimized chunk: by default 64 KiB for BC7 (6 chunks per Kodak
    /// image; 128 KiB is −0.8% to −1.3% BD-rate for the same CPU time, but fewer, longer chunks
    /// leave threads idle: 1.4-1.5× the wall time at 20 threads), 16 KiB for the others (1024 BC5
    /// or 2048 BC1/BC4 blocks, 24 or 12 chunks per Kodak image, about +0.03% BD-rate vs 64 KiB
    /// chunks at a 2 KiB window).
    pub fn chunk_bytes_for(&self, format: Format) -> usize {
        self.chunk_bytes
            .unwrap_or(if format == Format::Bc7 { 64 * 1024 } else { 16 * 1024 })
    }
}

/// What the RDO pass did.
#[derive(Clone, Copy, Debug, Default)]
pub struct RdoStats {
    pub blocks: usize,
    pub changed: usize,
    /// Chunks reverted to their base encoding because real Zstd did not shrink them.
    pub reverted_chunks: usize,
    /// Rate-model estimate for the base encoding, in bits.
    pub est_bits_before: f64,
    /// Rate-model estimate after RDO, in bits.
    pub est_bits_after: f64,
}

impl RdoStats {
    /// Sums two sets of statistics.
    pub fn merge(a: RdoStats, b: RdoStats) -> RdoStats {
        RdoStats {
            blocks: a.blocks + b.blocks,
            changed: a.changed + b.changed,
            reverted_chunks: a.reverted_chunks + b.reverted_chunks,
            est_bits_before: a.est_bits_before + b.est_bits_before,
            est_bits_after: a.est_bits_after + b.est_bits_after,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// LZ rate model
// ---------------------------------------------------------------------------------------------

const HASH_BITS: u32 = 13;

#[inline]
fn hash3(a: u8, b: u8, c: u8) -> usize {
    let v = a as u32 | (b as u32) << 8 | (c as u32) << 16;
    (v.wrapping_mul(2_654_435_761) >> (32 - HASH_BITS)) as usize
}

/// Parser state carried from one block to the next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LzState {
    /// Recent match offsets, most recent first (Zstd keeps three "repeat offsets" that are
    /// coded far more cheaply than a new offset).
    pub reps: [usize; 3],
    /// The previous block ended inside a match, so a repeat match at the start of the next
    /// block simply extends it (one long match in the real compressor).
    pub tail_match: bool,
}

/// A crude model of an LZ77 + entropy coder: greedy parse, order-0 literal costs,
/// offset cost ~ log2(distance), cheaper repeat-offset matches, and near-free extension of a
/// match that runs across block boundaries.
pub struct LzModel {
    hist: Vec<u8>,
    head: Vec<i32>,
    prev: Vec<i32>,
    counts: [u32; 256],
    max_count: u32,
    total: u32,
    /// `lit_cost` per byte value for the current statistics (refreshed on push).
    lit: Box<[f32; 256]>,
    /// `log2(offset)` for every offset up to the window.
    off_log2: Vec<f32>,
    /// Match length cost, `1 + 2·log2(len − 1)`, for lengths up to a 16-byte block.
    len: [f32; 17],
    state: LzState,
    window: usize,
    depth: usize,
    /// Literal statistics were fixed by [`set_literals`](Self::set_literals): `push` leaves them.
    frozen: bool,
    /// One extra match offset checked at every position, beyond the window if need be (0: none).
    pub hint: usize,
}

impl LzModel {
    pub fn new(window: usize, depth: usize) -> Self {
        Self {
            hist: Vec::new(),
            head: vec![-1; 1 << HASH_BITS],
            prev: Vec::new(),
            counts: [0; 256],
            max_count: 0,
            total: 0,
            lit: Box::new([(256f32).log2(); 256]),
            off_log2: (0..=window.max(16)).map(|o| (o as f32).log2()).collect(),
            len: std::array::from_fn(|l| 1.0 + 2.0 * (l.max(1) as f32 - 1.0).log2()),
            state: LzState::default(),
            window,
            depth,
            frozen: false,
            hint: 0,
        }
    }

    /// Fixes the literal statistics to those of `sample` (order-0, add-one smoothed) instead of
    /// learning them from the pushed bytes.
    pub fn set_literals(&mut self, sample: &[u8]) {
        self.counts = [0; 256];
        for &b in sample {
            self.counts[b as usize] += 1;
        }
        self.total = sample.len() as u32;
        self.max_count = self.counts.iter().copied().max().unwrap_or(0);
        self.refresh_literals();
        self.frozen = true;
    }

    fn refresh_literals(&mut self) {
        for b in 0..256 {
            let p = (self.counts[b] as f32 + 1.0) / (self.total as f32 + 256.0);
            self.lit[b] = (-p.log2()).min(10.0);
        }
    }

    /// Appends emitted bytes to the history and updates the match finder and statistics.
    pub fn push(&mut self, bytes: &[u8], state: LzState) {
        for &b in bytes {
            self.hist.push(b);
            self.prev.push(-1);
            if !self.frozen {
                self.counts[b as usize] += 1;
                self.max_count = self.max_count.max(self.counts[b as usize]);
                self.total += 1;
            }
            let n = self.hist.len();
            if n >= 3 {
                let q = n - 3;
                let h = hash3(self.hist[q], self.hist[q + 1], self.hist[q + 2]);
                self.prev[q] = self.head[h];
                self.head[h] = q as i32;
            }
        }
        self.state = state;
        if !self.frozen {
            self.refresh_literals();
        }
    }

    #[inline]
    fn lit_cost(&self, b: u8) -> f32 {
        self.lit[b as usize]
    }

    #[inline]
    fn len_cost(&self, len: usize) -> f32 {
        self.len[len]
    }

    /// A lower bound on [`cost`](Self::cost) for any `len`-byte candidate that cannot continue
    /// the previous block's match: the cheapest cover of `len` bytes by the cheapest literal and
    /// repeat-offset matches (every other token costs more).
    pub fn min_cost(&self, len: usize) -> f32 {
        let p = (self.max_count as f32 + 1.0) / (self.total as f32 + 256.0);
        let lit = (-p.log2()).min(10.0);
        let mut lb = [0f32; 17];
        for pos in (0..len).rev() {
            let mut b = lit + lb[pos + 1];
            for l in 3..=len - pos {
                b = b.min(3.0 + self.len_cost(l) + lb[pos + l]);
            }
            lb[pos] = b;
        }
        // Slack for float summation order.
        lb[0] * 0.999 - 0.01
    }

    /// Whether `cand` could start by continuing the previous block's match (a 1-bit token),
    /// which [`min_cost`](Self::min_cost) does not bound.
    #[inline]
    pub fn may_continue(&self, cand: &[u8]) -> bool {
        let r = self.state.reps[0];
        self.state.tail_match && r > 0 && r <= self.hist.len() && {
            let src = self.hist.len() - r;
            self.hist[src] == cand[0]
        }
    }

    /// Estimated bits to code `cand` after the current history, and the parser state after it.
    pub fn cost(&self, cand: &[u8]) -> (f32, LzState) {
        self.cost_until(cand, f32::INFINITY).unwrap()
    }

    /// [`cost`](Self::cost), or `None` as soon as the bits so far exceed `stop` (bits only
    /// grow along the parse, so the full cost would too).
    pub fn cost_until(&self, cand: &[u8], stop: f32) -> Option<(f32, LzState)> {
        let hist = &self.hist[..];
        let hl = hist.len();
        let at = |i: usize| if i < hl { hist[i] } else { cand[i - hl] };
        // Length of the match between absolute position `src` (history, or earlier in the
        // candidate) and candidate position `pos`, up to `max`.
        let match_len = |src: usize, pos: usize, max: usize| {
            let mut k = 0;
            if src < hl {
                let lim = max.min(hl - src);
                let (h, c) = (&hist[src..src + lim], &cand[pos..pos + lim]);
                while k < lim && h[k] == c[k] {
                    k += 1;
                }
                if k < lim {
                    return k;
                }
            }
            // The source has run into the candidate itself (src + k >= hl from here on).
            while k < max && cand[src + k - hl] == cand[pos + k] {
                k += 1;
            }
            k
        };
        let rep_of = |reps: &[usize; 3], off: usize| reps.iter().position(|&r| r == off);
        let mut bits = 0.0;
        let mut reps = self.state.reps;
        let mut tail = false;
        let mut pos = 0;
        while pos < cand.len() {
            let abs = hl + pos;
            let max = cand.len() - pos;
            // (len, offset, repeat-offset index)
            let mut best: (usize, usize, Option<usize>) = (0, 0, None);
            for (ri, &r) in reps.iter().enumerate() {
                // Cheap rejections first: this runs at every byte position.
                if r == 0 || r > abs || at(abs - r) != cand[pos] || reps[..ri].contains(&r) {
                    continue;
                }
                let l = match_len(abs - r, pos, max);
                let continuing = ri == 0 && pos == 0 && self.state.tail_match;
                if (l >= 3 || (continuing && l > 0)) && l > best.0 {
                    best = (l, r, Some(ri));
                }
            }
            if pos + 3 <= cand.len() {
                let h = hash3(cand[pos], cand[pos + 1], cand[pos + 2]);
                let mut q = self.head[h];
                let mut d = 0;
                while q >= 0 && d < self.depth {
                    let qs = q as usize;
                    let off = abs - qs;
                    if off > self.window {
                        break;
                    }
                    // Only a match of `need` bytes can replace the best one: check its last
                    // byte first (an exact, cheap rejection).
                    let need = best.0 + usize::from(best.2.is_some()) + 1;
                    if need <= max && at(qs + need - 1) == cand[pos + need - 1] {
                        let l = match_len(qs, pos, max);
                        if l >= need {
                            best = (l, off, rep_of(&reps, off));
                        }
                    }
                    q = self.prev[qs];
                    d += 1;
                    if best.0 == max {
                        break; // nothing can beat a match to the end
                    }
                }
                let hint = self.hint;
                if hint > self.window && hint <= abs && !reps.contains(&hint) {
                    let need = best.0 + usize::from(best.2.is_some()) + 1;
                    if need <= max && at(abs - hint + need - 1) == cand[pos + need - 1] {
                        let l = match_len(abs - hint, pos, max);
                        if l >= need {
                            best = (l, hint, None);
                        }
                    }
                }
                // Matches inside the candidate itself (e.g. BC5's G half repeating its R half).
                // The candidate is not in the hash chains, so scan its earlier bytes directly.
                for src in 0..pos {
                    let need = best.0 + usize::from(best.2.is_some()) + 1;
                    if need > max {
                        break;
                    }
                    if cand[src + need - 1] != cand[pos + need - 1] {
                        continue;
                    }
                    let l = match_len(hl + src, pos, max);
                    if l >= need {
                        let off = pos - src;
                        best = (l, off, rep_of(&reps, off));
                    }
                }
            }
            let continuing = pos == 0 && best.2 == Some(0) && self.state.tail_match;
            if best.0 >= 3 || (continuing && best.0 > 0) {
                bits += if continuing {
                    1.0 // continues the previous block's match
                } else {
                    match best.2 {
                        Some(0) => 3.0 + self.len_cost(best.0),
                        Some(_) => 4.0 + self.len_cost(best.0),
                        None => {
                            6.0 + self
                                .off_log2
                                .get(best.1)
                                .copied()
                                .unwrap_or_else(|| (best.1 as f32).log2())
                                + self.len_cost(best.0)
                        }
                    }
                };
                reps = match best.2 {
                    Some(0) => reps,
                    Some(1) => [reps[1], reps[0], reps[2]],
                    Some(_) => [reps[2], reps[0], reps[1]],
                    None => [best.1, reps[0], reps[1]],
                };
                pos += best.0;
                tail = pos == cand.len();
            } else {
                bits += self.lit_cost(cand[pos]);
                pos += 1;
                tail = false;
            }
            if bits > stop {
                return None;
            }
        }
        Some((bits, LzState { reps, tail_match: tail }))
    }
}

// ---------------------------------------------------------------------------------------------
// Candidate generation
// ---------------------------------------------------------------------------------------------

/// The bits above which a candidate with distortion `dist` cannot beat `best` at `lambda` (with
/// slack for the f32/f64 rounding, so it never cuts a winner short).
#[inline]
fn bit_budget(best: f64, dist: f64, lambda: f64) -> f32 {
    ((best - dist) / lambda * (1.0 + 1e-6) + 1e-3) as f32
}

/// Byte spans (start, length) of an 8-byte BC1/BC4 block (or BC5 half) to try copying from the
/// same position of a previous block: every span of 3-7 bytes (the whole block is tried anyway),
/// as bc7enc_rdo's ERT does. The rest of the block keeps its base bytes.
const BC4_SPANS: [(usize, usize); 20] = {
    let mut out = [(0usize, 0usize); 20];
    let mut n = 0;
    let mut len = 7;
    while len >= 3 {
        let mut k = 0;
        while k + len <= 8 {
            out[n] = (k, len);
            n += 1;
            k += 1;
        }
        len -= 1;
    }
    out
};

/// Byte spans (start, length) of a 16-byte BC7 block to try copying from a previous block.
const BC7_SPANS: &[(usize, usize)] = &[
    (0, 8),
    (8, 8),
    (0, 6),
    (0, 12),
    (4, 12),
    (2, 14),
    (6, 10),
    (10, 6),
    (12, 4),
    (4, 8),
];

/// BC7 base and previous block unpacked, and the previous block decoded: kept per final block of
/// a chunk so each is unpacked and decoded once rather than once per block that looks back at it.
#[derive(Clone, Copy)]
struct Bc7Pair<'a> {
    base: &'a bc7::Bc7Block,
    prev: &'a bc7::Bc7Block,
    prev_dec: &'a Block,
}

/// Final BC7 blocks of a chunk, unpacked and decoded (None for other formats).
fn unpack_final(format: Format, raw: &[u8; 16]) -> Option<(bc7::Bc7Block, Block)> {
    if format != Format::Bc7 {
        return None;
    }
    let u = bc7::unpack(raw)?;
    let mut d = [[0u8; 4]; 16];
    format.decode_block(raw, &mut d);
    Some((u, d))
}

/// Per-channel sums over a block's texels.
fn channel_sums(b: &Block) -> [i32; 4] {
    let mut t = [0i32; 4];
    for px in b {
        for c in 0..4 {
            t[c] += i32::from(px[c]);
        }
    }
    t
}

/// Marks a candidate whose error [`candidates`] did not compute.
const UNKNOWN: u64 = u64::MAX;

/// BC7: previous blocks (by distance, plus the one above) whose whole layout (mode, partition,
/// indices) is tried with this block's endpoints when it differs from the base's. Farther ones
/// rarely win (Kodak subset vs Oodle 2.9.16: 0.3% of the 0.4% fewer bits unlimited would give,
/// at a quarter more RDO time at λ 1).
const CROSS_BACK: usize = 8;

/// Candidate blocks for `base` built from `prev`, each with its weighted squared error when
/// generating it already gave that ([`UNKNOWN`] otherwise). Candidates whose error is known to
/// exceed `bound` may be left out.
#[expect(
    clippy::too_many_arguments,
    reason = "hot inner helper; a context struct would only rename the arguments"
)]
fn candidates(
    format: Format,
    base: &[u8; 16],
    prev: &[u8; 16],
    src: &Block,
    w: &[u32; 4],
    reduce_entropy: bool,
    spans: bool,
    cross: bool,
    fields: &mut FieldsSeen,
    bound: u64,
    unpacked: Option<Bc7Pair>,
    base_half: [u64; 2],
    out: &mut Vec<([u8; 16], u64)>,
) {
    let n = format.block_bytes();
    let mut full = [0u8; 16];
    full[..n].copy_from_slice(&prev[..n]);
    // BC4/BC5 with the usual weights: candidate errors come out of generating them (exact, or
    // some value above `bound`), as sums of half errors.
    let exact = match format {
        Format::Bc4 => w[1..] == [0, 0, 0],
        Format::Bc5 => w[2..] == [0, 0],
        _ => false,
    };
    let halves = if format == Format::Bc5 { 2 } else { 1 };
    let mut copy_err = [0u64; 2];
    // A half an earlier previous block already had only regenerates candidates already made
    // (copying it, its endpoints, its indices, its spans): those are skipped.
    let mut half_new = [true; 2];
    if exact {
        for h in 0..halves {
            let r = h * 8..h * 8 + 8;
            // The bound only shrinks over a block's search, so a stored error above an earlier
            // bound is still above the current one.
            let ((_, _, e), new) = fields.halves[h].get_or_insert(field_key(&prev[r.clone()]), || {
                ([0; 8], false, bc4::sse_bounded(&prev[r], src, h, w[h], bound))
            });
            copy_err[h] = e;
            half_new[h] = new;
        }
    }
    let full_err = if exact {
        copy_err[0] + copy_err[1]
    } else {
        unpacked.map_or(UNKNOWN, |u| block_sse(src, u.prev_dec, w))
    };
    out.push((full, full_err));
    match format {
        Format::Bc1 => {
            if fields.endpoints.insert(field_key(&prev[..4])) {
                let mut e = *base;
                e[..4].copy_from_slice(&prev[..4]);
                let mut c = [0u8; 16];
                let (r, err) = bc1::reindex_err(&e[..8], src, w);
                c[..8].copy_from_slice(&r);
                // block_err scores BC1 by RGB alone when alpha is unweighted, as reindex does.
                out.push((c, if w[3] == 0 { err } else { UNKNOWN }));
            }
            if fields.indices.insert(field_key(&prev[4..8])) {
                let mut ix = *base;
                ix[4..8].copy_from_slice(&prev[4..8]);
                if let Some(r) = bc1::refit(&ix[..8], src) {
                    let mut c = [0u8; 16];
                    c[..8].copy_from_slice(&r);
                    out.push((c, UNKNOWN));
                }
            }
            // Partial spans, copied as is.
            for &(k, len) in if spans { &BC4_SPANS[..] } else { &[] } {
                if base[k..k + len] != prev[k..k + len] {
                    let mut c = *base;
                    c[k..k + len].copy_from_slice(&prev[k..k + len]);
                    out.push((c, UNKNOWN));
                }
            }
        }
        Format::Bc4 | Format::Bc5 => {
            let first = out.len();
            let v = [bc4::channel(src, 0), bc4::channel(src, 1)];
            let known = |e: u64| if exact { e } else { UNKNOWN };
            // Per half: prev's endpoints re-indexed, and prev's indices with refit endpoints.
            // Both depend only on that field of prev (base and the source are fixed for the
            // block), so they are memoized by field value with their errors; a field an earlier
            // previous block already had would give the same candidate again, so that one is not
            // re-pushed.
            let mut reindexed = [[0u8; 8]; 2];
            let mut refit = [[0u8; 8]; 2];
            let mut re_err = [0u64; 2];
            let mut re_done = [true; 2];
            let mut rf_err = [0u64; 2];
            let lazy = exact && (format == Format::Bc4 || v[0] != v[1]);
            // Endpoints from prev with indices re-chosen for the source, and its error.
            let reindex = |h: usize| {
                let mut e = [0u8; 8];
                e.copy_from_slice(&base[h * 8..h * 8 + 8]);
                e[..2].copy_from_slice(&prev[h * 8..h * 8 + 2]);
                let (val, err) = bc4::reindex_channel_err(&e, &v[h]);
                (val, err * u64::from(w[h]))
            };
            for h in 0..halves {
                let r = h * 8..h * 8 + 8;
                let other = base_half[1 - h];
                // Copy the whole half.
                if half_new[h] {
                    let mut c = *base;
                    c[r.clone()].copy_from_slice(&prev[r.clone()]);
                    out.push((c, known(copy_err[h] + other)));
                }
                // Endpoints from prev, indices re-chosen.
                let ek = (h as u64) << 56 | u64::from(prev[h * 8]) << 8 | u64::from(prev[h * 8 + 1]);
                let ((val, done, err), new) = fields.endpoints.get_or_insert(ek, || {
                    // 8-value mode puts every palette value between the endpoints: when the
                    // distance to that interval alone already loses, the re-index is put off
                    // (flag false, the bound as its error) until a combined candidate needs it.
                    // Not with mirroring, which uses every candidate's bytes.
                    let (e0, e1) = (prev[h * 8], prev[h * 8 + 1]);
                    if lazy && e0 > e1 {
                        let lb = u64::from(w[h]) * interval_sse(&v[h], e1, e0);
                        if lb + other > bound {
                            return ([0; 8], false, lb);
                        }
                    }
                    let (val, err) = reindex(h);
                    (val, true, err)
                });
                reindexed[h] = val;
                re_err[h] = err;
                re_done[h] = done;
                if new && done {
                    let mut c = *base;
                    c[r.clone()].copy_from_slice(&val);
                    out.push((c, known(err + other)));
                }
                // Indices from prev, endpoints refit (when the fit fails, the half keeps
                // base's endpoints with prev's indices for the combined candidate below).
                let mut ik = 0u64;
                for &b in &prev[h * 8 + 2..h * 8 + 8] {
                    ik = ik << 8 | u64::from(b);
                }
                ik |= (h as u64) << 56;
                let ((val, ok, err), new) = fields.indices.get_or_insert(ik, || {
                    let mut ix = [0u8; 8];
                    ix.copy_from_slice(&base[r.clone()]);
                    ix[2..].copy_from_slice(&prev[h * 8 + 2..h * 8 + 8]);
                    let (val, ok) = match bc4::refit_channel(&ix, &v[h]) {
                        Some(rf) => (rf, true),
                        None => (ix, false),
                    };
                    let err = if exact {
                        bc4::sse_bounded(&val, src, h, w[h], u64::MAX)
                    } else {
                        0
                    };
                    (val, ok, err)
                });
                refit[h] = val;
                rf_err[h] = err;
                if new && ok {
                    let mut c = *base;
                    c[r.clone()].copy_from_slice(&val);
                    out.push((c, known(err + other)));
                }
                // Partial spans, copied as is.
                for &(k, len) in if spans && half_new[h] { &BC4_SPANS[..] } else { &[] } {
                    let (a, b) = (h * 8 + k, h * 8 + k + len);
                    if base[a..b] != prev[a..b] {
                        let mut c = *base;
                        c[a..b].copy_from_slice(&prev[a..b]);
                        out.push((c, UNKNOWN));
                    }
                }
            }
            if format == Format::Bc5 {
                // Edit both halves the same way, so blocks whose halves already agree stay
                // self-similar (Zstd matches the G half against the R half).
                for h in 0..2 {
                    if !re_done[h] && re_err[0] + re_err[1] <= bound {
                        (reindexed[h], re_err[h]) = reindex(h);
                        re_done[h] = true;
                    }
                }
                let mut e = *base;
                let mut ix = *base;
                for h in 0..2 {
                    let r = h * 8..h * 8 + 8;
                    e[r.clone()].copy_from_slice(&reindexed[h]);
                    ix[r].copy_from_slice(&refit[h]);
                }
                if re_done == [true, true] {
                    out.push((e, known(re_err[0] + re_err[1])));
                }
                out.push((ix, known(rf_err[0] + rf_err[1])));
                // Identical source channels: mirror the R half into the G half.
                if v[0] == v[1] {
                    let end = out.len();
                    for k in first..end {
                        let mut c = out[k].0;
                        let (lo, hi) = c.split_at_mut(8);
                        hi.copy_from_slice(lo);
                        out.push((c, UNKNOWN));
                    }
                }
            }
            if exact {
                // Known to lose: leave them out (after mirroring, which may still help them).
                out.retain(|&(_, e)| e == UNKNOWN || e <= bound);
            }
        }
        Format::Bc7 => {
            // Far back-references get only the endpoints re-indexed besides the whole-block copy
            // and the index copy below: the raw endpoint copy and the spans rarely win there
            // (dropping the raw copy: same polyhaven-test BD-rate, about 0.8x the time).
            // Spans within the first 8 bytes (copy and re-index) depend only on those bytes of
            // prev, and the index copy only on prev's indices: when an earlier previous block
            // had the same bytes, the candidates were already tried.
            for &(k, len) in if spans { BC7_SPANS } else { &BC7_SPANS[..1] } {
                let memo = match (k, len) {
                    (0, 8) => Some(&mut fields.endpoints),
                    (0, 6) => Some(&mut fields.spans),
                    _ => None,
                };
                if let Some(m) = memo
                    && !m.insert(field_key(&prev[k..k + len]))
                {
                    continue;
                }
                let mut c = *base;
                c[k..k + len].copy_from_slice(&prev[k..k + len]);
                if spans {
                    out.push((c, UNKNOWN));
                }
                if k + len <= 9 {
                    // Re-indexing yields the error too; past `bound` it would be rejected.
                    if let Some(r) = bc7::reindex_bounded(&c, src, w, bound, &mut fields.seg) {
                        out.push(r);
                    }
                }
            }
            if let Some(u) = unpacked {
                match bc7::index_key(u.base, u.prev) {
                    Some(key) => {
                        if fields.indices.insert(key) {
                            let r = bc7::with_indices(u.base, &u.prev.idx, src, w, reduce_entropy);
                            out.push((r, UNKNOWN));
                        }
                    }
                    // A different mode or partition: take prev's whole layout instead.
                    None if cross && bc7::layout_key(u.prev).is_some_and(|key| fields.indices.insert(key)) => {
                        let r = bc7::with_layout(u.prev, src, w, reduce_entropy);
                        out.push((r, UNKNOWN));
                    }
                    None => {}
                }
            }
        }
    }
}

/// Weighted SSE of `raw` against `src`. BC1/BC4/BC5 use fused kernels that may stop once the sum
/// passes `bound` (the result is then some value above `bound`); otherwise the result is exact.
#[inline]
fn block_err(
    format: Format,
    raw: &[u8],
    src: &Block,
    w: &[u32; 4],
    bound: u64,
    base: (&[u8; 16], [u64; 2]),
    dec: &mut Block,
) -> u64 {
    match format {
        Format::Bc1 if w[3] == 0 => bc1::sse_bounded(raw, src, [w[0], w[1], w[2]], bound),
        Format::Bc4 if w[1..] == [0, 0, 0] => bc4::sse_bounded(&raw[..8], src, 0, w[0], bound),
        Format::Bc5 if w[2..] == [0, 0] => {
            // Most candidates keep one half of the base block: reuse its error.
            let half = |h: usize, bound: u64| {
                let r = h * 8..h * 8 + 8;
                if raw[r.clone()] == base.0[r.clone()] {
                    base.1[h]
                } else {
                    bc4::sse_bounded(&raw[r], src, h, w[h], bound)
                }
            };
            let e = half(0, bound);
            if e > bound { e } else { e + half(1, bound - e) }
        }
        _ => {
            format.decode_block(raw, dec);
            block_sse(src, dec, w)
        }
    }
}

/// Per-half errors of a BC5 base block, which [`block_err`] reuses for candidates that keep a
/// half of it (zero for other formats, where they are unused).
fn base_half_err(format: Format, base: &[u8; 16], src: &Block, w: &[u32; 4]) -> [u64; 2] {
    if format == Format::Bc5 && w[2..] == [0, 0] {
        [0, 1].map(|h| bc4::sse_bounded(&base[h * 8..h * 8 + 8], src, h, w[h], u64::MAX))
    } else {
        [0; 2]
    }
}

/// Set of candidate blocks already evaluated for the current block: open addressing over a
/// fixed table, entries tagged with a generation so clearing is O(1). Far cheaper than a decode.
struct Seen {
    keys: Vec<[u8; 16]>,
    tags: Vec<u32>,
    generation: u32,
    bits: u32,
}

impl Seen {
    /// A table for up to about `n` keys (kept at most half full).
    fn new(n: usize) -> Self {
        let slots = (2 * n).next_power_of_two().max(64);
        Self {
            keys: vec![[0; 16]; slots],
            tags: vec![0; slots],
            generation: 1,
            bits: slots.trailing_zeros(),
        }
    }

    fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.tags.fill(0);
            self.generation = 1;
        }
    }

    /// Inserts `k`; false if it was already present. When the table is full, reports every key
    /// as new (correct, just no longer skipping).
    fn insert(&mut self, k: &[u8; 16]) -> bool {
        let lo = u64::from_le_bytes(k[..8].try_into().unwrap());
        let hi = u64::from_le_bytes(k[8..].try_into().unwrap());
        let h = (lo ^ hi.rotate_left(29)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mask = self.keys.len() - 1;
        let mut i = (h >> (64 - self.bits)) as usize;
        for _ in 0..self.keys.len() {
            if self.tags[i] != self.generation {
                self.tags[i] = self.generation;
                self.keys[i] = *k;
                return true;
            }
            if self.keys[i] == *k {
                return false;
            }
            i = (i + 1) & mask;
        }
        true
    }
}

/// Per-block memo of work on one field of a previous block (BC1 endpoints / indices, a BC4
/// half's endpoints / indices, BC7 leading spans / indices): open addressing on a `u64` key,
/// generation-tagged like [`Seen`].
struct Memo {
    keys: Vec<u64>,
    vals: Vec<([u8; 8], bool, u64)>,
    tags: Vec<u32>,
    generation: u32,
    bits: u32,
}

impl Memo {
    fn new(n: usize) -> Self {
        let slots = (2 * n).next_power_of_two().max(64);
        Self {
            keys: vec![0; slots],
            vals: vec![([0; 8], false, 0); slots],
            tags: vec![0; slots],
            generation: 1,
            bits: slots.trailing_zeros(),
        }
    }

    fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.tags.fill(0);
            self.generation = 1;
        }
    }

    /// The value stored for `k`, computing and storing it on first sight; the flag is true when
    /// it was computed now. A full table just computes every time (still correct).
    fn get_or_insert(&mut self, k: u64, f: impl FnOnce() -> ([u8; 8], bool, u64)) -> (([u8; 8], bool, u64), bool) {
        let mask = self.keys.len() - 1;
        let mut i = (k.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - self.bits)) as usize;
        for _ in 0..self.keys.len() {
            if self.tags[i] != self.generation {
                let v = f();
                self.tags[i] = self.generation;
                self.keys[i] = k;
                self.vals[i] = v;
                return (v, true);
            }
            if self.keys[i] == k {
                return (self.vals[i], false);
            }
            i = (i + 1) & mask;
        }
        (f(), true)
    }

    /// True the first time `k` is seen for the current block.
    fn insert(&mut self, k: u64) -> bool {
        self.get_or_insert(k, || ([0; 8], false, 0)).1
    }
}

/// Squared distance of 16 values to the interval `[lo, hi]`.
fn interval_sse(v: &[u8; 16], lo: u8, hi: u8) -> u64 {
    v.iter()
        .map(|&x| {
            let d = u64::from(lo.saturating_sub(x).max(x.saturating_sub(hi)));
            d * d
        })
        .sum()
}

/// Up to 8 bytes as a memo key.
fn field_key(field: &[u8]) -> u64 {
    field.iter().fold(0, |k, &b| k << 8 | u64::from(b))
}

/// Fields of previous blocks already used for the current block. A re-index candidate depends
/// only on a previous block's endpoints and a refit candidate only on its indices, so a field seen
/// before would only regenerate a candidate already evaluated.
struct FieldsSeen {
    endpoints: Memo,
    indices: Memo,
    /// BC7's second memoized span.
    spans: Memo,
    /// BC4/BC5 whole halves (per half), with their copy errors.
    halves: [Memo; 2],
    /// BC7 re-index bound state for the current source block.
    seg: Box<bc7::SegBound>,
}

impl FieldsSeen {
    fn new(n: usize) -> Self {
        // BC5 memoizes both halves.
        Self {
            endpoints: Memo::new(2 * n),
            indices: Memo::new(2 * n),
            spans: Memo::new(n),
            halves: [Memo::new(n), Memo::new(n)],
            seg: Box::new(bc7::SegBound::new(&[[0; 4]; 16], &[1; 4])),
        }
    }

    /// Starts a new source block.
    fn clear(&mut self, src: &Block, w: &[u32; 4], format: Format) {
        if format == Format::Bc7 {
            self.seg.reset(src, w);
        }
        self.endpoints.clear();
        self.indices.clear();
        self.spans.clear();
        self.halves[0].clear();
        self.halves[1].clear();
    }
}

/// Distortion multiplier for smooth blocks: banding on gradients is the most visible RDO artifact.
/// `1 + (smooth_scale - 1) * k² / (k² + std²)` with `k = smooth_std` and `std` the largest channel
/// standard deviation: full weight for flat blocks, half the extra at `k`, a long tail beyond it.
fn smooth_factor(src: &Block, w: &[u32; 4], p: &RdoParams) -> f32 {
    let mut max_std = 0f32;
    for c in 0..4 {
        if w[c] == 0 {
            continue;
        }
        let mean = src.iter().map(|t| t[c] as f32).sum::<f32>() / 16.0;
        let var = src.iter().map(|t| (t[c] as f32 - mean).powi(2)).sum::<f32>() / 16.0;
        max_std = max_std.max(var.sqrt());
    }
    let k2 = p.smooth_std * p.smooth_std;
    1.0 + (p.smooth_scale - 1.0) * k2 / (k2 + max_std * max_std)
}

/// Sum of absolute differences of the first `len` bytes (a multiple of 16).
#[inline]
fn sad(a: &[u8; 64], b: &[u8; 64], off: usize, len: usize) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of x86_64; loads are unaligned and within the arrays.
        unsafe {
            let mut acc = _mm_setzero_si128();
            for k in (off..off + len).step_by(16) {
                let x = _mm_loadu_si128(a.as_ptr().add(k).cast::<__m128i>());
                let y = _mm_loadu_si128(b.as_ptr().add(k).cast::<__m128i>());
                acc = _mm_add_epi64(acc, _mm_sad_epu8(x, y));
            }
            _mm_cvtsi128_si64(_mm_add_epi64(acc, _mm_unpackhi_epi64(acc, acc))) as u32
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a[off..off + len]
            .iter()
            .zip(&b[off..off + len])
            .map(|(x, y)| u32::from(x.abs_diff(*y)))
            .sum()
    }
}

/// Per weighted channel, the block's 16 values scaled to 0..=63 over the block's own range.
/// Weighted channels are packed first, so only the first 16·(weighted channels) bytes are used.
fn shape_sig(b: &Block, w: &[u32; 4]) -> [u8; 64] {
    let mut out = [0u8; 64];
    let mut slot = 0;
    for c in 0..4 {
        if w[c] == 0 {
            continue;
        }
        let o = slot * 16;
        slot += 1;
        let lo = b.iter().map(|t| t[c]).min().unwrap() as u32;
        let hi = b.iter().map(|t| t[c]).max().unwrap() as u32;
        if hi == lo {
            continue;
        }
        for t in 0..16 {
            out[o + t] = ((b[t][c] as u32 - lo) * 63 / (hi - lo)) as u8;
        }
    }
    out
}

fn pair<'a>(base: Option<&'a bc7::Bc7Block>, prev: Option<&'a (bc7::Bc7Block, Block)>) -> Option<Bc7Pair<'a>> {
    match (base, prev) {
        (Some(base), Some((prev, prev_dec))) => Some(Bc7Pair { base, prev, prev_dec }),
        _ => None,
    }
}

/// Endpoint pairs of a chunk's finished blocks, for the far-endpoint search. Fields are BC1's
/// endpoints or each BC4/BC5 half's. Per position and field: the decoded endpoints, scaled by the
/// square roots of the channel weights so plain distances are weighted errors, and whether it is
/// the latest occurrence of that pair.
struct FarEndpoints {
    format: Format,
    fields: usize,
    seg: Vec<[f32; 6]>,
    latest: Vec<bool>,
    last: std::collections::HashMap<(u8, [u8; 4]), usize>,
    sw: [f32; 3],
}

impl FarEndpoints {
    fn new(format: Format, cap: usize, w: &[u32; 4]) -> Self {
        let fields = if format == Format::Bc5 { 2 } else { 1 };
        Self {
            format,
            fields,
            seg: Vec::with_capacity(cap * fields),
            latest: Vec::with_capacity(cap * fields),
            last: std::collections::HashMap::with_capacity(cap * fields),
            sw: [(w[0] as f32).sqrt(), (w[1] as f32).sqrt(), (w[2] as f32).sqrt()],
        }
    }

    fn push(&mut self, i: usize, b: &[u8; 16]) {
        for h in 0..self.fields {
            let (k, valid, s) = if self.format == Format::Bc1 {
                let c0 = u16::from_le_bytes([b[0], b[1]]);
                let c1 = u16::from_le_bytes([b[2], b[3]]);
                let (a, z) = (bc1::unpack565(c0), bc1::unpack565(c1));
                let mut s = [0f32; 6];
                for c in 0..3 {
                    s[c] = a[c] as f32 * self.sw[c];
                    s[3 + c] = z[c] as f32 * self.sw[c];
                }
                // 3-color blocks also have black in their palette: no simple bound, never tried.
                ([b[0], b[1], b[2], b[3]], c0 > c1, s)
            } else {
                let (e0, e1) = (b[h * 8], b[h * 8 + 1]);
                let mut s = [0f32; 6];
                s[h] = e0 as f32 * self.sw[h];
                s[3 + h] = e1 as f32 * self.sw[h];
                // 6-value blocks also have 0 and 255: likewise never tried.
                ([e0, e1, 0, 0], e0 > e1, s)
            };
            if let Some(o) = self.last.insert((h as u8, k), i) {
                self.latest[o * self.fields + h] = false;
            }
            self.latest.push(valid);
            self.seg.push(s);
        }
    }

    /// The source block's scaled mean, and the slack for palette rounding: BC1's interpolated
    /// colors are truncated, up to 1 per channel off the segment; BC4's stay between the endpoints.
    fn mean(&self, s: &Block) -> ([f32; 3], f32) {
        let mut m = [0f32; 3];
        for px in s {
            for c in 0..3 {
                m[c] += px[c] as f32;
            }
        }
        for c in 0..3 {
            m[c] *= self.sw[c] / 16.0;
        }
        let slack = if self.format == Format::Bc1 {
            (self.sw[0] * self.sw[0] + self.sw[1] * self.sw[1] + self.sw[2] * self.sw[2]).sqrt()
        } else {
            0.0
        };
        (m, slack)
    }

    /// Lower bound on the error per texel of field `h` with position `j`'s endpoints and any
    /// indices: the mean of the chosen palette entries lies on the endpoints' segment, and the
    /// error is at least 16 times the squared distance between the two means. None when not tried.
    fn lower_bound(&self, j: usize, h: usize, m: &[f32; 3], slack: f32, px: &[[f32; 16]; 2]) -> Option<f64> {
        let at = j * self.fields + h;
        if !self.latest[at] {
            return None;
        }
        let s = &self.seg[at];
        if self.format != Format::Bc1 {
            // One channel: every texel is at least its distance from the endpoints' interval
            // away from any palette entry (a bound at least as tight as the mean's).
            let (lo, hi) = (s[3 + h].min(s[h]), s[3 + h].max(s[h]));
            let mut e = 0f32;
            for &x in &px[h] {
                let d = (lo - x).max(x - hi).max(0.0);
                e += d * d;
            }
            return Some(e as f64 / 16.0);
        }
        let ab = [s[3] - s[0], s[4] - s[1], s[5] - s[2]];
        let am = [m[0] - s[0], m[1] - s[1], m[2] - s[2]];
        let ll = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        let t = if ll > 0.0 {
            ((am[0] * ab[0] + am[1] * ab[1] + am[2] * ab[2]) / ll).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let d = [am[0] - t * ab[0], am[1] - t * ab[1], am[2] - t * ab[2]];
        let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let e = (dist - slack).max(0.0) as f64;
        Some(e * e)
    }

    /// `base` with field `h` taking position `j`'s endpoints, indices re-chosen.
    fn candidate(&self, base: &[u8; 16], prev: &[u8; 16], h: usize, s: &Block, w: &[u32; 4]) -> [u8; 16] {
        let mut c = *base;
        if self.format == Format::Bc1 {
            let mut e = *base;
            e[..4].copy_from_slice(&prev[..4]);
            c[..8].copy_from_slice(&bc1::reindex(&e[..8], s, w));
        } else {
            let r = h * 8..h * 8 + 8;
            let mut e = [0u8; 8];
            e.copy_from_slice(&base[r.clone()]);
            e[..2].copy_from_slice(&prev[h * 8..h * 8 + 2]);
            c[r].copy_from_slice(&bc4::reindex_channel(&e, &bc4::channel(s, h)));
        }
        c
    }
}

/// What the BC7 far search reads of the chunk's finished blocks.
struct Far7<'a> {
    fin: &'a [Option<(bc7::Bc7Block, Block)>],
    fin_sums: &'a [[i32; 4]],
    far_back: usize,
    w: &'a [u32; 4],
    lambda: f64,
}

/// BC7 far back-references, `back + 1..=far_back` blocks back: the whole-block copy and prev's
/// leading bytes (mode, partition, endpoints) re-indexed, the candidates that win there.
#[expect(
    clippy::too_many_arguments,
    reason = "the search state it updates is spread over the caller's locals"
)]
fn bc7_far(
    f7: &Far7,
    blocks: &[[u8; 16]],
    i: usize,
    back: usize,
    base: &[u8; 16],
    s: &Block,
    limit: u64,
    smooth: f64,
    min_bits: f64,
    fields: &mut FieldsSeen,
    seen: &mut Seen,
    model: &mut LzModel,
    best: &mut (f64, [u8; 16], f32, LzState),
) {
    let (fin, fin_sums, far_back, w, lambda) = (f7.fin, f7.fin_sums, f7.far_back, f7.w, f7.lambda);
    let (n, base) = (16, *base);
    let src_sums = channel_sums(s);
    for d in back + 1..=far_back.min(i) {
        let j = i - d;
        let prev = blocks[j];
        let mut pair = [([0u8; 16], UNKNOWN); 2];
        let mut np = 0;
        let bnd = limit.min((best.0 * 16.0 / smooth) as u64 + 2);
        if let Some((_, prev_dec)) = &fin[j] {
            // The difference of the means bounds the error from below (Cauchy-Schwarz):
            // most copies from far back lose on it alone.
            let ps = &fin_sums[j];
            let lb: u64 = (0..4)
                .map(|c| u64::from(w[c]) * ((src_sums[c] - ps[c]).pow(2) as u64))
                .sum::<u64>()
                / 16;
            if lb <= bnd {
                pair[np] = (prev, block_sse(s, prev_dec, w));
                np += 1;
            }
        }
        if fields.endpoints.insert(field_key(&prev[..8])) {
            let mut c = base;
            c[..8].copy_from_slice(&prev[..8]);
            if let Some(r) = bc7::reindex_bounded(&c, s, w, bnd, &mut fields.seg) {
                pair[np] = r;
                np += 1;
            }
        }
        for &(c, err) in &pair[..np] {
            if err > limit || c[..n] == base[..n] || !seen.insert(&c) {
                continue;
            }
            let dist = smooth * err as f64 / 16.0;
            if dist >= best.0 || (dist + min_bits >= best.0 && !model.may_continue(&c[..n])) {
                continue;
            }
            let Some((bits, rep)) = model.cost_until(&c[..n], bit_budget(best.0, dist, lambda)) else {
                continue;
            };
            let jj = dist + lambda * bits as f64;
            if jj < best.0 {
                *best = (jj, c, bits, rep);
            }
        }
    }
}

fn optimize_chunk(
    format: Format,
    src: &[Block],
    blocks: &mut [[u8; 16]],
    alts: &[&[[u8; 16]]],
    w: &[u32; 4],
    p: &RdoParams,
) -> RdoStats {
    let n = format.block_bytes();
    let window = p.window_for(format);
    let refine_back = p.refine_back_for(format);
    let span_back = p.span_back_for(format);
    let channels = w.iter().filter(|&&x| x > 0).count().max(1) as f64;
    let mut model = LzModel::new(window, p.chain_depth);
    // Literal costs from the chunk's base encoding rather than learned as the chunk goes: the
    // start of a chunk then has realistic costs, and RDO's own copies do not skew them. Kodak-24
    // BD-rate: BC4 −1.3%, BC5 −0.4%, BC7 −0.1%. BC1 keeps learned costs: frozen ones are even in
    // `mipple check` but −19.8% → −19.2% vs bc7enc_rdo (worse at high λ).
    if format != Format::Bc1 {
        let sample: Vec<u8> = blocks.iter().flat_map(|b| b[..n].iter().copied()).collect();
        model.set_literals(&sample);
    }
    let mut stats = RdoStats {
        blocks: blocks.len(),
        ..Default::default()
    };
    let mut cands = Vec::with_capacity(64);
    // Up to ~50 candidates per back-reference (BC5).
    let far_back = p.far_back_for(format).min(window / n);
    let max_back = p.max_back_for(format).min(window / n);
    let far_on = match format {
        Format::Bc1 => w[3] == 0,
        Format::Bc4 => w[1..] == [0, 0, 0],
        Format::Bc5 => w[2..] == [0, 0],
        Format::Bc7 => false,
    } && far_back > max_back;
    let mut far = FarEndpoints::new(format, if far_on { blocks.len() } else { 0 }, w);
    // Localized search (BC5's two channels): beyond the nearest `max_back` blocks, shape
    // signatures narrow the chunk's earlier blocks to `LOCAL_M`, the ranker to `local_k`, and
    // those (plus the blocks around the one above) get the full search. The ranker only chooses
    // where to look; every candidate is still scored exactly.
    let two = format == Format::Bc5 && w[0] > 0 && w[1] > 0;
    // BC4: the same with its own single-channel ranker ([`crate::ranker::bc4`]). BC7 has its own
    // ranker ([`crate::ranker::bc7`]), run after the far search.
    let one = format == Format::Bc4 && w[0] > 0;
    let bc7 = format == Format::Bc7;
    let local_k = if two || one || bc7 { p.local_k_for(format) } else { 0 };
    let dump = (two || one || bc7) && DUMP.lock().unwrap().is_some();
    let row_n = if local_k > 0 && p.row > 0 {
        2 * LOCAL_ROW_R + 1
    } else {
        0
    };
    let track = (two || one) && (local_k > 0 || dump);
    let track7 = bc7 && (local_k > 0 || dump);
    let mut sigs: Vec<[u8; 64]> = Vec::with_capacity(if track { blocks.len() } else { 0 });
    let mut rblocks: Vec<RankBlock> = Vec::with_capacity(if track && two && local_k > 0 { blocks.len() } else { 0 });
    let mut rblocks1: Vec<Rank1> = Vec::with_capacity(if track && one { blocks.len() } else { 0 });
    let mut rblocks7: Vec<RankBlock7> = Vec::with_capacity(if track7 { blocks.len() } else { 0 });
    let mut pre7: Vec<(u32, usize)> = Vec::new();
    let mut res7: Vec<(u32, usize)> = Vec::new();
    let mut decs: Vec<Block> = Vec::new();
    let mut scored: Vec<(f32, usize)> = Vec::new();
    let mut pick1 = crate::ranker::bc4::Scratch::default();
    let mut extra_ds: Vec<usize> = Vec::new();
    let mut gains: Vec<f32> = Vec::new();
    // A BC7 dump block searches the whole chunk.
    let all = if bc7 && dump { blocks.len() } else { 0 };
    let mut seen = Seen::new((max_back + local_k + row_n + all) * 50 + if far_on { far_back } else { 0 });
    let mut seen_prev = Seen::new(max_back + local_k + row_n + all);
    let mut fields = FieldsSeen::new(max_back.max(far_back) + local_k + row_n + all);
    let mut dec = [[0u8; 4]; 16];
    let lambda = p.lambda as f64;
    let original = p.verify_zstd_level.map(|_| blocks.to_vec());
    let mut fin: Vec<Option<(bc7::Bc7Block, Block)>> = Vec::with_capacity(blocks.len());
    // BC7: per-channel texel sums of each final block's decoding, for the far search's bound.
    let far7 = format == Format::Bc7 && far_back > max_back;
    let mut fin_sums: Vec<[i32; 4]> = Vec::new();

    for i in 0..blocks.len() {
        let s = &src[i];
        let smooth = smooth_factor(s, w, p) as f64;
        let score = |err: u64, bits: f32| smooth * err as f64 / 16.0 + lambda * bits as f64;
        let mut base = blocks[i];
        format.decode_block(&base[..n], &mut dec);
        let mut base_err = block_sse(s, &dec, w);
        let (mut base_bits, mut base_rep) = model.cost(&base[..n]);
        // Other base encodings (BC7: see `pipeline::encode`): the block starts from whichever
        // scores best.
        for a in alts.iter().map(|a| a[i]) {
            if a[..n] == base[..n] {
                continue;
            }
            format.decode_block(&a[..n], &mut dec);
            let a_err = block_sse(s, &dec, w);
            let (a_bits, a_rep) = model.cost(&a[..n]);
            if score(a_err, a_bits) < score(base_err, base_bits) {
                (base, base_err, base_bits, base_rep) = (a, a_err, a_bits, a_rep);
            }
        }
        let base_unpacked = unpack_final(format, &base).map(|(u, _)| u);
        let base_half = base_half_err(format, &base, s, w);
        let min_bits = lambda * model.min_cost(n) as f64;
        // Error budget: a multiple of the base error, plus a floor so near-lossless blocks still
        // have room to move.
        let limit = p.error_limit(base_err, channels);
        let mut best = (score(base_err, base_bits), base, base_bits, base_rep);

        let back = max_back.min(i);
        seen.clear();
        seen_prev.clear();
        fields.clear(s, w, format);
        extra_ds.clear();
        // BC4 (2048-block chunks): about one block in 2048, by a hash of its position and base
        // encoding, so every chunk position is sampled.
        let dump_blk = dump
            && i > back
            && if bc7 {
                dump7_pick(s)
            } else if one {
                (field_key(&base[..8]) ^ i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 53 == 0
            } else {
                i % DUMP_EVERY == DUMP_EVERY - 1
            };
        if dump_blk {
            extra_ds.extend(back + 1..=i);
        } else if local_k > 0 && i > back && one {
            let rs = RankSrc1::new(s, base_err, lambda as f32, p.row);
            let picks = rs.pick(&rblocks1, i, back, LOCAL_M1, local_k, &mut pick1);
            extra_ds.extend(picks.iter().map(|&(_, d)| d));
            if p.row > 0 {
                for d in p.row.saturating_sub(LOCAL_ROW_R)..=p.row + LOCAL_ROW_R {
                    if d > back && d <= i && !extra_ds.contains(&d) {
                        extra_ds.push(d);
                    }
                }
            }
            extra_ds.sort_unstable();
        } else if local_k > 0 && i > back {
            let by = |a: &(f32, usize), b: &(f32, usize)| a.0.total_cmp(&b.0);
            let m = LOCAL_M.max(local_k);
            scored.clear();
            if bc7 {
                // The value SAD and the refit residual each narrow the nearest `LOCAL7_REACH`
                // blocks to `LOCAL7_M`; the ranker picks `local_k` of their union.
                let rs = RankSrc7::new(s, &base, base_err, smooth as f32, lambda as f32);
                rs.prefilter_top(
                    &rblocks7,
                    i,
                    back + 1..=i.min(LOCAL7_REACH),
                    LOCAL7_M,
                    &mut pre7,
                    &mut res7,
                );
                for l in [&pre7, &res7] {
                    scored.extend(l.iter().map(|&(_, d)| (0.0, d)));
                }
                scored.sort_unstable_by_key(|e| e.1);
                scored.dedup_by_key(|e| e.1);
                let wb = window / n;
                for e in &mut scored {
                    e.0 = -rs.mlp(&rs.features(&rblocks7[i - e.1], e.1, e.1 > wb));
                }
            } else {
                let sig = shape_sig(s, w);
                scored.extend((back + 1..=i).map(|d| (sad(&sigs[i - d], &sig, 0, 32) as f32, d)));
                if m < scored.len() {
                    scored.select_nth_unstable_by(m, by);
                    scored.truncate(m);
                }
                let rs = RankSrc::new(s, [0, 1], base_half, lambda as f32);
                for e in &mut scored {
                    e.0 = -rs.mlp(&rs.features(&rblocks[i - e.1], e.1));
                }
            }
            if local_k < scored.len() {
                scored.select_nth_unstable_by(local_k, by);
                scored.truncate(local_k);
            }
            extra_ds.extend(scored.iter().map(|&(_, d)| d));
            // The ranker rarely picks the blocks around the one above, yet in photos they win
            // often (Kodak-8 BD-rate −1.6% with radius 2 at max_back 128 and K = 16).
            if p.row > 0 {
                for d in p.row.saturating_sub(LOCAL_ROW_R)..=p.row + LOCAL_ROW_R {
                    if d > back && d <= i && !extra_ds.contains(&d) {
                        extra_ds.push(d);
                    }
                }
            }
            extra_ds.sort_unstable();
        }
        gains.clear();
        gains.resize(extra_ds.len(), 0.0);
        let mut near = best;
        let mut overall = best;
        let mut xi = usize::MAX;
        let mut far_done = false;
        for d in (1..=back).chain(extra_ds.iter().copied()) {
            // BC7's far search before the picks: they get the candidates it does not try, and
            // the training dump scores them against the near and far searches together.
            if far7 && !far_done && d > back {
                far_done = true;
                model.hint = 0;
                let f7 = Far7 {
                    fin: &fin,
                    fin_sums: &fin_sums,
                    far_back,
                    w,
                    lambda,
                };
                bc7_far(
                    &f7,
                    blocks,
                    i,
                    back,
                    &base,
                    s,
                    limit,
                    smooth,
                    min_bits,
                    &mut fields,
                    &mut seen,
                    &mut model,
                    &mut best,
                );
            }
            model.hint = if d > back { d * n } else { 0 };
            if dump_blk && d > back {
                if xi == usize::MAX {
                    near = best;
                    overall = best;
                }
                xi = xi.wrapping_add(1);
                if best.0 < overall.0 {
                    overall = best;
                }
                best = near;
            }
            let prev = blocks[i - d];
            // Candidates depend only on (base, prev, source): a previous block identical to a
            // nearer one (flat or repeated areas) adds nothing new. Its nearer twin also gives
            // the shorter LZ offset, so skipping it never loses a better match.
            if !seen_prev.insert(&prev) {
                continue;
            }
            cands.clear();
            // Errors above this are rejected below anyway (over the limit, or distortion alone
            // no better than the best score, which only improves), so kernels may stop there.
            // The +2 covers rounding in the division.
            let bound = |best: f64| limit.min((best * 16.0 / smooth) as u64 + 2);
            candidates(
                format,
                &base,
                &prev,
                s,
                w,
                p.reduce_entropy,
                d <= span_back,
                d <= CROSS_BACK || d == p.row,
                &mut fields,
                bound(best.0),
                pair(base_unpacked.as_ref(), fin[i - d].as_ref()),
                base_half,
                &mut cands,
            );
            for (c, known) in &cands {
                // Different previous blocks often yield the same candidate (e.g. a copied span
                // they share); within one block the model is unchanged, so a repeat scores the same.
                if c[..n] == base[..n] || !seen.insert(c) {
                    continue;
                }
                let err = match *known {
                    UNKNOWN => {
                        let b = bound(best.0);
                        block_err(format, &c[..n], s, w, b, (&base, base_half), &mut dec)
                    }
                    e => e,
                };
                if err > limit {
                    continue;
                }
                let dist = smooth * err as f64 / 16.0;
                if dist >= best.0 {
                    continue;
                }
                // Even the cheapest possible coding would not beat the best.
                if dist + min_bits >= best.0 && !model.may_continue(&c[..n]) {
                    continue;
                }
                let Some((bits, rep)) = model.cost_until(&c[..n], bit_budget(best.0, dist, lambda)) else {
                    continue;
                };
                let j = dist + lambda * bits as f64;
                if j < best.0 {
                    best = (j, *c, bits, rep);
                }
            }
            if dump_blk && d > back {
                gains[xi] = (near.0 - best.0) as f32;
            }
        }
        if dump_blk && xi != usize::MAX {
            if best.0 < overall.0 {
                overall = best;
            }
            best = overall;
            if bc7 {
                let rs = RankSrc7::new(s, &base, base_err, smooth as f32, lambda as f32);
                let g = DUMP_GROUP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Rows for the blocks within `DUMP7_M` of the best by either prefilter key (the
                // ranker sees those within `LOCAL7_M`), then one d = 0 row: the best gain of all,
                // and the block count.
                pre7.clear();
                res7.clear();
                for (k, &d) in extra_ds.iter().enumerate() {
                    let (v, r) = rs.prefilter(&rblocks7[i - d]);
                    pre7.push((v, k));
                    res7.push((r, k));
                }
                let mut rank = vec![u32::MAX; extra_ds.len()];
                for l in [&mut pre7, &mut res7] {
                    l.sort_unstable();
                    for (r, &(_, k)) in l.iter().enumerate() {
                        rank[k] = rank[k].min(r as u32);
                    }
                }
                let mut buf = Vec::with_capacity(4 * DUMP7_M * 92);
                for (k, &d) in extra_ds.iter().enumerate() {
                    if (rank[k] as usize) < DUMP7_M {
                        rs.dump_row(&mut buf, g, &rblocks7[i - d], d, d > window / n, gains[k], rank[k]);
                    }
                }
                buf.extend_from_slice(&g.to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
                buf.extend(std::iter::repeat_n(0u8, 4 * crate::ranker::bc7::NIN));
                buf.extend_from_slice(&gains.iter().fold(0f32, |a, &b| a.max(b)).to_le_bytes());
                buf.extend_from_slice(&(extra_ds.len() as f32).to_le_bytes());
                use std::io::Write;
                DUMP.lock().unwrap().as_mut().unwrap().write_all(&buf).unwrap();
            } else if one {
                let rs = RankSrc1::new(s, base_err, lambda as f32, p.row);
                if let Some(wr) = DUMP.lock().unwrap().as_mut() {
                    let g = DUMP_GROUP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    crate::ranker::bc4::dump_rows(wr, g, &rs, &rblocks1, i, &extra_ds, &gains, near.0);
                }
            } else {
                dump_rows(s, i, &extra_ds, &gains, near.0, &sigs, &decs, w, base_half, lambda);
            }
        }

        model.hint = 0;
        // Far endpoints: earlier blocks' endpoints beyond the main search, re-indexed. Each
        // distinct pair is tried at its nearest position only, and only when a lower bound on its
        // error (the block mean's distance from the endpoints' segment) leaves it a chance.
        if far_on && i > back {
            let (m, slack) = far.mean(s);
            let px: [[f32; 16]; 2] = [0, 1].map(|c| std::array::from_fn(|t| s[t][c] as f32 * far.sw[c]));
            for d in back + 1..=far_back.min(i) {
                let j = i - d;
                for h in 0..far.fields {
                    let Some(lb) = far.lower_bound(j, h, &m, slack, &px) else {
                        continue;
                    };
                    // A BC5 candidate keeps base's other half.
                    let lb = lb
                        + if far.fields == 2 {
                            base_half[1 - h] as f64 / 16.0
                        } else {
                            0.0
                        };
                    if smooth * lb >= best.0 || (16.0 * lb) as u64 > limit {
                        continue;
                    }
                    let c = far.candidate(&base, &blocks[j], h, s, w);
                    if c[..n] == base[..n] || !seen.insert(&c) {
                        continue;
                    }
                    let bnd = limit.min((best.0 * 16.0 / smooth) as u64 + 2);
                    let err = block_err(format, &c[..n], s, w, bnd, (&base, base_half), &mut dec);
                    if err > limit {
                        continue;
                    }
                    let dist = smooth * err as f64 / 16.0;
                    if dist >= best.0 || (dist + min_bits >= best.0 && !model.may_continue(&c[..n])) {
                        continue;
                    }
                    let Some((bits, rep)) = model.cost_until(&c[..n], bit_budget(best.0, dist, lambda)) else {
                        continue;
                    };
                    let jj = dist + lambda * bits as f64;
                    if jj < best.0 {
                        best = (jj, c, bits, rep);
                    }
                }
            }
        }

        if far7 && !far_done && i > back {
            let f7 = Far7 {
                fin: &fin,
                fin_sums: &fin_sums,
                far_back,
                w,
                lambda,
            };
            bc7_far(
                &f7,
                blocks,
                i,
                back,
                &base,
                s,
                limit,
                smooth,
                min_bits,
                &mut fields,
                &mut seen,
                &mut model,
                &mut best,
            );
        }

        // Refinement: the search copies whole blocks, whole fields with the other refit, and
        // nearby spans; mixing one field of an earlier block into the chosen one can still win.
        if refine_back > 0 && best.1[..n] != base[..n] {
            let flds: &[(usize, usize)] = match format {
                Format::Bc1 => &[(0, 4), (4, 8)],
                Format::Bc4 => &[(0, 2), (2, 8)],
                Format::Bc5 => &[(0, 2), (2, 8), (8, 10), (10, 16)],
                // Mode 6 (most RDO output): mode, endpoints and P0; then P1 and the indices.
                Format::Bc7 => &[(0, 8), (8, 16)],
            };
            let cur = best.1;
            // Each candidate changes one field, so for BC5 it keeps one half of `cur`.
            let cur_half = base_half_err(format, &cur, s, w);
            // BC7 reaches past the exhaustive depth (32) to the whole window.
            let depth = if bc7 {
                refine_back.min(i).min(window / n)
            } else {
                back.min(refine_back)
            };
            // Mode 6 halves are scored without a decode (most BC7 RDO output is mode 6).
            let m6 = (bc7 && bc7::is_mode6(&cur)).then(|| (bc7::Mode6Lo::new(&cur, s, w), bc7::Mode6Hi::new(&cur)));
            let half = |c: &[u8; 16], k: usize| u64::from_le_bytes(c[k * 8..k * 8 + 8].try_into().unwrap());
            for &(a, b) in flds {
                for d in 1..=depth {
                    let prev = &blocks[i - d];
                    if prev[a..b] == cur[a..b] {
                        continue;
                    }
                    let mut c = cur;
                    c[a..b].copy_from_slice(&prev[a..b]);
                    if c[..n] == base[..n] || !seen.insert(&c) {
                        continue;
                    }
                    let bnd = limit.min((best.0 * 16.0 / smooth) as u64 + 2);
                    let err = match &m6 {
                        Some((lo, _)) if a == 8 => lo.err(half(&c, 1), bnd),
                        Some((_, hi)) if bc7::is_mode6(prev) => hi.err(half(&c, 0), s, w, bnd),
                        _ => block_err(format, &c[..n], s, w, bnd, (&cur, cur_half), &mut dec),
                    };
                    if err > limit {
                        continue;
                    }
                    let dist = smooth * err as f64 / 16.0;
                    if dist >= best.0 || (dist + min_bits >= best.0 && !model.may_continue(&c[..n])) {
                        continue;
                    }
                    let Some((bits, rep)) = model.cost_until(&c[..n], bit_budget(best.0, dist, lambda)) else {
                        continue;
                    };
                    let j = dist + lambda * bits as f64;
                    if j < best.0 {
                        best = (j, c, bits, rep);
                    }
                }
            }
        }
        stats.est_bits_before += base_bits as f64;
        stats.est_bits_after += best.2 as f64;
        if best.1[..n] != base[..n] {
            stats.changed += 1;
        }
        blocks[i] = best.1;
        if track {
            format.decode_block(&best.1[..n], &mut dec);
            if !one {
                sigs.push(shape_sig(&dec, w));
            }
            if dump && !one {
                decs.push(dec);
            }
            if one {
                rblocks1.push(Rank1::new(&dec, &best.1[..8]));
            } else if local_k > 0 {
                rblocks.push(RankBlock::new(&dec, [0, 1]));
            }
        }
        if track7 {
            format.decode_block(&best.1[..n], &mut dec);
            rblocks7.push(RankBlock7::new(&dec, &best.1));
        }
        fin.push(unpack_final(format, &best.1));
        if far7 {
            fin_sums.push(fin[i].as_ref().map_or([0; 4], |(_, d)| channel_sums(d)));
        }
        model.push(&best.1[..n], best.3);
        if far_on {
            far.push(i, &best.1);
        }
    }

    // Safety net: the rate model is an estimate. If real Zstd says RDO did not help this
    // chunk, keep the base encoding (better quality at the same or smaller size).
    if let (Some(level), Some(original)) = (p.verify_zstd_level, original) {
        let size = |bs: &[[u8; 16]]| {
            let flat: Vec<u8> = bs.iter().flat_map(|b| b[..n].iter().copied()).collect();
            zstd::bulk::compress(&flat, level).map_or(usize::MAX, |v| v.len())
        };
        if size(blocks) >= size(&original) {
            blocks.copy_from_slice(&original);
            stats.changed = 0;
            stats.est_bits_after = stats.est_bits_before;
            stats.reverted_chunks = 1;
        }
    }
    stats
}

/// Localized search: blocks the shape signatures pass to the ranker.
const LOCAL_M: usize = 128;
/// Localized BC4 search: blocks the shape and value SADs pass to the ranker (half each).
const LOCAL_M1: usize = 128;
/// Localized search: blocks either side of the one directly above that are also searched.
const LOCAL_ROW_R: usize = 2;
/// Ranker training dumps: one block in this many searches every far block of its chunk.
const DUMP_EVERY: usize = 512;

/// BC7 localized search: blocks each prefilter key passes to the ranker (32 matched 16's size and quality on
/// Kodak-24 against Oodle at 6% more CPU time).
const LOCAL7_M: usize = 16;
/// Localized BC7 search: how far back the prefilter looks. Half a 64 KiB chunk (the chunk size the ranker was
/// trained on): the whole chunk matched it on Kodak-24 but cost more, and with larger `--chunk-bytes`,
/// scanning the whole chunk would cost quadratically more.
const LOCAL7_REACH: usize = 2048;
/// BC7 ranker training dumps: blocks dumped per prefilter key (see `optimize_chunk`).
const DUMP7_M: usize = 64;

/// BC7 ranker training dumps: about one block in 256, chosen by a hash of its texels (fixed
/// positions would sample the same columns of every chunk).
fn dump7_pick(s: &Block) -> bool {
    let h = s.iter().fold(0x811c_9dc5u32, |h, t| {
        (h ^ u32::from_le_bytes(*t)).wrapping_mul(0x0100_0193)
    });
    h.wrapping_mul(0x9e37_79b1) >> 24 == 0
}

/// Ranker training rows (`MIPPLE_LOCAL_DUMP=path`; BC4, BC5 and BC7), appended by every encode in
/// the process: a development hook for `scripts/train_ranker*.py`, not used otherwise. Every
/// `DUMP_EVERY`-th block searches every far block of its chunk, each scored on its own against
/// the near search's best, and writes one row per far block.
pub static DUMP: std::sync::LazyLock<std::sync::Mutex<Option<std::io::BufWriter<std::fs::File>>>> =
    std::sync::LazyLock::new(|| {
        std::sync::Mutex::new(
            std::env::var_os("MIPPLE_LOCAL_DUMP").map(|p| {
                std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap())
            }),
        )
    });
static DUMP_GROUP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// One row per far block j = i − d (two weighted channels): u32 group, u16 × 18 features
/// [d, shape SAD ×2, decoded-vs-source SAD ×2, |sum diff| ×2, |range diff| ×2, source range ×2,
/// refit residual ×2 (least squares of the source on j's shape), copy SSE / 4 ×2,
/// base half SSE / 4 ×2, 100·λ],
/// f32 gain (near best J − best J with j's candidates, ≥ 0), f32 near best J.
#[expect(
    clippy::too_many_arguments,
    reason = "diagnostic dump of everything the search saw for one block"
)]
fn dump_rows(
    s: &Block,
    i: usize,
    ds: &[usize],
    gains: &[f32],
    near_j: f64,
    sigs: &[[u8; 64]],
    decs: &[Block],
    w: &[u32; 4],
    base_half: [u64; 2],
    lambda: f64,
) {
    use std::io::Write;
    let chs: Vec<usize> = (0..4).filter(|&c| w[c] > 0).collect();
    let sig = shape_sig(s, w);
    let stat = |b: &Block, c: usize| {
        let (mut lo, mut hi, mut sum) = (255u32, 0u32, 0u32);
        for t in b {
            let v = t[c] as u32;
            lo = lo.min(v);
            hi = hi.max(v);
            sum += v;
        }
        (hi - lo, sum)
    };
    let g = DUMP_GROUP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut buf = Vec::with_capacity(ds.len() * 48);
    for (k, &d) in ds.iter().enumerate() {
        let (o, p) = (&sigs[i - d], &decs[i - d]);
        let mut f = [0u16; 18];
        f[0] = d as u16;
        for (slot, &c) in chs.iter().enumerate() {
            f[1 + slot] = sad(o, &sig, slot * 16, 16) as u16;
            f[3 + slot] = (0..16).map(|t| s[t][c].abs_diff(p[t][c]) as u16).sum();
            let ((rs, ss), (rp, sp)) = (stat(s, c), stat(p, c));
            f[5 + slot] = ss.abs_diff(sp) as u16;
            f[7 + slot] = rs.abs_diff(rp) as u16;
            f[9 + slot] = rs as u16;
            let q = &o[slot * 16..slot * 16 + 16];
            let (mut s1, mut s2, mut q1, mut q2, mut sq) = (0f64, 0f64, 0f64, 0f64, 0f64);
            let mut cp = 0u32;
            for t in 0..16 {
                let (x, y) = (s[t][c] as f64, q[t] as f64);
                s1 += x;
                s2 += x * x;
                q1 += y;
                q2 += y * y;
                sq += x * y;
                cp += (s[t][c] as i32 - p[t][c] as i32).pow(2) as u32;
            }
            let vs = s2 - s1 * s1 / 16.0;
            let vq = q2 - q1 * q1 / 16.0;
            let cov = sq - s1 * q1 / 16.0;
            let res = if vq > 0.0 { vs - cov * cov / vq } else { vs };
            f[11 + slot] = res.clamp(0.0, 65535.0) as u16;
            f[13 + slot] = (cp / 4).min(65535) as u16;
            f[15 + slot] = (base_half[slot] / 4).min(65535) as u16;
        }
        f[17] = (lambda * 100.0).min(65535.0) as u16;
        buf.extend_from_slice(&g.to_le_bytes());
        for x in f {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&gains[k].to_le_bytes());
        buf.extend_from_slice(&(near_j as f32).to_le_bytes());
    }
    if let Some(wr) = DUMP.lock().unwrap().as_mut() {
        wr.write_all(&buf).unwrap();
        wr.flush().unwrap();
    }
}

/// Runs RDO over one mip level's blocks in place. `src` holds the source texels per block.
pub fn optimize(format: Format, src: &[Block], blocks: &mut [[u8; 16]], w: &[u32; 4], p: &RdoParams) -> RdoStats {
    optimize_with_alt(format, src, blocks, &[], w, p)
}

/// [`optimize`] with other base encodings per block (each of `alts` is as long as `blocks`): each
/// block starts from whichever scores best.
pub fn optimize_with_alt(
    format: Format,
    src: &[Block],
    blocks: &mut [[u8; 16]],
    alts: &[Vec<[u8; 16]>],
    w: &[u32; 4],
    p: &RdoParams,
) -> RdoStats {
    assert_eq!(src.len(), blocks.len());
    if !p.enabled() || blocks.is_empty() {
        return RdoStats {
            blocks: blocks.len(),
            ..Default::default()
        };
    }
    let chunk = (p.chunk_bytes_for(format) / format.block_bytes()).max(1);
    let alt_chunks: Vec<Vec<&[[u8; 16]]>> = (0..src.len().div_ceil(chunk))
        .map(|k| {
            alts.iter()
                .map(|a| &a[k * chunk..((k + 1) * chunk).min(a.len())])
                .collect()
        })
        .collect();
    src.par_chunks(chunk)
        .zip(blocks.par_chunks_mut(chunk))
        .zip(alt_chunks)
        .map(|((s, b), a)| optimize_chunk(format, s, b, &a, w, p))
        .reduce(RdoStats::default, RdoStats::merge)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Throwaway profile of one RDO chunk (BC7, λ = 1, entropy-reduced base) over a Kodak image:
    /// time and counts per stage of the candidate loop. Mirrors `optimize_chunk` with timers.
    #[test]
    #[ignore = "timing report; run by hand with --nocapture on a quiet machine"]
    fn profile_rdo_stages() {
        use std::time::{Duration, Instant};
        let path = std::env::var("MIPPLE_PROFILE_IMG").unwrap_or("test-images/kodak/kodim01.png".into());
        let img = image::open(path).unwrap().to_rgba8();
        let mut surf = crate::surface::Surface::new(img.width(), img.height());
        for (x, y, p) in img.enumerate_pixels() {
            surf.set(x, y, p.0);
        }
        let env = |k: &str, d: &str| std::env::var(k).unwrap_or(d.into());
        let format = match env("MIPPLE_PROFILE_FMT", "bc7").as_str() {
            "bc1" => Format::Bc1,
            "bc4" => Format::Bc4,
            "bc5" => Format::Bc5,
            _ => Format::Bc7,
        };
        let opts = crate::format::BlockOpts {
            effort: crate::format::DEFAULT_EFFORT,
            reduce_entropy: true,
        };
        let (src, mut blocks) = crate::pipeline::encode_base(&surf, format, opts);
        let (src, blocks) = (&src[..4096], &mut blocks[..4096]);
        let w = format.error_weights();
        let p = RdoParams {
            reduce_entropy: true,
            max_back: Some(env("MIPPLE_PROFILE_BACK", "32").parse().unwrap()),
            span_back: Some(env("MIPPLE_PROFILE_SPAN_BACK", "100000").parse().unwrap()),
            window: Some(env("MIPPLE_PROFILE_WINDOW", "2048").parse().unwrap()),
            ..RdoParams::with_lambda(env("MIPPLE_PROFILE_LAMBDA", "1").parse().unwrap())
        };
        let lambda = p.lambda as f64;
        let n = format.block_bytes();
        let mut model = LzModel::new(p.window_for(format), p.chain_depth);
        let (mut t_gen, mut t_dec, mut t_cost) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        let (mut n_gen, mut n_dec, mut n_cost, mut n_dup) = (0usize, 0usize, 0usize, 0usize);
        let mut n_lb = 0usize;
        let (mut n_limit, mut n_dist) = (0usize, 0usize);
        let back = p.max_back_for(format).min(p.window_for(format) / n);
        let mut wins = std::collections::BTreeMap::new();
        let mut seen = Seen::new(back * 50);
        let mut seen_prev = Seen::new(back);
        let mut fields = FieldsSeen::new(back);
        let mut cands = Vec::new();
        let mut dec = [[0u8; 4]; 16];
        let mut fin: Vec<Option<(bc7::Bc7Block, Block)>> = Vec::with_capacity(blocks.len());
        let t_all = Instant::now();
        for i in 0..blocks.len() {
            let base = blocks[i];
            let s = &src[i];
            let base_unpacked = unpack_final(format, &base).map(|(u, _)| u);
            format.decode_block(&base[..n], &mut dec);
            let base_err = block_sse(s, &dec, &w);
            let smooth = smooth_factor(s, &w, &p) as f64;
            let channels = w.iter().filter(|&&x| x > 0).count().max(1) as f64;
            let limit = p.error_limit(base_err, channels);
            let (base_bits, base_rep) = model.cost(&base[..n]);
            let mut best = (
                smooth * base_err as f64 / 16.0 + lambda * base_bits as f64,
                base,
                base_bits,
                base_rep,
            );
            let base_half = base_half_err(format, &base, s, &w);
            let min_bits = lambda * model.min_cost(n) as f64;
            seen.clear();
            seen_prev.clear();
            fields.clear(s, &w, format);
            let mut win = None;
            for d in 1..=back.min(i) {
                let prev = blocks[i - d];
                if !seen_prev.insert(&prev) {
                    continue;
                }
                let t0 = Instant::now();
                cands.clear();
                candidates(
                    format,
                    &base,
                    &prev,
                    s,
                    &w,
                    p.reduce_entropy,
                    d <= p.span_back_for(format),
                    false,
                    &mut fields,
                    limit.min((best.0 * 16.0 / smooth) as u64 + 2),
                    pair(base_unpacked.as_ref(), fin[i - d].as_ref()),
                    base_half,
                    &mut cands,
                );
                t_gen += t0.elapsed();
                n_gen += cands.len();
                for (ci, (c, known)) in cands.iter().enumerate() {
                    if c[..n] == base[..n] || !seen.insert(c) {
                        n_dup += 1;
                        continue;
                    }
                    let t1 = Instant::now();
                    let bound = limit.min((best.0 * 16.0 / smooth) as u64 + 2);
                    let err = match *known {
                        UNKNOWN => block_err(format, &c[..n], s, &w, bound, (&base, base_half), &mut dec),
                        e => e,
                    };
                    t_dec += t1.elapsed();
                    n_dec += 1;
                    if err > limit {
                        n_limit += 1;
                        continue;
                    }
                    let dist = smooth * err as f64 / 16.0;
                    if dist >= best.0 {
                        n_dist += 1;
                        continue;
                    }
                    if dist + min_bits >= best.0 && !model.may_continue(&c[..n]) {
                        n_lb += 1;
                        continue;
                    }
                    let t2 = Instant::now();
                    let r = model.cost_until(&c[..n], bit_budget(best.0, dist, lambda));
                    t_cost += t2.elapsed();
                    n_cost += 1;
                    let Some((bits, rep)) = r else { continue };
                    let j = dist + lambda * bits as f64;
                    if j < best.0 {
                        best = (j, *c, bits, rep);
                        win = Some((d, ci));
                    }
                }
            }
            if let Some((d, ci)) = win {
                let bucket = match d {
                    1..=8 => 0,
                    9..=32 => 1,
                    33..=128 => 2,
                    _ => 3,
                };
                *wins.entry((bucket, ci.min(63))).or_insert(0usize) += 1;
            }
            blocks[i] = best.1;
            fin.push(unpack_final(format, &best.1));
            model.push(&best.1[..n], best.3);
        }
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        println!("blocks {}  total {:.1} ms", blocks.len(), ms(t_all.elapsed()));
        println!("generate  {:8.1} ms  {n_gen} candidates", ms(t_gen));
        println!("decode+sse{:8.1} ms  {n_dec} decoded ({n_dup} duplicates)", ms(t_dec));
        println!("          over limit {n_limit}, distortion alone worse {n_dist}, rate bound {n_lb}");
        println!("lz cost   {:8.1} ms  {n_cost} costed", ms(t_cost));
        // Winning candidates by distance bucket (1-8, 9-32, 33-128, 129+) and candidate index.
        for b in 0..4 {
            let row: Vec<String> = wins
                .iter()
                .filter(|((bb, _), _)| *bb == b)
                .map(|((_, ci), v)| format!("{ci}:{v}"))
                .collect();
            println!("wins d-bucket {b}: {}", row.join(" "));
        }
    }

    #[test]
    fn fused_error_kernels_match_decode() {
        // xorshift: deterministic random blocks and sources.
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut dec = [[0u8; 4]; 16];
        for format in [Format::Bc1, Format::Bc4, Format::Bc5] {
            let w = format.error_weights();
            for _ in 0..2000 {
                let mut raw = [0u8; 16];
                for b in &mut raw {
                    *b = next() as u8;
                }
                let mut src = [[0u8; 4]; 16];
                src.iter_mut().flatten().for_each(|b| *b = next() as u8);
                let n = format.block_bytes();
                format.decode_block(&raw[..n], &mut dec);
                let exact = block_sse(&src, &dec, &w);
                let mut scratch = [[0u8; 4]; 16];
                // A base block sharing one half (or neither) with the candidate.
                let mut base = [0u8; 16];
                for b in &mut base {
                    *b = next() as u8;
                }
                let keep = next() % 3;
                if keep < 2 {
                    let r = keep as usize * 8..keep as usize * 8 + 8;
                    base[r.clone()].copy_from_slice(&raw[r]);
                }
                let base = (&base, base_half_err(format, &base, &src, &w));
                assert_eq!(
                    block_err(format, &raw[..n], &src, &w, u64::MAX, base, &mut scratch),
                    exact
                );
                // A bound below the error may stop early but must still report more than it.
                let bound = next() % (exact + 1);
                let e = block_err(format, &raw[..n], &src, &w, bound, base, &mut scratch);
                assert!(if exact > bound { e > bound } else { e == exact });
            }
        }
    }

    #[test]
    fn repeated_data_is_cheap() {
        let mut m = LzModel::new(1024, 16);
        let block = [7u8, 1, 200, 33, 90, 14, 15, 16];
        let (fresh, s0) = m.cost(&block);
        m.push(&block, s0);
        let (again, s1) = m.cost(&block);
        assert!(again < fresh / 3.0, "fresh {fresh} again {again}");
        assert_eq!(s1.reps[0], 8);
        assert!(s1.tail_match);
        // Once a run is established, extending it is nearly free.
        m.push(&block, s1);
        let (extend, _) = m.cost(&block);
        assert!(extend <= 1.0, "extend {extend}");
    }

    #[test]
    fn repeats_inside_a_candidate_are_cheap() {
        let m = LzModel::new(1024, 16);
        let half = [7u8, 1, 200, 33, 90, 14, 15, 16];
        let mut twice = [0u8; 16];
        twice[..8].copy_from_slice(&half);
        twice[8..].copy_from_slice(&half);
        let mut distinct = twice;
        distinct[8..].copy_from_slice(&[99, 98, 97, 96, 95, 94, 93, 92]);
        let (a, _) = m.cost(&twice);
        let (b, _) = m.cost(&distinct);
        assert!(a < b * 0.75, "repeated {a} vs distinct {b}");
    }
}
