//! Decoding ASTC LDR blocks to RGBA8.

use super::{COLOR_UNQUANT, Symbolic, WEIGHT_UNQUANT, cem_is_hdr, cem_values, partition_of};
use crate::surface::{Rgba, Surface};

/// The LDR profile's error color.
pub const ERROR: Rgba = [255, 0, 255, 255];

fn bit_transfer_signed(a: i32, b: i32) -> (i32, i32) {
    let b = (b >> 1) | (a & 0x80);
    let mut a = (a >> 1) & 0x3f;
    if a & 0x20 != 0 {
        a -= 0x40;
    }
    (a, b)
}

fn blue_contract(c: [i32; 4]) -> [i32; 4] {
    [(c[0] + c[2]) >> 1, (c[1] + c[2]) >> 1, c[2], c[3]]
}

fn clamp(c: [i32; 4]) -> [i32; 4] {
    c.map(|x| x.clamp(0, 255))
}

/// The two endpoints (RGBA, 0..=255) of LDR color endpoint mode `cem` from its unquantized values,
/// or `None` for HDR modes.
pub fn endpoints(cem: u8, v: &[i32]) -> Option<([i32; 4], [i32; 4])> {
    Some(match cem {
        0 => ([v[0], v[0], v[0], 255], [v[1], v[1], v[1], 255]),
        1 => {
            let l0 = (v[0] >> 2) | (v[1] & 0xc0);
            let l1 = (l0 + (v[1] & 0x3f)).min(255);
            ([l0, l0, l0, 255], [l1, l1, l1, 255])
        }
        4 => ([v[0], v[0], v[0], v[2]], [v[1], v[1], v[1], v[3]]),
        5 => {
            let (d0, b0) = bit_transfer_signed(v[1], v[0]);
            let (d1, b1) = bit_transfer_signed(v[3], v[2]);
            (clamp([b0, b0, b0, b1]), clamp([b0 + d0, b0 + d0, b0 + d0, b1 + d1]))
        }
        6 => (
            [(v[0] * v[3]) >> 8, (v[1] * v[3]) >> 8, (v[2] * v[3]) >> 8, 255],
            [v[0], v[1], v[2], 255],
        ),
        8 | 12 => {
            let a = if cem == 12 { [v[6], v[7]] } else { [255, 255] };
            let e0 = [v[0], v[2], v[4], a[0]];
            let e1 = [v[1], v[3], v[5], a[1]];
            if v[1] + v[3] + v[5] >= v[0] + v[2] + v[4] {
                (e0, e1)
            } else {
                (blue_contract(e1), blue_contract(e0))
            }
        }
        9 | 13 => {
            let (d0, b0) = bit_transfer_signed(v[1], v[0]);
            let (d1, b1) = bit_transfer_signed(v[3], v[2]);
            let (d2, b2) = bit_transfer_signed(v[5], v[4]);
            let (d3, b3) = if cem == 13 {
                bit_transfer_signed(v[7], v[6])
            } else {
                (0, 255)
            };
            let e0 = [b0, b1, b2, b3];
            let e1 = [b0 + d0, b1 + d1, b2 + d2, b3 + d3];
            if d0 + d1 + d2 >= 0 {
                (clamp(e0), clamp(e1))
            } else {
                (clamp(blue_contract(e1)), clamp(blue_contract(e0)))
            }
        }
        10 => (
            [(v[0] * v[3]) >> 8, (v[1] * v[3]) >> 8, (v[2] * v[3]) >> 8, v[4]],
            [v[0], v[1], v[2], v[5]],
        ),
        _ => return None,
    })
}

/// Bilinear infill of a `gx` × `gy` weight grid (stride `planes`, offset `plane`) to texel (`s`, `t`)
/// of a `bw` × `bh` block, per the spec's fixed-point rule.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the spec's infill inputs, kept flat for the per-texel hot loop"
)]
pub fn infill(w: &[u8], gx: u32, gy: u32, planes: usize, plane: usize, bw: u32, bh: u32, s: u32, t: u32) -> u32 {
    if gx == bw && gy == bh {
        return w[(t * gx + s) as usize * planes + plane] as u32;
    }
    let ds = (1024 + bw / 2) / (bw - 1);
    let dt = (1024 + bh / 2) / (bh - 1);
    let gs = (ds * s * (gx - 1) + 32) >> 6;
    let gt = (dt * t * (gy - 1) + 32) >> 6;
    let (js, fs, jt, ft) = (gs >> 4, gs & 15, gt >> 4, gt & 15);
    let at = |x: u32, y: u32| -> u32 {
        if x < gx && y < gy {
            w[(y * gx + x) as usize * planes + plane] as u32
        } else {
            0
        }
    };
    let w11 = (fs * ft + 8) >> 4;
    let w10 = ft - w11;
    let w01 = fs - w11;
    let w00 = 16 - fs - ft + w11;
    (at(js, jt) * w00 + at(js + 1, jt) * w01 + at(js, jt + 1) * w10 + at(js + 1, jt + 1) * w11 + 8) >> 4
}

/// The texel value of endpoints `e0`, `e1` at weight `w` (0..=64), as UNORM8: the spec's 16-bit
/// interpolation, then rounded to eight bits (linear) or truncated (sRGB RGB, before conversion).
#[inline]
pub fn lerp(e0: [i32; 4], e1: [i32; 4], w: [u32; 4], srgb: bool) -> Rgba {
    std::array::from_fn(|c| lerp1(e0[c], e1[c], w[c], srgb && c < 3) as u8)
}

/// One channel of [`lerp`]: `srgb` for an sRGB color channel (not alpha).
#[inline]
pub fn lerp1(e0: i32, e1: i32, w: u32, srgb: bool) -> i32 {
    if srgb {
        let (c0, c1) = ((e0 << 8 | 0x80) as u32, (e1 << 8 | 0x80) as u32);
        ((c0 * (64 - w) + c1 * w + 32) >> 14) as i32
    } else {
        // The spec's ((e * 257 interpolated) * 255 + 32767) / 65535, which is exactly this for
        // eight-bit endpoints (checked over every input).
        ((e0 as u32 * (64 - w) + e1 as u32 * w + 32) >> 6) as i32
    }
}

/// Decodes one block of a `bw` × `bh` footprint into `out` (row-major, `bw * bh` texels).
pub fn decode_block(raw: &[u8; 16], bw: u32, bh: u32, srgb: bool, out: &mut [Rgba]) {
    let v = u128::from_le_bytes(*raw);
    let n = (bw * bh) as usize;
    if v & 0x1ff == 0x1fc {
        // Void extent: a constant color (UNORM16), unless HDR or the reserved bits are wrong.
        let hdr = (v >> 9) & 1 != 0;
        let c = |i: u32| ((v >> (64 + 16 * i)) & 0xffff) as u32;
        let px = if hdr || (v >> 10) & 3 != 3 {
            ERROR
        } else {
            std::array::from_fn(|i| {
                let x = c(i as u32);
                if srgb && i < 3 {
                    (x >> 8) as u8
                } else {
                    ((x * 255 + 32767) / 65535) as u8
                }
            })
        };
        out[..n].fill(px);
        return;
    }
    let Some(sym) = Symbolic::read(v, bw, bh) else {
        out[..n].fill(ERROR);
        return;
    };
    let p = sym.partitions as usize;
    let mut eps = [([0i32; 4], [0i32; 4]); 4];
    let mut at = 0;
    let unq = &COLOR_UNQUANT[sym.color_quant as usize];
    for (i, ep) in eps.iter_mut().enumerate().take(p) {
        let cem = sym.cems[i];
        let k = cem_values(cem);
        if cem_is_hdr(cem) {
            out[..n].fill(ERROR);
            return;
        }
        let vals: Vec<i32> = sym.colors[at..at + k].iter().map(|&q| unq[q as usize] as i32).collect();
        at += k;
        *ep = endpoints(cem, &vals).expect("LDR mode");
    }
    let wu = &WEIGHT_UNQUANT[sym.mode.quant as usize];
    let mut w = [0u8; 64];
    for i in 0..sym.mode.weights() {
        w[i] = wu[sym.weights[i] as usize];
    }
    let planes = 1 + sym.mode.dual as usize;
    let (gx, gy) = (sym.mode.gx as u32, sym.mode.gy as u32);
    for t in 0..bh {
        for s in 0..bw {
            let w0 = infill(&w, gx, gy, planes, 0, bw, bh, s, t);
            let mut ws = [w0; 4];
            if sym.mode.dual {
                ws[sym.ccs as usize] = infill(&w, gx, gy, planes, 1, bw, bh, s, t);
            }
            let part = partition_of(sym.seed as u32, s, t, sym.partitions as u32, bw * bh) as usize;
            let (e0, e1) = eps[part];
            out[(t * bw + s) as usize] = lerp(e0, e1, ws, srgb);
        }
    }
}

/// Decodes a whole image of `bw` × `bh` blocks (raster order) at `width` × `height`.
pub fn decode_rgba(data: &[u8], width: u32, height: u32, bw: u32, bh: u32, srgb: bool) -> Surface {
    let mut s = Surface::new(width, height);
    let bx = width.div_ceil(bw);
    let mut px = vec![[0u8; 4]; (bw * bh) as usize];
    for (i, raw) in data.as_chunks::<16>().0.iter().enumerate() {
        let (x0, y0) = ((i as u32 % bx) * bw, (i as u32 / bx) * bh);
        decode_block(raw, bw, bh, srgb, &mut px);
        for t in 0..bh {
            for u in 0..bw {
                let (x, y) = (x0 + u, y0 + t);
                if x < width && y < height {
                    s.set(x, y, px[(t * bw + u) as usize]);
                }
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn void_extent_is_its_color() {
        // 0x1fc, LDR, reserved bits set, then R, G, B, A as UNORM16.
        let mut v: u128 = 0x1fc | (3 << 10) | (((1u128 << 52) - 1) << 12);
        v |= 0xffffu128 << 64 | 0x8080u128 << 80 | 0xffffu128 << 112; // B = 0
        let mut out = [[0u8; 4]; 16];
        decode_block(&v.to_le_bytes(), 4, 4, false, &mut out);
        assert_eq!(out[0], [255, 128, 0, 255]);
        assert!(out.iter().all(|&p| p == out[0]));
    }

    #[test]
    fn reserved_blocks_decode_to_the_error_color() {
        let mut out = [[0u8; 4]; 16];
        decode_block(&[0u8; 16], 4, 4, false, &mut out);
        assert_eq!(out[0], ERROR);
    }

    #[test]
    fn linear_lerp_matches_the_spec() {
        for e0 in 0..256u32 {
            for e1 in 0..256u32 {
                for w in 0..=64 {
                    let v = ((e0 * (64 - w) + e1 * w) * 257 + 32) >> 6;
                    let spec = ((v * 255 + 32767) / 65535) as i32;
                    assert_eq!(lerp1(e0 as i32, e1 as i32, w, false), spec, "{e0} {e1} {w}");
                }
            }
        }
    }

    #[test]
    fn infill_hits_the_grid_at_the_corners() {
        // An 8x8 block from a 5x5 grid: corner texels take the corner weights exactly.
        let w: Vec<u8> = (0..25).map(|i| (i * 2) as u8).collect();
        assert_eq!(infill(&w, 5, 5, 1, 0, 8, 8, 0, 0), 0);
        assert_eq!(infill(&w, 5, 5, 1, 0, 8, 8, 7, 0), 8);
        assert_eq!(infill(&w, 5, 5, 1, 0, 8, 8, 0, 7), 40);
        assert_eq!(infill(&w, 5, 5, 1, 0, 8, 8, 7, 7), 48);
    }

    #[test]
    fn endpoint_modes_match_hand_examples() {
        // Direct RGB with the second sum lower: blue contraction and swapped endpoints.
        let (e0, e1) = endpoints(8, &[200, 10, 100, 20, 50, 30]).unwrap();
        assert_eq!(e0, [(10 + 30) >> 1, (20 + 30) >> 1, 30, 255]);
        assert_eq!(e1, [(200 + 50) >> 1, (100 + 50) >> 1, 50, 255]);
        // Luminance base + offset.
        let (e0, e1) = endpoints(1, &[0x40, 0xc5]).unwrap();
        assert_eq!(e0[0], (0x40 >> 2) | 0xc0);
        assert_eq!(e1[0], ((0x40 >> 2) | 0xc0) + 5);
        assert!(endpoints(7, &[0; 8]).is_none());
    }
}
