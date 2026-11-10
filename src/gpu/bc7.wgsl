// BC7 base encoder, efforts 0-2: one invocation per block. A port of src/bc7/fast.rs (encode8 with
// mode 6, its least-squares refits, and mode 1 on bc7f's partition estimate), which the CPU encoder
// defines in exact integers: covariances cut to 11 bits, power iterations on a 10-bit vector, a
// 7-bit axis, endpoints at the extreme projections on the axis line through the mean (in 1/16
// units), least-squares refits as exact rationals, the quantizer's rounding, and each index the
// better of the two palette entries around the projection. So the output matches the CPU encoder
// block for block on any GPU. Blocks with alpha are re-encoded by the host (the CPU sends them to
// its scalar encoder). The host prepends PART2 (2-subset partitions, bit i = texel i in subset
// 1) and ANCHOR2 (the second subset's anchor texel).

struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    count: u32,  // blocks in this dispatch
    power: u32,  // power iterations of the mode-6 fit
    first: u32,  // first block of this dispatch; block g writes output words 4g..4g + 3
    // Mode-6 refits | mode-1 partition candidates << 8 (0: mode 6 only) | ranked (else bc7f's
    // estimate) << 16 | mode-1 refits << 20 | refinement passes << 24 | mode-6 refinement << 28.
    stages: u32,
    busy: u32,   // mode 1 is tried when the mode-6 error exceeds this (and the block has spread)
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> px: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

var<private> t: array<vec3<i32>, 16>;
var<private> sum: vec3<i32>;

// Interpolation weights round(64 j / (n - 1)) of 4-bit (mode 6) and 3-bit (mode 1) indices.
const W4 = array<i32, 16>(0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64);
const W3 = array<i32, 8>(0, 9, 18, 27, 37, 46, 55, 64);

fn load(block: u32) {
    let bx = block % P.blocks_x;
    let by = block / P.blocks_x;
    sum = vec3(0);
    for (var i = 0u; i < 16u; i++) {
        // Edge texels repeat when the image is not a multiple of 4 (Surface::read_block).
        let tx = min(bx * 4u + (i & 3u), P.width - 1u);
        let ty = min(by * 4u + (i >> 2u), P.height - 1u);
        let p = px[ty * P.width + tx];
        t[i] = vec3(i32(p & 0xffu), i32((p >> 8u) & 0xffu), i32((p >> 16u) & 0xffu));
        sum += t[i];
    }
}

// floor(a / d) for d > 0 (WGSL division truncates).
fn floor_div(a: i32, d: i32) -> i32 {
    let q = a / d;
    return select(q, q - 1, q * d > a);
}

fn floor_div3(a: vec3<i32>, d: i32) -> vec3<i32> {
    let q = a / d;
    return select(q, q - 1, q * d > a);
}

// max(0, floor(log2 m) - bits): the shift that cuts m >= 0 to at most bits + 1 bits.
fn shrink(m: i32, bits: i32) -> u32 {
    return u32(max(firstLeadingBit(m) - bits, 0));
}

fn abs_max(v: vec3<i32>) -> i32 {
    let a = abs(v);
    return max(a.x, max(a.y, a.z));
}

// Principal axis of a covariance (rows c, diagonal diag): cut to 11 bits by the largest diagonal,
// power iterations from the largest-diagonal row (the first on ties), each step cut to 10 bits;
// (1, 1, 1) when it vanishes (`principal_axis`).
fn axis(c: array<vec3<i32>, 3>, power: u32) -> vec3<i32> {
    let diag = vec3(c[0].x, c[1].y, c[2].z);
    let cs = vec3(shrink(max(diag.x, max(diag.y, diag.z)), 10));
    let s = array<vec3<i32>, 3>(c[0] >> cs, c[1] >> cs, c[2] >> cs);
    var v = s[0];
    if diag.x < diag.y {
        v = s[1];
    }
    if max(diag.x, diag.y) < diag.z {
        v = s[2];
    }
    for (var it = 0u; it < power; it++) {
        let w = vec3(dot(s[0], v), dot(s[1], v), dot(s[2], v));
        v = w >> vec3(shrink(abs_max(w), 9));
    }
    if all(v == vec3(0)) {
        v = vec3(1);
    }
    return v;
}

// One channel of a line's endpoint in 1/16 units: E = s + q k + floor(r k / d) for a ratio
// q + r / d times k, clamped to 0..=255 n, then floor(16 E / n) (`line_point`).
// Per channel. (No loops writing vector components by a variable index: D3D's FXC cannot address
// them and would have to unroll the loop.)
fn line_point(s: vec3<i32>, q: vec3<i32>, r: vec3<i32>, k: vec3<i32>, d: i32, n: i32) -> vec3<i32> {
    let e = clamp(s + q * k + floor_div3(r * k, d), vec3(0), vec3(255 * n));
    return floor_div3(16 * e, n);
}

// A quantized mode-6 endpoint pair: 7-bit RGB levels, p-bits, 8-bit values 2q + p.
struct Ends {
    q0: vec3<i32>,
    q1: vec3<i32>,
    p0: i32,
    p1: i32,
    v0: vec3<i32>,
    v1: vec3<i32>,
}

// Nearest mode-6 levels of an endpoint (RGB e16 / 16, alpha 255) for the better p-bit: per p,
// q = floor((x - p) / 2 + 1/2) clamped to 0..=127, scored by the squared endpoint error in
// 1/16 units (alpha decodes to 254 with p = 0, an error of 1); p = 1 only when strictly better.
fn quant(e16: vec3<i32>) -> vec4<i32> {
    var best = vec4(0);
    var best_e = 0;
    for (var p = 0; p < 2; p++) {
        let q = clamp((e16 - 16 * p + 16) >> vec3(5u), vec3(0), vec3(127));
        let d = 16 * (2 * q + p) - e16;
        let err = dot(d, d) + select(0, 256, p == 0);
        if p == 0 || err < best_e {
            best_e = err;
            best = vec4(q, p);
        }
    }
    return best;
}

fn quantize(lo: vec3<i32>, hi: vec3<i32>) -> Ends {
    let qa = quant(lo);
    let qb = quant(hi);
    var e: Ends;
    e.q0 = qa.xyz;
    e.p0 = qa.w;
    e.q1 = qb.xyz;
    e.p1 = qb.w;
    e.v0 = 2 * e.q0 + e.p0;
    e.v1 = 2 * e.q1 + e.p1;
    return e;
}

// Squared RGB error of texel i against the palette entry of weight w: a + floor((w d + 32) / 64)
// = ((64 - w) a + w b + 32) >> 6, the decoder's interpolation.
fn entry_err(i: u32, a: vec3<i32>, d: vec3<i32>, w: i32) -> i32 {
    let e = t[i] - a - ((w * d + 32) >> vec3(6u));
    return dot(e, e);
}

// Mode-6 indices and their error: j = floor(15 (x - a)·d / |d|²) clamped to 0..=14, then j or
// j + 1, whichever entry is nearer (j on ties).
struct Fit6 {
    e: Ends,
    idx: array<i32, 16>,
    err: i32,
}

fn project6(e: Ends) -> Fit6 {
    var f: Fit6;
    f.e = e;
    f.err = 0;
    let d = e.v1 - e.v0;
    let dd = max(dot(d, d), 1);
    // A local array: FXC cannot write a struct's array member by a variable index in a loop.
    var idx: array<i32, 16>;
    for (var i = 0u; i < 16u; i++) {
        let j = clamp(floor_div(15 * dot(t[i] - e.v0, d), dd), 0, 14);
        let e0 = entry_err(i, e.v0, d, W4[j]);
        let e1 = entry_err(i, e.v0, d, W4[j + 1]);
        idx[i] = select(j, j + 1, e1 < e0);
        f.err += min(e0, e1);
    }
    f.idx = idx;
    return f;
}

// Least-squares endpoints (1/16 units) for indices idx of weights w over the members of `mask`
// (n of them): E(w) = Σt + num (n w - Σw) / D in 1/n units, num = n Σwt - Σw Σt and
// D = n Σw² - (Σw)² (cut to 13 bits, num by the same shift); usable when D >= 5 and every slope
// is at most 4096 (`ls_ends`).
struct Ls {
    ok: bool,
    lo: vec3<i32>,
    hi: vec3<i32>,
}

fn ls_ends(w: array<i32, 16>, mask: u32, n: i32, s: vec3<i32>) -> Ls {
    var sw = 0;
    var sww = 0;
    var swt = vec3(0);
    for (var i = 0u; i < 16u; i++) {
        if (mask & (1u << i)) != 0u {
            sw += w[i];
            sww += w[i] * w[i];
            swt += w[i] * t[i];
        }
    }
    let d = n * sww - sw * sw;
    let sh = shrink(d, 12);
    let ds = max(d >> sh, 1);
    let num = (n * swt - sw * s) >> vec3(sh);
    let q = floor_div3(num, ds);
    let r = num - q * ds;
    var o: Ls;
    o.ok = d >= 5 && abs_max(q) <= 4096;
    o.lo = line_point(s, q, r, vec3(-sw), ds, n);
    o.hi = line_point(s, q, r, vec3(64 * n - sw), ds, n);
    return o;
}

// One mode-1 subset (members `mask`): 6-bit levels with a shared p-bit, 8-bit values, 3-bit
// indices of the members and their error.
struct Sub {
    q0: vec3<i32>,
    q1: vec3<i32>,
    p: i32,
    idx: array<i32, 16>,
    err: i32,
}

// Nearest 6-bit level of an endpoint channel x = e16 / 16 with p-bit p: the rounded level
// floor((127 x / 255 - p) / 2 + 1/2) (in integers) and its neighbours, by squared error in 1/16
// units (the first on ties); returns (level, error).
fn quant6p(e16: i32, p: i32) -> vec2<i32> {
    let c = max(floor_div(127 * e16 - 4080 * p + 4080, 8160), 0);
    var best = vec2(0, 0x7fffffff);
    for (var off = -1; off <= 1; off++) {
        let q = clamp(c + off, 0, 63);
        let v7 = 2 * q + p;
        let d = 16 * (2 * v7 + (v7 >> 6u)) - e16;
        if d * d < best.y {
            best = vec2(q, d * d);
        }
    }
    return best;
}

// quant6p per channel: the levels and the summed error.
fn quant6p3(e16: vec3<i32>, p: i32) -> vec4<i32> {
    let x = quant6p(e16.x, p);
    let y = quant6p(e16.y, p);
    let z = quant6p(e16.z, p);
    return vec4(x.x, y.x, z.x, x.y + y.y + z.y);
}

// Expands a 6-bit level with p-bit to 8 bits: (v7 << 1) | (v7 >> 6) of v7 = 2q + p.
fn dequant6p(q: vec3<i32>, p: i32) -> vec3<i32> {
    let v7 = 2 * q + p;
    return 2 * v7 + (v7 >> vec3(6u));
}

// Indices and error of a subset's members for levels q0, q1 and p-bit p: j = floor(7 (x - a)·d
// / |d|²) clamped to 0..=6, then j or j + 1 (`index_two`).
fn index_sub(mask: u32, q0: vec3<i32>, q1: vec3<i32>, p: i32) -> Sub {
    var o: Sub;
    o.q0 = q0;
    o.q1 = q1;
    o.p = p;
    let a = dequant6p(q0, p);
    let d = dequant6p(q1, p) - a;
    let dd = max(dot(d, d), 1);
    o.err = 0;
    var idx: array<i32, 16>;
    for (var i = 0u; i < 16u; i++) {
        idx[i] = 0;
        if (mask & (1u << i)) != 0u {
            let j = clamp(floor_div(7 * dot(t[i] - a, d), dd), 0, 6);
            let e0 = entry_err(i, a, d, W3[j]);
            let e1 = entry_err(i, a, d, W3[j + 1]);
            idx[i] = select(j, j + 1, e1 < e0);
            o.err += min(e0, e1);
        }
    }
    o.idx = idx;
    return o;
}

// Mode-1 levels of endpoints lo, hi (1/16 units) with the shared p-bit of the lower total
// endpoint error (0 on ties), then the members' indices (`quant_two`).
fn quant_sub(mask: u32, lo: vec3<i32>, hi: vec3<i32>) -> Sub {
    let a0 = quant6p3(lo, 0);
    let b0 = quant6p3(hi, 0);
    let a1 = quant6p3(lo, 1);
    let b1 = quant6p3(hi, 1);
    let one = a1.w + b1.w < a0.w + b0.w;
    return index_sub(mask, select(a0.xyz, a1.xyz, one), select(b0.xyz, b1.xyz, one), select(0, 1, one));
}

fn encode_two(mask: u32, refits: u32) -> Sub {
    // Masked line fit: n Σxy - Σx Σy, three power iterations, a 7-bit axis, the members'
    // extreme projections (`encode_two`, `subset_ends`).
    var n = 0;
    var s = vec3(0);
    var c = array<vec3<i32>, 3>();
    for (var i = 0u; i < 16u; i++) {
        if (mask & (1u << i)) != 0u {
            n += 1;
            s += t[i];
            c[0] += t[i].x * t[i];
            c[1] += t[i].y * t[i];
            c[2] += t[i].z * t[i];
        }
    }
    n = max(n, 1);
    c[0] = n * c[0] - s.x * s;
    c[1] = n * c[1] - s.y * s;
    c[2] = n * c[2] - s.z * s;
    let v = axis(c, 3u);
    let ws = v >> vec3(shrink(abs_max(v), 6));
    var pmin = 0x7fffffff;
    var pmax = -0x7fffffff;
    for (var i = 0u; i < 16u; i++) {
        if (mask & (1u << i)) != 0u {
            let p = dot(t[i], ws);
            pmin = min(pmin, p);
            pmax = max(pmax, p);
        }
    }
    let pm = dot(s, ws);
    let n2 = dot(ws, ws);
    let qh = floor_div(pmax * n - pm, n2);
    let rh = pmax * n - pm - qh * n2;
    let ql = floor_div(pmin * n - pm, n2);
    let rl = pmin * n - pm - ql * n2;
    let hi = line_point(s, vec3(qh), vec3(rh), ws, n2, n);
    let lo = line_point(s, vec3(ql), vec3(rl), ws, n2, n);
    var best = quant_sub(mask, lo, hi);
    // Least-squares refits on the members' weights, each kept when it lowers the error.
    for (var r = 0u; r < refits; r++) {
        var w: array<i32, 16>;
        for (var i = 0u; i < 16u; i++) {
            w[i] = W3[best.idx[i]];
        }
        let ls = ls_ends(w, mask, n, s);
        if !ls.ok {
            break;
        }
        let cand = quant_sub(mask, ls.lo, ls.hi);
        if cand.err >= best.err {
            break;
        }
        best = cand;
    }
    return best;
}

// Greedy ±1 moves of each endpoint channel level (endpoint, channel, +1 then -1), then the
// shared p-bit flip, each kept when it lowers the error; up to `passes` sweeps (`refine_two`).
fn refine_sub(mask: u32, sub: Sub, passes: u32) -> Sub {
    var b = sub;
    for (var it = 0u; it < passes; it++) {
        var any = false;
        for (var mv = 0; mv < 12; mv++) {
            let k = mv / 6;
            let c = (mv / 2) % 3;
            // The move as a vector (no component writes by a variable index, for FXC); the other
            // channels are in range, so the clamp leaves them.
            let dv = select(vec3(0), vec3(select(1, -1, (mv & 1) != 0)), vec3(c) == vec3(0, 1, 2));
            let q0 = clamp(b.q0 + select(vec3(0), dv, k == 0), vec3(0), vec3(63));
            let q1 = clamp(b.q1 + select(vec3(0), dv, k == 1), vec3(0), vec3(63));
            let cand = index_sub(mask, q0, q1, b.p);
            if cand.err < b.err {
                b = cand;
                any = true;
            }
        }
        let cand = index_sub(mask, b.q0, b.q1, 1 - b.p);
        if cand.err < b.err {
            b = cand;
            any = true;
        }
        if !any {
            break;
        }
    }
    return b;
}

// Greedy ±1 moves of the mode-6 RGB levels (endpoint, channel, +1 then -1), then flips of the
// first, second and both p-bits, each kept when it lowers the error; up to `passes` sweeps
// (`refine6`).
fn refine6(f0: Fit6, passes: u32) -> Fit6 {
    var f = f0;
    for (var it = 0u; it < passes; it++) {
        var any = false;
        for (var mv = 0; mv < 15; mv++) {
            var e = f.e;
            if mv < 12 {
                let k = mv / 6;
                let c = (mv / 2) % 3;
                let dv = select(vec3(0), vec3(select(1, -1, (mv & 1) != 0)), vec3(c) == vec3(0, 1, 2));
                e.q0 = clamp(e.q0 + select(vec3(0), dv, k == 0), vec3(0), vec3(127));
                e.q1 = clamp(e.q1 + select(vec3(0), dv, k == 1), vec3(0), vec3(127));
            } else {
                if mv != 13 {
                    e.p0 = 1 - e.p0;
                }
                if mv != 12 {
                    e.p1 = 1 - e.p1;
                }
            }
            e.v0 = 2 * e.q0 + e.p0;
            e.v1 = 2 * e.q1 + e.p1;
            let cand = project6(e);
            if cand.err < f.err {
                f = cand;
                any = true;
            }
        }
        if !any {
            break;
        }
    }
    return f;
}

// Writes the low n bits of v at bit position pos of a 128-bit block.
fn put(o: ptr<function, array<u32, 4>>, pos: u32, v: u32, n: u32) {
    let w = pos >> 5u;
    let b = pos & 31u;
    (*o)[w] |= v << b;
    if b + n > 32u {
        (*o)[w + 1u] |= v >> (32u - b);
    }
}

fn pack6(f: Fit6) -> array<u32, 4> {
    var idx = f.idx;
    // The anchor (texel 0) index must be below 8: otherwise swap the endpoints and invert.
    var q0 = vec4<u32>(vec3<u32>(f.e.q0), 127u);
    var q1 = vec4<u32>(vec3<u32>(f.e.q1), 127u);
    var p0 = u32(f.e.p0);
    var p1 = u32(f.e.p1);
    if idx[0] > 7 {
        let tq = q0;
        q0 = q1;
        q1 = tq;
        let tp = p0;
        p0 = p1;
        p1 = tp;
        for (var i = 0u; i < 16u; i++) {
            idx[i] = 15 - idx[i];
        }
    }
    // Mode 6 = bit 6; R0 7, R1 14, G0 21, G1 28, B0 35, B1 42, A0 49, A1 56; P0 63, P1 64;
    // index 0 (3 bits) at 65, index i >= 1 at 64 + 4i.
    var o: array<u32, 4>;
    o[0] = (1u << 6u) | (q0.x << 7u) | (q1.x << 14u) | (q0.y << 21u) | ((q1.y & 15u) << 28u);
    o[1] = (q1.y >> 4u) | (q0.z << 3u) | (q1.z << 10u) | (q0.w << 17u) | (q1.w << 24u) | (p0 << 31u);
    o[2] = p1 | (u32(idx[0]) << 1u);
    o[3] = 0u;
    for (var i = 1u; i < 8u; i++) {
        o[2] |= u32(idx[i]) << (4u * i);
    }
    for (var i = 8u; i < 16u; i++) {
        o[3] |= u32(idx[i]) << (4u * (i - 8u));
    }
    return o;
}

// Mode 1: mode bits 10, partition (6), RGB endpoints channel-major (subset, endpoint; 6 bits),
// the two shared p-bits, then 3-bit indices (2 bits at the anchors). An anchor index of 4 or more
// swaps its subset's endpoints and inverts its indices.
fn pack1(part: u32, s0: Sub, s1: Sub) -> array<u32, 4> {
    let anchor = ANCHOR2[part];
    let code = PART2[part];
    let sw0 = s0.idx[0] >= 4;
    let sw1 = s1.idx[anchor] >= 4;
    let e = array<array<vec3<i32>, 2>, 2>(
        array(select(s0.q0, s0.q1, sw0), select(s0.q1, s0.q0, sw0)),
        array(select(s1.q0, s1.q1, sw1), select(s1.q1, s1.q0, sw1)),
    );
    var idx: array<i32, 16>;
    for (var i = 0u; i < 16u; i++) {
        let in1 = ((code >> i) & 1u) != 0u;
        let j = select(s0.idx[i], s1.idx[i], in1);
        idx[i] = select(j, 7 - j, select(sw0, sw1, in1));
    }
    var o = array<u32, 4>(0u, 0u, 0u, 0u);
    put(&o, 0u, 2u, 2u);
    put(&o, 2u, part, 6u);
    var pos = 8u;
    for (var c = 0; c < 3; c++) {
        for (var s = 0; s < 2; s++) {
            for (var k = 0; k < 2; k++) {
                put(&o, pos, u32(e[s][k][c]), 6u);
                pos += 6u;
            }
        }
    }
    put(&o, pos, u32(s0.p), 1u);
    put(&o, pos + 1u, u32(s1.p), 1u);
    pos += 2u;
    for (var i = 0u; i < 16u; i++) {
        let n = select(3u, 2u, i == 0u || i == anchor);
        put(&o, pos, u32(idx[i]), n);
        pos += n;
    }
    return o;
}

// One subset's 3-bit ranking estimate and its scale shift (`rank_fits`): n S = n Σttᵀ - Σt Σtᵀ
// (n texels, sums s, products q0 = (rr, rg, rb) and q1 = (gg, gb, bb)) cut to 10 bits by its
// largest diagonal, one power step from the axis cut to 5 bits, λ = floor(v1·v2 / |v1|²), then
// floor(16 (49 tr - 48 λ) / n).
fn subset_est(n: i32, s: vec3<i32>, q0: vec3<i32>, q1: vec3<i32>, ax: vec3<i32>) -> vec2<i32> {
    let c0 = n * q0 - s.x * s;
    let c1 = vec3(c0.y, n * q1.x - s.y * s.y, n * q1.y - s.y * s.z);
    let c2 = vec3(c0.z, c1.z, n * q1.z - s.z * s.z);
    let sh = shrink(max(c0.x, max(c1.y, c2.z)), 9);
    let r0 = c0 >> vec3(sh);
    let r1 = c1 >> vec3(sh);
    let r2 = c2 >> vec3(sh);
    var v1 = vec3(dot(r0, ax), dot(r1, ax), dot(r2, ax));
    v1 = v1 >> vec3(shrink(abs_max(v1), 4));
    let v2 = vec3(dot(r0, v1), dot(r1, v1), dot(r2, v1));
    let lam = max(floor_div(dot(v1, v2), max(dot(v1, v1), 1)), 0);
    let x = max(49 * (r0.x + r1.y + r2.z) - 48 * lam, 0);
    return vec2(16 * x / n, i32(sh));
}

// The K best 2-subset partitions for 3-bit indices by the subsets' estimates, each scaled back
// by its shift and summed with one f32 rounding (the products are exact), as `rank_fits`.
fn rank_fits(ax: vec3<i32>, parts: u32) -> array<u32, 4> {
    var tq0 = vec3(0);
    var tq1 = vec3(0);
    for (var i = 0u; i < 16u; i++) {
        tq0 += t[i].x * t[i];
        tq1 += vec3(t[i].y * t[i].y, t[i].y * t[i].z, t[i].z * t[i].z);
    }
    var best = array<f32, 4>(3e38, 3e38, 3e38, 3e38);
    var best_p = array<u32, 4>(0u, 0u, 0u, 0u);
    for (var p = 0u; p < 64u; p++) {
        let bits = PART2[p];
        var s = vec3(0);
        var q0 = vec3(0);
        var q1 = vec3(0);
        for (var i = 0u; i < 16u; i++) {
            if (bits & (1u << i)) != 0u {
                s += t[i];
                q0 += t[i].x * t[i];
                q1 += vec3(t[i].y * t[i].y, t[i].y * t[i].z, t[i].z * t[i].z);
            }
        }
        let n1 = i32(countOneBits(bits));
        let e0 = subset_est(16 - n1, sum - s, tq0 - q0, tq1 - q1, ax);
        let e1 = subset_est(n1, s, q0, q1, ax);
        var e = f32(e0.x) * f32(1 << u32(e0.y)) + f32(e1.x) * f32(1 << u32(e1.y));
        var id = p;
        for (var k = 0u; k < parts; k++) {
            if e < best[k] {
                let ob = best[k];
                let op = best_p[k];
                best[k] = e;
                best_p[k] = id;
                e = ob;
                id = op;
            }
        }
    }
    return best_p;
}

fn encode() -> array<u32, 4> {
    // Covariance scaled by 16.
    var c = array<vec3<i32>, 3>();
    for (var i = 0u; i < 16u; i++) {
        c[0] += t[i].x * t[i];
        c[1] += t[i].y * t[i];
        c[2] += t[i].z * t[i];
    }
    c[0] = 16 * c[0] - sum.x * sum;
    c[1] = 16 * c[1] - sum.y * sum;
    c[2] = 16 * c[2] - sum.z * sum;
    let v = axis(c, P.power);
    // The axis cut to 7 bits; endpoints in 1/16 units at the extreme projections.
    let ws = v >> vec3(shrink(abs_max(v), 6));
    var pmin = dot(t[0], ws);
    var pmax = pmin;
    for (var i = 1u; i < 16u; i++) {
        let p = dot(t[i], ws);
        pmin = min(pmin, p);
        pmax = max(pmax, p);
    }
    let pm = dot(sum, ws);
    let n2 = dot(ws, ws);
    let hi = clamp(sum + floor_div3(ws * (16 * pmax - pm), n2), vec3(0), vec3(4080));
    let lo = clamp(sum + floor_div3(ws * (16 * pmin - pm), n2), vec3(0), vec3(4080));
    var f = project6(quantize(lo, hi));
    // Least-squares refits, each kept when it lowers the error.
    let refits = P.stages & 255u;
    for (var r = 0u; r < refits; r++) {
        var w: array<i32, 16>;
        for (var i = 0u; i < 16u; i++) {
            w[i] = W4[f.idx[i]];
        }
        let ls = ls_ends(w, 0xffffu, 16, sum);
        if !ls.ok {
            break;
        }
        let cand = project6(quantize(ls.lo, ls.hi));
        if cand.err >= f.err {
            break;
        }
        f = cand;
    }
    // Mode 1 for blocks with error left and some spread, on the K partitions nearest (in
    // Hamming distance, either polarity) to the split by the side of the mean along the axis.
    let parts = (P.stages >> 8u) & 255u;
    let two_refits = (P.stages >> 20u) & 15u;
    let busy = i32(P.busy) < f.err && 1024 < max(c[0].x, max(c[1].y, c[2].z));
    if parts == 0u || !busy {
        return pack6(f);
    }
    var best_p = array<u32, 4>(0u, 0u, 0u, 0u);
    if ((P.stages >> 16u) & 15u) != 0u {
        best_p = rank_fits(v, parts);
    } else {
        var bits = 0u;
        for (var i = 0u; i < 16u; i++) {
            if dot(16 * t[i] - sum, v) > 0 {
                bits |= 1u << i;
            }
        }
        var best = array<i32, 4>(99, 99, 99, 99);
        for (var p = 0u; p < 64u; p++) {
            let h = i32(countOneBits(bits ^ PART2[p]));
            var d = min(h, 16 - h);
            var id = p;
            for (var k = 0u; k < parts; k++) {
                if d < best[k] {
                    let ob = best[k];
                    let op = best_p[k];
                    best[k] = d;
                    best_p[k] = id;
                    d = ob;
                    id = op;
                }
            }
        }
    }
    var win_err = 0;
    var win_part = 0u;
    var w0: Sub;
    var w1: Sub;
    for (var k = 0u; k < parts; k++) {
        let part = best_p[k];
        let s0 = encode_two(~PART2[part] & 0xffffu, two_refits);
        let s1 = encode_two(PART2[part], two_refits);
        let e = s0.err + s1.err;
        if k == 0u || e < win_err {
            win_err = e;
            win_part = part;
            w0 = s0;
            w1 = s1;
        }
    }
    // ±1 refinement of the winner (it never raises the error, so the winner stays).
    let refine = (P.stages >> 24u) & 15u;
    if win_err < f.err {
        if refine > 0u {
            w0 = refine_sub(~PART2[win_part] & 0xffffu, w0, refine);
            w1 = refine_sub(PART2[win_part], w1, refine);
        }
        return pack1(win_part, w0, w1);
    }
    if refine > 0u && (P.stages >> 28u) != 0u {
        f = refine6(f, refine);
    }
    return pack6(f);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if i >= P.count {
        return;
    }
    let g = P.first + i;
    load(g);
    let o = encode();
    for (var k = 0u; k < 4u; k++) {
        out[4u * g + k] = o[k];
    }
}
