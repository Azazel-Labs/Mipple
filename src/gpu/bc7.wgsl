// BC7 base encoder, effort 0 (mode 6 only): one invocation per block. A port of the mode-6 stage
// of src/bc7/fast.rs (encode8 with no refits and no multi-subset search), which the CPU encoder
// defines in exact integers: the covariance cut to 11 bits, power iterations on a 10-bit vector, a
// 7-bit axis, endpoints at the extreme projections on the axis line through the mean (in 1/16
// units), the p-bit with the lower endpoint error, and each index the better of the two palette
// entries around the projection. So the output matches the CPU encoder block for block on any
// GPU. Blocks with alpha are re-encoded by the host (the CPU sends them to its scalar encoder).

struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    count: u32,  // blocks in this dispatch
    power: u32,  // power iterations
    first: u32,  // first block of this dispatch; block g writes output words 4g..4g + 3
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> px: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

var<private> t: array<vec3<i32>, 16>;
var<private> sum: vec3<i32>;

// Mode-6 interpolation weights (4-bit indices).
const W4 = array<i32, 16>(0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64);

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

// max(0, floor(log2 m) - bits): the shift that cuts m >= 0 to at most bits + 1 bits.
fn shrink(m: i32, bits: i32) -> u32 {
    return u32(max(firstLeadingBit(m) - bits, 0));
}

fn abs_max(v: vec3<i32>) -> i32 {
    let a = abs(v);
    return max(a.x, max(a.y, a.z));
}

// A quantized endpoint pair: 7-bit RGB levels, p-bits, 8-bit values 2q + p.
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

// Squared RGB error of texel i against the palette entry of weight w: a + floor((w d + 32) / 64)
// = ((64 - w) a + w b + 32) >> 6, the decoder's interpolation.
fn entry_err(i: u32, a: vec3<i32>, d: vec3<i32>, w: i32) -> i32 {
    let e = t[i] - a - ((w * d + 32) >> vec3(6u));
    return dot(e, e);
}

fn encode() -> array<u32, 4> {
    // Covariance scaled by 16, cut to 11 bits by the largest diagonal.
    var c = array<vec3<i32>, 3>();
    for (var i = 0u; i < 16u; i++) {
        c[0] += t[i].x * t[i];
        c[1] += t[i].y * t[i];
        c[2] += t[i].z * t[i];
    }
    c[0] = 16 * c[0] - sum.x * sum;
    c[1] = 16 * c[1] - sum.y * sum;
    c[2] = 16 * c[2] - sum.z * sum;
    let diag = vec3(c[0].x, c[1].y, c[2].z);
    let cs = vec3(shrink(max(diag.x, max(diag.y, diag.z)), 10));
    let s = array<vec3<i32>, 3>(c[0] >> cs, c[1] >> cs, c[2] >> cs);
    // Power iterations from the largest-diagonal row (the first on ties), each step cut to 10 bits.
    var v = s[0];
    if diag.x < diag.y {
        v = s[1];
    }
    if max(diag.x, diag.y) < diag.z {
        v = s[2];
    }
    for (var it = 0u; it < P.power; it++) {
        let w = vec3(dot(s[0], v), dot(s[1], v), dot(s[2], v));
        v = w >> vec3(shrink(abs_max(w), 9));
    }
    if all(v == vec3(0)) {
        v = vec3(1);
    }
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
    var hi = vec3(0);
    var lo = vec3(0);
    for (var k = 0; k < 3; k++) {
        hi[k] = clamp(sum[k] + floor_div(ws[k] * (16 * pmax - pm), n2), 0, 4080);
        lo[k] = clamp(sum[k] + floor_div(ws[k] * (16 * pmin - pm), n2), 0, 4080);
    }
    let qa = quant(lo);
    let qb = quant(hi);
    var e: Ends;
    e.q0 = qa.xyz;
    e.p0 = qa.w;
    e.q1 = qb.xyz;
    e.p1 = qb.w;
    e.v0 = 2 * e.q0 + e.p0;
    e.v1 = 2 * e.q1 + e.p1;
    // Indices: j = floor(15 (x - a)·d / |d|²) clamped to 0..=14, then j or j + 1, whichever
    // entry is nearer (j on ties).
    let d = e.v1 - e.v0;
    let dd = max(dot(d, d), 1);
    var idx = array<u32, 16>();
    for (var i = 0u; i < 16u; i++) {
        let j = clamp(floor_div(15 * dot(t[i] - e.v0, d), dd), 0, 14);
        let e0 = entry_err(i, e.v0, d, W4[j]);
        let e1 = entry_err(i, e.v0, d, W4[j + 1]);
        idx[i] = u32(select(j, j + 1, e1 < e0));
    }
    // The anchor (texel 0) index must be below 8: otherwise swap the endpoints and invert.
    var q0 = vec4<u32>(vec3<u32>(e.q0), 127u);
    var q1 = vec4<u32>(vec3<u32>(e.q1), 127u);
    var p0 = u32(e.p0);
    var p1 = u32(e.p1);
    if idx[0] > 7u {
        let tq = q0;
        q0 = q1;
        q1 = tq;
        let tp = p0;
        p0 = p1;
        p1 = tp;
        for (var i = 0u; i < 16u; i++) {
            idx[i] = 15u - idx[i];
        }
    }
    // Mode 6 = bit 6; R0 7, R1 14, G0 21, G1 28, B0 35, B1 42, A0 49, A1 56; P0 63, P1 64;
    // index 0 (3 bits) at 65, index i >= 1 at 64 + 4i.
    var o: array<u32, 4>;
    o[0] = (1u << 6u) | (q0.x << 7u) | (q1.x << 14u) | (q0.y << 21u) | ((q1.y & 15u) << 28u);
    o[1] = (q1.y >> 4u) | (q0.z << 3u) | (q1.z << 10u) | (q0.w << 17u) | (q1.w << 24u) | (p0 << 31u);
    o[2] = p1 | (idx[0] << 1u);
    o[3] = 0u;
    for (var i = 1u; i < 8u; i++) {
        o[2] |= idx[i] << (4u * i);
    }
    for (var i = 8u; i < 16u; i++) {
        o[3] |= idx[i] << (4u * (i - 8u));
    }
    return o;
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
