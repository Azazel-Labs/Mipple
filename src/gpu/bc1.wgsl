// BC1 base encoder, efforts 0 and 1: one invocation per block. A port of the line fit and
// least-squares refits of src/bc1/batch.rs (encode8 with no cluster fit), which the CPU encoder
// defines in exact integers (its f32 values are integers below 2^24): the covariance cut to
// 11 bits, one power iteration, a 7-bit axis, endpoints at the inset extreme projections on the
// axis line through the mean, steps by exact thresholds, and refits by bc1::batch::ls_levels.
// Solid blocks use the single-color tables (MATCH5 / MATCH6, prepended to this source by the
// host). So the output matches the CPU encoder block for block on any GPU.

struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    count: u32,   // blocks in this dispatch
    effort: u32,  // 0: one refit kept unconditionally; 1: up to two, each kept if it helps
    first: u32,   // first block of this dispatch; block g writes output words 2g and 2g + 1
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> px: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

var<private> t: array<vec3<i32>, 16>;
var<private> sum: vec3<i32>;

// Palette index of each step (0 = c0, 1 = 1/3, 2 = 2/3, 3 = c1).
const STEP_INDEX = array<u32, 4>(0u, 2u, 3u, 1u);
const LEVELS = vec3<i32>(31, 63, 31);

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

// Bit-replicated 8-bit values of 565 levels.
fn expand(q: vec3<i32>) -> vec3<i32> {
    return vec3((q.x << 3u) | (q.x >> 2u), (q.y << 2u) | (q.y >> 4u), (q.z << 3u) | (q.z >> 2u));
}

fn pack(q: vec3<i32>) -> u32 {
    return (u32(q.x) << 11u) | (u32(q.y) << 5u) | u32(q.z);
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

// A candidate: packed endpoints in 4-color order (c0 >= c1), their decoded values, and each
// texel's step from c0 (0) to c1 (3).
struct Fit {
    c0: u32,
    c1: u32,
    a: vec3<i32>,
    b: vec3<i32>,
    step: array<i32, 16>,
}

// Orders two endpoints (565 levels) and assigns steps by projection: with d = b - a and
// n = 6 (p - a).d, step >= k when n >= (2k - 1) |d|^2.
fn assign(q0: vec3<i32>, q1: vec3<i32>) -> Fit {
    var f: Fit;
    let p0 = pack(q0);
    let p1 = pack(q1);
    let swap = p0 < p1;
    f.c0 = select(p0, p1, swap);
    f.c1 = select(p1, p0, swap);
    f.a = expand(select(q0, q1, swap));
    f.b = expand(select(q1, q0, swap));
    let d = f.b - f.a;
    let n2 = dot(d, d);
    let base = dot(f.a, d) * 6;
    for (var i = 0u; i < 16u; i++) {
        let n = dot(t[i], d) * 6 - base;
        var s = 0;
        if n2 > 0 {
            s = select(0, 1, n >= n2) + select(0, 1, n >= 3 * n2) + select(0, 1, n >= 5 * n2);
        }
        f.step[i] = s;
    }
    return f;
}

// Squared error against the decoder palette: ((3 - s) a + s b) / 3 per channel.
fn error(f: Fit) -> i32 {
    var e = 0;
    for (var i = 0u; i < 16u; i++) {
        let s = f.step[i];
        let d = t[i] - ((3 - s) * f.a + s * f.b) / 3;
        e += dot(d, d);
    }
    return e;
}

struct Refit {
    ok: bool,
    q0: vec3<i32>,
    q1: vec3<i32>,
}

// Least-squares 565 levels for the fit's steps, exactly bc1::batch::ls_levels.
fn refit(f: Fit) -> Refit {
    var j = 0;
    var jj = 0;
    var jx = vec3(0);
    for (var i = 0u; i < 16u; i++) {
        let s = f.step[i];
        j += s;
        jj += s * s;
        jx += s * t[i];
    }
    var r: Refit;
    let s00 = 144 - 6 * j + jj;
    let s01 = 3 * j - jj;
    let d = s00 * jj - s01 * s01;
    r.ok = d >= 41;
    if !r.ok {
        return r;
    }
    let s0x = 3 * sum - jx;
    let num0 = jj * s0x - s01 * jx;
    let num1 = s00 * jx - s01 * s0x;
    for (var c = 0; c < 3; c++) {
        let l = LEVELS[c];
        let n0 = 6 * l * num0[c] + 255 * d;
        let n1 = 6 * l * num1[c] + 255 * d;
        r.q0[c] = select(min(n0 / (510 * d), l), 0, n0 <= 0);
        r.q1[c] = select(min(n1 / (510 * d), l), 0, n1 <= 0);
    }
    return r;
}

// The line fit's two endpoints as 565 levels (see line_points in src/bc1/batch.rs).
fn line_fit() -> array<vec3<i32>, 2> {
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
    let cs = shrink(max(diag.x, max(diag.y, diag.z)), 10);
    let s = array<vec3<i32>, 3>(c[0] >> vec3(cs), c[1] >> vec3(cs), c[2] >> vec3(cs));
    // One power iteration from the largest-diagonal row (the first on ties).
    var v = s[0];
    if diag.x < diag.y {
        v = s[1];
    }
    if max(diag.x, diag.y) < diag.z {
        v = s[2];
    }
    var w = vec3(dot(s[0], v), dot(s[1], v), dot(s[2], v));
    if all(w == vec3(0)) {
        w = vec3(1);
    }
    let ws = w >> vec3(shrink(max(abs(w.x), max(abs(w.y), abs(w.z))), 6));
    var pmin = dot(t[0], ws);
    var pmax = pmin;
    for (var i = 1u; i < 16u; i++) {
        let p = dot(t[i], ws);
        pmin = min(pmin, p);
        pmax = max(pmax, p);
    }
    let pm = dot(sum, ws);
    let n2 = dot(ws, ws);
    let th = 15 * pmax + pmin - pm;
    let tl = pmax + 15 * pmin - pm;
    var e = array<vec3<i32>, 2>();
    for (var k = 0; k < 3; k++) {
        let l = LEVELS[k];
        let hi = clamp(sum[k] + floor_div(ws[k] * th, n2), 0, 4080);
        let lo = clamp(sum[k] + floor_div(ws[k] * tl, n2), 0, 4080);
        e[0][k] = (2 * l * hi + 4080) / 8160;
        e[1][k] = (2 * l * lo + 4080) / 8160;
    }
    return e;
}

// Solid blocks: the single-color tables' endpoints, then the best palette entry (bc1 finalize).
fn solid() -> vec2<u32> {
    let v = vec3<u32>(t[0]);
    let m = vec3(MATCH5[v.x], MATCH6[v.y], MATCH5[v.z]);
    let qa = vec3<i32>(m & vec3(0xffu));
    let qb = vec3<i32>(m >> vec3(8u));
    var c0 = pack(qa);
    var c1 = pack(qb);
    var a = expand(qa);
    var b = expand(qb);
    if c0 < c1 {
        let tc = c0;
        c0 = c1;
        c1 = tc;
        let ta = a;
        a = b;
        b = ta;
    }
    var pal = array<vec3<i32>, 4>(a, b, (2 * a + b) / 3, (a + 2 * b) / 3);
    if c0 <= c1 {
        pal[2] = (a + b) / 2;
        pal[3] = vec3(0);
    }
    var bi = 0u;
    var be = 1 << 30;
    for (var k = 0u; k < 4u; k++) {
        let d = t[0] - pal[k];
        let e = dot(d, d);
        if e < be {
            be = e;
            bi = k;
        }
    }
    // Every texel is the same color, so every index is bi.
    return vec2(c0 | (c1 << 16u), bi * 0x55555555u);
}

fn encode() -> vec2<u32> {
    let e = line_fit();
    var f = assign(e[0], e[1]);
    if P.effort == 0u {
        let r = refit(f);
        if r.ok {
            f = assign(r.q0, r.q1);
        }
    } else {
        var err = error(f);
        for (var it = 0; it < 2; it++) {
            let r = refit(f);
            if !r.ok {
                break;
            }
            let g = assign(r.q0, r.q1);
            let ge = error(g);
            if ge >= err {
                break;
            }
            f = g;
            err = ge;
        }
    }
    var bits = 0u;
    if f.c0 > f.c1 {
        for (var i = 0u; i < 16u; i++) {
            bits |= STEP_INDEX[f.step[i]] << (2u * i);
        }
    }
    return vec2(f.c0 | (f.c1 << 16u), bits);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if i >= P.count {
        return;
    }
    let g = P.first + i;
    load(g);
    // Solid when every texel has the first one's color (the CPU's zero covariance trace).
    var flat = true;
    for (var k = 1u; k < 16u; k++) {
        flat = flat && all(t[k] == t[0]);
    }
    var o: vec2<u32>;
    if flat {
        o = solid();
    } else {
        o = encode();
    }
    out[2u * g] = o.x;
    out[2u * g + 1u] = o.y;
}
