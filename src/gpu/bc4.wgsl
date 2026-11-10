// BC4 / BC5 base encoder: one invocation per (block, channel). A port of src/bc4/batch.rs:
// effort 0 = proj8, effort 1 = fast8, efforts 2-9 = exact8 with endpoint window radius P.radius.
// Everything is integer math: palettes and errors (the CPU encoder's floats hold the same small
// integers exactly), and the projections and least-squares fits, which the CPU encoder also
// defines in exact integers (bc4::ls_steps). So the output matches the CPU encoder block for block
// on any GPU, whatever its float rounding.

struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    count: u32,     // invocations in this dispatch
    channels: u32,  // 1 = BC4, 2 = BC5
    effort: u32,
    radius: i32,
    first: u32,     // first invocation of this dispatch; invocation g is channel g % channels
                    // of block g / channels, and writes output words 2g and 2g + 1
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> px: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

var<private> x: array<i32, 16>;

// Step toward e1 (0..7) of each 8-value-mode index: index 0 is e0, 1 is e1, k >= 2 is step k - 1.
const STEP8 = array<i32, 8>(0, 7, 1, 2, 3, 4, 5, 6);
// Palette index of each projection step: step 0 is e0, step 7 is e1.
const STEP_INDEX = array<u32, 8>(0u, 2u, 3u, 4u, 5u, 6u, 7u, 1u);

fn load(block: u32, c: u32) {
    let bx = block % P.blocks_x;
    let by = block / P.blocks_x;
    for (var i = 0u; i < 16u; i++) {
        // Edge texels repeat when the image is not a multiple of 4 (Surface::read_block).
        let tx = min(bx * 4u + (i & 3u), P.width - 1u);
        let ty = min(by * 4u + (i >> 2u), P.height - 1u);
        x[i] = i32((px[ty * P.width + tx] >> (8u * c)) & 0xffu);
    }
}

// The decoder's palette: 8-value mode when a > b, else 6-value mode with literal 0 and 255.
fn palette(a: i32, b: i32) -> array<i32, 8> {
    var p = array<i32, 8>(a, b, a, a, a, a, 0, 255);
    if a > b {
        for (var k = 1; k < 7; k++) {
            p[k + 1] = ((7 - k) * a + k * b) / 7;
        }
    } else {
        for (var k = 1; k < 5; k++) {
            p[k + 1] = ((5 - k) * a + k * b) / 5;
        }
    }
    return p;
}

struct Fit {
    a: i32,
    b: i32,
    err: i32,
    idx: array<u32, 16>,
}

// Nearest palette entry per texel (lowest index on ties) and the summed squared error.
fn assign(a: i32, b: i32) -> Fit {
    let p = palette(a, b);
    var f: Fit;
    f.a = a;
    f.b = b;
    f.err = 0;
    for (var i = 0u; i < 16u; i++) {
        var d = x[i] - p[0];
        var best = d * d;
        var bi = 0u;
        for (var k = 1u; k < 8u; k++) {
            d = x[i] - p[k];
            let e = d * d;
            if e < best {
                best = e;
                bi = k;
            }
        }
        f.idx[i] = bi;
        f.err += best;
    }
    return f;
}

fn error(a: i32, b: i32) -> i32 {
    let p = palette(a, b);
    var err = 0;
    for (var i = 0u; i < 16u; i++) {
        var d = x[i] - p[0];
        var best = d * d;
        for (var k = 1u; k < 8u; k++) {
            d = x[i] - p[k];
            best = min(best, d * d);
        }
        err += best;
    }
    return err;
}

// 8-value-mode indices by projection onto the line from a (step 0) to b (step 7), with the error
// against the decoder's palette; steps[i] is each texel's step:
// floor(clamp((a - x) * 7 / span + 0.5, 0, 7)) = (14 (a - x) + span) / (2 span), clamped.
var<private> steps: array<i32, 16>;

fn project(a: i32, b: i32) -> Fit {
    let span = a - b;
    var f: Fit;
    f.a = a;
    f.b = b;
    f.err = 0;
    for (var i = 0u; i < 16u; i++) {
        var j = 0;
        if span > 0 {
            j = min(max(14 * (a - x[i]) + span, 0) / (2 * span), 7);
        }
        steps[i] = j;
        let d = x[i] - (7 * a + j * (b - a)) / 7;
        f.err += d * d;
        f.idx[i] = STEP_INDEX[j];
    }
    return f;
}

// Least-squares endpoints from the steps (weight toward e1 = step / 7), rounded half up and clamped
// to 0..255, exactly as bc4::ls_steps; (0, 0) when every step is equal.
fn ls() -> vec2<i32> {
    var j = 0;
    var jj = 0;
    var jx = 0;
    var sx = 0;
    for (var i = 0u; i < 16u; i++) {
        j += steps[i];
        jj += steps[i] * steps[i];
        jx += steps[i] * x[i];
        sx += x[i];
    }
    let s00 = 784 - 14 * j + jj;
    let s01 = 7 * j - jj;
    let s0x = 7 * sx - jx;
    let d = s00 * jj - s01 * s01;
    if d <= 0 {
        return vec2(0, 0);
    }
    let na = 7 * (jj * s0x - s01 * jx);
    let nb = 7 * (s00 * jx - s01 * s0x);
    return vec2(select(min((2 * na + d) / (2 * d), 255), 0, na <= 0), select(min((2 * nb + d) / (2 * d), 255), 0, nb <= 0));
}

fn min_max() -> vec2<i32> {
    var lo = x[0];
    var hi = x[0];
    for (var i = 1u; i < 16u; i++) {
        lo = min(lo, x[i]);
        hi = max(hi, x[i]);
    }
    return vec2(lo, hi);
}

// Effort 0: min/max endpoints, indices by projection rounded at 0.45 of a step (the decoder
// truncates): floor((hi - x) * 7 / span + 0.45) = (140 (hi - x) + 9 span) / (20 span).
fn proj8() -> Fit {
    let m = min_max();
    let span = m.y - m.x;
    var f: Fit;
    f.a = m.y;
    f.b = m.x;
    for (var i = 0u; i < 16u; i++) {
        f.idx[i] = STEP_INDEX[(140 * (m.y - x[i]) + 9 * span) / max(20 * span, 1)];
    }
    return f;
}

// Effort 1: projection from min/max, then one least-squares refit kept when it lowers the error.
fn fast8() -> Fit {
    let m = min_max();
    let f = project(m.y, m.x);
    let e = ls();
    if !(e.y < e.x && f.err > 0) {
        return f;
    }
    let g = project(e.x, e.y);
    if g.err < f.err {
        return g;
    }
    return f;
}

// Efforts 2-9: exactly the scalar reference (min/max 8-value fit, 6-value fit around literal
// 0/255, two least-squares refits, then an exhaustive endpoint window of radius r).
fn exact8(r: i32) -> Fit {
    var lo = x[0];
    var hi = x[0];
    var ilo = 255;
    var ihi = 0;
    var extreme = false;
    for (var i = 0u; i < 16u; i++) {
        lo = min(lo, x[i]);
        hi = max(hi, x[i]);
        if x[i] > 0 && x[i] < 255 {
            ilo = min(ilo, x[i]);
            ihi = max(ihi, x[i]);
        } else {
            extreme = true;
        }
    }
    var best = assign(hi, lo);
    if extreme {
        let none = ihi < ilo;
        let f = assign(select(ilo, 0, none), select(ihi, 0, none));
        if f.err < best.err {
            best = f;
        }
    }
    for (var it = 0; it < 2; it++) {
        for (var i = 0u; i < 16u; i++) {
            steps[i] = STEP8[best.idx[i]];
        }
        let e = ls();
        if !(best.a > best.b && e.x > e.y && best.err > 0) {
            break;
        }
        let f = assign(e.x, e.y);
        if f.err < best.err {
            best = f;
        }
    }
    if r > 0 && best.err > 0 {
        let b0 = best.a;
        let b1 = best.b;
        let m8 = b0 > b1;
        var ba = b0;
        var bb = b1;
        var be = best.err;
        for (var d0 = -r; d0 <= r; d0++) {
            let a = b0 + d0;
            for (var d1 = -r; d1 <= r; d1++) {
                let b = b1 + d1;
                // In range and in the starting mode.
                if a < 0 || a > 255 || b < 0 || b > 255 || (a > b) != m8 {
                    continue;
                }
                let e = error(a, b);
                if e < be {
                    ba = a;
                    bb = b;
                    be = e;
                }
            }
        }
        return assign(ba, bb);
    }
    return best;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.x + gid.y * nwg.x * 64u;
    if t >= P.count {
        return;
    }
    let g = P.first + t;
    let c = g % P.channels;
    let block = g / P.channels;
    load(block, c);
    var f: Fit;
    if P.effort == 0u {
        f = proj8();
    } else if P.effort == 1u {
        f = fast8();
    } else {
        f = exact8(P.radius);
    }
    // e0, e1, then 3-bit indices: texels 0-7 and 8-15 in 24 bits each.
    var lo = 0u;
    var hi = 0u;
    for (var i = 0u; i < 8u; i++) {
        lo |= f.idx[i] << (3u * i);
        hi |= f.idx[i + 8u] << (3u * i);
    }
    let o = 2u * g;
    out[o] = u32(f.a) | (u32(f.b) << 8u) | (lo << 16u);
    out[o + 1u] = (lo >> 16u) | (hi << 8u);
}
