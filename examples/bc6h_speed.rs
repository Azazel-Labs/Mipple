//! BC6H encode speed: `cargo run --release --example bc6h_speed -- EFFORT REPEATS a.hdr b.hdr ...`.
//! Prints the best of REPEATS encode times per image, summed, and nanoseconds per block (set
//! RAYON_NUM_THREADS=1 and pin the process for single-core numbers).

use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let effort: u8 = args[0].parse().expect("effort");
    let reps: usize = args[1].parse().expect("repeats");
    let mut total = 0.0;
    let mut blocks = 0usize;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    // Q=1 also reports mean mPSNR against the representable source, as `mipple score` does.
    let quality = std::env::var("Q").is_ok_and(|v| v == "1");
    let mut psnr = 0.0;
    for path in &args[2..] {
        let s = mipple::hdr::HdrSurface::load(Path::new(path), true).expect("load");
        let mut best = f64::INFINITY;
        for r in 0..reps {
            let t = Instant::now();
            let out = mipple::bc6h::encode_surface(&s, effort);
            best = best.min(t.elapsed().as_secs_f64());
            if r == 0 {
                for &b in &out {
                    hash = (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
                }
                if quality {
                    let src = s.representable();
                    let dec = mipple::bc6h::decode_surface(&out, s.width, s.height);
                    psnr += mipple::hdr::mpsnr(&src, &dec, mipple::hdr::exposure_range(&src));
                }
            }
            std::hint::black_box(out);
        }
        total += best;
        blocks += s.block_count();
    }
    let n = (args.len() - 2) as f64;
    println!(
        "effort {effort}: {total:.4} s, {:.1} ns/block, {blocks} blocks, hash {hash:016x}{}",
        total * 1e9 / blocks as f64,
        if quality {
            format!(", mPSNR {:.3}", psnr / n)
        } else {
            String::new()
        }
    );
}
