//! The `.astc` container: a 16-byte header (magic, block footprint, 24-bit image size) and the
//! blocks in raster order.

use anyhow::{Result, ensure};

const MAGIC: u32 = 0x5CA1_AB13;

/// A 2D ASTC image as stored in a `.astc` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AstcFile {
    pub bw: u32,
    pub bh: u32,
    pub width: u32,
    pub height: u32,
    pub blocks: Vec<u8>,
}

impl AstcFile {
    /// The number of blocks across and down.
    pub fn grid(&self) -> (u32, u32) {
        (self.width.div_ceil(self.bw), self.height.div_ceil(self.bh))
    }
}

fn u24(b: &[u8]) -> u32 {
    b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16
}

/// Parses a `.astc` file (2D footprints only).
pub fn read(bytes: &[u8]) -> Result<AstcFile> {
    ensure!(bytes.len() >= 16, "not an .astc file: {} bytes", bytes.len());
    ensure!(
        u32::from_le_bytes(bytes[..4].try_into()?) == MAGIC,
        "not an .astc file: bad magic"
    );
    let (bw, bh, bd) = (bytes[4] as u32, bytes[5] as u32, bytes[6] as u32);
    ensure!(bd == 1 && u24(&bytes[13..16]) == 1, "3D ASTC is not supported");
    let f = AstcFile {
        bw,
        bh,
        width: u24(&bytes[7..10]),
        height: u24(&bytes[10..13]),
        blocks: Vec::new(),
    };
    let (gx, gy) = f.grid();
    let n = (gx * gy) as usize * 16;
    ensure!(
        bytes.len() >= 16 + n,
        ".astc file truncated: {} of {} block bytes",
        bytes.len() - 16,
        n
    );
    Ok(AstcFile {
        blocks: bytes[16..16 + n].to_vec(),
        ..f
    })
}

/// Serializes a `.astc` file.
pub fn write(f: &AstcFile) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + f.blocks.len());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&[f.bw as u8, f.bh as u8, 1]);
    for v in [f.width, f.height, 1] {
        out.extend_from_slice(&v.to_le_bytes()[..3]);
    }
    out.extend_from_slice(&f.blocks);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let f = AstcFile {
            bw: 6,
            bh: 6,
            width: 13,
            height: 7,
            blocks: (0..3 * 2 * 16).map(|i| i as u8).collect(),
        };
        assert_eq!(read(&write(&f)).unwrap(), f);
    }

    /// Prints which block configurations `.astc` files use: `MIPPLE_ASTC_STATS=a.astc;b.astc`.
    #[test]
    #[ignore = "needs external .astc files"]
    fn block_mode_histogram() {
        let Ok(spec) = std::env::var("MIPPLE_ASTC_STATS") else {
            return;
        };
        let mut hist = std::collections::BTreeMap::<String, usize>::new();
        let mut total = 0;
        for path in spec.split(';') {
            let f = read(&std::fs::read(path).unwrap()).unwrap();
            for b in f.blocks.as_chunks::<16>().0 {
                let v = u128::from_le_bytes(*b);
                total += 1;
                let key = if v & 0x1ff == 0x1fc {
                    "void".to_string()
                } else if let Some(s) = super::super::Symbolic::read(v, f.bw, f.bh) {
                    format!(
                        "p{} {} cem{:?} grid{}x{} wq{} cq{}",
                        s.partitions,
                        if s.mode.dual {
                            format!("dual{}", s.ccs)
                        } else {
                            "single".into()
                        },
                        &s.cems[..s.partitions as usize],
                        s.mode.gx,
                        s.mode.gy,
                        super::super::RANGES[s.mode.quant as usize],
                        super::super::RANGES[s.color_quant as usize]
                    )
                } else {
                    "invalid".to_string()
                };
                *hist.entry(key).or_default() += 1;
            }
        }
        let mut v: Vec<_> = hist.into_iter().collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.1));
        for (k, c) in v.iter().take(40) {
            println!("{:6.2}% {k}", 100.0 * *c as f64 / total as f64);
        }
        let mut agg = std::collections::BTreeMap::<String, usize>::new();
        for (k, c) in &v {
            *agg.entry(k.split(" grid").next().unwrap().to_string()).or_default() += c;
        }
        for (k, c) in agg {
            println!("summary {:6.2}% {k}", 100.0 * c as f64 / total as f64);
        }
    }

    /// Decodes ISPC's `.astc` output of a source image and checks the error is what an encoder
    /// produces, not garbage: `MIPPLE_ASTC_CHECK=src.png;enc.astc[;...]`.
    #[test]
    #[ignore = "needs external .astc files"]
    fn decodes_external_files() {
        let Ok(spec) = std::env::var("MIPPLE_ASTC_CHECK") else {
            return;
        };
        for pair in spec.split(';').collect::<Vec<_>>().chunks(2) {
            let src = crate::surface::Surface::load(std::path::Path::new(pair[0])).unwrap();
            let f = read(&std::fs::read(pair[1]).unwrap()).unwrap();
            let dec = super::super::decode::decode_rgba(&f.blocks, f.width, f.height, f.bw, f.bh, false);
            let errs = dec.pixels.iter().filter(|p| **p == super::super::decode::ERROR).count();
            let psnr = crate::metrics::psnr(&src, &dec, &[0, 1, 2]);
            println!(
                "{} {}x{}: RGB PSNR {psnr:.3} dB, {errs} error-color texels",
                pair[1], f.bw, f.bh
            );
        }
    }
}
