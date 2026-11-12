//! mipple's BC6H decoder against Microsoft's texconv (DirectXTex), the reference for the D3D
//! specification: random blocks of all 14 modes, unsigned (BC6H_UF16) and signed (BC6H_SF16), must
//! decode to the same half floats. The expected decodes are stored as one 32-bit hash per block in
//! `tests/data/bc6h_texconv.txt`; `cargo test --test bc6h_reference -- --ignored regen` rebuilds
//! that file with `third_party/bin/texconv.exe` (Windows).

use mipple::bc6h::{self, HdrBlock};

/// Random blocks per mode.
const PER_MODE: usize = 32;
/// The 14 mode values in D3D order.
const MODE_VALUES: [u128; 14] = [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15];
const GOLDEN: &str = "tests/data/bc6h_texconv.txt";

/// Blocks in raster order of a (4 * PER_MODE) x 56 image: one row of blocks per mode.
fn blocks() -> Vec<[u8; 16]> {
    let mut seed = 0x6a09_e667_f3bc_c908u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut out = Vec::new();
    for v in MODE_VALUES {
        let bits = if v < 2 { 2 } else { 5 };
        for _ in 0..PER_MODE {
            let r = u128::from(next()) | u128::from(next()) << 64;
            out.push(((r & !((1u128 << bits) - 1)) | v).to_le_bytes());
        }
    }
    out
}

fn hash(b: &HdrBlock) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for t in b {
        for c in t {
            for byte in c.to_le_bytes() {
                h = (h ^ u32::from(byte)).wrapping_mul(0x0100_0193);
            }
        }
    }
    h
}

fn golden() -> Vec<(bool, u32)> {
    let text = std::fs::read_to_string(GOLDEN).expect("golden file");
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let (s, h) = l.split_once(' ').expect("'u|s hash'");
            (s == "s", u32::from_str_radix(h.trim(), 16).expect("hex hash"))
        })
        .collect()
}

#[test]
fn decoder_matches_texconv() {
    let g = golden();
    let bs = blocks();
    assert_eq!(g.len(), 2 * bs.len(), "golden file out of date: regenerate it");
    let mut bad = Vec::new();
    for (k, &(signed, want)) in g.iter().enumerate() {
        let i = k % bs.len();
        assert_eq!(signed, k >= bs.len());
        let got = hash(&bc6h::decode(&bs[i], signed));
        if got != want {
            bad.push(format!(
                "{} mode {} block {}",
                if signed { "SF16" } else { "UF16" },
                i / PER_MODE + 1,
                i % PER_MODE
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} blocks differ: {:?}",
        bad.len(),
        g.len(),
        &bad[..bad.len().min(20)]
    );
}

/// Every block mipple's packer writes unpacks to the same fields, and packing what it unpacks
/// gives back the same bits for every random block of every mode (no unused or shared bits).
#[test]
fn random_blocks_repack_exactly() {
    for b in blocks() {
        let u = bc6h::unpack(&b).expect("valid mode");
        assert_eq!(bc6h::pack(&u), b);
    }
}

/// A DDS of `blocks` with DXGI format `dxgi` (95 BC6H_UF16, 96 BC6H_SF16).
fn dds(blocks: &[[u8; 16]], dxgi: u32) -> Vec<u8> {
    let tex = mipple::EncodedTexture {
        format: mipple::Format::Bc6h,
        srgb: false,
        width: 4 * PER_MODE as u32,
        height: 4 * MODE_VALUES.len() as u32,
        levels: vec![mipple::Level {
            width: 4 * PER_MODE as u32,
            height: 4 * MODE_VALUES.len() as u32,
            data: blocks.concat(),
        }],
    };
    let mut out = mipple::dds::write(&tex).unwrap();
    out[128..132].copy_from_slice(&dxgi.to_le_bytes());
    out
}

#[test]
#[ignore = "regenerates the golden file with texconv (Windows)"]
fn regen() {
    let dir = std::env::temp_dir().join("mipple_bc6h_regen");
    std::fs::create_dir_all(&dir).unwrap();
    let bs = blocks();
    let (w, h) = (4 * PER_MODE, 4 * MODE_VALUES.len());
    let mut lines = vec![
        "# texconv (DirectXTex) decodes of tests/bc6h_reference.rs's random blocks: u = BC6H_UF16, s = BC6H_SF16,".into(),
        "# FNV-1a 32 of each block's 16 texels x RGB half floats. Regenerate: cargo test --test bc6h_reference -- --ignored regen".into(),
    ];
    for (tag, dxgi) in [("u", 95u32), ("s", 96)] {
        let src = dir.join(format!("bc6h_{tag}.dds"));
        std::fs::write(&src, dds(&bs, dxgi)).unwrap();
        let out = dir.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let st = std::process::Command::new("third_party/bin/texconv.exe")
            .args(["-nologo", "-y", "-f", "R16G16B16A16_FLOAT", "-o"])
            .arg(&out)
            .arg(&src)
            .status()
            .expect("running texconv");
        assert!(st.success());
        let dec = std::fs::read(out.join(format!("bc6h_{tag}.dds"))).unwrap();
        let off = if &dec[84..88] == b"DX10" { 148 } else { 128 };
        let px = &dec[off..off + w * h * 8];
        let half = |x: usize, y: usize, c: usize| {
            let i = (y * w + x) * 8 + c * 2;
            u16::from_le_bytes([px[i], px[i + 1]])
        };
        for i in 0..bs.len() {
            let (bx, by) = (i % PER_MODE, i / PER_MODE);
            let b: HdrBlock = std::array::from_fn(|t| std::array::from_fn(|c| half(bx * 4 + t % 4, by * 4 + t / 4, c)));
            lines.push(format!("{tag} {:08x}", hash(&b)));
        }
    }
    std::fs::create_dir_all("tests/data").unwrap();
    std::fs::write(GOLDEN, lines.join("\n") + "\n").unwrap();
}
