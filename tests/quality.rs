//! Quality floors and RDO behavior on the synthetic corpus.

mod common;

use common::corpus;
use mipple::pipeline::{decode_blocks, encode_base, flatten};
use mipple::rdo::{self, RdoParams};
use mipple::surface::block_sse;
use mipple::{Format, metrics};

fn psnr_for(format: Format, effort: u8, img: &mipple::Surface) -> f64 {
    let (_, blocks) = encode_base(img, format, effort);
    let dec = decode_blocks(format, &flatten(format, &blocks), img.width, img.height, false, false);
    metrics::psnr(img, &dec, &format.metric_channels(img.has_alpha()))
}

#[test]
fn base_encoders_meet_quality_floors() {
    // Floors are deliberately conservative; the benchmark tracks the real numbers.
    for (name, img) in corpus(128) {
        let bc7 = psnr_for(Format::Bc7, 6, &img);
        let bc1 = psnr_for(Format::Bc1, 6, &img);
        let bc4 = psnr_for(Format::Bc4, 6, &img);
        assert!(bc7 > 38.0, "{name}: BC7 {bc7:.2} dB");
        assert!(bc4 > 38.0, "{name}: BC4 {bc4:.2} dB");
        if name != "ui_alpha" {
            assert!(bc1 > 30.0, "{name}: BC1 {bc1:.2} dB");
            assert!(bc7 > bc1, "{name}: BC7 {bc7:.2} should beat BC1 {bc1:.2}");
        }
    }
}

#[test]
fn higher_effort_is_not_worse() {
    for (name, img) in corpus(64) {
        for format in [Format::Bc1, Format::Bc4, Format::Bc7] {
            let lo = psnr_for(format, 0, &img);
            let hi = psnr_for(format, 3, &img);
            assert!(hi + 0.05 >= lo, "{name} {format}: effort 3 {hi:.2} < effort 0 {lo:.2}");
        }
    }
}

#[test]
fn rdo_trades_quality_for_size() {
    for (name, img) in corpus(128) {
        // ASTC has no RDO (and its blocks are not 4x4).
        for format in Format::ALL.into_iter().filter(|f| !f.is_astc()) {
            let w = format.error_weights();
            let (src, base) = encode_base(&img, format, 1);
            let base_data = flatten(format, &base);
            let base_z = zstd::bulk::compress(&base_data, 19).unwrap().len();
            let base_dec = decode_blocks(format, &base_data, img.width, img.height, false, false);
            let ch = format.metric_channels(img.has_alpha());
            let base_psnr = metrics::psnr(&img, &base_dec, &ch);

            // lambda = 0 is a no-op.
            let mut same = base.clone();
            rdo::optimize(format, &src, &mut same, &w, &RdoParams::with_lambda(0.0));
            assert_eq!(same, base);

            let params = RdoParams::with_lambda(4.0);
            let mut blocks = base.clone();
            let stats = rdo::optimize(format, &src, &mut blocks, &w, &params);
            let data = flatten(format, &blocks);
            let z = zstd::bulk::compress(&data, 19).unwrap().len();
            let dec = decode_blocks(format, &data, img.width, img.height, false, false);
            let psnr = metrics::psnr(&img, &dec, &ch);
            assert!(z <= base_z, "{name} {format}: RDO grew zstd size {base_z} -> {z}");
            assert!(
                psnr <= base_psnr + 1.0,
                "{name} {format}: PSNR {base_psnr:.2} -> {psnr:.2}"
            );
            assert!(stats.est_bits_after <= stats.est_bits_before + 1e-6);

            // Safety invariant: no block exceeds ratio² × its base error + the floor.
            let n = format.block_bytes();
            let channels = w.iter().filter(|&&x| x > 0).count() as f64;
            let ratio2 = (params.max_rms_ratio as f64).powi(2);
            let min2 = (params.min_rms as f64).powi(2);
            let (mut a, mut b) = ([[0u8; 4]; 16], [[0u8; 4]; 16]);
            for i in 0..src.len() {
                format.decode_block(&base[i][..n], &mut a);
                format.decode_block(&blocks[i][..n], &mut b);
                let e0 = block_sse(&src[i], &a, &w) as f64;
                let e1 = block_sse(&src[i], &b, &w) as f64;
                assert!(
                    e1 <= e0 * ratio2 + 16.0 * channels * min2 + 0.5,
                    "{name} {format} block {i}: {e0} -> {e1}"
                );
            }
        }
    }
}
