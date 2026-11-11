#!/usr/bin/env python3
"""Render docs/img/rdo.webp, the README animation of what BC7 RDO does to an image.

    python scripts/rdo_anim.py [--exe target/release/mipple.exe] [--out docs/img/rdo.webp]

Encodes Kodak 23 twice with the mipple binary (BC7 effort 8 without RDO, and BC7 at the default
effort with lambda 1), both with raw blocks (--zstd 0) so the block streams can be read back, then
sweeps the image block by block. Every number on screen is measured, not illustrative:

  * each block is classed by comparing the two streams: an exact copy of an earlier RDO block
    (cyan), a changed block sharing a run of >= 5 aligned bytes with one of the previous 1024
    blocks (magenta), or kept (grey); the loupe draws a link from the block it reuses;
  * size = Zstd-19 size of the stream whose first k blocks come from the RDO encode and the rest
    from the plain encode, in bits per texel; PSNR is computed on the same mix of decoded pixels.

The output is a lossless animated WebP, so the magnified pixels are the decoder's exact output.
Needs numpy, pillow, zstandard, ffmpeg (with libwebp) and the Segoe UI fonts from Windows.
"""
import argparse
import math
import os
import struct
import subprocess
import tempfile

import numpy as np
import zstandard as zstd
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--exe', default=os.path.join(ROOT, 'target', 'release', 'mipple.exe' if os.name == 'nt' else 'mipple'))
ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'img', 'rdo.webp'))
ap.add_argument('--frames', help='also write the PNG frames to this directory')
args = ap.parse_args()
SRC = os.path.join(ROOT, 'test-images', 'kodak', 'kodim23.png')
TMP = tempfile.mkdtemp(prefix='rdo_anim_')


def encode(name, *flags):
    ktx = os.path.join(TMP, name + '.ktx2')
    subprocess.run([args.exe, 'encode', SRC, '-o', ktx, '-f', 'bc7', '--zstd', '0', '--no-mips', *flags],
                   check=True, capture_output=True)
    png = os.path.join(TMP, name + '.png')
    subprocess.run([args.exe, 'decode', ktx, '-o', png], check=True, capture_output=True)
    b = open(ktx, 'rb').read()
    w, h = struct.unpack_from('<II', b, 20)
    off, ln, _ = struct.unpack_from('<QQQ', b, 80)  # level 0 entry of the KTX2 level index
    return w, h, np.frombuffer(b[off:off + ln], np.uint8).reshape(-1, 16), png


SHARE, BACK = 5, 1024


def classify(base, rdo):
    """Per block: 0 kept, 1 changed and sharing >= SHARE aligned bytes with one of the previous BACK
    blocks, 2 exact copy of an earlier block. src = the earlier block it links to (-1 none)."""
    n = len(rdo); cls = np.zeros(n, np.int8); src = np.full(n, -1); run = np.zeros(n, np.int8)
    last = {}
    for i in range(n):
        k = rdo[i].tobytes()
        if k in last and i - last[k] <= 4096:
            cls[i], src[i], run[i] = 2, last[k], 16
        elif (base[i] != rdo[i]).any() and i:
            lo = max(0, i - BACK)
            eq = rdo[lo:i] == rdo[i]                     # (m,16)
            best = np.zeros(i - lo, np.int8); cur = np.zeros(i - lo, np.int8)
            for b in range(16):
                cur = np.where(eq[:, b], cur + 1, 0); best = np.maximum(best, cur)
            j = int(np.argmax(best[::-1])); j = i - 1 - j   # nearest among the longest
            if best.max() >= SHARE: cls[i], src[i], run[i] = 1, j, best.max()
        last[k] = i
    return cls, src, run

W_, H_ = 936, 576
BG = (11, 12, 28)
INK, INK2, MUTED = (240, 242, 255), (176, 180, 214), (110, 114, 150)
CYAN, MAG, VIO, GREY = (34, 228, 240), (255, 62, 165), (160, 92, 255), (120, 124, 150)
F = 'C:/Windows/Fonts/'
def font(name, size): return ImageFont.truetype(F + name, size)
f_title, f_sub, f_body, f_small = font('segoeuib.ttf', 26), font('segoeui.ttf', 16), font('segoeui.ttf', 15), font('segoeui.ttf', 13)
f_num, f_big = font('segoeuisl.ttf', 22), font('segoeuib.ttf', 30)


w, h, b0, png0 = encode('plain', '-e', '8', '-l', '0')
_, _, b1, png1 = encode('rdo', '-l', '1')
src = np.asarray(Image.open(SRC).convert('RGB')).astype(np.float32)
dec0 = np.asarray(Image.open(png0).convert('RGB')).astype(np.float32)
dec1 = np.asarray(Image.open(png1).convert('RGB')).astype(np.float32)
cls, srcb, _ = classify(b0, b1)
BW, BH = w // 4, h // 4
N = BW * BH
cctx = zstd.ZstdCompressor(level=19)


def bpt(k): return len(cctx.compress(b1[:k].tobytes() + b0[k:].tobytes())) * 8 / (w * h)


def mask_img(k):
    m = (np.arange(N) < k).reshape(BH, BW)
    return np.kron(m, np.ones((4, 4), bool))[..., None]


def psnr(img): return 10 * math.log10(255 ** 2 / np.mean((img - src) ** 2))


# Layout
S = 0.75
IX, IY, IW, IH = 24, 84, int(w * S), int(h * S)          # main image 576x384
LX, LY, LB, LPX = IX + IW + 24, IY, 9, 32                 # loupe: 9x9 blocks at 32 px
LS = LB * LPX                                             # 288
BARY = IY + IH + 30

# Main image layers at display size.
def disp(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((IW, IH), Image.LANCZOS)
src_d, dec1_d = disp(src), disp(dec1)
lum = (0.2126 * dec1[..., 0] + 0.7152 * dec1[..., 1] + 0.0722 * dec1[..., 2])[..., None]
tint = np.zeros((BH, BW, 3), np.float32); alpha = np.zeros((BH, BW, 1), np.float32)
c2 = cls.reshape(BH, BW)
tint[c2 == 2], alpha[c2 == 2] = CYAN, 0.62
tint[c2 == 1], alpha[c2 == 1] = MAG, 0.50
tint[c2 == 0], alpha[c2 == 0] = (40, 42, 70), 0.30
T = np.kron(tint, np.ones((4, 4, 1))); A = np.kron(alpha, np.ones((4, 4, 1)))
mapped = lum * 0.55 * (1 - A) + T * A
mapped_d = disp(mapped)

def sweep_frame(k):
    """Main image with blocks < k mapped, the rest original."""
    m = np.kron((np.arange(N) < k).reshape(BH, BW), np.ones((3, 3), np.uint8)) * 255
    return Image.composite(mapped_d, src_d, Image.fromarray(m.astype(np.uint8)).resize((IW, IH)))

# Loupe path: block-row of the scan -> loupe centre column (cosine-eased keypoints).
KEYS = [(0, 150), (34, 150), (50, 127), (62, 53), (92, 56), (108, 120), (128, 150)]
def loupe_cx(r):
    for (r0, c0), (r1, c1) in zip(KEYS, KEYS[1:]):
        if r <= r1:
            t = (r - r0) / max(r1 - r0, 1)
            return c0 + (c1 - c0) * (0.5 - 0.5 * math.cos(math.pi * min(max(t, 0), 1)))
    return KEYS[-1][1]

def loupe(k, row_f, fade=1.0, at=None):
    """Magnified 9x9 block window just above the scan line, drawn at 2x then reduced."""
    U = 2
    cx = loupe_cx(row_f)
    bx0 = int(round(min(max(cx - LB / 2, 0), BW - LB)))
    by0 = int(round(min(max(row_f - LB + 1.5, 0), BH - LB)))
    if at: bx0, by0 = at
    px = slice(by0 * 4, (by0 + LB) * 4), slice(bx0 * 4, (bx0 + LB) * 4)
    m = mask_img(k)[px]
    pix = np.where(m, dec1[px], src[px])
    im = Image.fromarray(pix.astype(np.uint8)).resize((LS * U, LS * U), Image.NEAREST).convert('RGBA')
    ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    q = LPX * U
    # dim processed kept blocks slightly so links read
    for j in range(LB):
        for i in range(LB):
            b = (by0 + j) * BW + bx0 + i
            if b >= k: continue
            x, y = i * q, j * q
            c = cls[b]
            if c == 2: dr.rectangle([x + 3, y + 3, x + q - 4, y + q - 4], outline=CYAN + (255,), width=5)
            elif c == 1: dr.rectangle([x + 3, y + 3, x + q - 4, y + q - 4], outline=MAG + (235,), width=4)
    # grid
    for i in range(LB + 1):
        dr.line([(i * q, 0), (i * q, LS * U)], fill=(11, 12, 28, 110), width=2)
        dr.line([(0, i * q), (LS * U, i * q)], fill=(11, 12, 28, 110), width=2)
    # links: source block -> this block
    for j in range(LB):
        for i in range(LB):
            b = (by0 + j) * BW + bx0 + i
            if b >= k or cls[b] == 0: continue
            s = srcb[b]
            if not (bx0 <= s % BW < bx0 + LB and by0 <= s // BW < by0 + LB): continue
            sx, sy = (s % BW - bx0 + 0.5) * q, (s // BW - by0 + 0.5) * q
            tx, ty = (i + 0.5) * q, (j + 0.5) * q
            col = CYAN if cls[b] == 2 else MAG
            a = 255 if cls[b] == 2 else 170
            L = math.hypot(tx - sx, ty - sy)
            if L < 1: continue
            ux, uy = (tx - sx) / L, (ty - sy) / L
            ex, ey = tx - ux * q * 0.18, ty - uy * q * 0.18
            dr.line([(sx, sy), (ex, ey)], fill=(11, 12, 28, 200), width=9)
            dr.line([(sx, sy), (ex, ey)], fill=col + (a,), width=5 if cls[b] == 2 else 3)
            hl = 18
            p1 = (ex - ux * hl - uy * hl * 0.6, ey - uy * hl + ux * hl * 0.6)
            p2 = (ex - ux * hl + uy * hl * 0.6, ey - uy * hl - ux * hl * 0.6)
            dr.polygon([(ex, ey), p1, p2], fill=col + (a,))
            dr.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=col + (a,))
    # scan line inside the loupe
    srow, scol = k // BW, k % BW
    if by0 <= srow < by0 + LB:
        y0, y1 = (srow - by0) * q, (srow - by0 + 1) * q
        xc = min(max((scol - bx0) * q, 0), LS * U)
        dr.line([(xc, y0), (LS * U, y0)], fill=INK + (255,), width=4)
        dr.line([(0, y1), (xc, y1)], fill=INK + (255,), width=4)
        dr.line([(xc, y0), (xc, y1)], fill=INK + (255,), width=4)
    im = Image.alpha_composite(im, ov) if fade >= 1 else Image.blend(im, Image.alpha_composite(im, ov), fade)
    return im.resize((LS, LS), Image.LANCZOS).convert('RGB'), (bx0, by0)

def frame(main, lp, lbox, k, stats, caption, final=0.0):
    im = Image.new('RGB', (W_, H_), BG); dr = ImageDraw.Draw(im)
    dr.text((IX, 22), 'mipple', font=f_title, fill=INK)
    tw = dr.textlength('mipple', font=f_title)
    dr.text((IX + tw + 12, 31), 'BC7 rate-distortion optimization  ·  λ = 1  ·  Kodak 23', font=f_sub, fill=INK2)
    cw = dr.textlength(caption, font=f_sub)
    dr.text((W_ - 24 - cw, 31), caption, font=f_sub, fill=CYAN if final < 0.5 else INK)
    im.paste(main, (IX, IY))
    if lp is not None:
        bx0, by0 = lbox
        r = [IX + bx0 * 4 * S, IY + by0 * 4 * S, IX + (bx0 + LB) * 4 * S, IY + (by0 + LB) * 4 * S]
        dr.rectangle(r, outline=INK, width=1)
        im.paste(lp, (LX, LY))
        dr.rectangle([LX - 1, LY - 1, LX + LS, LY + LS], outline=(70, 74, 120), width=1)
    # scan line on the main image
    if 0 < k < N:
        srow, scol = k // BW, k % BW
        y0, y1, xc = IY + srow * 3, IY + srow * 3 + 3, IX + scol * 3
        dr.line([(xc, y0), (IX + IW - 1, y0)], fill=INK, width=1)
        dr.line([(IX, y1), (xc, y1)], fill=INK, width=1)
        dr.rectangle([xc - 2, y0 - 2, xc + 4, y1 + 2], fill=INK)
    # legend
    n_done = max(k, 1)
    rows = [(CYAN, 'exact copy of an earlier block', (cls[:k] == 2).sum()),
            (MAG, 'shares bytes with an earlier one', (cls[:k] == 1).sum()),
            (GREY, 'kept as encoded', (cls[:k] == 0).sum())]
    ly = LY + LS + 18
    for col, label, cnt in rows:
        dr.rounded_rectangle([LX, ly + 4, LX + 12, ly + 16], radius=3, fill=col)
        dr.text((LX + 20, ly), label, font=f_body, fill=INK2)
        pct = f'{100 * cnt / n_done:.0f}%' if k else '–'
        dr.text((LX + LS - dr.textlength(pct, font=f_body), ly), pct, font=f_body, fill=INK)
        ly += 24
    # size bar
    b_base, b_now, p_base, p_now = stats
    x0, x1 = IX + 118, IX + IW + 24 + LS - 150
    sc = (x1 - x0) / 8.0
    dr.text((IX, BARY - 2), 'size', font=f_body, fill=INK2)
    dr.text((IX, BARY + 16), 'zstd bits/texel', font=f_small, fill=MUTED)
    dr.text((x0 + b_base * sc - 78, BARY + 32), 'without RDO', font=f_small, fill=MUTED)
    dr.rounded_rectangle([x0, BARY + 4, x0 + b_base * sc, BARY + 30], radius=4, outline=(80, 84, 130), width=1)
    gx1 = x0 + b_now * sc
    bar = Image.new('RGB', (max(int(gx1 - x0), 1), 26))
    g = np.linspace(0, 1, bar.width)[None, :, None]
    bar = Image.fromarray((np.array(CYAN) * (1 - g) + np.array(VIO) * g).repeat(26, 0).astype(np.uint8))
    m = Image.new('L', bar.size, 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, bar.width - 1, 25], radius=4, fill=255)
    im.paste(bar, (x0, BARY + 4), m)
    dr.text((x1 + 14, BARY), f'{b_now:.2f}', font=f_num, fill=INK)
    red = 100 * (1 - b_now / b_base)
    dr.text((x1 + 72, BARY + 6), f'−{red:.0f}%' if red >= 0.5 else 'base', font=f_body, fill=CYAN if red >= 0.5 else MUTED)
    dr.text((IX, BARY + 42), 'quality', font=f_body, fill=INK2)
    dr.text((x0, BARY + 42), f'PSNR {p_now:.1f} dB', font=f_body, fill=INK)
    dr.text((x0 + 120, BARY + 42), f'(without RDO: {p_base:.1f} dB)', font=f_body, fill=MUTED)
    return im

# Timeline
FPS = 20
INTRO, SWEEP, XF, HOLD, BACK = 18, 120, 11, 56, 8
b_base, p_base = bpt(0), psnr(dec0)
print('base', b_base, p_base, 'rdo', bpt(N), psnr(dec1))
FRAME_DIR = args.frames or os.path.join(TMP, 'frames')
os.makedirs(FRAME_DIR, exist_ok=True)
fi = 0


def save(im):
    global fi
    im.save(os.path.join(FRAME_DIR, f'f{fi:04d}.png'))
    fi += 1

lp0, lb0 = loupe(0, 0)
for _ in range(INTRO):
    save(frame(src_d, lp0, lb0, 0, (b_base, b_base, p_base, p_base), '1  plain BC7 encode, best quality per block'))
for f in range(1, SWEEP + 1):
    t = f / SWEEP
    k = int(round(N * t))
    row_f = k / BW
    mixed = np.where(mask_img(k), dec1, dec0)
    lp, lb = loupe(k, row_f)
    save(frame(sweep_frame(k), lp, lb, k, (b_base, bpt(k), p_base, psnr(mixed)), '2  RDO: re-encode blocks so zstd finds repeats'))
    if f % 25 == 0: print('sweep', f)
b_rdo, p_rdo = bpt(N), psnr(dec1)
lpN, lbN = loupe(N, 0, at=(123, 38))
for f in range(XF):
    t = 0.5 - 0.5 * math.cos(math.pi * (f + 1) / XF)
    save(frame(Image.blend(mapped_d, dec1_d, t), lpN, lbN, N, (b_base, b_rdo, p_base, p_rdo), '3  decoded result', final=t))
for f in range(HOLD):
    save(frame(dec1_d, lpN, lbN, N, (b_base, b_rdo, p_base, p_rdo), '3  decoded result', final=1))
for f in range(BACK):
    t = (f + 1) / BACK
    save(frame(Image.blend(dec1_d, src_d, t), lpN, lbN, N, (b_base, b_rdo, p_base, p_rdo), '3  decoded result', final=1))
# ffmpeg's libwebp_anim encodes lossless frame deltas ~10% smaller than Pillow's, and in seconds.
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS), '-i', os.path.join(FRAME_DIR, 'f%04d.png'),
                '-c:v', 'libwebp_anim', '-lossless', '1', '-compression_level', '6', '-loop', '0', args.out], check=True)
print(f'{fi} frames -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)')
