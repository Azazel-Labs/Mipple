"""Minimal DX10 DDS writer for a single mip of BC1/BC4/BC5/BC7 blocks (used by run_ue_oodle.py)."""
import struct

# DXGI_FORMAT values (UNORM, non-sRGB)
DXGI = {"bc1": 71, "bc4": 80, "bc5": 83, "bc7": 98}
BLOCK_BYTES = {"bc1": 8, "bc4": 8, "bc5": 16, "bc7": 16}


def level_bytes(fmt, w, h):
    return ((w + 3) // 4) * ((h + 3) // 4) * BLOCK_BYTES[fmt]


def write_dds(path, fmt, w, h, blocks):
    size = level_bytes(fmt, w, h)
    if len(blocks) != size:
        raise ValueError(f"{path}: {len(blocks)} bytes, expected {size} for {fmt} {w}x{h}")
    DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PIXELFORMAT, DDSD_LINEARSIZE = 0x1, 0x2, 0x4, 0x1000, 0x80000
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    pf = struct.pack("<II4sIIIII", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
    hdr = struct.pack("<IIIIIII", 124, flags, h, w, size, 0, 1) + b"\0" * 44 + pf
    hdr += struct.pack("<IIIII", 0x1000, 0, 0, 0, 0)  # caps: TEXTURE
    dx10 = struct.pack("<IIIII", DXGI[fmt], 3, 0, 1, 0)  # TEXTURE2D, arraySize 1
    with open(path, "wb") as f:
        f.write(b"DDS " + hdr + dx10 + bytes(blocks))
