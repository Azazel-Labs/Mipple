// bcn_harness: runs third-party BCn block encoders on a PNG and writes a DX10 DDS, so
// `mipple score` and scripts/head2head.py can grade them exactly like mipple's own output.
//
// Usage: bcn_harness <encoder> <in.png> <out.dds> [threads]
//   encoder:
//     bc7f:<fastest|faster|fast|default|partial|nonanalytical>   Binomial bc7f (Basis Universal)
//     ispc_bc7:<ultrafast|veryfast|fast|basic|slow>               Intel ISPC Texture Compressor
//     ispc_bc1 | ispc_bc4 | ispc_bc5
//     icbc:<0-9>                                                  Ignacio Castano's icbc (BC1)
//     stb_bc1 | stb_bc1hq | stb_bc4 | stb_bc5                     stb_dxt
// threads (default 1): block rows are split across this many threads.
// Prints "secs=<seconds>": wall time of the block-encoding loop only (no PNG load, no file write).
//
// BC4 encodes the red channel, BC5 red + green; BC1 is opaque RGB (3-colour mode allowed where the
// encoder supports it, as in mipple and rgbcx; scoring is RGB-only).
//
// Build: tools/competitors/CMakeLists.txt.

#include "lodepng.h"

#ifdef HAVE_BC7F
#include "basisu_transcoder.h"
#include "basisu_transcoder_internal.h"
#endif
#ifdef HAVE_ISPC
#include "ispc_texcomp.h"
#endif
#define ICBC_IMPLEMENTATION
#include "icbc.h"
#define STB_DXT_IMPLEMENTATION
#include "stb_dxt.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

struct Image
{
	uint32_t w = 0, h = 0;       // source size
	uint32_t pw = 0, ph = 0;     // padded to a multiple of 4 (edge replicated)
	std::vector<uint8_t> rgba;   // padded, pw * ph * 4
	void block(uint32_t bx, uint32_t by, uint8_t out[64]) const
	{
		for (uint32_t i = 0; i < 16; i++)
			memcpy(out + i * 4, &rgba[((by * 4 + (i >> 2)) * pw + bx * 4 + (i & 3)) * 4], 4);
	}
};

static void put32(std::vector<uint8_t>& v, uint32_t x)
{
	for (int i = 0; i < 4; i++) v.push_back((uint8_t)(x >> (8 * i)));
}

static bool write_dds(const char* path, uint32_t w, uint32_t h, uint32_t dxgi, const std::vector<uint8_t>& blocks)
{
	std::vector<uint8_t> out = { 'D', 'D', 'S', ' ' };
	uint32_t hdr[31] = {};
	hdr[0] = 124;
	hdr[1] = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000;
	hdr[2] = h;
	hdr[3] = w;
	hdr[4] = (uint32_t)blocks.size();
	hdr[6] = 1;
	hdr[18] = 32;
	hdr[19] = 0x4;
	hdr[20] = 0x30315844; // "DX10"
	hdr[26] = 0x1000;
	for (uint32_t v : hdr) put32(out, v);
	for (uint32_t v : { dxgi, 3u, 0u, 1u, 0u }) put32(out, v);
	out.insert(out.end(), blocks.begin(), blocks.end());
	FILE* f = fopen(path, "wb");
	if (!f) return false;
	const bool ok = fwrite(out.data(), 1, out.size(), f) == out.size();
	fclose(f);
	return ok;
}

// Runs fn(row_begin, row_end) over block rows on `threads` threads.
static void parallel_rows(uint32_t rows, int threads, const std::function<void(uint32_t, uint32_t)>& fn)
{
	threads = std::max(1, std::min<int>(threads, (int)rows));
	if (threads == 1)
	{
		fn(0, rows);
		return;
	}
	std::vector<std::thread> pool;
	for (int t = 0; t < threads; t++)
	{
		const uint32_t a = rows * t / threads, b = rows * (t + 1) / threads;
		pool.emplace_back(fn, a, b);
	}
	for (auto& th : pool) th.join();
}

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "usage: bcn_harness <encoder> <in.png> <out.dds> [threads]\n");
		return 1;
	}
	const std::string enc = argv[1];
	const int threads = argc > 4 ? atoi(argv[4]) : 1;
	const std::string family = enc.substr(0, enc.find(':'));
	const std::string param = enc.find(':') == std::string::npos ? "" : enc.substr(enc.find(':') + 1);

	Image img;
	{
		std::vector<unsigned char> src;
		if (lodepng::decode(src, img.w, img.h, argv[2]) != 0)
		{
			fprintf(stderr, "failed to load %s\n", argv[2]);
			return 1;
		}
		img.pw = (img.w + 3) & ~3u;
		img.ph = (img.h + 3) & ~3u;
		img.rgba.resize(img.pw * img.ph * 4);
		for (uint32_t y = 0; y < img.ph; y++)
			for (uint32_t x = 0; x < img.pw; x++)
				memcpy(&img.rgba[(y * img.pw + x) * 4], &src[(std::min(y, img.h - 1) * img.w + std::min(x, img.w - 1)) * 4], 4);
	}
	const uint32_t bw = img.pw / 4, bh = img.ph / 4;

	uint32_t block_bytes = 16, dxgi = 98;
	std::function<void(uint32_t, uint32_t, uint8_t*)> encode_rows; // rows [a, b) into dst (start of row a)
	std::function<void()> init = [] {};

	if (family == "bc7f")
	{
#ifdef HAVE_BC7F
		uint32_t flags;
		if (param == "fastest") flags = basist::bc7f::cPackBC7FlagDefaultFastest;
		else if (param == "faster") flags = basist::bc7f::cPackBC7FlagDefaultFaster;
		else if (param == "fast") flags = basist::bc7f::cPackBC7FlagDefaultFast;
		else if (param == "default" || param.empty()) flags = basist::bc7f::cPackBC7FlagDefault;
		else if (param == "partial") flags = basist::bc7f::cPackBC7FlagDefaultPartiallyAnalytical;
		else if (param == "nonanalytical") flags = basist::bc7f::cPackBC7FlagDefaultNonAnalytical;
		else { fprintf(stderr, "unknown bc7f preset %s\n", param.c_str()); return 2; }
		init = [] { basist::basisu_transcoder_init(); };
		encode_rows = [&, flags](uint32_t a, uint32_t b, uint8_t* dst) {
			for (uint32_t by = a; by < b; by++)
				for (uint32_t bx = 0; bx < bw; bx++)
				{
					uint8_t px[64];
					img.block(bx, by, px);
					basist::color_rgba c[16];
					for (int i = 0; i < 16; i++) c[i].set(px[i * 4], px[i * 4 + 1], px[i * 4 + 2], px[i * 4 + 3]);
					basist::bc7f::fast_pack_bc7_auto_rgba(dst + ((by - a) * bw + bx) * 16, c, flags);
				}
		};
#else
		fprintf(stderr, "built without bc7f\n");
		return 2;
#endif
	}
	else if (family.rfind("ispc_", 0) == 0)
	{
#ifdef HAVE_ISPC
		bool alpha = false;
		for (size_t i = 3; i < img.rgba.size(); i += 4) alpha |= img.rgba[i] != 255;
		static bc7_enc_settings s;
		if (family == "ispc_bc7")
		{
			if (param == "ultrafast") alpha ? GetProfile_alpha_ultrafast(&s) : GetProfile_ultrafast(&s);
			else if (param == "veryfast") alpha ? GetProfile_alpha_veryfast(&s) : GetProfile_veryfast(&s);
			else if (param == "fast") alpha ? GetProfile_alpha_fast(&s) : GetProfile_fast(&s);
			else if (param == "basic" || param.empty()) alpha ? GetProfile_alpha_basic(&s) : GetProfile_basic(&s);
			else if (param == "slow") alpha ? GetProfile_alpha_slow(&s) : GetProfile_slow(&s);
			else { fprintf(stderr, "unknown ispc_bc7 profile %s\n", param.c_str()); return 2; }
		}
		else if (family == "ispc_bc1") { block_bytes = 8; dxgi = 71; }
		else if (family == "ispc_bc4") { block_bytes = 8; dxgi = 80; }
		else if (family == "ispc_bc5") { block_bytes = 16; dxgi = 83; }
		else { fprintf(stderr, "unknown encoder %s\n", enc.c_str()); return 2; }
		// BC4 reads an R8 surface and BC5 an RG8 surface. The plane is allocated before timing but
		// filled inside the timed region, row range by row range, so ISPC starts from the same RGBA
		// image as every other encoder here (they all read RGBA inside their timed loop).
		static std::vector<uint8_t> planes;
		const uint32_t bpp = family == "ispc_bc4" ? 1 : family == "ispc_bc5" ? 2 : 4;
		uint8_t* base = img.rgba.data();
		if (bpp < 4)
		{
			planes.resize(img.pw * img.ph * bpp);
			base = planes.data();
		}
		encode_rows = [&, family, bpp, base](uint32_t a, uint32_t b, uint8_t* dst) {
			// Plain strided copies the compiler can vectorize.
			const uint8_t* src = img.rgba.data();
			uint8_t* pl = planes.data();
			const uint32_t i0 = a * 4 * img.pw, i1 = b * 4 * img.pw;
			if (bpp == 1)
				for (uint32_t i = i0; i < i1; i++) pl[i] = src[i * 4];
			else if (bpp == 2)
				for (uint32_t i = i0; i < i1; i++)
				{
					uint16_t v;
					memcpy(&v, src + i * 4, 2);
					memcpy(pl + i * 2, &v, 2);
				}
			rgba_surface surf{ base + a * 4 * img.pw * bpp, (int32_t)img.pw, (int32_t)((b - a) * 4), (int32_t)(img.pw * bpp) };
			if (family == "ispc_bc7") CompressBlocksBC7(&surf, dst, &s);
			else if (family == "ispc_bc1") CompressBlocksBC1(&surf, dst);
			else if (family == "ispc_bc4") CompressBlocksBC4(&surf, dst);
			else CompressBlocksBC5(&surf, dst);
		};
#else
		fprintf(stderr, "built without ISPC Texture Compressor\n");
		return 2;
#endif
	}
	else if (family == "icbc")
	{
		block_bytes = 8;
		dxgi = 71;
		const int level = param.empty() ? 8 : atoi(param.c_str());
		init = [] { icbc::init(icbc::Decoder_D3D10); };
		encode_rows = [&, level](uint32_t a, uint32_t b, uint8_t* dst) {
			const float weights[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
			const float cw[3] = { 1, 1, 1 };
			for (uint32_t by = a; by < b; by++)
				for (uint32_t bx = 0; bx < bw; bx++)
				{
					uint8_t px[64];
					img.block(bx, by, px);
					float f[64];
					for (int i = 0; i < 64; i++) f[i] = px[i] / 255.0f;
					icbc::compress_bc1((icbc::Quality)level, f, weights, cw, true, true, dst + ((by - a) * bw + bx) * 8);
				}
		};
	}
	else if (family.rfind("stb_", 0) == 0)
	{
		if (family == "stb_bc1" || family == "stb_bc1hq") { block_bytes = 8; dxgi = 71; }
		else if (family == "stb_bc4") { block_bytes = 8; dxgi = 80; }
		else if (family == "stb_bc5") { block_bytes = 16; dxgi = 83; }
		else { fprintf(stderr, "unknown encoder %s\n", enc.c_str()); return 2; }
		encode_rows = [&, family](uint32_t a, uint32_t b, uint8_t* dst) {
			for (uint32_t by = a; by < b; by++)
				for (uint32_t bx = 0; bx < bw; bx++)
				{
					uint8_t px[64];
					img.block(bx, by, px);
					uint8_t* o = dst + ((by - a) * bw + bx) * block_bytes;
					if (family == "stb_bc1") stb_compress_dxt_block(o, px, 0, STB_DXT_NORMAL);
					else if (family == "stb_bc1hq") stb_compress_dxt_block(o, px, 0, STB_DXT_HIGHQUAL);
					else
					{
						uint8_t r[16], rg[32];
						for (int i = 0; i < 16; i++) { r[i] = px[i * 4]; rg[i * 2] = px[i * 4]; rg[i * 2 + 1] = px[i * 4 + 1]; }
						if (family == "stb_bc4") stb_compress_bc4_block(o, r);
						else stb_compress_bc5_block(o, rg);
					}
				}
		};
	}
	else
	{
		fprintf(stderr, "unknown encoder %s\n", enc.c_str());
		return 2;
	}

	init();
	std::vector<uint8_t> blocks(bw * bh * block_bytes);
	const auto t0 = std::chrono::high_resolution_clock::now();
	parallel_rows(bh, threads, [&](uint32_t a, uint32_t b) { encode_rows(a, b, &blocks[a * bw * block_bytes]); });
	const double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

	if (!write_dds(argv[3], img.w, img.h, dxgi, blocks))
	{
		fprintf(stderr, "failed to write %s\n", argv[3]);
		return 1;
	}
	printf("secs=%.6f\n", secs);
	return 0;
}
