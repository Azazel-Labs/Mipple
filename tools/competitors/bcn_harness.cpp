// bcn_harness: runs third-party BCn block encoders on a PNG and writes a DX10 DDS, so
// `mipple score` and scripts/compete.py can grade them exactly like mipple's own output.
//
// Usage: bcn_harness <encoder> <in.png> <out.dds> [threads]
//   encoder:
//     bc7f:<fastest|faster|fast|default|partial|nonanalytical>   Binomial bc7f (Basis Universal)
//     ispc_bc7:<ultrafast|veryfast|fast|basic|slow>               Intel ISPC Texture Compressor
//     ispc_bc1 | ispc_bc4 | ispc_bc5
//     ispc_bc6h:<veryfast|fast|basic|slow|veryslow>               ISPC BC6H_UF16 from a Radiance .hdr
//     ispc_astc:<W>x<H>[:fast|alpha_fast|alpha_slow]              ISPC ASTC LDR; writes a .astc file
//     icbc:<0-9>                                                  Ignacio Castano's icbc (BC1)
//     stb_bc1 | stb_bc1hq | stb_bc4 | stb_bc5                     stb_dxt
// threads (default 1): block rows are split across this many threads.
// Prints "secs=<seconds>": wall time of the block-encoding loop only (no PNG load, no file write).
//
// BC4 encodes the red channel, BC5 red + green; BC1 is opaque RGB (3-colour mode allowed where the
// encoder supports it, as in mipple and rgbcx; scoring is RGB-only). BC6H reads a Radiance .hdr (stb_image,
// the same RGBE decode as the image crate mipple uses) and converts to half floats inside the timed loop
// (round to nearest even; negatives and NaN to 0; clamped to 65504), as mipple does.
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
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_HDR
#include "stb_image.h"

#include <algorithm>
#include <cmath>
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
	std::vector<float> rgbf;     // HDR input instead: padded, pw * ph * 3
	void block(uint32_t bx, uint32_t by, uint8_t out[64]) const
	{
		for (uint32_t i = 0; i < 16; i++)
			memcpy(out + i * 4, &rgba[((by * 4 + (i >> 2)) * pw + bx * 4 + (i & 3)) * 4], 4);
	}
};

// f32 to an unsigned half: round to nearest even, NaN and negatives to 0, clamped to 65504.
static uint16_t to_uf16(float f)
{
	if (!(f > 0.0f)) return 0;
	if (f >= 65504.0f) return 0x7bff;
	uint32_t x;
	memcpy(&x, &f, 4);
	const int e = (int)(x >> 23) - 127;
	if (e < -24) return 0;
	if (e < -14)
	{
		// Denormal half: value / 2^-24, rounded half to even.
		const float v = f * 16777216.0f;
		float r = std::nearbyint(v); // default rounding mode: nearest even
		return (uint16_t)r;
	}
	uint32_t m = x & 0x7fffff;
	uint32_t h = (uint32_t)((e + 15) << 10) | (m >> 13);
	const uint32_t rest = m & 0x1fff;
	if (rest > 0x1000 || (rest == 0x1000 && (h & 1))) h++;
	return (uint16_t)std::min<uint32_t>(h, 0x7bff);
}

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

// The .astc container: magic, block footprint (x, y, z), then the image size as three 24-bit values.
static bool write_astc(const char* path, uint32_t w, uint32_t h, uint32_t bw, uint32_t bh, const std::vector<uint8_t>& blocks)
{
	std::vector<uint8_t> out;
	put32(out, 0x5CA1AB13);
	out.push_back((uint8_t)bw);
	out.push_back((uint8_t)bh);
	out.push_back(1);
	for (uint32_t v : { w, h, 1u })
		for (int i = 0; i < 3; i++) out.push_back((uint8_t)(v >> (8 * i)));
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
	if (family == "ispc_bc6h")
	{
		int w, h, n;
		float* f = stbi_loadf(argv[2], &w, &h, &n, 3);
		if (!f)
		{
			fprintf(stderr, "failed to load %s\n", argv[2]);
			return 1;
		}
		img.w = w;
		img.h = h;
		img.pw = (img.w + 3) & ~3u;
		img.ph = (img.h + 3) & ~3u;
		img.rgbf.resize(img.pw * img.ph * 3);
		for (uint32_t y = 0; y < img.ph; y++)
			for (uint32_t x = 0; x < img.pw; x++)
				memcpy(&img.rgbf[(y * img.pw + x) * 3], &f[(std::min(y, img.h - 1) * img.w + std::min(x, img.w - 1)) * 3], 12);
		stbi_image_free(f);
	}
	else
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
	else if (family == "ispc_bc6h")
	{
#ifdef HAVE_ISPC
		static bc6h_enc_settings s;
		if (param == "veryfast") GetProfile_bc6h_veryfast(&s);
		else if (param == "fast") GetProfile_bc6h_fast(&s);
		else if (param == "basic" || param.empty()) GetProfile_bc6h_basic(&s);
		else if (param == "slow") GetProfile_bc6h_slow(&s);
		else if (param == "veryslow") GetProfile_bc6h_veryslow(&s);
		else { fprintf(stderr, "unknown ispc_bc6h profile %s\n", param.c_str()); return 2; }
		dxgi = 95;
		static std::vector<uint16_t> halves;
		halves.resize(img.pw * img.ph * 4);
		encode_rows = [&](uint32_t a, uint32_t b, uint8_t* dst) {
			for (uint32_t i = a * 4 * img.pw; i < b * 4 * img.pw; i++)
			{
				for (int c = 0; c < 3; c++) halves[i * 4 + c] = to_uf16(img.rgbf[i * 3 + c]);
				halves[i * 4 + 3] = 0x3c00;
			}
			rgba_surface surf{ (uint8_t*)&halves[a * 4 * img.pw * 4], (int32_t)img.pw, (int32_t)((b - a) * 4), (int32_t)(img.pw * 8) };
			CompressBlocksBC6H(&surf, dst, &s);
		};
#else
		fprintf(stderr, "built without ISPC Texture Compressor\n");
		return 2;
#endif
	}
	else if (family == "ispc_astc")
	{
#ifdef HAVE_ISPC
		// The footprint, then the profile. The image is padded (edge texels repeated) to whole blocks
		// before timing, since ISPC reads whole blocks.
		int abw = 4, abh = 4;
		if (sscanf(param.c_str(), "%dx%d", &abw, &abh) != 2) { fprintf(stderr, "ispc_astc needs <W>x<H>\n"); return 2; }
		const std::string prof = param.find(':') == std::string::npos ? "fast" : param.substr(param.find(':') + 1);
		static astc_enc_settings as;
		if (prof == "fast") GetProfile_astc_fast(&as, abw, abh);
		else if (prof == "alpha_fast") GetProfile_astc_alpha_fast(&as, abw, abh);
		else if (prof == "alpha_slow") GetProfile_astc_alpha_slow(&as, abw, abh);
		else { fprintf(stderr, "unknown ispc_astc profile %s\n", prof.c_str()); return 2; }
		const uint32_t apw = (img.w + abw - 1) / abw * abw, aph = (img.h + abh - 1) / abh * abh;
		std::vector<uint8_t> pad(apw * aph * 4);
		for (uint32_t y = 0; y < aph; y++)
			for (uint32_t x = 0; x < apw; x++)
				memcpy(&pad[(y * apw + x) * 4], &img.rgba[(std::min(y, img.h - 1) * img.pw + std::min(x, img.w - 1)) * 4], 4);
		const uint32_t abx = apw / abw, aby = aph / abh;
		std::vector<uint8_t> out(abx * aby * 16);
		const auto t0 = std::chrono::high_resolution_clock::now();
		parallel_rows(aby, threads, [&](uint32_t a, uint32_t b) {
			rgba_surface surf{ pad.data() + a * abh * apw * 4, (int32_t)apw, (int32_t)((b - a) * abh), (int32_t)(apw * 4) };
			CompressBlocksASTC(&surf, &out[a * abx * 16], &as);
		});
		const double asecs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
		if (!write_astc(argv[3], img.w, img.h, abw, abh, out)) { fprintf(stderr, "failed to write %s\n", argv[3]); return 1; }
		printf("secs=%.6f\n", asecs);
		return 0;
#else
		fprintf(stderr, "built without ISPC Texture Compressor\n");
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
