// bc7f_encode: encodes a PNG to BC7 with Binomial's analytical bc7f encoder (Basis Universal,
// Apache 2.0) and writes a DX10 DDS, so `mipple score` can grade it like any other encoder.
//
// Usage: bc7f_encode <in.png> <out.dds> [preset]
//   preset: fastest | faster | fast | default | partial | nonanalytical   (default: default)
// Prints "secs=<encode seconds>" (single thread, block loop only) on stdout.
//
// Build: see build.sh. Needs basis_universal/transcoder and lodepng (from bc7enc_rdo).

#include "basisu_transcoder.h"
#include "basisu_transcoder_internal.h"
#include "lodepng.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace basist;

static uint32_t preset_flags(const std::string& name)
{
	if (name == "fastest") return bc7f::cPackBC7FlagDefaultFastest;
	if (name == "faster") return bc7f::cPackBC7FlagDefaultFaster;
	if (name == "fast") return bc7f::cPackBC7FlagDefaultFast;
	if (name == "default") return bc7f::cPackBC7FlagDefault;
	if (name == "partial") return bc7f::cPackBC7FlagDefaultPartiallyAnalytical;
	if (name == "nonanalytical") return bc7f::cPackBC7FlagDefaultNonAnalytical;
	fprintf(stderr, "unknown preset '%s'\n", name.c_str());
	exit(2);
}

static void put32(std::vector<uint8_t>& v, uint32_t x)
{
	for (int i = 0; i < 4; i++) v.push_back((uint8_t)(x >> (8 * i)));
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "usage: bc7f_encode <in.png> <out.dds> [fastest|faster|fast|default|partial|nonanalytical]\n");
		return 1;
	}
	const uint32_t flags = preset_flags(argc > 3 ? argv[3] : "default");

	std::vector<unsigned char> rgba;
	unsigned w = 0, h = 0;
	if (lodepng::decode(rgba, w, h, argv[1]) != 0)
	{
		fprintf(stderr, "failed to load %s\n", argv[1]);
		return 1;
	}

	basisu_transcoder_init();

	const uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4;
	std::vector<uint8_t> blocks(bw * bh * 16);

	auto t0 = std::chrono::high_resolution_clock::now();
	for (uint32_t by = 0; by < bh; by++)
	{
		for (uint32_t bx = 0; bx < bw; bx++)
		{
			color_rgba px[16];
			for (uint32_t i = 0; i < 16; i++)
			{
				const uint32_t x = std::min(bx * 4 + (i & 3), w - 1), y = std::min(by * 4 + (i >> 2), h - 1);
				const unsigned char* p = &rgba[(y * w + x) * 4];
				px[i].set(p[0], p[1], p[2], p[3]);
			}
			bc7f::fast_pack_bc7_auto_rgba(&blocks[(by * bw + bx) * 16], px, flags);
		}
	}
	const double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

	// DDS with DX10 header: BC7_UNORM, one mip level.
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
	for (uint32_t v : { 98u, 3u, 0u, 1u, 0u }) put32(out, v);
	out.insert(out.end(), blocks.begin(), blocks.end());

	FILE* f = fopen(argv[2], "wb");
	if (!f || fwrite(out.data(), 1, out.size(), f) != out.size())
	{
		fprintf(stderr, "failed to write %s\n", argv[2]);
		return 1;
	}
	fclose(f);
	printf("secs=%.6f\n", secs);
	return 0;
}
