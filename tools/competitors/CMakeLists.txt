# Builds bcn_harness (third-party BCn encoders behind one CLI) for scripts/compete.py.
#
#   cmake -S tools/competitors -B third_party/build-harness -DTP=<abs path to third_party> -DISPC=<ispc.exe>
#   cmake --build third_party/build-harness --config Release
#
# TP must contain clones of: bc7enc_rdo (lodepng), basis_universal (bc7f), ISPCTextureCompressor,
# icbc, stb. ISPC is optional; without it the ispc_* encoders are left out.
cmake_minimum_required(VERSION 3.16)
project(bcn_harness CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TP "" CACHE PATH "third_party directory with the encoder clones")
set(ISPC "" CACHE FILEPATH "ispc compiler (optional)")

add_executable(bcn_harness
  bcn_harness.cpp
  ${TP}/bc7enc_rdo/lodepng.cpp
  ${TP}/basis_universal/transcoder/basisu_transcoder.cpp)
target_include_directories(bcn_harness PRIVATE
  ${TP}/bc7enc_rdo ${TP}/basis_universal/transcoder ${TP}/icbc ${TP}/stb)
target_compile_definitions(bcn_harness PRIVATE HAVE_BC7F
  BASISD_SUPPORT_KTX2=0 BASISD_SUPPORT_KTX2_ZSTD=0 ICBC_SIMD=4 NDEBUG)  # ICBC_SIMD 4 = AVX2
if(MSVC)
  target_compile_options(bcn_harness PRIVATE /O2 /arch:AVX2 /W1 /bigobj)
else()
  target_compile_options(bcn_harness PRIVATE -O3 -mavx2 -mfma)
  find_package(Threads REQUIRED)
  target_link_libraries(bcn_harness PRIVATE Threads::Threads)
endif()

if(ISPC)
  set(ISPC_DIR ${TP}/ISPCTextureCompressor/ispc_texcomp)
  set(KOBJ ${CMAKE_BINARY_DIR}/kernel${CMAKE_CXX_OUTPUT_EXTENSION})
  set(KOBJS ${KOBJ}
    ${CMAKE_BINARY_DIR}/kernel_sse4${CMAKE_CXX_OUTPUT_EXTENSION}
    ${CMAKE_BINARY_DIR}/kernel_avx2${CMAKE_CXX_OUTPUT_EXTENSION})
  add_custom_command(OUTPUT ${KOBJS} ${CMAKE_BINARY_DIR}/kernel_ispc.h
    COMMAND ${ISPC} -O2 ${ISPC_DIR}/kernel.ispc -o ${KOBJ} -h ${CMAKE_BINARY_DIR}/kernel_ispc.h
            --target=sse4,avx2 --opt=fast-math --pic
    DEPENDS ${ISPC_DIR}/kernel.ispc)
  set(AOBJ ${CMAKE_BINARY_DIR}/kernel_astc${CMAKE_CXX_OUTPUT_EXTENSION})
  set(AOBJS ${AOBJ}
    ${CMAKE_BINARY_DIR}/kernel_astc_sse4${CMAKE_CXX_OUTPUT_EXTENSION}
    ${CMAKE_BINARY_DIR}/kernel_astc_avx2${CMAKE_CXX_OUTPUT_EXTENSION})
  add_custom_command(OUTPUT ${AOBJS} ${CMAKE_BINARY_DIR}/kernel_astc_ispc.h
    COMMAND ${ISPC} -O2 ${ISPC_DIR}/kernel_astc.ispc -o ${AOBJ} -h ${CMAKE_BINARY_DIR}/kernel_astc_ispc.h
            --target=sse4,avx2 --opt=fast-math --pic
    DEPENDS ${ISPC_DIR}/kernel_astc.ispc)
  target_sources(bcn_harness PRIVATE ${ISPC_DIR}/ispc_texcomp.cpp ${ISPC_DIR}/ispc_texcomp_astc.cpp
    ${KOBJS} ${AOBJS} ${CMAKE_BINARY_DIR}/kernel_ispc.h ${CMAKE_BINARY_DIR}/kernel_astc_ispc.h)
  target_include_directories(bcn_harness PRIVATE ${ISPC_DIR} ${CMAKE_BINARY_DIR})
  target_compile_definitions(bcn_harness PRIVATE HAVE_ISPC)
endif()
