// Standalone micro-benchmark: RDNA2 SF4 sdot4 GEMM (original vs candidate kernels). Not part of Strata.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include "ink_hip_codecs.hip.h"

using namespace strata::inkling2::hip;

// ---------------------------------------------------------------------------------------------
// Candidate v2: KB K-blocks (16 elements each) per sync, BMxBN output tile, 256 threads,
// per-thread TMxTN register tile. Same int8 quantization and same per-block fma order as the original => identical results.
// ---------------------------------------------------------------------------------------------
template <int BM, int BN, int KB>
__global__ void __launch_bounds__(256) k_sf4_sdot4_v2(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scales, float outer_scale,
    const half_t* __restrict__ X, float* __restrict__ Y, int M, int N, int K) {
    constexpr int TM = 4;
    constexpr int TN = (BM * BN) / (256 * TM);
    constexpr int PAD = 4;                       // ints of row padding (bank-conflict relief)
    constexpr int RS = KB * 4 + PAD;             // row stride in ints
    __shared__ alignas(16) int sXq[BM * RS];
    __shared__ alignas(16) int sWq[BN * RS];
    __shared__ float sXd[BM * KB];
    __shared__ float sWs[BN * KB];

    const int tid = threadIdx.x;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = 0.5f * outer_scale;
    constexpr int COLS = BN / TN;                // threads per tile row
    const int t_row = (tid / COLS) * TM;
    const int t_col = (tid % COLS) * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int nblk16 = K >> 4;
    for (int gb0 = 0; gb0 < nblk16; gb0 += KB) {
        const int nb = min(KB, nblk16 - gb0);
        // ---- stage X: BM*KB 16-element blocks, quantized to int8 with per-block scale
        for (int t = tid; t < BM * KB; t += 256) {
            const int r = t / KB, kb = t % KB;
            const int gm = m_blk + r;
            int4 p4 = make_int4(0, 0, 0, 0);
            float d = 0.0f;
            if (gm < M && kb < nb) {
                const half_t* src = X + static_cast<size_t>(gm) * K + ((gb0 + kb) << 4);
                const uint4 u0 = *reinterpret_cast<const uint4*>(src);
                const uint4 u1 = *reinterpret_cast<const uint4*>(src + 8);
                float xf[16];
                h2_unpack(u0.x, xf[0], xf[1]);   h2_unpack(u0.y, xf[2], xf[3]);
                h2_unpack(u0.z, xf[4], xf[5]);   h2_unpack(u0.w, xf[6], xf[7]);
                h2_unpack(u1.x, xf[8], xf[9]);   h2_unpack(u1.y, xf[10], xf[11]);
                h2_unpack(u1.z, xf[12], xf[13]); h2_unpack(u1.w, xf[14], xf[15]);
                float am = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(xf[i]));
                d = am * (1.0f / 127.0f);
                const float inv = am > 0.0f ? (127.0f / am) : 0.0f;
                p4.x = pack4_i8(clamp_i8(xf[0] * inv), clamp_i8(xf[1] * inv), clamp_i8(xf[2] * inv), clamp_i8(xf[3] * inv));
                p4.y = pack4_i8(clamp_i8(xf[4] * inv), clamp_i8(xf[5] * inv), clamp_i8(xf[6] * inv), clamp_i8(xf[7] * inv));
                p4.z = pack4_i8(clamp_i8(xf[8] * inv), clamp_i8(xf[9] * inv), clamp_i8(xf[10] * inv), clamp_i8(xf[11] * inv));
                p4.w = pack4_i8(clamp_i8(xf[12] * inv), clamp_i8(xf[13] * inv), clamp_i8(xf[14] * inv), clamp_i8(xf[15] * inv));
            }
            *reinterpret_cast<int4*>(&sXq[r * RS + kb * 4]) = p4;
            sXd[r * KB + kb] = d;
        }
        // ---- stage W: BN*KB blocks of 16 SF4 weights -> int8 (x2 exact), with block scale
        for (int t = tid; t < BN * KB; t += 256) {
            const int r = t / KB, kb = t % KB;
            const int gn = n_blk + r;
            int4 p4 = make_int4(0, 0, 0, 0);
            float s = 0.0f;
            if (gn < N && kb < nb) {
                const int gb = gb0 + kb;
                s = e4m3fn_to_f32(scales[static_cast<size_t>(gn) * nblk16 + gb]) * half_outer;
                const uint2 raw = *reinterpret_cast<const uint2*>(packed + static_cast<size_t>(gn) * (K >> 1) + gb * 8);
                p4.x = pack4_i8(sf4_e2m1_i8_x2(raw.x & 15u), sf4_e2m1_i8_x2((raw.x >> 4) & 15u),
                                sf4_e2m1_i8_x2((raw.x >> 8) & 15u), sf4_e2m1_i8_x2((raw.x >> 12) & 15u));
                p4.y = pack4_i8(sf4_e2m1_i8_x2((raw.x >> 16) & 15u), sf4_e2m1_i8_x2((raw.x >> 20) & 15u),
                                sf4_e2m1_i8_x2((raw.x >> 24) & 15u), sf4_e2m1_i8_x2((raw.x >> 28) & 15u));
                p4.z = pack4_i8(sf4_e2m1_i8_x2(raw.y & 15u), sf4_e2m1_i8_x2((raw.y >> 4) & 15u),
                                sf4_e2m1_i8_x2((raw.y >> 8) & 15u), sf4_e2m1_i8_x2((raw.y >> 12) & 15u));
                p4.w = pack4_i8(sf4_e2m1_i8_x2((raw.y >> 16) & 15u), sf4_e2m1_i8_x2((raw.y >> 20) & 15u),
                                sf4_e2m1_i8_x2((raw.y >> 24) & 15u), sf4_e2m1_i8_x2((raw.y >> 28) & 15u));
            }
            *reinterpret_cast<int4*>(&sWq[r * RS + kb * 4]) = p4;
            sWs[r * KB + kb] = s;
        }
        __syncthreads();
        for (int kb = 0; kb < nb; ++kb) {
            int4 a[TM]; float xd[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = *reinterpret_cast<const int4*>(&sXq[(t_row + i) * RS + kb * 4]);
                xd[i] = sXd[(t_row + i) * KB + kb];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int4 b = *reinterpret_cast<const int4*>(&sWq[(t_col + j) * RS + kb * 4]);
                const float ws = sWs[(t_col + j) * KB + kb];
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    int idot = 0;
                    idot = sdot4_i32(a[i].x, b.x, idot);
                    idot = sdot4_i32(a[i].y, b.y, idot);
                    idot = sdot4_i32(a[i].z, b.z, idot);
                    idot = sdot4_i32(a[i].w, b.w, idot);
                    acc[i][j] = fmaf(static_cast<float>(idot), xd[i] * ws, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = m_blk + t_row + i;
        if (gm < M) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gn = n_blk + t_col + j;
                if (gn < N) Y[static_cast<size_t>(gm) * N + gn] = acc[i][j];
            }
        }
    }
}

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

template <typename F>
static float time_ms(F&& f, int iters = 20) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    f(); CK(hipDeviceSynchronize());   // warm
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b));
    return ms / iters;
}

int main(int argc, char** argv) {
    struct Shape { int M, N, K; };
    std::vector<Shape> shapes = {{256, 2304, 2304}, {256, 6144, 2304}, {256, 2304, 6144}, {256, 4608, 2304}};
    if (argc == 4) shapes = {{atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}};
    std::mt19937 rng(1234);
    for (auto s : shapes) {
        const int M = s.M, N = s.N, K = s.K;
        std::vector<uint8_t> hp((size_t)N * (K / 2)), hs((size_t)N * (K / 16));
        std::vector<uint16_t> hx((size_t)M * K);
        for (auto& v : hp) v = rng() & 0xFF;
        for (auto& v : hs) v = 0x30 + (rng() % 16);          // plausible small positive E4M3 scales
        for (auto& v : hx) { float f = ((int)(rng() % 2001) - 1000) / 500.0f; _Float16 h = (_Float16)f; memcpy(&v, &h, 2); }
        uint8_t *dp, *ds; uint16_t* dx; float *dy0, *dy1;
        CK(hipMalloc(&dp, hp.size())); CK(hipMalloc(&ds, hs.size())); CK(hipMalloc(&dx, hx.size() * 2));
        CK(hipMalloc(&dy0, (size_t)M * N * 4)); CK(hipMalloc(&dy1, (size_t)M * N * 4));
        CK(hipMemcpy(dp, hp.data(), hp.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(ds, hs.data(), hs.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        const half_t* X = reinterpret_cast<const half_t*>(dx);
        const float outer = 0.37f;

        auto orig = [&]() {
            k_sf4_sdot4_tiled<16, 32><<<dim3((N + 31) / 32, (M + 15) / 16), 128>>>(dp, ds, outer, X, dy0, M, N, K);
        };
        float t0 = time_ms(orig);
        double gmac = (double)M * N * K / 1e9;
        printf("shape M=%d N=%d K=%d | original 16x32: %.3f ms (%.2f TMAC/s)\n", M, N, K, t0, gmac / t0);

        std::vector<float> ref((size_t)M * N), got((size_t)M * N);
        CK(hipMemcpy(ref.data(), dy0, ref.size() * 4, hipMemcpyDeviceToHost));

        auto check = [&](const char* name, auto&& launch) {
            CK(hipMemset(dy1, 0, (size_t)M * N * 4));
            launch(); CK(hipDeviceSynchronize());
            CK(hipMemcpy(got.data(), dy1, got.size() * 4, hipMemcpyDeviceToHost));
            double maxd = 0; size_t bad = 0;
            for (size_t i = 0; i < ref.size(); ++i) { double d = fabs((double)ref[i] - got[i]); if (d > maxd) maxd = d; if (d > 1e-3 * (1 + fabs(ref[i]))) ++bad; }
            float t = time_ms(launch);
            printf("   %-22s %.3f ms (%.2f TMAC/s) speedup %.2fx  maxdiff=%.3g mismatches=%zu\n", name, t, gmac / t, t0 / t, maxd, bad);
        };
#define V2(BMv, BNv, KBv) check("v2 " #BMv "x" #BNv " kb" #KBv, [&]() { k_sf4_sdot4_v2<BMv, BNv, KBv><<<dim3((N + BNv - 1) / BNv, (M + BMv - 1) / BMv), 256>>>(dp, ds, outer, X, dy1, M, N, K); })
        V2(32, 32, 4);
        V2(32, 64, 4);
        V2(64, 64, 4);
        V2(64, 64, 8);
        V2(64, 32, 8);
        V2(64, 128, 4);
        V2(128, 64, 4);
        hipFree(dp); hipFree(ds); hipFree(dx); hipFree(dy0); hipFree(dy1);
    }
    return 0;
}
