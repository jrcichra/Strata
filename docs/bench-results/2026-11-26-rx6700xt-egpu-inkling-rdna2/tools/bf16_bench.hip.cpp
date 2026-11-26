// Standalone micro-benchmark: RDNA2 bf16 fdot2 GEMM (original vs candidate kernels). Not part of Strata.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include "ink_hip_codecs.hip.h"

using namespace strata::inkling2::hip;

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

template <typename F>
static float time_ms(F&& f, int iters = 20) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    f(); CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b));
    return ms / iters;
}

static uint16_t f32_to_bf16_host(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding) >> 16);
}

int main(int argc, char** argv) {
    struct Shape { int M, N, K; };
    std::vector<Shape> shapes = {{256, 2304, 2304}, {256, 6144, 2304}, {256, 2304, 6144}, {256, 4608, 2304}};
    if (argc == 4) shapes = {{atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}};
    std::mt19937 rng(1234);
    for (auto s : shapes) {
        const int M = s.M, N = s.N, K = s.K;
        std::vector<uint16_t> hw((size_t)N * K);
        std::vector<float> hx((size_t)M * K);
        for (auto& v : hw) {
            float f = ((int)(rng() % 4001) - 2000) / 1000.0f;
            v = f32_to_bf16_host(f);
        }
        for (auto& v : hx) v = ((int)(rng() % 4001) - 2000) / 1000.0f;
        uint16_t* dW; float *dX, *dY0, *dY1;
        CK(hipMalloc(&dW, hw.size() * 2)); CK(hipMalloc(&dX, hx.size() * 4));
        CK(hipMalloc(&dY0, (size_t)M * N * 4)); CK(hipMalloc(&dY1, (size_t)M * N * 4));
        CK(hipMemcpy(dW, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dX, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));

        auto orig = [&]() {
            k_bf16_fdot2_tiled<16, 32><<<dim3((N + 31) / 32, (M + 15) / 16), 128>>>(dW, dX, nullptr, dY0, M, N, K);
        };
        float t0 = time_ms(orig);
        double gmac = (double)M * N * K / 1e9;
        printf("shape M=%d N=%d K=%d | original 16x32: %.3f ms (%.2f TMAC/s)\n", M, N, K, t0, gmac / t0);

        std::vector<float> ref((size_t)M * N), got((size_t)M * N);
        CK(hipMemcpy(ref.data(), dY0, ref.size() * 4, hipMemcpyDeviceToHost));

        auto check = [&](const char* name, auto&& launch) {
            CK(hipMemset(dY1, 0, (size_t)M * N * 4));
            launch(); CK(hipDeviceSynchronize());
            CK(hipMemcpy(got.data(), dY1, got.size() * 4, hipMemcpyDeviceToHost));
            double maxd = 0; size_t bad = 0;
            for (size_t i = 0; i < ref.size(); ++i) { double d = fabs((double)ref[i] - got[i]); if (d > maxd) maxd = d; if (d > 1e-3 * (1 + fabs(ref[i]))) ++bad; }
            float t = time_ms(launch);
            printf("   %-22s %.3f ms (%.2f TMAC/s) speedup %.2fx  maxdiff=%.3g mismatches=%zu\n", name, t, gmac / t, t0 / t, maxd, bad);
        };
#define V2(BMv, BNv, KBv) check("v2 " #BMv "x" #BNv " kb" #KBv, [&]() { k_bf16_fdot2_v2<BMv, BNv, KBv><<<dim3((N + BNv - 1) / BNv, (M + BMv - 1) / BMv), 256>>>(dW, dX, nullptr, dY1, M, N, K); })
        V2(32, 32, 4);
        V2(32, 32, 2);
        V2(32, 64, 4);
        V2(32, 64, 2);
        hipFree(dW); hipFree(dX); hipFree(dY0); hipFree(dY1);
    }
    return 0;
}
