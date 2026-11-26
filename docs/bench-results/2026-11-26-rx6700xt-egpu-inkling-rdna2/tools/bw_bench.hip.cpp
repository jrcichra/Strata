// Standalone achievable-bandwidth probe for the eGPU (RX 6700 XT / gfx1031). Not part of Strata.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

template <typename F>
static float time_ms(F&& f, int iters = 10) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    f(); CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b));
    return ms / iters;
}

__global__ void k_read(const uint4* __restrict__ p, size_t n, float* out) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a = 0, b = 0, c = 0, d = 0;
    for (; i < n; i += stride) {
        const uint4 v = p[i];
        a += v.x; b += v.y; c += v.z; d += v.w;
    }
    if ((a | b | c | d) == 0xDEADBEEFu) *out = 1.0f;
}

// Same traffic but consumed 64 B per warp with a 70 B stride: mimics S2_K expert-weight streaming.
__global__ void k_read_64b70(const uint8_t* __restrict__ p, size_t bytes, float* out) {
    const size_t warp = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const size_t nwarp = ((size_t)gridDim.x * blockDim.x) >> 5;
    const size_t lane = threadIdx.x & 31;
    uint32_t acc = 0;
    for (size_t blk = warp; blk * 70 < bytes; blk += nwarp) {
        const size_t off = blk * 70 + 6 + lane * 2;   // skip 6 B of scales, then 64 B payload
        if (off + 1 < bytes) {
            acc += *reinterpret_cast<const uint16_t*>(p + off);   // off is always even
        }
    }
    if (acc == 0xDEADBEEFu) *out = 1.0f;
}

// Per-thread 2-byte loads (what k_moe_grouped_w13_situ actually issues for W).
__global__ void k_read_u16(const uint16_t* __restrict__ p, size_t n, float* out) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (; i < n; i += stride) acc += p[i];
    if (acc == 0xDEADBEEFu) *out = 1.0f;
}

int main() {
    const size_t GB = 1024ull * 1024 * 1024;
    size_t bytes = 1024ull * GB / 100;   // 10.24 GB is too much VRAM here; use ~1 GB
    bytes = GB;                          // 1 GiB working set
    uint4* d = nullptr;
    CK(hipMalloc(&d, bytes));
    CK(hipMemset(d, 0x11, bytes));
    float* out = nullptr; CK(hipMalloc(&out, 4));
    const size_t n4 = bytes / 16;
    const int blocks = 40 * 16;
    for (int b : {40 * 4, 40 * 8, 40 * 16, 40 * 32}) {
        float ms = time_ms([&]{ k_read<<<b, 256>>>(d, n4, out); });
        printf("read uint4   blocks=%4d : %7.2f ms -> %6.1f GB/s\n", b, ms, bytes / (ms * 1e-3) / 1e9);
    }
    float ms16 = time_ms([&]{ k_read_u16<<<40 * 16, 256>>>((const uint16_t*)d, bytes / 2, out); });
    printf("read uint16  blocks=%4d : %7.2f ms -> %6.1f GB/s\n", 40 * 16, ms16, bytes / (ms16 * 1e-3) / 1e9);
    float ms64 = time_ms([&]{ k_read_64b70<<<40 * 16, 256>>>((const uint8_t*)d, bytes, out); });
    printf("read 64B/70B warp stride      : %7.2f ms -> %6.1f GB/s (payload only)\n", ms64, (bytes * 64.0 / 70.0) / (ms64 * 1e-3) / 1e9);
    CK(hipFree(d)); CK(hipFree(out));
    return 0;
}
