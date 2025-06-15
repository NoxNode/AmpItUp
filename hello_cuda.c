// gcc -o hello_cuda.exe hello_cuda.c -lcuda -lnvrtc -L"/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" && ./hello_cuda.exe
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Minimal CUDA/NVRTC forward declarations

typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;
typedef int CUresult;
typedef struct _nvrtcProgram* nvrtcProgram;
typedef int nvrtcResult;

#define NVRTC_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75

extern CUresult cuInit(unsigned int Flags);
extern CUresult cuDeviceGet(CUdevice* device, int ordinal);
extern CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
extern CUresult cuModuleLoadData(CUmodule* module, const void* image);
extern CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
extern CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
extern CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
extern CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
extern CUresult cuMemFree(CUdeviceptr dptr);
extern CUresult cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUcontext hStream,
    void** kernelParams,
    void** extra);

extern nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src,
    const char* name, int numHeaders, const char** headers, const char** includeNames);
extern nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options);
extern nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet);
extern nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx);
extern const char* nvrtcGetErrorString(nvrtcResult result);

// CUDA kernel as string
const char* kernel_code =
"extern \"C\" __global__ void add_one(int* data) {\n"
"    int idx = threadIdx.x;\n"
"    data[idx] += 1;\n"
"}\n";

int main() {
    // Compile CUDA kernel with NVRTC
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, kernel_code, "add_one.cu", 0, NULL, NULL) != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to create NVRTC program\n");
        return 1;
    }

    if (nvrtcCompileProgram(prog, 0, NULL) != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to compile program\n");
        return 1;
    }

    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = (char*)malloc(ptx_size);
    nvrtcGetPTX(prog, ptx);
	printf("%s\n", ptx);

    // CUDA Driver API setup
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction func;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);
    cuModuleLoadData(&mod, ptx);
    cuModuleGetFunction(&func, mod, "add_one");

    // Setup input/output data
    int data[4] = { 1, 2, 3, 4 };
    CUdeviceptr dev_data;
    cuMemAlloc(&dev_data, sizeof(data));
    cuMemcpyHtoD(dev_data, data, sizeof(data));

    // Launch kernel
    void* args[] = { &dev_data };
    cuLaunchKernel(func, 1, 1, 1, 4, 1, 1, 0, 0, args, NULL);

    // Copy result back
    cuMemcpyDtoH(data, dev_data, sizeof(data));
    cuMemFree(dev_data);

    // Print result
    for (int i = 0; i < 4; ++i) {
        printf("data[%d] = %d\n", i, data[i]);
    }

    free(ptx);
    return 0;
}

