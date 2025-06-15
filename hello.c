// gcc -g -o hello.exe hello.c -lcuda -lcudart -L"/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" && ./hello.exe
#include <stdio.h>
#include <stdlib.h>

// TODO: debug this code and see if we can see more about what's happening
	// maybe we can output cubins

typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;
typedef int i32;

typedef struct cudaDeviceProp {
    char         name[256];                  /**< ASCII string identifying device */
    char         uuid[16];                   /**< 16-byte unique identifier */
    char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int          regsPerBlock;               /**< 32-bit registers available per block */
    int          warpSize;                   /**< Warp size in threads */
    size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int          clockRate;                  /**< Deprecated, Clock frequency in kilohertz */
    size_t       totalConstMem;              /**< Constant memory available on device in bytes */
    int          major;                      /**< Major compute capability */
    int          minor;                      /**< Minor compute capability */
    size_t       textureAlignment;           /**< Alignment requirement for textures */
    size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int          deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int          multiProcessorCount;        /**< Number of multiprocessors on device */
    int          kernelExecTimeoutEnabled;   /**< Deprecated, Specified whether there is a run time limit on kernels */
    int          integrated;                 /**< Device is integrated as opposed to discrete */
    int          canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int          computeMode;                /**< Deprecated, Compute mode (See ::cudaComputeMode) */
    int          maxTexture1D;               /**< Maximum 1D texture size */
    int          maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int          maxTexture1DLinear;         /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int          maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int          maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int          maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int          maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int          maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
    int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int          maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int          maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
    int          maxSurface1D;               /**< Maximum 1D surface size */
    int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int          maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int          maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
    int          maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
    size_t       surfaceAlignment;           /**< Alignment requirements for surfaces */
    int          concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int          ECCEnabled;                 /**< Device has ECC support enabled */
    int          pciBusID;                   /**< PCI bus ID of the device */
    int          pciDeviceID;                /**< PCI device ID of the device */
    int          pciDomainID;                /**< PCI domain ID of the device */
    int          tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int          asyncEngineCount;           /**< Number of asynchronous engines */
    int          unifiedAddressing;          /**< Device shares a unified address space with the host */
    int          memoryClockRate;            /**< Deprecated, Peak memory clock frequency in kilohertz */
    int          memoryBusWidth;             /**< Global memory bus width in bits */
    int          l2CacheSize;                /**< Size of L2 cache in bytes */
    int          persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int          streamPrioritiesSupported;  /**< Device supports stream priorities */
    int          globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int          localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t       sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int          regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int          managedMemory;              /**< Device supports allocating managed memory on this system */
    int          isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int          multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
    int          hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
    int          singleToDoublePrecisionPerfRatio; /**< Deprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    int          concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
    int          computePreemptionSupported; /**< Device supports Compute Preemption */
    int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
    int          cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    int          cooperativeMultiDeviceLaunch; /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    size_t       sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
    int          pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
    int          directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
    int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
    int          accessPolicyMaxWindowSize;  /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
    size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by CUDA driver per block in bytes */
    int          hostRegisterSupported;      /**< Device supports host memory registration via ::cudaHostRegister. */
    int          sparseCudaArraySupported;   /**< 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise */
    int          hostRegisterReadOnlySupported; /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    int          timelineSemaphoreInteropSupported; /**< External timeline semaphore interop is supported on the device */
    int          memoryPoolsSupported;       /**< 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise */
    int          gpuDirectRDMASupported;     /**< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise */
    unsigned int gpuDirectRDMAFlushWritesOptions; /**< Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    int          gpuDirectRDMAWritesOrdering;/**< See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values */
    unsigned int memoryPoolSupportedHandleTypes; /**< Bitmask of handle types supported with mempool-based IPC */
    int          deferredMappingCudaArraySupported; /**< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    int          ipcEventSupported;          /**< Device supports IPC Events. */
    int          clusterLaunch;              /**< Indicates device supports cluster launch */
    int          unifiedFunctionPointers;    /**< Indicates device supports unified pointers */
    int          reserved[63];               /**< Reserved for future use */
} cudaDeviceProp;

void printCudaDeviceProperties(cudaDeviceProp prop) {
	printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
	printf("  Total Global Memory: %zu bytes\n", prop.totalGlobalMem);
	printf("  Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
	printf("  Shared Memory per Multiprocessor: %zu bytes\n", prop.sharedMemPerMultiprocessor);
	printf("  Registers per Block: %d\n", prop.regsPerBlock);
	printf("  Registers per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
	printf("  Warp Size: %d\n", prop.warpSize);
	printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
	printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("  Max Blocks per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
	printf("  Max Threads Dimensions: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("  Max Grid Size: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("  Clock Rate: %d kHz\n", prop.clockRate);
	printf("  Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
	printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
	printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
	printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
	printf("  Compute Preemption Supported: %d\n", prop.computePreemptionSupported);
	printf("  Concurrent Kernels: %d\n", prop.concurrentKernels);
	printf("  Concurrent Managed Access: %d\n", prop.concurrentManagedAccess);
	printf("  Unified Addressing: %d\n", prop.unifiedAddressing);
	printf("  Async Engine Count: %d\n", prop.asyncEngineCount);
	printf("  Can Map Host Memory: %d\n", prop.canMapHostMemory);
	printf("  PCI Bus ID: %d\n", prop.pciBusID);
	printf("  PCI Device ID: %d\n", prop.pciDeviceID);
	printf("  PCI Domain ID: %d\n", prop.pciDomainID);
	printf("  TCC Driver: %d\n", prop.tccDriver);
	printf("  UUID: ");
	for (int j = 0; j < 16; ++j)
		printf("%02x", (unsigned char)prop.uuid[j]);
	printf("\n");

	printf("\n");
}

extern i32 cuInit(unsigned int Flags);
extern i32 cuDeviceGet(CUdevice* device, int ordinal);
extern i32 cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
extern i32 cuModuleLoad(CUmodule *module, const char *fname);
extern i32 cuModuleLoadData(CUmodule* module, const void* image);
extern i32 cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
extern i32 cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
extern i32 cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
extern i32 cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
extern i32 cuMemFree(CUdeviceptr dptr);
extern i32 cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUcontext hStream,
    void** kernelParams,
    void** extra);
extern i32 cudaGetDeviceProperties(cudaDeviceProp *prop, int device);


// Hardcoded PTX code for a simple kernel "add_one"
const char* ptx_code =
".version 6.0\n"
".target sm_30\n"
".address_size 64\n"
"\n"
".visible .entry add_one(\n"
"    .param .u64 _param_data\n"
")\n"
"{\n"
"    .reg .u32 %r<3>;\n"
"    .reg .u64 %rd<3>;\n"
"    ld.param.u64 %rd1, [_param_data];\n"
"    mov.u32 %r1, %tid.x;\n"
"    mul.wide.u32 %rd2, %r1, 4;\n"
"    add.u64 %rd1, %rd1, %rd2;\n"
"    ld.global.u32 %r2, [%rd1];\n"
"    add.u32 %r2, %r2, 1;\n"
"    st.global.u32 [%rd1], %r2;\n"
"    ret;\n"
"}\n";

// Load cubin file into memory
void* load_file(const char* filename, size_t* size_out) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);
    void* data = malloc(size);
    fread(data, 1, size, f);
    fclose(f);
    if (size_out) *size_out = size;
    return data;
}

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction func;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printCudaDeviceProperties(prop);
	//printf("Compute capability: %d.%d\n", prop.major, prop.minor);

	size_t cubin_size;
	void* cubin_data = load_file("hello.cubin", &cubin_size);
	i32 res = cuModuleLoadData(&mod, cubin_data);
    //i32 res = cuModuleLoadData(&mod, ptx_code);
    //i32 res = cuModuleLoad(&mod, "hello.cubin");
    if (res != 0) {
        fprintf(stderr, "Failed to load PTX module\n");
        return 1;
    }

    res = cuModuleGetFunction(&func, mod, "add_one");
    if (res != 0) {
        fprintf(stderr, "Failed to get kernel function\n");
        return 1;
    }

    int data[4] = {1, 2, 3, 4};
    CUdeviceptr dev_data;
    cuMemAlloc(&dev_data, sizeof(data));
    cuMemcpyHtoD(dev_data, data, sizeof(data));

    void* args[] = { &dev_data };
    cuLaunchKernel(func, 1, 1, 1, 4, 1, 1, 0, 0, args, NULL);
    cuMemcpyDtoH(data, dev_data, sizeof(data));
    cuMemFree(dev_data);
    for (int i = 0; i < 4; ++i) {
        printf("data[%d] = %d\n", i, data[i]);
    }
    return 0;
}

