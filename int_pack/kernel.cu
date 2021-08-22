
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <nvfunctional>

#include <iostream>
#include <math.h>
#include <omp.h>
#include <vector>

using namespace std;

constexpr int MAX_N = 10;
char errorString[256];

// global host memory arrays.
int* g_symbolsOut;
int* g_countsOut;
int* g_in;
int* g_decompressed;

// Device memory used in PARLE
int* d_totalRuns;
int* d_symbolsOut;
int* d_countsOut;
int* d_in;
int* d_backwardMask;
int* d_scannedBackwardMask;
int* d_compactedBackwardMask;
int numSMs;

void printArray(int* arr, int n) {
	for (int i = 0; i < n; ++i) {
		printf("%d, ", arr[i]);
	}
	printf("\n");
}

int* generateData() 
{
	std::random_device rd;     // only used once to initialise (seed) engine
	std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	std::uniform_int_distribution<int> uni(1, 100); // guaranteed unbiased

	for (size_t i = 0; i < MAX_N; ++i) {
		//g_in[i] = uni(rng);
		g_in[i] = 1;
	}

	return g_in;
}

int rleCpu(int *in, int n, int* symbolsOut, int* countsOut) {

	if (n == 0)
		return 0; // nothing to compress!

	int outIndex = 0;
	int symbol = in[0];
	int count = 1;

	for (int i = 1; i < n; ++i) {
		if (in[i] != symbol) {
			// run is over.
			// So output run.
			symbolsOut[outIndex] = symbol;
			countsOut[outIndex] = count;
			outIndex++;

			// and start new run:
			symbol = in[i];
			count = 1;
		}
		else {
			++count; // run is not over yet.
		}
	}

	// output last run.
	symbolsOut[outIndex] = symbol;
	countsOut[outIndex] = count;
	outIndex++;

	return outIndex;
}

__global__ void compactKernel(int* g_in, int* g_scannedBackwardMask, int* g_compactedBackwardMask, int* g_totalRuns, int n) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {

		if (i == (n - 1)) {
			g_compactedBackwardMask[g_scannedBackwardMask[i] + 0] = i + 1;
			*g_totalRuns = g_scannedBackwardMask[i];
		}

		if (i == 0) {
			g_compactedBackwardMask[0] = 0;
		}
		else if (g_scannedBackwardMask[i] != g_scannedBackwardMask[i - 1]) {
			g_compactedBackwardMask[g_scannedBackwardMask[i] - 1] = i;
		}
	}
}

__global__ void scatterKernel(int* g_compactedBackwardMask, int* g_totalRuns, int* g_in, int* g_symbolsOut, int* g_countsOut) {
	int n = *g_totalRuns;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		int a = g_compactedBackwardMask[i];
		int b = g_compactedBackwardMask[i + 1];

		g_symbolsOut[i] = g_in[a];
		g_countsOut[i] = b - a;
	}
}

__global__ void maskKernel(int *g_in, int* g_backwardMask, int n) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		if (i == 0) {
			g_backwardMask[i] = 1;
		} else {
			g_backwardMask[i] = (g_in[i] != g_in[i - 1]);
		}
	}
}

__global__ void maskKernelVec(int* in, int* out, int n) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		if (i == 0) {
			out[i] = 1;
		}
		else {
			out[i] = (in[i] != in[i - 1]);
		}
	}
}

thrust::device_vector<int> gpuEncoding(thrust::device_vector<int> rle) {
	thrust::device_vector<int> arrayCompressed(rle.size());

	//auto f =  __device__() {
	//	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rle.size(); i += blockDim.x * gridDim.x) {
	//		if (i == 0) {
	//			arrayCompressed[i] = 1;
	//		}
	//		else {
	//			arrayCompressed[i] = (rle[i] != rle[i - 1]);
	//		}
	//	}
	//};
	//// GPU - Elegant Pair Function
	//int prev = -1;

	//auto gpuElegantPair = [prev] __device__(int element) {

	//	int result = element;
	//	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//	printf("%d\n",i);

	//	printf("curr: %d, prev: %d\n", element, prev);

	//	if (i == 0) {
	//		result = 1;
	//	}
	//	else {
	//		//result = 
	//	}

	//	prev = element;

	//	return element;
	//};
	
	//thrust::transform(rle.begin(), rle.end(), arrayCompressed.begin(), gpuElegantPair);

	return arrayCompressed;
}

//native scan
__global__ void scan(int *g_odata, int *g_idata, int n) {
	extern __shared__ float temp[]; // allocated on invocation    
	int thid = threadIdx.x;   
	int pout = 0, pin = 1;   
	// Load input into shared memory.    
	// This is exclusive scan, so shift right by one    
	// and set first element to 0   
	temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;   
	__syncthreads();   
	for (int offset = 1; offset < n; offset *= 2)   {     
		pout = 1 - pout; // swap double buffer indices     
		pin = 1 - pout;     
		if (thid >= offset)       
			temp[pout*n+thid] += temp[pin*n+thid - offset];     
		else       
			temp[pout*n+thid] = temp[pin*n+thid];     
		__syncthreads();   
	}   
	g_odata[thid] = temp[pout*n+thid]; // write output 
} 

// run parle on the GPU
void parleDevice(int *d_in, int n,
	int* d_symbolsOut,
	int* d_countsOut,
	int* d_totalRuns
) {
	int tmp[MAX_N];
	const int blocks = 32 * numSMs;

	thrust::host_vector<int> rle(MAX_N);

	// Initialize Vectors CPU
	for (int i = 0; i < MAX_N; i++) {
		rle[i] = 1;
	}

	thrust::device_vector<int> d_rle = rle;

	thrust::device_vector<int> arrayCompressedDevice;

	arrayCompressedDevice.resize(rle.size());

	int* _in = thrust::raw_pointer_cast(d_rle.data());
	int* _out = thrust::raw_pointer_cast(arrayCompressedDevice.data());

	maskKernelVec<<<blocks, 256>>>(_in, _out, n);

	thrust::host_vector<int> arrayCompressedHost = arrayCompressedDevice;

	for (int i = 0; i < arrayCompressedHost.size(); i++) {
		cout << arrayCompressedHost[i] << endl;
	}
	
	maskKernel<<<blocks,256>>>(d_in, d_backwardMask, n);

	cudaMemcpy(tmp, d_backwardMask, n * sizeof(int), cudaMemcpyDeviceToHost);

	printArray(tmp, n);

	cudaMemcpy(tmp, d_in, n * sizeof(int), cudaMemcpyDeviceToHost);

	printArray(tmp, n);

	scan<<<blocks,256>>>(d_backwardMask, d_scannedBackwardMask, n);
	compactKernel<<<blocks,256>>>(d_in, d_scannedBackwardMask, d_compactedBackwardMask, d_totalRuns, n);
	scatterKernel<<<blocks,256>>>(d_compactedBackwardMask, d_totalRuns, d_in, d_symbolsOut, d_countsOut);
}

bool verifyCompression(
	int* original, int n,
	int* compressedSymbols, int* compressedCounts, int totalRuns) {

	// decompress.
	int j = 0;

	int sum = 0;
	for (int i = 0; i < totalRuns; ++i) {
		sum += compressedCounts[i];
	}

	if (sum != n) {
		sprintf(errorString, "Decompressed and original size not equal %d != %d\n", n, sum);

		for (int i = 0; i < totalRuns; ++i) {
			int symbol = compressedSymbols[i];
			int count = compressedCounts[i];

			printf("%d, %d\n", count, symbol);
		}
		return false;
	}

	for (int i = 0; i < totalRuns; ++i) {
		int symbol = compressedSymbols[i];
		int count = compressedCounts[i];

		for (int k = 0; k < count; ++k) {
			g_decompressed[j++] = symbol;
		}
	}

	// verify the compression.
	for (int i = 0; i < n; ++i) {
		if (original[i] != g_decompressed[i]) {

			sprintf(errorString, "Decompressed and original not equal at %d, %d != %d\n", i, original[i], g_decompressed[i]);
			return false;
		}
	}

	return true;
}

// On the CPU do preparation to run parle, launch PARLE on GPU, and then transfer the result data to the CPU. 
void parleHost(int *h_in, int n,
	int* h_symbolsOut,
	int* h_countsOut) 
{
	int h_totalRuns;

	
	// transer input data to device.
	d_in = generateData();
	cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);


	// RUN.    
	parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns);

	cudaDeviceSynchronize();

	// transer result data to host.
	cudaMemcpy(h_symbolsOut, d_symbolsOut, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_countsOut, d_countsOut, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_totalRuns, d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost);

	printf("n = %d\n", n);
	printf("Original Size  : %d\n", n);
	printf("Compressed Size: %d\n", h_totalRuns * 2);

	if (!verifyCompression(
		d_in, n,
		g_symbolsOut, g_countsOut, h_totalRuns)) {
		printf("Failed test %s\n", errorString);
		printArray(d_in, n);

		exit(1);
	}
	else {
		printf("passed test!\n\n");
	}


}

void printError(const char* msg, cudaError_t err) 
{
	fprintf(stderr, "Error: %s, Desription: %s", msg, cudaGetErrorString(err));
}


int main()
{
	cudaError_t cudaStatus;

	cudaStatus = cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	if (cudaStatus != cudaSuccess) {
		printError("cudaDeviceGetAttribute failed!", cudaStatus);
		return 1;
	}

	cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		printError("cudaSetDevice failed!", cudaStatus);
		return 1;
	}

	// allocate resources on device. These arrays are used globally thoughouts the program.
	cudaMalloc((void**)&d_backwardMask, MAX_N * sizeof(int));
	cudaMalloc((void**)&d_scannedBackwardMask, MAX_N * sizeof(int));
	cudaMalloc((void**)&d_compactedBackwardMask, (MAX_N + 1) * sizeof(int));
	cudaMalloc((void**)&d_totalRuns, sizeof(int));
	cudaMalloc((void**)&d_in, MAX_N * sizeof(int));
	cudaMalloc((void**)&d_countsOut, MAX_N * sizeof(int));
	cudaMalloc((void**)&d_symbolsOut, MAX_N * sizeof(int));

	// allocate resources on the host. 
	g_in = new int[MAX_N];
	g_decompressed = new int[MAX_N];
	g_symbolsOut = new int[MAX_N];
	g_countsOut = new int[MAX_N];
			
	parleHost(d_in, MAX_N, d_symbolsOut, d_countsOut);

	cudaFree(d_backwardMask);
	cudaFree(d_scannedBackwardMask);
	cudaFree(d_compactedBackwardMask);
	cudaFree(d_in);
	cudaFree(d_countsOut);
	cudaFree(d_symbolsOut);
	cudaFree(d_totalRuns);

	cudaDeviceReset();

	// free host memory.
	delete[] g_in;
	delete[] g_decompressed;

	delete[] g_symbolsOut;
	delete[] g_countsOut;

	return 0;
}

