#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <random>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include <nvfunctional>

#include <iostream>
#include <math.h>
#include <omp.h>
#include <vector>

using namespace std;

int generateCompressibleRandomData() {
	int val = rand() % 100;

	if (rand() % 10 == 0) {
		val = rand() % 100;
	}

	return 10;
}

__global__ void mask(int* d_in, int* d_mask, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		if (i == 0) {
			d_mask[i] = 1;
		}
		else {
			d_mask[i] = (d_in[i] != d_in[i - 1]);
		}
	}
}

__global__ void compact(int* d_mask, int* d_compact_mask, int* d_total_pairs, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{

		if (i == (n - 1)) {
			d_compact_mask[d_mask[i]] = i + 1;
			*d_total_pairs = d_mask[i];
		}

		if (i == 0) {
			d_compact_mask[0] = 0;
		}
		else if (d_mask[i] != d_mask[i - 1]) {
			d_compact_mask[d_mask[i] - 1] = i;
		}
	}
}

__global__ void scatter(int* d_comact_mask, int* d_total_pairs, int* d_in, int* d_comact_rle_chars, int* d_compact_rle_counts) {
	int n = *d_total_pairs;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		int a = d_comact_mask[i];
		int b = d_comact_mask[i + 1];

		d_comact_rle_chars[i] = d_in[a];
		d_compact_rle_counts[i] = b - a;
	}
}

__device__ int counter = 0;

__global__ void decompress(int* d_compressed_symbols, int* d_compressed_counts, int* d_total_pairs, int* d_decompressed) {
	int n = *d_total_pairs;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {

		int j = 0;

		for (int _i = 0; _i < i; ++_i) {
			int count = d_compressed_counts[_i];
			j += count;
		}

		int symbol = d_compressed_symbols[i];
		int count = d_compressed_counts[i];

		for (int k = 0; k < count; ++k) {

			int current_val = atomicAdd(&counter, 1);

			d_decompressed[j++] = symbol;
			//printf("%d\n", current_val);
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


