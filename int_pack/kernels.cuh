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

__global__ void encode_gpu(int* d_comact_mask, int d_total_pairs, int* d_in, int* d_comact_rle_chars, int* d_compact_rle_counts) {

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_total_pairs; i += blockDim.x * gridDim.x) {
		int a = d_comact_mask[i];
		int b = d_comact_mask[i + 1];

		d_comact_rle_chars[i] = d_in[a];
		d_compact_rle_counts[i] = b - a;
	}
}

__global__ void decode_gpu(int* d_compressed_symbols, int* d_compressed_counts, int* d_decomp_mask, int d_total_pairs, int* d_decompressed) {
	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_total_pairs; i += blockDim.x * gridDim.x) {
		
		int j = d_decomp_mask[i];

		int symbol = d_compressed_symbols[i];
		int count = d_compressed_counts[i];

		for (int k = 0; k < count; ++k) {

			d_decompressed[j++] = symbol;
		}
	}	
}



