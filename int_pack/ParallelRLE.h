#pragma once

#include "IArchiver.h"
#include "ParallelRLE.cuh"
#include "cuda_util.h"

class ParallelRLE :public IArchiver
{
private:
	int _mp_cnt;
	bool _verbose;

	void print_host_vector(const std::string& array_name, thrust::host_vector<int>& vec)
	{
		if (_verbose) {
			cout << "printng " << array_name << endl;
			for (int i = 0; i < vec.size(); i++) {
				cout << vec[i] << " ";
			}

			cout << endl << endl;
		}
	}

	bool verify_compression(thrust::host_vector<int> original, thrust::host_vector<int> compressedSymbols, thrust::host_vector<int> compressedCounts, int totalRuns) {

		// decompress.
		thrust::host_vector<int> g_decompressed(original.size());

		int sum = 0;
		for (int i = 0; i < totalRuns; ++i) {
			sum += compressedCounts[i];
		}

		if (sum != original.size()) {
			cout << "Decompressed and original size not equal " << original.size() << " != " << sum;

			for (int i = 0; i < totalRuns; ++i) {
				int symbol = compressedSymbols[i];
				int count = compressedCounts[i];

				cout << count << "," << symbol << endl;
			}
			return false;
		}

		int j = 0;
		for (int i = 0; i < totalRuns; ++i) {
			int symbol = compressedSymbols[i];
			int count = compressedCounts[i];

			for (int k = 0; k < count; ++k) {
				g_decompressed[j++] = symbol;
			}
		}

		// verify the compression.
		for (int i = 0; i < original.size(); ++i) {
			if (original[i] != g_decompressed[i]) {

				cout << "Decompressed and original not equal at " << i << original[i] << " != " <<  g_decompressed[i] << endl;
				return false;
			}
		}

		return true;
	}

public:

	ParallelRLE(bool verbose = false): _mp_cnt(0), _verbose(verbose)
	{
		try {
			_mp_cnt = cuda::get_mp_count(0);
			cuda::init(0);
		}
		catch (const cuda::cuda_exception& e) {
			throw e;
		}
	}

	void encode(const std::vector<int>& in, std::vector<RLE>& out) override
	{
		const int blocks = 32 * _mp_cnt;

		thrust::host_vector<int> h_rle(in.begin(), in.end());

		print_host_vector("h_rle", h_rle);

		thrust::device_vector<int> d_rle = h_rle;
		thrust::device_vector<int> d_mask;

		d_mask.resize(h_rle.size());

		int* d_rle_ptr = thrust::raw_pointer_cast(d_rle.data());
		int* d_mask_ptr = thrust::raw_pointer_cast(d_mask.data());

		mask << <blocks, 256 >> > (d_rle_ptr, d_mask_ptr, h_rle.size());

		thrust::host_vector<int> h_tmp = d_mask;

		print_host_vector("d_mask", h_tmp);

		thrust::inclusive_scan(thrust::device, d_mask.begin(), d_mask.end(), d_mask.begin());

		h_tmp = d_mask;

		print_host_vector("d_mask scanned", h_tmp);
		
		thrust::device_vector<int> d_compact_mask;

		d_compact_mask.resize(h_rle.size());

		thrust::device_vector<int> d_total_pairs(1);

		int* d_compact_mask_ptr = thrust::raw_pointer_cast(d_compact_mask.data());
		int* d_total_pairs_ptr = thrust::raw_pointer_cast(d_total_pairs.data());

		compact << <blocks, 256 >> > (d_mask_ptr, d_compact_mask_ptr, d_total_pairs_ptr, h_rle.size());

		h_tmp = d_compact_mask;

		print_host_vector("d_compact_mask",h_tmp);

		h_tmp = d_total_pairs;

		int h_total_pairs = h_tmp[0];

		thrust::device_vector<int> d_compact_rle_chars;

		d_compact_rle_chars.resize(h_rle.size());

		thrust::device_vector<int> d_compact_rle_counts;

		d_compact_rle_counts.resize(h_rle.size());

		int* d_compact_rle_chars_ptr = thrust::raw_pointer_cast(d_compact_rle_chars.data());
		int* d_compact_rle_counts_ptr = thrust::raw_pointer_cast(d_compact_rle_counts.data());

		scatter << <blocks, 256 >> > (d_compact_mask_ptr, d_total_pairs_ptr, d_rle_ptr, d_compact_rle_chars_ptr, d_compact_rle_counts_ptr);

		thrust::device_vector<int> d_decomp_rle;

		d_decomp_rle.resize(h_rle.size());

		int* d_decomp_rle_ptr = thrust::raw_pointer_cast(d_decomp_rle.data());

		decompress << <blocks, 256 >> > (d_compact_rle_chars_ptr, d_compact_rle_counts_ptr, d_total_pairs_ptr, d_decomp_rle_ptr);

		h_tmp = d_decomp_rle;

		if (h_tmp.size() != h_rle.size()) {
			cout << "Size mismatch " << h_tmp.size() << " != " << h_rle.size() << endl;
		}

		for (int i = 0; i< h_rle.size(); ++i) {
			if (h_rle[i] != h_tmp[i]) {
				cout << "Test failed!" << endl;
				break;
			}
		}


		print_host_vector("d_decomp_rle", h_tmp);

		cudaDeviceSynchronize();

		d_compact_rle_chars.resize(h_total_pairs);
		d_compact_rle_counts.resize(h_total_pairs);

		h_tmp = d_compact_rle_chars;

		print_host_vector("d_compact_rle_chars", h_tmp);

		h_tmp = d_compact_rle_counts;

		print_host_vector("d_compact_rle_counts", h_tmp);

		cout << "Original size: " << h_rle.size() << endl;
		cout << "Compressed size: " << h_total_pairs * 2 << endl;

		thrust::host_vector<int> h_compact_rle_chars = d_compact_rle_chars;
		thrust::host_vector<int> h_compact_rle_counts = d_compact_rle_counts;

		if (!verify_compression(h_rle, h_compact_rle_chars, h_compact_rle_counts, h_total_pairs)) {
			cout << "Failed test!" << endl;
		}
		else {
			cout << "passed test!" << endl;
		}
	} 

	void decode() override 
	{

	}

	~ParallelRLE()
	{
		cuda::uninit();
	}
};

