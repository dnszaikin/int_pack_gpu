#pragma once

#include "IArchiver.h"
#include "ParallelRLE.cuh"
#include "cuda_util.h"

class ParallelRLE :public IArchiver
{
private:
	int _blocks;
	bool _verbose;

	void print_vector(const std::string& array_name, const thrust::device_vector<int>& vec) const
	{
		if (_verbose) {
			thrust::host_vector<int> h_tmp = vec;
			print_vector(array_name, h_tmp);
		}
	}
	void print_vector(const std::string& array_name, const thrust::host_vector<int>& vec) const
	{
		if (_verbose) {
			cout << "printng " << array_name << endl;
			for (int i = 0; i < vec.size(); i++) {
				cout << vec[i] << " ";
			}

			cout << endl << endl;
		}
	}
public:

	ParallelRLE(bool verbose = false): _blocks(0), _verbose(verbose)
	{
		try {
			_blocks = 32*cuda::get_mp_count(0);
			cuda::init(0);
		}
		catch (const cuda::cuda_exception& e) {
			throw e;
		}
	}

	void encode(const thrust::host_vector<int>& h_in, thrust::host_vector<int>& h_symbols, thrust::host_vector<int>& h_counts) override
	{
		
		print_vector("h_rle", h_in);

		thrust::device_vector<int> d_rle { h_in };
		thrust::device_vector<int> d_mask {};

		d_mask.resize(h_in.size());

		int* d_rle_ptr { thrust::raw_pointer_cast(d_rle.data()) };
		int* d_mask_ptr { thrust::raw_pointer_cast(d_mask.data()) };

		mask << <_blocks, 256 >> > (d_rle_ptr, d_mask_ptr, h_in.size());

		print_vector("d_mask", d_mask);

		thrust::inclusive_scan(thrust::device, d_mask.begin(), d_mask.end(), d_mask.begin());

		print_vector("d_mask scanned", d_mask);
		
		thrust::device_vector<int> d_compact_mask {};

		d_compact_mask.resize(h_in.size());

		thrust::device_vector<int> d_total_pairs(1);

		int* d_compact_mask_ptr{ thrust::raw_pointer_cast(d_compact_mask.data()) };
		int* d_total_pairs_ptr{ thrust::raw_pointer_cast(d_total_pairs.data()) };

		compact << <_blocks, 256 >> > (d_mask_ptr, d_compact_mask_ptr, d_total_pairs_ptr, h_in.size());
				
		print_vector("d_compact_mask", d_compact_mask);

		thrust::host_vector<int> h_tmp = d_total_pairs;

		int h_total_pairs{ h_tmp[0] };

		thrust::device_vector<int> d_compact_rle_chars{};

		d_compact_rle_chars.resize(h_in.size());

		thrust::device_vector<int> d_compact_rle_counts{};

		d_compact_rle_counts.resize(h_in.size());

		int* d_compact_rle_chars_ptr = thrust::raw_pointer_cast(d_compact_rle_chars.data());
		int* d_compact_rle_counts_ptr = thrust::raw_pointer_cast(d_compact_rle_counts.data());

		scatter << <_blocks, 256 >> > (d_compact_mask_ptr, h_total_pairs, d_rle_ptr, d_compact_rle_chars_ptr, d_compact_rle_counts_ptr);

		cudaDeviceSynchronize();

		d_compact_rle_chars.resize(h_total_pairs);
		d_compact_rle_counts.resize(h_total_pairs);

		print_vector("d_compact_rle_chars", d_compact_rle_chars);

		print_vector("d_compact_rle_counts", d_compact_rle_counts);

		cout << "Original size: " << h_in.size() << endl;
		cout << "Compressed size: " << h_total_pairs * 2 << endl;

		h_symbols = d_compact_rle_chars;
		h_counts = d_compact_rle_counts;
	} 

	void decode(const thrust::host_vector<int>& h_symbols, const thrust::host_vector<int>& h_counts, thrust::host_vector<int>& h_out) override
	{
		thrust::device_vector<int> d_decomp_rle{};

		d_decomp_rle.resize(h_out.size());

		thrust::device_vector<int> d_decomp_mask{};

		d_decomp_mask.resize(h_out.size());

		thrust::device_vector<int> d_compact_rle_chars{};

		d_compact_rle_chars = h_symbols;

		thrust::device_vector<int> d_compact_rle_counts{};

		d_compact_rle_counts = h_counts;

		thrust::device_vector<int> d_total_pairs(1);

		d_total_pairs[0] = h_symbols.size();

		int* d_total_pairs_ptr{ thrust::raw_pointer_cast(d_total_pairs.data()) };

		int* d_compact_rle_chars_ptr = thrust::raw_pointer_cast(d_compact_rle_chars.data());
		int* d_compact_rle_counts_ptr = thrust::raw_pointer_cast(d_compact_rle_counts.data());
		int* d_decomp_rle_ptr{ thrust::raw_pointer_cast(d_decomp_rle.data()) };
		int* d_decomp_mask_ptr{ thrust::raw_pointer_cast(d_decomp_mask.data()) };

		for (int i = 0; i < h_counts.size(); ++i) {
			int j = 0;
			for (int _i = 0; _i < i; ++_i) {
				int count = h_counts[_i];
				j += count;
			}
			d_decomp_mask[i] = j;
		}

		decompress <<<_blocks, 256>>> (d_compact_rle_chars_ptr, d_compact_rle_counts_ptr, d_decomp_mask_ptr, h_symbols.size(), d_decomp_rle_ptr);

		h_out = d_decomp_rle;

		print_vector("d_decomp_rle", h_out);
	}

	~ParallelRLE()
	{
		cuda::uninit();
	}
};

