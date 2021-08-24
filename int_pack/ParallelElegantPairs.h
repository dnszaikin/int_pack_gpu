#pragma once

#include "RLEHelper.h"
#include "kernels.cuh"

class ParallelElegantPairs: public RLEHelper {
public:
	void encode(const thrust::host_vector<int>& h_symbols, const thrust::host_vector<int>& h_counts, thrust::host_vector<int>& h_out) 
	{
		thrust::device_vector<int> d_out(h_symbols.size());
		thrust::device_vector<int> d_symbols = h_symbols;
		thrust::device_vector<int> d_counts = h_counts;


		auto gpu_elegant_pair = [=] __device__(thrust::tuple<int, int> t) {
			int x = thrust::get<0>(t);
			int y = thrust::get<1>(t);
			//printf("x=%d, y=%d\n", x, y);
			x = x >= 0 ? x * 2 : (x * -2) - 1;
			y = y >= 0 ? y * 2 : (y * -2) - 1;

			return (x >= y) ? ((x * x) + x + y) : ((y * y) + x);
		};

		auto first = thrust::make_zip_iterator(thrust::make_tuple(d_symbols.begin(), d_counts.begin()));
		auto last = thrust::make_zip_iterator(thrust::make_tuple(d_symbols.end(), d_counts.end()));
		
		thrust::transform(first, last, d_out.begin(), gpu_elegant_pair);

		h_out = d_out;

		print_vector("h_out", h_out);
	};

	void decode(const thrust::host_vector<int>& h_in, thrust::host_vector<int>& h_symbols, thrust::host_vector<int>& h_counts) 
	{
		thrust::device_vector<int> d_in = h_in;
		thrust::device_vector<int> d_symbols(h_symbols.size());
		thrust::device_vector<int> d_counts(h_counts.size());

		auto gpu_elegant_unpair = [=] __device__(int z) {
			thrust::tuple<int, int> t;
			int& x = thrust::get<0>(t);
			int& y = thrust::get<1>(t);

			int sqrtz = floor(sqrt((float)z));
			int sqz = sqrtz * sqrtz;

			if ((z - sqz) >= sqrtz) {
				x = sqrtz;
				y = z - sqz - sqrtz;
			}
			else {
				x = z - sqz;
				y = sqrtz;
			}

			x = x % 2 == 0 ? x / 2 : (x + 1) / -2;
			y = y % 2 == 0 ? y / 2 : (y + 1) / -2;

			return t;
		};

		auto first = thrust::make_zip_iterator(thrust::make_tuple(d_symbols.begin(), d_counts.begin()));
		auto last = thrust::make_zip_iterator(thrust::make_tuple(d_symbols.end(), d_counts.end()));

		thrust::transform(d_in.begin(), d_in.end(), first, gpu_elegant_unpair);

		h_counts = d_counts;
		h_symbols = d_symbols;

		print_vector("h_counts", h_counts);
		print_vector("d_symbols", d_symbols);
	}
};