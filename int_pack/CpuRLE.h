#pragma once

#include "IArchiver.h"

class CpuRLE : public IArchiver {
public:

	void encode(const thrust::host_vector<int>& h_in, thrust::host_vector<int>& h_symbols, thrust::host_vector<int>& h_counts) override
	{
		if (h_in.size() == 0) {
			return;
		}		

		int symbol = h_in[0];
		int count = 1;

		for (int i = 1; i < h_in.size(); ++i) {
			if (h_in[i] != symbol) {
				h_symbols.push_back(symbol);
				h_counts.push_back(count);

				symbol = h_in[i];
				count = 1;
			}
			else {
				++count;
			}
		}

		h_symbols.push_back(symbol);
		h_counts.push_back(count);
	}

	void decode(const thrust::host_vector<int>& h_symbols, const thrust::host_vector<int>& h_counts, thrust::host_vector<int>& h_out) override
	{
		for (int i = 0; i < h_symbols.size(); ++i) {
			int symbol = h_symbols[i];
			int count = h_counts[i];

			for (int k = 0; k < count; ++k) {
				h_out.push_back(symbol);
			}
		}
	}
};