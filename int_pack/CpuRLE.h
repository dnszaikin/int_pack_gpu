#pragma once

#include "IArchiver.h"

class CpuRLE : public IArchiver {
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
public:

	void encode(const std::vector<int>& in, std::vector<RLE>& out) override {
	}

	void decode() override {

	}
};