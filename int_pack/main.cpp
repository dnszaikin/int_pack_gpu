#include <iostream>

#include "ParallelRLE.h"
#include "CpuRLE.h"

using namespace std;

//int* generateData() 
//{
//	std::random_device rd;     // only used once to initialise (seed) engine
//	std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
//	std::uniform_int_distribution<int> uni(1, 100); // guaranteed unbiased
//
//	for (size_t i = 0; i < MAX_N; ++i) {
//		//g_in[i] = uni(rng);
//		g_in[i] = 1;
//	}
//
//	return g_in;
//}

int main()
{
	try {
		ParallelRLE parle(false);
		//std::vector<int> data { 1,2,3,6,6,6,5,5 };

		std::vector<int> data(1'000'000);

		std::for_each(data.begin(), data.end(), [](int& i) { i = rand() % 100;  });

		std::vector<RLE> out_data;
		parle.encode(data, out_data);
	}
	catch (const std::exception& e) {
		cerr << e.what() << endl;
	}

	return 0;
}

