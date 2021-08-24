#include <iostream>

#include <omp.h>

#if __cplusplus >= 201703L
#include <execution>
#endif

#include <chrono>

#include "ParallelRLE.h"
#include "CpuRLE.h"
#include "ParallelElegantPairs.h"

using namespace std;

using hrc = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<std::chrono::steady_clock>;

class timer {
private:
	time_point _start;
	time_point _finish;
public:
	void start() {
		_start = hrc::now();
	}
	void stop() {
		_finish = hrc::now();
	}

	long long duration_nanos() const {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(_finish - _start).count();
	}

	long long duration_millis() const {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(_finish - _start).count()/1000/1000;
	}

};

constexpr int MAX_N{ 1'000'000 };

int main()
{
	std::random_device rd;     
	std::mt19937 rng(rd());    
	std::uniform_int_distribution<int> uni(1, 100); 

	try {

		timer t;

		t.start();
		thrust::host_vector<int> data(MAX_N);
		thrust::host_vector<int> out_data(MAX_N); //suppose we have some meta data to determine decomp size

		//generating badly compressible data
#if __cplusplus >= 201703L

		std::for_each(std::execution::par_unseq, data.begin(), data.end(), [&](int& i) { i = uni(rng);  });
#else
		long i{ 0 };

		#pragma omp parallel for
		for (i = 0; i < data.size(); ++i) {
			data[i] = uni(rng);
		}
#endif

		t.stop();

		cout << "Prepeare data takes: " << t.duration_millis() << " millis\n";
		
		t.start();

		thrust::host_vector<int> h_symbols{};
		thrust::host_vector<int> h_counts{};

		ParallelRLE parle{};

		//parle.set_verbose();

		parle.encode(data, h_symbols, h_counts);

		t.stop();

		cout << "GPU Encode data takes: " << t.duration_millis() << " millis\n";

		ParallelElegantPairs papairs;

		//papairs.set_verbose();

		t.start();
		
		papairs.encode(h_symbols, h_counts, out_data);

		t.stop();

		cout << "Parallel elegant pairs encode takes: " << t.duration_millis() << " millis\n";
		cout << "Original size: " << data.size() << endl;
		cout << "Compressed size: " << h_counts.size()*2 << endl;
		cout << "Compressed size with elegant pairs: " << out_data.size() << endl;

		t.start();
		size_t s{ h_symbols.size() };

		h_symbols.clear();
		h_symbols.resize(s);
		h_counts.clear();
		h_counts.resize(s);

		papairs.decode(out_data, h_symbols, h_counts);

		t.stop();

		cout << "Parallel elegant pairs decode takes: " << t.duration_millis() << " millis\n";
		cout << "Uncompressed size with elegant pairs: " << h_symbols.size() * 2 << endl;
		
		t.start();

		out_data.clear();
		out_data.resize(MAX_N);

		parle.decode(h_symbols, h_counts, out_data);

		t.stop();

		cout << "GPU Decode data takes: " << t.duration_millis() << " millis\n";

		t.start();

		volatile  bool equal = true;

		if (data.size() == out_data.size()) {
			
#if __cplusplus >= 201703L
			equal = std::equal(std::execution::par_unseq, data.begin(), data.end(), out_data.begin());
#else
			long i{ 0 };

			#pragma omp parallel for shared(equal)
			for (i = 0; i < data.size(); ++i) {
				if (!equal) continue;
				if (data[i] == out_data[i]) {
					equal = false;
				}
			}
#endif
		}
		else {
			equal = false;
		}

		t.stop();

		cout << "Parallel Verify data takes: " << t.duration_millis() << " millis\n";
		cout << "Vectors equals: " << std::boolalpha << equal << endl;

		h_symbols.clear();
		h_counts.clear();
		out_data.clear();
		
		CpuRLE rle;

		t.start();

		rle.encode(data, h_symbols, h_counts);

		t.stop();

		cout << "CPU Encode data takes: " << t.duration_millis() << " millis\n";

		t.start();
		
		out_data.resize(MAX_N);

		rle.decode(h_symbols, h_counts, out_data);

		t.stop();

		cout << "CPU Decode data takes: " << t.duration_millis() << " millis\n";

		t.start();

		auto success{ std::equal(data.begin(), data.end(), out_data.begin()) };

		t.stop();

		cout << "CPU Verify data takes: " << t.duration_millis() << " millis\n";

		cout << "Vectors equals: " << std::boolalpha << success << endl;

	}
	catch (const std::exception& e) {
		cerr << e.what() << endl;
	}

	return 0;
}

