#include <iostream>

#include <omp.h>

#if __cplusplus == 201703L
#include <execution>
#endif

#include <chrono>

#include "ParallelRLE.h"
#include "CpuRLE.h"

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
		ParallelRLE parle(false);

		timer t;

		t.start();
		thrust::host_vector<int> data(MAX_N);

#if __cplusplus == 201703L

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

		parle.encode(data, h_symbols, h_counts);

		t.stop();

		cout << "GPU Encode data takes: " << t.duration_millis() << " millis\n";

		t.start();

		thrust::host_vector<int> out_data(MAX_N); //suppose we have meta data to determine size

		parle.decode(h_symbols, h_counts, out_data);

		t.stop();

		cout << "GPU Decode data takes: " << t.duration_millis() << " millis\n";

		t.start();

		volatile  bool equal = true;

		if (data.size() == out_data.size()) {
			
#if __cplusplus == 201703L
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
			
		rle.decode(h_symbols, h_counts, out_data);

		t.stop();

		cout << "CPU Decode data takes: " << t.duration_millis() << " millis\n";

		t.start();

		auto success{ std::equal(data.begin(), data.end(), out_data.begin()) };

		t.stop();

		cout << "CPU Verify data takes: " << t.duration_millis() << " millis\n";

		cout << "Vectors equals: " << std::boolalpha << equal << endl;

	}
	catch (const std::exception& e) {
		cerr << e.what() << endl;
	}

	return 0;
}

