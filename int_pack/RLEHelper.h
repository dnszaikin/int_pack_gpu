#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <string>

class RLEHelper {
protected:
	bool _verbose = false;

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
			std::cout << "printng " << array_name << std::endl;
			for (int i = 0; i < vec.size(); i++) {
				std::cout << vec[i] << " ";
			}

			std::cout << std::endl << std::endl;
		}
	}

public:
	RLEHelper() {

	}

	void set_verbose() {
		_verbose = true;
	}

	virtual ~RLEHelper() {

	}
};