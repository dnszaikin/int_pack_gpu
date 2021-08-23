#pragma once
#include <thrust/host_vector.h>

class IArchiver {
public:
	IArchiver() {

	}

	virtual void encode(const thrust::host_vector<int>&, thrust::host_vector<int>&, thrust::host_vector<int>&) = 0;
	virtual void decode(const thrust::host_vector<int>&, const thrust::host_vector<int>&, thrust::host_vector<int>&) = 0;

	virtual ~IArchiver() {

	}
};