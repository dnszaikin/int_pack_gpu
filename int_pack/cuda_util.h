#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

namespace cuda {

	class cuda_exception : public std::exception {
	private:
		std::string _err_msg;

	public:

		cuda_exception(std::string msg, cudaError_t err)  {
			_err_msg = "Error: " + msg + ", Desription: " + cudaGetErrorString(err);
		}

		const char * what() const throw ()
		{
			return _err_msg.c_str();
		}
	};

	inline int get_mp_count(int device = 0)
	{
		cudaError_t cudaStatus;

		int mp_cnt;

		cudaStatus = cudaDeviceGetAttribute(&mp_cnt, cudaDevAttrMultiProcessorCount, device);

		if (cudaStatus != cudaSuccess) {
			throw cuda_exception("cudaDeviceGetAttribute failed!", cudaStatus);
		}

		return mp_cnt;
	}

	inline void init(int device = 0)
	{
		cudaError_t cudaStatus;

		cudaStatus = cudaSetDevice(device);

		if (cudaStatus != cudaSuccess) {
			throw cuda_exception("cudaSetDevice failed!", cudaStatus);
		}
	}

	inline void uninit()
	{
		cudaDeviceReset();
	}
}

