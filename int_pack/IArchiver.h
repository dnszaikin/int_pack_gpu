#pragma once
#include <vector>

struct RLE {
	int symbol;
	int count;
};

class IArchiver {
public:
	IArchiver() {

	}

	virtual void encode(const std::vector<int>&, std::vector<RLE>&) = 0;
	virtual void decode() = 0;

	virtual ~IArchiver() {

	}
};