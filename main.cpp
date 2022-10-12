#include "custom_array.hpp" 
#include <iostream> 
#include <string>
#include <vector> 

template <typename T>
GPU_KERNEL void gpu_print(T* arr, const int n)
{
	const auto ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < n) {
		arr[ii] += ii * 2;
		printf("gpu element %d reads: %d\n", ii, arr[ii]);
	}

}

template <typename T>
void cpu_print(T* arr, const int n)
{
	const auto ii = 0;
	if (ii < n) {
		arr[ii] += 2;
		printf("cpu element %d reads: %d\n", ii, arr[ii]);
	}
}
// Driver code
int main()
{	
	std::vector<int> vec(10);
	speculum::ndarray<int> arr1d(10, 0);
	speculum::ndarray<speculum::ndarray<double>> arr2d(10, speculum::ndarray<double>(10, 0));
	speculum::ndarray<speculum::ndarray<speculum::ndarray<double>>> arr3d(2, speculum::ndarray<speculum::ndarray<double>>(10, speculum::ndarray<double>(10, 0)));
	std::cout << arr1d << "\n";
	std::cout << arr2d << "\n";
	std::cout << arr3d << "\n";
	const auto n = 1 << 4;
	auto q = speculum::ndarray<int>(n, 0);
	q.copyToGpu();
	int nthreads = 256;
	int nblocks  = (n + nthreads  - 1) / nthreads;
	gpu_print<<<nblocks, nthreads>>>(q.dev_data(), q.size());
	q.copyFromGpu();
	std::cout << "host values: " << q << "\n";
	gpuErrchk(gpuDeviceSynchronize());
	return 0;
}