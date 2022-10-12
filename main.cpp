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
	simbi::ndarray<int> arr1d(10, 0);
	simbi::ndarray<simbi::ndarray<double>> arr2d(10, simbi::ndarray<double>(10, 0));
	simbi::ndarray<simbi::ndarray<simbi::ndarray<double>>> arr3d(2, simbi::ndarray<simbi::ndarray<double>>(10, simbi::ndarray<double>(10, 0)));
	std::cout << arr1d << "\n";
	std::cout << arr2d << "\n";
	std::cout << arr3d << "\n";
	#ifdef __HIPCC__
    printf("__HIPCC__ is defined\n");
	#endif
	#ifdef __CUDACC__
		printf("__CUDACC__ is defined\n");
	#endif
	#ifdef __HCC__
		printf("__HCC__ is defined\n");
	#endif
	// const auto n = 1 << 4;
	// auto q = simbi::ndarray<int>(n, 0);
	// q.copyToGpu();
	// int nthreads = 256;
	// int nblocks  = (n + nthreads  - 1) / nthreads;
	// gpu_print<<<nblocks, nthreads>>>(q.dev_data(), q.size());
	// q.copyFromGpu();
	// std::cout << "host values: " << q << "\n";
	// gpuErrchk(gpuDeviceSynchronize());
	return 0;
}