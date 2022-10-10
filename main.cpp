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
	const auto n = 1 << 15;
	simbi::ndarray<int> q(n);
	simbi::ndarray<double> a = {1.0, 2.0, 3.0};
	simbi::ndarray<double> b = {5.0, 6.0, 7.0};
	simbi::ndarray<simbi::ndarray<double>> m(10, simbi::ndarray<double>(10));
	simbi::ndarray<simbi::ndarray<simbi::ndarray<double>>> k(2, simbi::ndarray<simbi::ndarray<double>>(10, simbi::ndarray<double>(10, 0)));
	a.push_back(4.0);
	a += b;
	a.swap(b);
	std::cout << a << "\n";
	std::cout << b << "\n";
	a.resize(100, 10.0);
	std::cout << a << "\n";
	std::cout << k.ndim() << ", "<<  m.ndim() << ", " << a.ndim() << "\n";
	std::cout << k << "\n";
	// q.copyToGpu();
	// int nthreads = 256;
	// int nblocks  = (n + nthreads  - 1) / nthreads;
	// gpu_print<<<nblocks, nthreads>>>(q.get_dev_data(), q.size());
	// q.copyFromGpu();
	// std::cout << "host values: " << q << "\n";
	// gpuErrchk(gpuDeviceSynchronize());
	return 0;
}