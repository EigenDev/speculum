// C++ program to implement Custom array
// class
// import <iostream>
#ifndef CUSTOM_array_HPP
#define CUSTOM_array_HPP
#include <iostream>
#include <algorithm> // for copy
#include <iterator>  // for ostream_iterator
#include <memory>
#ifdef __HIPCC__
#include "hip/hip_runtime.h"
#define gpuMalloc               hipMalloc
#define gpuMallocManaged        hippMallocManaged
#define gpuError_t              hipError_t
#define gpuGetErrorString       hipGetErrorString
#define gpuSuccess              hipSuccess
#define gpuFree                 hipFree
#define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
#define gpuMemcpuDeviceToHost   hipMemcpyDeviceToHost
#define gpuDeviceSynchronize    hipDeviceSynchronize
#define gpuMemcpuDeviceToDevice hipMemcpyDeviceToDevice
#elif defined(__NVCC__)
#include "cuda_runtime.h"
#define gpuMalloc               cudaMalloc
#define gpuMallocManaged        cudaMallocManaged
#define gpuError_t              cudaError_t
#define gpuGetErrorString       cudaGetErrorString
#define gpuSuccess              cudaSuccess
#define gpuFree                 cudaFree
#define gpuMemcpy               cudaMemcpy
#define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuDeviceSynchronize    cudaDeviceSynchronize
#else 
#define gpuMalloc cudaMalloc
#define gpuMallocManaged cudaMallocManaged
#define gpuError_t cudaError_t
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuDeviceSynchronize  cudaDeviceSynchronize
#endif 

#define GPU_KERNEL   __global__

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(gpuError_t code, const char *file, int line, bool abort=true)
{
   if (code != gpuSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", gpuGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

enum class Platform {CPU = 0, GPU = 1};
constexpr Platform BuildPlatform = Platform::GPU;
using real      = double;
using size_type = size_t;




namespace simbi
{
	// Template class to create array of
	// different data_type
	template <typename DT>
	class ndarray {
	template <typename Deleter>
	using unique_p = std::unique_ptr<DT[], Deleter>;
	private:
		// Variable to store the current capacity
		// of the array
		size_type nd_capacity;

		size_type dimensions;

		// Variable to store the size of the
		// array
		size_type sz;

		std::unique_ptr<DT[]> arr;
		// Device-side array
		auto myGpuMalloc(size_type size) { 
			if constexpr(BuildPlatform == Platform::GPU) {
				void* ptr; 
				gpuMalloc((void**)&ptr, size);
				return ptr; 
			}
		};

		// Device-side array
		auto myGpuMallocManaged(size_type size) { 
			if constexpr(BuildPlatform == Platform::GPU) {
				void* ptr; 
				gpuMallocManaged((void**)&ptr, size);
				return ptr; 
			}
		};
		
		struct gpuDeleter {
			void operator()(DT* ptr) {
				if constexpr(BuildPlatform == Platform::GPU) {
					gpuFree(ptr); 
				}
			 }
		};

		unique_p<gpuDeleter> dev_arr;
	public:
		ndarray();
		~ndarray();
		// Assignment operator
		ndarray& operator=(ndarray rhs);

		void swap(ndarray& rhs);

		// Initilizer list constructor
		ndarray(std::initializer_list<DT> list);

		// Zero-initialize the array with defined size
		ndarray(size_type size);

		// Fill-initialize the array with defined size
		ndarray(size_type size, const DT val);

		// Copy-constructor for array
		ndarray(const ndarray& rhs);

		// Function that returns the number of
		// elements in array after pushing the data
		constexpr void push_back(const DT&);

		// function that returns the popped element
		constexpr void pop_back();

		// fucntion to resize ndarray
		constexpr void resize(size_type new_size);

		// fucntion to resize ndarray
		constexpr void resize(size_type new_size, const DT new_value);

		// Function that return the size of array
		constexpr size_type size() const;
		constexpr size_type capacity() const;
		constexpr size_type ndim() const;

		// Access operator (mutable)
		template <typename IndexType>
		constexpr DT& operator[](IndexType);

		// Const-access operator (read-only)
		template<typename IndexType>
		constexpr DT operator[](IndexType) const ;

		// Some math operator overloads
		constexpr ndarray& operator*(double);
		constexpr ndarray& operator*=(double);
		constexpr ndarray& operator/(double);
		constexpr ndarray& operator/=(double);
		constexpr ndarray& operator+=(const ndarray& rhs);



		// Check if ndarray is empty
		bool empty() const;

		// get pointers to underlying data ambigiously, on host, or on gpu
		DT* data();
		DT* host_data();
		DT* dev_data();
		// Iterator Class
		class iterator {
		private:
			// Dynamic array using pointers
			DT* ptr;
		public:
			using iterator_category = std::forward_iterator_tag;;
			using value_type        = DT;
			using difference_type   = void;
			using pointer           = void;
			using reference         = void;
			explicit iterator()
				: ptr(nullptr)
			{
			}
			explicit iterator(DT* p)
				: ptr(p)
			{
			}
			bool operator==(const iterator& rhs) const
			{
				return ptr == rhs.ptr;
			}
			bool operator!=(const iterator& rhs) const
			{
				return !(*this == rhs);
			}
			DT operator*() const
			{
				return *ptr;
			}
			iterator& operator++()
			{
				++ptr;
				return *this;
			}
			iterator operator++(int)
			{
				iterator temp(*this);
				++*this;
				return temp;
			}
		};

		// Begin iterator
		iterator begin() const;

		// End iterator
		iterator end() const;
		
		// back of container
		DT  back() const;
		DT& back();
		DT  front() const;
		DT& front();


		// GPU memeory copy helpers
		void copyToGpu();
		void copyFromGpu();
		void copyBetweenGpu(const ndarray &rhs);

	}; // end ndarray class declaration

} // namespace simbi

// Type trait 
template <typename T>
struct is_ndarray {
	static constexpr bool value = false;
};

template <typename T>
struct is_2darray {
	static constexpr bool value = false;
};

template <typename T>
struct is_3darray {
	static constexpr bool value = false;
};

template <typename T>
struct is_1darray {
	static constexpr bool value = false;
};

template<typename U>
struct is_ndarray<simbi::ndarray<U>>
{
	static constexpr bool value = true;
};

template<typename U>
struct is_1darray<simbi::ndarray<U>>
{
	static constexpr bool value = true;
};

template<typename U>
struct is_2darray<simbi::ndarray<simbi::ndarray<U>>>
{
	static constexpr bool value = true;
};

template<typename U>
struct is_3darray<simbi::ndarray<simbi::ndarray<simbi::ndarray<U>>>>
{
	static constexpr bool value = true;
};


#include "custom_array.tpp"
#endif 