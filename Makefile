# Compiler settings
CXX = g++
NVCC = nvcc

CXXFLAGS_COMMON = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
CXXFLAGS_SIMD = -mavx2 -mfma -O3 -march=native
CXXFLAGS_OPENMP = -fopenmp
CUDAFLAGS = -std=c++11 -O3 -DENABLE_TIMING_1

# Targets
TARGETS = canny_serial canny_simd canny_openmp canny_simd_openmp canny_cuda

# Default target
all: $(TARGETS)

# Canny serial (C++ with OpenCV)
canny_serial: cannySerial.cpp
	$(CXX) $< -o $@ $(CXXFLAGS_COMMON)

# Canny SIMD (C++ with OpenCV + SIMD optimizations)
canny_simd: cannySIMD.cpp
	$(CXX) $< -o $@ $(CXXFLAGS_COMMON) $(CXXFLAGS_SIMD)

# Canny OpenMP (C++ with OpenCV + OpenMP support)
canny_openmp: cannyOpenMP.cpp
	$(CXX) $< -o $@ $(CXXFLAGS_COMMON) $(CXXFLAGS_OPENMP)

# Canny SIMD + OpenMP (C++ with OpenCV + SIMD + OpenMP support)
canny_simd_openmp: cannySIMD_OpenMP.cpp
	$(CXX) $< -o $@ $(CXXFLAGS_COMMON) $(CXXFLAGS_SIMD) $(CXXFLAGS_OPENMP)

# Canny CUDA (CUDA with OpenCV)
canny_cuda: cannyCUDA.cu
	$(NVCC) $< -o $@ $(CUDAFLAGS) $(CXXFLAGS_COMMON)

# Clean up generated files
clean:
	rm -f $(TARGETS)

.PHONY: all clean

