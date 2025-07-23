# Accelerating Canny Edge Detection with Parallelization

## Table of Contents
- [Project Overview](#project-overview)  
- [How to Reproduce the Experiment](#how-to-reproduce-the-experiment)
  * [How to Build Canny Edge Detection Algorithm](#how-to-build-canny-edge-detection-algorithm)
  * [How to Execute the Canny Edge Detection Algorithm](#how-to-execute-the-canny-edge-detection-algorithm)

---

## Project Overview  
This project aims to accelerate the Canny edge detection algorithm using various parallelization techniques to improve performance and make it more suitable for real-time applications. The Canny edge detection algorithm is a widely used method for detecting edges in images, but its computational intensity can limit its real-time applicability. To address this, the project explores the use of multiple parallel computing paradigms:

- SIMD (Single Instruction, Multiple Data): This technique enhances performance by processing multiple data points simultaneously, reducing the time required for operations such as gradient calculation.
- OpenMP (Open Multi-Processing): OpenMP leverages multithreading to efficiently utilize multiple CPU cores, speeding up steps like Gaussian Blur and Non-Maximum Suppression.
- CUDA (Compute Unified Device Architecture): CUDA is employed to parallelize the algorithm on the GPU, allowing for massive parallel execution, which significantly accelerates edge detection tasks.

The implementation parallelizes key steps of the Canny algorithm, including Gaussian Blur, Gradient Calculation using the Sobel Operator, Non-Maximum Suppression, Double Thresholding, and Edge Tracking by Hysteresis. By optimizing these steps across different parallel computing paradigms, the project demonstrates significant speedups, making the Canny edge detection more efficient and capable of handling real-time image processing tasks.

## How to reproduce the experiment
### How to Build Canny Edge Detection Algorithm

### 1. **make all**
This is the default target. It will build all the targets listed below.
```
make all
```

### 2. **make canny_serial**
This target builds the serial version of the Canny edge detection algorithm (C++ with OpenCV).
```
make canny_serial
```

### 3. **make canny_simd**
This target builds the SIMD-optimized version of the Canny edge detection algorithm (C++ with OpenCV + SIMD optimizations).
```
make canny_simd
```

### 4. **make canny_openmp**
This target builds the OpenMP-parallelized version of the Canny edge detection algorithm (C++ with OpenCV + OpenMP support).
```
make canny_openmp
```

### 5. **make canny_simd_openmp**
This target builds the version of Canny edge detection with both SIMD and OpenMP optimizations (C++ with OpenCV + SIMD + OpenMP support).
```
make canny_simd_openmp
```

### 6. **make canny_cuda**
This target builds the CUDA version of the Canny edge detection algorithm (CUDA with OpenCV).
```
make canny_cuda
```

### 7. **make clean**
This target removes all the generated files.
```
make clean
```

### How to Execute the Canny Edge Detection Algorithm

### 1. **canny_simd, canny_openmp, canny_simd_openmp, canny_serial**
```
./canny_simd
./canny_openmp
./canny_simd_openmp
./canny_serial
```
- The input file should be named ```input.jpg```

### 2. **canny_cuda**
```
./canny_cuda input.jpg output.jpg
```
- ```./canny_cuda```: This runs the compiled CUDA version of the Canny edge detection algorithm.
- ```input.jpg```: This is the input image file you want to process. Replace this with the path to your own image.
- ```output.jpg```: This is the output image where the result of the edge detection will be saved. You can specify the desired output file name here.
