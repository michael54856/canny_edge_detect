#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
// #include <cmath>
#include <cuda.h>

// CUDA 核心函數：高斯模糊
__global__ void gaussian_blur_kernel(const uchar* input, uchar* output, int width, int height, const float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
    // if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int half_kernel = kernel_size / 2;
        float sum = 0.0f;
        for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
            for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                // similar to padding with border values
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                // int nx = x + kx;
                // int ny = y + ky;
                sum += input[ny * width + nx] * kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
            }
        }
        output[y * width + x] = static_cast<uchar>(min(255.0f, max(0.0f, sum)));
    }
}

// CUDA 核心函數：Sobel 梯度計算
// __global__ void sobel_kernel(const uchar* input, float* grad_x, float* grad_y, int width, int height) {
__global__ void sobel_kernel(const uchar* input, float* magnitude, float* direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X kernel
        float gx =  -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)] 
                    -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                    -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];
        // Sobel Y kernel
        float gy =  -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                    +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

        // combine gradient_magnitude_direction_kernel into here
        // to reduce the kernel initialization overhead and reduce memory access and usage
        
        // grad_x[y * width + x] = gx;
        // grad_y[y * width + x] = gy;
        int idx = y * width + x;
        magnitude[idx] = sqrtf(gx * gx + gy * gy);
        direction[idx] = atan2f(gy, gx) * 180.0f / M_PI;
    }
}

// CUDA 核心函數：非極大值抑制
// __global__ void non_maximum_suppression_kernel(const float* magnitude, const float* direction, float* output, int width, int height) {
__global__ void non_maximum_suppression_kernel(const float* magnitude, const float* direction, uchar* output, int width, int height,
    float lowThreshold, float highThreshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        float angle = direction[idx];
        float mag = magnitude[idx];
        float q = 255.0f, r = 255.0f;

        angle = angle < 0 ? angle + 180 : angle;

        // 根據角度選擇方向
        if ((angle >= 0 && angle < 22.5) || angle >= 157.5) {
            q = magnitude[idx + 1];
            r = magnitude[idx - 1];
        } else if (angle >= 112.5) {
            q = magnitude[idx + width + 1];
            r = magnitude[idx - width - 1];
        } else if (angle >= 67.5) {
            q = magnitude[idx + width];
            r = magnitude[idx - width];
        } else if (angle >= 22.5) {
            q = magnitude[idx + width - 1];
            r = magnitude[idx - width + 1];
        }
        // Combine doubleThresholdKernel into here to reduce kernel initialization overhead
        // and reduce memory access and usage
        
        // output[idx] = (mag >= q && mag >= r) ? mag : 0.0f;
        float pixel = (mag >= q && mag >= r) ? mag : 0.0f;
        if (pixel < lowThreshold) {
            output[idx] = 0;
        } else if (pixel > highThreshold) {
            output[idx] = 255;
        } else {
            output[idx] = 128;
        }
    }
}

// CUDA kernel for Edge Tracking
__global__ void edgeTrackingKernel(unsigned char* input, unsigned char* output, 
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        if (input[y * width + x] == 255) {
            output[y * width + x] = 255;
            
            // Check 8-neighborhood
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (input[(y + dy) * width + (x + dx)] == 128) {
                        output[(y + dy) * width + (x + dx)] = 255;
                    }
                }
            }
        }
    }
}

// Main CUDA Canny Edge Detection Function
void cudaCannyEdgeDetection(cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;
    size_t size = width * height * sizeof(unsigned char);
    size_t floatSize = width * height * sizeof(float);

    // Device memory allocation
    unsigned char *d_input, *d_blurred, *d_edges, *d_output;
    // float *d_gradX, *d_gradY, *d_magnitude, *d_direction, *d_suppressed;
    float *d_magnitude, *d_direction;
    float *d_kernel;

#ifdef ENABLE_TIMING_1
    // 創建 CUDA 事件用於計時
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
#endif
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_blurred, size);
    // cudaMalloc(&d_gradX, floatSize);
    // cudaMalloc(&d_gradY, floatSize);
    cudaMalloc(&d_magnitude, floatSize);
    cudaMalloc(&d_direction, floatSize);
    // cudaMalloc(&d_suppressed, floatSize);
    cudaMalloc(&d_edges, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_kernel, 25 * sizeof(float));  // 5x5 kernel

#ifdef ENABLE_TIMING_1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsCudaMalloc = 0;
    cudaEventElapsedTime(&millisecondsCudaMalloc, start, stop);
    printf("cudaMalloc Execution Time: %f ms\n", millisecondsCudaMalloc);
    
    cudaEventRecord(start);
#endif
    // Copy input to device
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);

#ifdef ENABLE_TIMING_1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsCudaInputCopy = 0;
    cudaEventElapsedTime(&millisecondsCudaInputCopy, start, stop);
    printf("cuda input copy Execution Time: %f ms\n", millisecondsCudaInputCopy);

    cudaEventRecord(start);
#endif    

    // Create Gaussian kernel on host
    float h_kernel[25];
    float sigma = 1.0f;
    float sum = 0.0f;
    int kernelSize = 5;
    int center = kernelSize / 2;
    
    for (int x = 0; x < kernelSize; x++) {
        for (int y = 0; y < kernelSize; y++) {
            int x_dist = x - center;
            int y_dist = y - center;
            h_kernel[x * kernelSize + y] = exp(-(x_dist * x_dist + y_dist * y_dist) / (2 * sigma * sigma));
            sum += h_kernel[x * kernelSize + y];
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        h_kernel[i] /= sum;
    }
    
    // Copy kernel to device
    cudaMemcpy(d_kernel, h_kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    
#ifdef ENABLE_TIMING_1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsCudaMakeKernel = 0;
    cudaEventElapsedTime(&millisecondsCudaMakeKernel, start, stop);
    printf("cuda make kernel Execution Time: %f ms\n", millisecondsCudaMakeKernel);

    cudaEventRecord(start);
#endif
    // 1. Gaussian Blur
    gaussian_blur_kernel<<<gridSize, blockSize>>>(
        d_input, d_blurred, width, height, d_kernel, kernelSize
    );
#ifdef ENABLE_TIMING_1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsCudaBlur = 0;
    cudaEventElapsedTime(&millisecondsCudaBlur, start, stop);
    printf("Gaussian Blur Kernel Execution Time: %f ms\n", millisecondsCudaBlur);
    
    cudaEventRecord(start);
#endif
    // 2. Sobel Gradient
    // 3. Gradient Magnitude & Direction
    sobel_kernel<<<gridSize, blockSize>>>(
        d_blurred, d_magnitude, d_direction, width, height
    );

#ifdef ENABLE_TIMING_1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsCudaSobel = 0;
    cudaEventElapsedTime(&millisecondsCudaSobel, start, stop);
    printf("Sobel Gradient Kernel Execution Time: %f ms\n", millisecondsCudaSobel);
    
    cudaEventRecord(start);
#endif
    
    // 4. Non-Maximum Suppression
    // 5. Double Thresholding
    non_maximum_suppression_kernel<<<gridSize, blockSize>>>(
        d_magnitude, d_direction, d_edges, width, height, 20.0f, 50.0f
    );

#ifdef ENABLE_TIMING_1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsCudaNMS = 0;
    cudaEventElapsedTime(&millisecondsCudaNMS, start, stop);
    printf("Non-Maximum Suppression Kernel Execution Time: %f ms\n", millisecondsCudaNMS);
    
    cudaEventRecord(start);
#endif
    // 6. Edge Tracking
    edgeTrackingKernel<<<gridSize, blockSize>>>(
        d_edges, d_output, width, height
    );

#ifdef ENABLE_TIMING_1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsCudaEdgeTracking = 0;
    cudaEventElapsedTime(&millisecondsCudaEdgeTracking, start, stop);
    printf("Edge Tracking Kernel Execution Time: %f ms\n", millisecondsCudaEdgeTracking);

    cudaEventRecord(start);
#endif
    // Copy result back to host
    output = cv::Mat(height, width, CV_8UC1);
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);

#ifdef ENABLE_TIMING_1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsCudaOutputCopy = 0;
    cudaEventElapsedTime(&millisecondsCudaOutputCopy, start, stop);
    printf("cuda output copy Execution Time: %f ms\n", millisecondsCudaOutputCopy);
    
    cudaEventRecord(start);
#endif

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blurred);
    // cudaFree(d_gradX);
    // cudaFree(d_gradY);
    cudaFree(d_magnitude);
    cudaFree(d_direction);
    // cudaFree(d_suppressed);
    cudaFree(d_edges);
    cudaFree(d_kernel);

#ifdef ENABLE_TIMING_1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsCudaFree = 0;
    cudaEventElapsedTime(&millisecondsCudaFree, start, stop);
    printf("Edge Free Mem Execution Time: %f ms\n", millisecondsCudaFree);

    // 計算總執行時間
    float totalKernelTime = millisecondsCudaBlur + millisecondsCudaSobel + 
                             millisecondsCudaNMS + millisecondsCudaEdgeTracking +
                             millisecondsCudaMalloc +
                             millisecondsCudaInputCopy + millisecondsCudaOutputCopy +
                             millisecondsCudaMakeKernel + millisecondsCudaFree;
    printf("Total Kernel Execution Time (not accurate, please use `make timing2`): %f ms\n", totalKernelTime);

    // 釋放 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("\n\n");
#endif
}

int main(int argc, char** argv) {

    // 檢查命令行參數是否正確
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << " <out_path> " << std::endl;
        return -1;
    }

    // Read input image
    // cv::Mat input = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    
    if (input.empty()) {
        std::cerr << "Cannot read image" << std::endl;
        return -1;
    }
    
    cv::Mat output;

#ifdef ENABLE_TIMING_1
    for(int i=0;i<30;i++)
    {
        cudaCannyEdgeDetection(input, output);
    }
#elif defined(ENABLE_TIMING_2)
    cudaEvent_t start, stop;
    float min_time=1e4;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i=0;i<30;i++)
    {
        cudaEventRecord(start);
        cudaCannyEdgeDetection(input, output);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Total Execution Time: %f ms\n", milliseconds);
        min_time = std::min(min_time, milliseconds);
    }
    // 釋放 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Min Execution Time: %f ms\n", min_time);
#else
    cudaCannyEdgeDetection(input, output);
#endif
    
    // Save output image
    cv::imwrite(argv[2], output);
    
    return 0;
}