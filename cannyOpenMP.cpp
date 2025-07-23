#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>


// 輔助函數：獲取二維索引
inline int get_index(int y, int x, int width) {
    return y * width + x;
}

// 從OpenCV Mat轉換
float* mat_to_array(const cv::Mat& mat) {
    int size = mat.rows * mat.cols;
    float* data = new float[size];
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            data[get_index(y, x, mat.cols)] = mat.at<uchar>(y, x);
        }
    }
    return data;
}

// 轉換回OpenCV Mat
cv::Mat array_to_mat(float* data, int height, int width) {
    cv::Mat mat(height, width, CV_8UC1);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            mat.at<uchar>(y, x) = static_cast<uchar>(std::min(255.0f, std::max(0.0f, data[get_index(y, x, width)])));
        }
    }
    return mat;
}

// 生成高斯濾波kernel
float* generate_gaussian_kernel(int size, float sigma) {
    float* kernel = new float[size * size];
    float sum = 0.0f;
    int center = size / 2;

    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            int x_dist = x - center;
            int y_dist = y - center;
            kernel[get_index(x, y, size)] = std::exp(-(x_dist * x_dist + y_dist * y_dist) / (2 * sigma * sigma));
            sum += kernel[get_index(x, y, size)];
        }
    }

    // 歸一化
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            kernel[get_index(x, y, size)] /= sum;
        }
    }

    return kernel;
}

// 高斯模糊
float* gaussian_blur(float* input, int height, int width, int kernel_size = 5, float sigma = 1.0f) {
    float* kernel = generate_gaussian_kernel(kernel_size, sigma);
    float* blurred = new float[height * width]();
    int offset = kernel_size / 2;

	#pragma omp parallel for 
    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            float pixel_value = 0.0f;
            for (int ky = -offset; ky <= offset; ky++) {
                for (int kx = -offset; kx <= offset; kx++) {
                    pixel_value += input[get_index(y + ky, x + kx, width)] * kernel[get_index(ky + offset, kx + offset, kernel_size)];
                }
            }
            blurred[get_index(y, x, width)] = std::min(255.0f, std::max(0.0f, pixel_value));
        }
    }

    delete[] kernel;
    return blurred;
}

// Sobel梯度
std::pair<float*, float*> sobel_gradient(float* input, int height, int width) {
    float* gradient_x = new float[height * width]();
    float* gradient_y = new float[height * width]();

    // Sobel x方向kernel
    int kernel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    // Sobel y方向kernel
    int kernel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

	#pragma omp parallel for 
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    gx += input[get_index(y + dy, x + dx, width)] * kernel_x[dy + 1][dx + 1];
                    gy += input[get_index(y + dy, x + dx, width)] * kernel_y[dy + 1][dx + 1];
                }
            }
            gradient_x[get_index(y, x, width)] = gx;
            gradient_y[get_index(y, x, width)] = gy;
        }
    }

    return {gradient_x, gradient_y};
}

// 計算梯度幅值和方向
std::pair<float*, float*> compute_gradient_magnitude_and_direction
(
    float* gradient_x, float* gradient_y, int height, int width) {
    
    float* magnitude = new float[height * width];
    float* direction = new float[height * width];

	#pragma omp parallel for 
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float gx = gradient_x[get_index(y, x, width)];
            float gy = gradient_y[get_index(y, x, width)];

            magnitude[get_index(y, x, width)] = std::sqrt(gx * gx + gy * gy);
            direction[get_index(y, x, width)] = std::atan2(gy, gx) * 180.0f / M_PI;
        }
    }

    return {magnitude, direction};
}

// 非極大值抑制
float* non_maximum_suppression(float* magnitude, float* direction, int height, int width) 
{
    float* suppressed = new float[height * width]();

	#pragma omp parallel for 
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float current_angle = direction[get_index(y, x, width)];
            float current_magnitude = magnitude[get_index(y, x, width)];

            // 標準化角度到 [0, 180]
            current_angle = current_angle < 0 ? current_angle + 180 : current_angle;

            float q = 255.0f, r = 255.0f;
            if ((0 <= current_angle && current_angle < 22.5f) || 157.5f <= current_angle) {
                q = magnitude[get_index(y, x+1, width)];
                r = magnitude[get_index(y, x-1, width)];
            }
            else if (112.5f <= current_angle) {
                q = magnitude[get_index(y-1, x-1, width)];
                r = magnitude[get_index(y+1, x+1, width)];
            }
            else if (67.5f <= current_angle) {
                q = magnitude[get_index(y+1, x, width)];
                r = magnitude[get_index(y-1, x, width)];
            }
            else if (22.5f <= current_angle) {
                q = magnitude[get_index(y+1, x-1, width)];
                r = magnitude[get_index(y-1, x+1, width)];
            }

            suppressed[get_index(y, x, width)] = (current_magnitude >= q && current_magnitude >= r) ? current_magnitude : 0;
        }
    }

    return suppressed;
}

// 雙閾值檢測
float* double_threshold(float* input, int height, int width, float low_ratio = 0.1f, float high_ratio = 0.3f) 
{
    float low_threshold = 20.0f;
    float high_threshold = 50.0f;

    float* thresholded = new float[height * width];
	
	#pragma omp parallel for 
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float pixel = input[get_index(y, x, width)];
            if (pixel < low_threshold) {
                thresholded[get_index(y, x, width)] = 0.0f;
            }
            else if (pixel > high_threshold) {
                thresholded[get_index(y, x, width)] = 255.0f;
            }
            else {
                thresholded[get_index(y, x, width)] = 128.0f;  // 不確定邊緣
            }
        }
    }

    return thresholded;
}

// 邊緣追蹤
float* edge_tracking(float* thresholded, int height, int width) 
{
    float* edges = new float[height * width]();

	#pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (thresholded[get_index(y, x, width)] == 255.0f) {
                edges[get_index(y, x, width)] = 255.0f;
                // 追蹤8鄰域
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (thresholded[get_index(y+dy, x+dx, width)] == 128.0f) {
                            edges[get_index(y+dy, x+dx, width)] = 255.0f;
                        }
                    }
                }
            }
        }
    }

    return edges;
}

// Canny邊緣檢測主函數
float* canny_edge_detection(float* input, int height, int width) {
	
	auto time1 = std::chrono::high_resolution_clock::now();
    // 1. 高斯模糊
    float* blurred = gaussian_blur(input, height, width);
	
	auto time2 = std::chrono::high_resolution_clock::now();

    // 2. Sobel梯度
    auto gradients = sobel_gradient(blurred, height, width);
    float* gradient_x = gradients.first;
    float* gradient_y = gradients.second;

    // 3. 計算梯度幅值和方向
    auto grad_mag_dir = compute_gradient_magnitude_and_direction(gradient_x, gradient_y, height, width);
    float* magnitude = grad_mag_dir.first;
    float* direction = grad_mag_dir.second;
	
	auto time3 = std::chrono::high_resolution_clock::now();

    // 4. 非極大值抑制
    float* suppressed = non_maximum_suppression(magnitude, direction, height, width);

    // 5. 雙閾值檢測
    float* thresholded = double_threshold(suppressed, height, width);
	
	auto time4 = std::chrono::high_resolution_clock::now();

    // 6. 邊緣追蹤
    float* edges = edge_tracking(thresholded, height, width);
	
	auto time5 = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<double> elapsed_1 = time2 - time1;
	std::chrono::duration<double> elapsed_2 = time3 - time2;
	std::chrono::duration<double> elapsed_3 = time4 - time3;
	std::chrono::duration<double> elapsed_4 = time5 - time4;
	std::chrono::duration<double> elapsed_final = time5 - time1;
	
	std::cout << "Step 1: Gaussian Blur    : \t\t" << elapsed_1.count() << std::endl;
	std::cout << "Step 2: Sobel & gradient : \t\t" << elapsed_2.count() << std::endl;
	std::cout << "Step 3: NMS & Threshold  : \t\t" << elapsed_3.count() << std::endl;
	std::cout << "Step 4: Edge Tracking    : \t\t" << elapsed_4.count() << std::endl;
	std::cout << "Final :                  : \t\t" << elapsed_final.count() << std::endl;

    // 釋放中間結果的記憶體
    delete[] blurred;
    delete[] gradient_x;
    delete[] gradient_y;
    delete[] magnitude;
    delete[] direction;
    delete[] suppressed;
    delete[] thresholded;

    return edges;
}

int main() {
    // 讀取圖像 - 使用OpenCV讀取灰度圖
    cv::Mat input_mat = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    
    if (input_mat.empty()) {
        std::cerr << "can't read image" << std::endl;
        return -1;
    }

    // 轉換為陣列
    float* input_image = mat_to_array(input_mat);

    // 執行Canny邊緣檢測
    float* edges = canny_edge_detection(input_image, input_mat.rows, input_mat.cols);

    // 轉換回OpenCV Mat並保存
    cv::Mat edges_mat = array_to_mat(edges, input_mat.rows, input_mat.cols);
    cv::imwrite("output.jpg", edges_mat);

    // 釋放記憶體
    delete[] input_image;
    delete[] edges;

    return 0;
}