#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <chrono>

// 輔助函數：獲取二維索引
inline int get_index(int y, int x, int width) {
    return y * width + x;
}

// 從OpenCV Mat轉換
float* mat_to_array(const cv::Mat& mat) {
    int size = mat.rows * mat.cols;
	
	float* data = static_cast<float*>(std::aligned_alloc(32, size * sizeof(float)));

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


// Polynomial approximation of exponential function for AVX2
__m256 avx2_exp_ps(__m256 x) {
    // Constants for exponential approximation
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    
    // Polynomial coefficients
    const __m256 cephes_LOG2EF = _mm256_set1_ps(1.44269504088896340735f);
    const __m256 cephes_exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    
    const __m256 cephes_exp_p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 cephes_exp_p1 = _mm256_set1_ps(1.4352314417e-3f);
    const __m256 cephes_exp_p2 = _mm256_set1_ps(8.3496402606e-3f);
    const __m256 cephes_exp_p3 = _mm256_set1_ps(4.1602886268e-2f);
    const __m256 cephes_exp_p4 = _mm256_set1_ps(0.166665036920f);
    const __m256 cephes_exp_p5 = _mm256_set1_ps(0.499999999505f);
    
    // Clamp input
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);
    
    // Decompose exponential
    __m256 fx = _mm256_mul_ps(x, cephes_LOG2EF);
    fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    
    __m256 x_sub = _mm256_fnmadd_ps(fx, cephes_exp_C1, x);
    x_sub = _mm256_fnmadd_ps(fx, cephes_exp_C2, x_sub);
    
    // Polynomial approximation
    __m256 z = _mm256_mul_ps(x_sub, x_sub);
    
    __m256 p = _mm256_fmadd_ps(cephes_exp_p0, x_sub, cephes_exp_p1);
    p = _mm256_fmadd_ps(p, x_sub, cephes_exp_p2);
    p = _mm256_fmadd_ps(p, x_sub, cephes_exp_p3);
    p = _mm256_fmadd_ps(p, x_sub, cephes_exp_p4);
    p = _mm256_fmadd_ps(p, x_sub, cephes_exp_p5);
    p = _mm256_fmadd_ps(p, z, x_sub);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));
    
    // Reconstruct exponential
    __m256i imm0 = _mm256_cvtps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(127));
    imm0 = _mm256_slli_epi32(imm0, 23);
    
    __m256 pow2 = _mm256_castsi256_ps(imm0);
    return _mm256_mul_ps(p, pow2);
}


// 生成高斯濾波kernel
float* generate_gaussian_kernel(int size, float sigma)
{
    // Aligned allocation
	float* kernel = static_cast<float*>(std::aligned_alloc(32, size * 8 * sizeof(float)));
	
	float* temp_sum = static_cast<float*>(std::aligned_alloc(32,  8 * sizeof(float)));

    float sum = 0.0f;
    int center = size / 2;
	
	// Precompute constants
    __m256 sigma_sq_vec = _mm256_set1_ps(2.0f * sigma * sigma);
    __m256 zero_vec = _mm256_setzero_ps();

    for (int x = 0; x < size; x++) 
	{
		int x_dist = x - center;
		
		// Vectorized distance calculation
		__m256 x_dist_vec = _mm256_set1_ps(static_cast<float>(x_dist));
		
		// Prepare y distances 
		__m256 y_dist_vec = _mm256_setr_ps(
			static_cast<float>(0 - center), 
			static_cast<float>(1 - center), 
			static_cast<float>(2 - center), 
			static_cast<float>(3 - center),
			static_cast<float>(4 - center), 
			static_cast<float>(0), 
			static_cast<float>(0), 
			static_cast<float>(0)
		);

		// Calculate squared distances
		__m256 x_sq = _mm256_mul_ps(x_dist_vec, x_dist_vec);
		__m256 y_sq = _mm256_mul_ps(y_dist_vec, y_dist_vec);

		// Sum of squared distances
		__m256 dist_sq = _mm256_add_ps(x_sq, y_sq);

		// Exponential calculation: exp(-(x²+y²)/(2*σ²))
		__m256 exp_arg = _mm256_div_ps(_mm256_sub_ps(zero_vec, dist_sq), sigma_sq_vec);
		__m256 kernel_vals = avx2_exp_ps(exp_arg);

		// Store results
		_mm256_store_ps(&kernel[x*8], kernel_vals);


		_mm256_store_ps(temp_sum, kernel_vals);
		for (int i = 0; i < size; i++) 
		{
			sum += temp_sum[i];
		}
    }
	
	 // Normalization (vectorized)
    __m256 sum_vec = _mm256_set1_ps(sum);

    for (int x = 0; x < size; x++) 
	{
		__m256 kernel_vals = _mm256_load_ps(&kernel[x*8]);
		__m256 normalized_vals = _mm256_div_ps(kernel_vals, sum_vec);
		_mm256_store_ps(&kernel[x*8], normalized_vals);
    }
	
	free(temp_sum);

    return kernel;
}

// 高斯模糊
float* gaussian_blur(float* input, int height, int width, int kernel_size = 5, float sigma = 1.0f) 
{
    float* kernel = generate_gaussian_kernel(kernel_size, sigma);
	
	float* blurred = static_cast<float*>(std::aligned_alloc(32, height * width * sizeof(float)));

    int offset = kernel_size / 2;
	
	#pragma omp parallel for 
    for (int y = offset; y < height - offset; y++) 
	{
        for (int x = offset; x < width - offset; x += 8) // 每次處理 8 個像素
		{ 
			if(x+8 <= width-offset)
			{
				 __m256 pixel_value = _mm256_setzero_ps();

				for (int ky = -offset; ky <= offset; ky++) 
				{
					for (int kx = -offset; kx <= offset; kx++) 
					{
						// 載入輸入數據
						__m256 input_values = _mm256_loadu_ps(&input[get_index(y + ky, x + kx, width)]);

						// 加載 kernel 值
						__m256 kernel_value = _mm256_set1_ps(kernel[get_index(ky + offset, kx + offset, 8)]);
						

						// 融合乘加：pixel_value += input * kernel
						pixel_value = _mm256_fmadd_ps(input_values, kernel_value, pixel_value);
					}
				}

				// 範圍限制到 [0, 255]
				pixel_value = _mm256_min_ps(_mm256_set1_ps(255.0f), _mm256_max_ps(_mm256_set1_ps(0.0f), pixel_value));

				// 儲存到輸出
				_mm256_storeu_ps(&blurred[get_index(y, x, width)], pixel_value);
			}
			else
			{
				for ( ; x < width - offset; x++) 
				{
					float pixel_value = 0.0f;
					for (int ky = -offset; ky <= offset; ky++) 
					{
						for (int kx = -offset; kx <= offset; kx++) 
						{
							pixel_value += input[get_index(y + ky, x + kx, width)] * kernel[get_index(ky + offset, kx + offset, 8)];
						}
					}
					blurred[get_index(y, x, width)] = std::min(255.0f, std::max(0.0f, pixel_value));
				}
			}
           
        }
    }


    return blurred;
}

// Sobel梯度
std::pair<float*, float*> sobel_gradient(float* input, int height, int width) 
{

	float* gradient_x = static_cast<float*>(std::aligned_alloc(32, height * width * sizeof(float)));
	float* gradient_y = static_cast<float*>(std::aligned_alloc(32, height * width * sizeof(float)));

	
	 // Sobel x方向kernel
    const float kernel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

    // Sobel y方向kernel
    const float kernel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

	
	#pragma omp parallel for 
    for (int y = 1; y < height - 1; y++) 
	{
        for (int x = 1; x < width - 1; x += 8) //每次處理 8 個像素
		{
			if(x+8 <= width-1)
			{
				__m256 gx = _mm256_setzero_ps();
				__m256 gy = _mm256_setzero_ps();

				for (int ky = 0; ky < 3; ky++) 
				{
					for (int kx = 0; kx < 3; kx++) 
					{
						// 計算 kernel 的索引
						int kernel_idx = ky * 3 + kx;

						// 載入輸入數據（非對齊載入）
						__m256 input_values = _mm256_loadu_ps(&input[get_index(y + ky - 1, x + kx - 1, width)]);

						// 加載 kernel 值
						__m256 kx_value = _mm256_set1_ps(kernel_x[kernel_idx]);
						__m256 ky_value = _mm256_set1_ps(kernel_y[kernel_idx]);

						// 累積 gx 和 gy
						gx = _mm256_fmadd_ps(input_values, kx_value, gx);
						gy = _mm256_fmadd_ps(input_values, ky_value, gy);
					}
				}

				// 儲存結果
				_mm256_storeu_ps(&gradient_x[get_index(y,x,width)], gx);
				_mm256_storeu_ps(&gradient_y[get_index(y,x,width)], gy);
			}
			else
			{
				for ( ; x < width - 1; x++) 
				{
					float gx = 0.0f, gy = 0.0f;
					int counter = 0;
					for (int dy = -1; dy <= 1; dy++) {
						for (int dx = -1; dx <= 1; dx++) {
							gx += input[get_index(y + dy, x + dx, width)] * kernel_x[counter];
							gy += input[get_index(y + dy, x + dx, width)] * kernel_y[counter];
							counter++;
						}
					}
					gradient_x[get_index(y,x,width)] = gx;
					gradient_y[get_index(y,x,width)] = gy;
				}
			}
           
        }
    }

    return {gradient_x, gradient_y};
}



// 計算梯度幅值和方向
std::pair<float*, float*> compute_gradient_magnitude_and_direction(float* gradient_x, float* gradient_y, int height, int width) 
{		
	float* magnitude = static_cast<float*>(std::aligned_alloc(32, height * width * sizeof(float)));
	float* direction = static_cast<float*>(std::aligned_alloc(32, height * width * sizeof(float)));
	

	#pragma omp parallel for 
    for (int y = 0; y < height; y++) 
	{		
		float atan2_val[8]; 
		int x = 0;
		// 處理向量化部分
        for (; x <= width - 8; x += 8) 
		{
            // 加載 gradient_x 和 gradient_y 的 8 個值
            __m256 gx = _mm256_load_ps(&gradient_x[get_index(y,x,width)]);
            __m256 gy = _mm256_load_ps(&gradient_y[get_index(y,x,width)]);

            // 計算幅值 magnitude = sqrt(gx^2 + gy^2)
            __m256 gx2 = _mm256_mul_ps(gx, gx);
            __m256 gy2 = _mm256_mul_ps(gy, gy);
            __m256 sum = _mm256_add_ps(gx2, gy2);
            __m256 mag = _mm256_sqrt_ps(sum);


			
			for(int i = 0; i < 8; i++)
			{
				atan2_val[i] = std::atan2(gradient_y[get_index(y,x+i,width)], gradient_x[get_index(y,x+i,width)]);
			}
			__m256 dir = _mm256_loadu_ps(atan2_val); 
			
			
            __m256 scale = _mm256_set1_ps(180.0f / M_PI);
            dir = _mm256_mul_ps(dir, scale);

            // 儲存結果到 magnitude 和 direction
            _mm256_store_ps(&magnitude[get_index(y,x,width)], mag);
            _mm256_store_ps(&direction[get_index(y,x,width)], dir);
        }

        // 處理剩餘部分（寬度非 8 的倍數）
        for (; x < width; x++) 
		{
            float gx = gradient_x[get_index(y,x,width)];
            float gy = gradient_y[get_index(y,x,width)];
            magnitude[get_index(y,x,width)] = std::sqrt(gx * gx + gy * gy);
            direction[get_index(y,x,width)] = std::atan2(gy, gx) * 180.0f / M_PI;
        }
    }

    return {magnitude, direction};
}




float* non_maximum_suppression_simd(float* magnitude, float* direction, int height, int width) 
{
    float* suppressed = static_cast<float*>(std::aligned_alloc(32, height * width * sizeof(float)));

	#pragma omp parallel for 
	for(int i = 0; i < width; i++)
	{
		suppressed[get_index(0,i,width)] = 0;
		suppressed[get_index(height-1,i,width)] = 0;
	}
	
	#pragma omp parallel for 
	for(int i = 0; i < height; i++)
	{
		suppressed[get_index(i,0,width)] = 0;
		suppressed[get_index(i,width-1,width)] = 0;
	}

    const __m256 angle0 = _mm256_set1_ps(0.0f);
    const __m256 angle22_5 = _mm256_set1_ps(22.5f);
    const __m256 angle67_5 = _mm256_set1_ps(67.5f);
    const __m256 angle112_5 = _mm256_set1_ps(112.5f);
    const __m256 angle157_5 = _mm256_set1_ps(157.5f);
    const __m256 angle180 = _mm256_set1_ps(180.0f);


	#pragma omp parallel for 
    for (int y = 1; y < height - 1; y++) 
	{
		int x = 1;
        for ( ; x < width - 1; x += 8) // 每次處理 8 個像素
		{ 
            if (x + 8 <= width - 1) 
			{
				// 加載方向和幅值
				__m256 dir = _mm256_loadu_ps(&direction[get_index(y, x, width)]);
				__m256 mag = _mm256_loadu_ps(&magnitude[get_index(y, x, width)]);

				// 標準化角度到 [0, 180]
				dir = _mm256_add_ps(dir, _mm256_and_ps(_mm256_cmp_ps(dir, angle0, _CMP_LT_OQ), angle180));

				// 初始化鄰近幅值
				__m256 q = _mm256_set1_ps(255.0f);
				__m256 r = _mm256_set1_ps(255.0f);

				// 判斷方向，設定 q 和 r
				__m256 mask0 = _mm256_or_ps(_mm256_and_ps(_mm256_cmp_ps(dir, angle0, _CMP_GE_OQ), _mm256_cmp_ps(dir, angle22_5, _CMP_LT_OQ)),
											_mm256_cmp_ps(dir, angle157_5, _CMP_GE_OQ));
															
				__m256 mask45 = _mm256_and_ps(_mm256_cmp_ps(dir, angle22_5, _CMP_GE_OQ), _mm256_cmp_ps(dir, angle67_5, _CMP_LT_OQ));
				__m256 mask90 = _mm256_and_ps(_mm256_cmp_ps(dir, angle67_5, _CMP_GE_OQ), _mm256_cmp_ps(dir, angle112_5, _CMP_LT_OQ));
				__m256 mask135 = _mm256_and_ps(_mm256_cmp_ps(dir, angle112_5, _CMP_GE_OQ), _mm256_cmp_ps(dir, angle157_5, _CMP_LT_OQ));

				q = _mm256_blendv_ps(q, _mm256_loadu_ps(&magnitude[get_index(y, x + 1, width)]), mask0);
				r = _mm256_blendv_ps(r, _mm256_loadu_ps(&magnitude[get_index(y, x - 1, width)]), mask0);

				q = _mm256_blendv_ps(q, _mm256_loadu_ps(&magnitude[get_index(y + 1, x - 1, width)]), mask45);
				r = _mm256_blendv_ps(r, _mm256_loadu_ps(&magnitude[get_index(y - 1, x + 1, width)]), mask45);

				q = _mm256_blendv_ps(q, _mm256_loadu_ps(&magnitude[get_index(y + 1, x, width)]), mask90);
				r = _mm256_blendv_ps(r, _mm256_loadu_ps(&magnitude[get_index(y - 1, x, width)]), mask90);

				q = _mm256_blendv_ps(q, _mm256_loadu_ps(&magnitude[get_index(y - 1, x - 1, width)]), mask135);
				r = _mm256_blendv_ps(r, _mm256_loadu_ps(&magnitude[get_index(y + 1, x + 1, width)]), mask135);

				// 非極大值抑制
				__m256 mask = _mm256_and_ps(_mm256_cmp_ps(mag, q, _CMP_GE_OQ), _mm256_cmp_ps(mag, r, _CMP_GE_OQ));
				__m256 result = _mm256_and_ps(mask, mag);

				// 儲存結果
				_mm256_storeu_ps(&suppressed[get_index(y, x, width)], result);
            }
			else
			{
				for( ; x < width - 1; x++)
				{
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
        }
    }

    return suppressed;
}

// 雙閾值檢測
float* double_threshold(float* input, int height, int width, float low_ratio = 0.1f, float high_ratio = 0.3f) 
{
    float low_threshold = 20.0f;
    float high_threshold = 50.0f;
	
	float* thresholded = static_cast<float*>(std::aligned_alloc(32, height * width * sizeof(float)));
	
	// 加載低閾值和高閾值到 AVX 寄存器
    __m256 low_threshold_vec = _mm256_set1_ps(low_threshold);
    __m256 high_threshold_vec = _mm256_set1_ps(high_threshold);
    __m256 zero_vec = _mm256_set1_ps(0.0f);
    __m256 weak_edge_vec = _mm256_set1_ps(128.0f);
    __m256 strong_edge_vec = _mm256_set1_ps(255.0f);
	
	int totalPixel = height*width;

	#pragma omp parallel for 
	for(int i = 0;  i < totalPixel; i+=8)
	{
		if(i+8 <= totalPixel)
		{
			// 加載輸入像素到 AVX 寄存器
			__m256 pixel_vec = _mm256_load_ps(input + i);

			// 執行閾值判斷
			__m256 mask_low = _mm256_cmp_ps(pixel_vec, low_threshold_vec, _CMP_LT_OQ); // pixel < low_threshold
			__m256 mask_high = _mm256_cmp_ps(pixel_vec, high_threshold_vec, _CMP_GT_OQ); // pixel > high_threshold

			// 預設為弱邊緣
			__m256 result = weak_edge_vec;

			// 將強邊緣和非邊緣應用到結果
			result = _mm256_blendv_ps(result, zero_vec, mask_low); // pixel < low_threshold => 0.0f
			result = _mm256_blendv_ps(result, strong_edge_vec, mask_high); // pixel > high_threshold => 255.0f

			// 將結果存回
			_mm256_store_ps(thresholded + i, result);
		}
		else
		{
			for(int j = i; j < totalPixel; j++)
			{
				float pixel = input[j];
				if (pixel < low_threshold) 
				{
					thresholded[j] = 0.0f;
				}
				else if (pixel > high_threshold) 
				{
					thresholded[j] = 255.0f;
				}
				else 
				{
					thresholded[j] = 128.0f;  // 不確定邊緣
				}
			}
		}
		
	}
	
	


    return thresholded;
}

// 邊緣追蹤
float* edge_tracking(float* thresholded, int height, int width) 
{

	float* edges = static_cast<float*>(std::aligned_alloc(32, height * width * sizeof(float)));
	std::fill(edges, edges + height * width, 0.0f);
	
	#pragma omp parallel for 
	for (int y = 1; y < height - 1; y++) 
	{
		int x = 1;
        for ( ; x+8 <= width - 1; x += 8) // 一次處理 8 個像素
		{
            // 加載 thresholded 值到 SIMD 寄存器
            __m256 center = _mm256_loadu_ps(&thresholded[get_index(y, x, width)]);
            
            // 檢查是否等於 255.0f
            __m256 cmp255 = _mm256_cmp_ps(center, _mm256_set1_ps(255.0f), _CMP_EQ_OQ);
            
            // 設置結果
            __m256 result = _mm256_and_ps(cmp255, _mm256_set1_ps(255.0f));
            _mm256_storeu_ps(&edges[get_index(y, x, width)], result);
			
			// 將浮點比較結果轉換為整數掩碼
			__m256i cmp_mask = _mm256_castps_si256(cmp255);
			
			if( _mm256_testz_si256(cmp_mask, cmp_mask) == true)
			{
				continue;
			}

            // 8 鄰域追蹤 (要對應的thresholded是255才能夠執行，幫我修改一下)
            for (int dy = -1; dy <= 1; dy++) 
			{
                for (int dx = -1; dx <= 1; dx++) 
				{

                    __m256 neighbor = _mm256_loadu_ps(&thresholded[get_index(y + dy, x + dx, width)]);
                    __m256 cmp128 = _mm256_cmp_ps(neighbor, _mm256_set1_ps(128.0f), _CMP_EQ_OQ);
                    __m256 neighbor_result = _mm256_and_ps(cmp128, _mm256_set1_ps(255.0f));
					
					neighbor_result = _mm256_and_ps(neighbor_result, cmp255);
                    
                    // 合併到結果中
                    __m256 edges_prev = _mm256_loadu_ps(&edges[get_index(y + dy, x + dx, width)]);
                    __m256 edges_new = _mm256_or_ps(edges_prev, neighbor_result);
                    _mm256_storeu_ps(&edges[get_index(y + dy, x + dx, width)], edges_new);
                }
            }
        }
		for( ; x < width-1; x++)
		{
			if (thresholded[get_index(y, x, width)] == 255.0f) 
			{
                edges[get_index(y, x, width)] = 255.0f;
                // 追蹤8鄰域
                for (int dy = -1; dy <= 1; dy++) 
				{
                    for (int dx = -1; dx <= 1; dx++) 
					{
                        if (thresholded[get_index(y+dy, x+dx, width)] == 128.0f) 
						{
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
float* canny_edge_detection(float* input, int height, int width) 
{
	auto time1 = std::chrono::high_resolution_clock::now();
    // 1. 高斯模糊
    float* blurred = gaussian_blur(input, height, width);
	
	auto time2 = std::chrono::high_resolution_clock::now();

    // 2. Sobel梯度
    std::pair<float*, float*> gradients = sobel_gradient(blurred, height, width);
    float* gradient_x = gradients.first;
    float* gradient_y = gradients.second;

    // 3. 計算梯度幅值和方向
    std::pair<float*, float*> grad_mag_dir = compute_gradient_magnitude_and_direction(gradient_x, gradient_y, height, width);
    float* magnitude = grad_mag_dir.first;
    float* direction = grad_mag_dir.second;

	auto time3 = std::chrono::high_resolution_clock::now();
	
    // 4. 非極大值抑制
    float* suppressed = non_maximum_suppression_simd(magnitude, direction, height, width);

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

int main() 
{
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