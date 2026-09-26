"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: implement elementwise c[i] = a[i] + b[i]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < n; i += blockDim.x){
        c[i] = a[i] + b[i];
    }
    
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;              // total threads in the grid
   
    for (int i = idx; i < n; i += stride) {
        a[i] = a[i] * scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // TODO: replace each a[i] with expf(a[i])
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        a[i] = expf(a[i]);
    }
}

# Step 4 - row_max
#include <cuda_runtime.h>
#include <cfloat>

__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) return;

    const float* row_in = matrix + (size_t)row * cols;
    float m = -FLT_MAX;
    for (int c = 0; c < cols; c++) {
        m = fmaxf(m, row_in[c]);
    }

    out[row] = m;

}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    // TODO: write out[r] = sum of matrix row r
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* row_in = matrix + (size_t)row * cols;

    float local_sum = 0.0f;

    for (int c = tid; c < cols; c += blockDim.x) {
        local_sum += row_in[c];
    }
    sdata[tid] = local_sum;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s>>=1) {
        if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[row] = sdata[0];


    
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    // TODO: return the dot product of a and b
    float product = 0.0f;
    for (int i = 0; i < n; i++) {
        product += a[i] * b[i];
    }

    return product; 
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    // TODO: compute C = A * B for row-major matrices
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // which row of C (0..m-1)
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // which col of C (0..n-1)
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    for (int i = 0; i < k; i++){
        sum += a[(size_t)row * k + i] * b[(size_t)i * n + col];
    }

    c[(size_t)row * n + col] = sum;
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    // TODO: write out[c*rows + r] = in[r*cols + c]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) return;

    
    out[(size_t)col * rows + row] = in[(size_t)row * cols + col];
   
 }

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    // TODO: compute scores[i, j] = dot(q_row_i, k_row_j) / sqrt(head_dim)
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= seq_len || i >= seq_len) return;

    const float* q_row = q + (size_t)i * head_dim;
    const float* k_row = k + (size_t)j * head_dim;

    float dot = dot_product(q_row, k_row, head_dim);
    scores[(size_t)i * seq_len + j] = dot / sqrtf((float)head_dim);


}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    // TODO: implement numerically stable row-wise softmax in place
    extern __shared__  float sdata[];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    if(row >= rows) return;

    float* row_ptr = matrix + (size_t)row * cols; // in-place so not const

    float local_max = -FLT_MAX;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, row_ptr[c]);
    }
    sdata[tid] = local_max;
    __syncthreads();
     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];               // broadcast max into every thread's register
    __syncthreads();  


    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_sum += expf(row_ptr[c] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>=1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
   
    for (int c = tid; c < cols; c += blockDim.x) {
        row_ptr[c] = expf(row_ptr[c] - row_max) / row_sum;
    }

}

# Step 11 - pv_matmul (not yet solved)
# TODO: implement

# Step 12 - naive_attention (not yet solved)
# TODO: implement

# Step 13 - online_max (not yet solved)
# TODO: implement

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

