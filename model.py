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

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    // TODO: compute out[i, d] = sum_j p[i, j] * v[j, d]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= seq_len || col >= head_dim) return;

    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        sum += p[(size_t)row * seq_len + j] * v[(size_t)j * head_dim + col];
    }
    out[(size_t)row * head_dim + col] = sum;
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    // TODO: allocate scratch, launch qk_scores -> softmax_rows -> pv_matmul, free scratch
    float* d_scores = nullptr;
    size_t scores_bytes = (size_t)seq_len * seq_len * sizeof(float);
    cudaMalloc(&d_scores, scores_bytes);

    dim3 threads2d(16,16);
    dim3 qk_blocks((seq_len + threads2d.x - 1) / threads2d.x,
                    (seq_len + threads2d.y - 1) / threads2d.y);
    qk_scores<<<qk_blocks, threads2d>>>(d_q, d_k, d_scores, seq_len, head_dim);

    int sm_threads = 256;
    int sm_blocks = seq_len;
    size_t sm_shmem = sm_threads * sizeof(float);
    softmax_rows<<<sm_blocks, sm_threads, sm_shmem>>>(d_scores, seq_len, seq_len);

    dim3 pv_blocks((head_dim + threads2d.x - 1) / threads2d.x,
                    (seq_len + threads2d.y - 1) / threads2d.y );
    pv_matmul<<<pv_blocks, threads2d>>>(d_scores, d_v, d_out, seq_len, head_dim);

    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    // TODO: return the running max of old_max and new_val
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    // TODO: return the scalar used to rescale running statistics
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    // TODO: combine the rescaled old sum with the new block sum
    float z_new = correction * old_sum + block_sum;
    return z_new;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    // TODO: multiply each of the head_dim entries of out_row by correction in place
    for (int d = 0; d < head_dim; d++)
        out_row[d] *= correction;
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    // TODO: cooperatively copy the tile into shared_dst, zero-filling out-of-bounds positions.
    int total = tile_rows * tile_cols;              // flat tile size

    // grid-stride over flattened tile positions
    for (int t = thread_id; t < total; t += num_threads) {
        int tr = t / tile_cols;                     // tile row
        int tc = t % tile_cols;                     // tile col

        int src_row = src_row_start + tr;           // where this maps in the source
        int src_col = src_col_start + tc;

        float val;
        if (src_row < src_rows && src_col < src_cols   // in-bounds (and >= 0 below)
            && src_row >= 0 && src_col >= 0) {
            val = src[(size_t)src_row * src_cols + src_col];   // row-major: row*width + col
        } else {
            val = 0.0f;                             // out-of-bounds -> zero-fill
        }

        shared_dst[tr * tile_cols + tc] = val;      // dest is always in-tile
    }
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    // TODO: cooperatively fill s_tile[i, j] = scale * dot(q_tile[i, :], k_tile[j, :])
    int total = tile_q * tile_k;

    for (int t = thread_id; t < total; t +=num_threads) {
        int tr = t / tile_k;
        int tc = t % tile_k; 

        const float* q_row = q_tile + (size_t)tr * head_dim;
        const float* k_row = k_tile + (size_t)tc * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++){
            dot += q_row[d] * k_row[d];
        }

        s_tile[(size_t)tr * tile_k + tc] = scale * dot;
    }

}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    // TODO: write row_max_out[r] = max over c of s_tile[r, c]
    for (int r = thread_id; r < tile_q; r += num_threads) {
        const float* s_row = s_tile + (size_t) r * tile_k;

        float m = -FLT_MAX;
        for (int c = 0; c < tile_k; c++) {
            m = fmaxf(m, s_row[c]);
        }

        row_max_out[r] = m;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    // TODO: for each (r, c) in the tile, set s_tile[r*tile_k+c] = expf(s_tile[r*tile_k+c] - row_max[r])
    int total = tile_q * tile_k;

    for (int t = thread_id; t < total; t +=num_threads) {
        int tr = t / tile_k;
        int tc = t % tile_k; 

        s_tile[t] = expf(s_tile[t] - row_max[tr]);
    }

}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    // TODO: cooperatively fill row_sum_out[r] with the sum of p_tile row r 
    
    for (int r = thread_id; r < tile_q; r += num_threads) {
        const float* p_row = p_tile + (size_t)r * tile_k;
        
        float local_sum = 0.0f;
        for (int c = 0; c < tile_k; c++) {
            local_sum += p_row[c];
        }
        row_sum_out[r] = local_sum;
    }
    
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {
    // TODO: cooperatively add P_tile * V_tile into out_acc
    int total = tile_q * head_dim;
    
    for (int t = thread_id; t < total; t += num_threads) {
        int i = t / head_dim; // query idx
        int d = t % head_dim; // feature dims


        float sum = 0.0f;
        for (int j = 0; j < tile_k; j++) {  // contract over keys in this tile
        sum += p_tile[(size_t)i * tile_k + j] // P[i][j], p_tile is tile_q x tile_k,
                    * v_tile[(size_t)j * head_dim + d];  // V[j][d], v_tile is tile_k x head_dim
        }
        out_acc[(size_t)i * head_dim + d] += sum;
    }
}

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

