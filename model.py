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

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    // TODO: tiled fused attention using shared memory and online softmax.
    extern __shared__ float smem[];                 // one dynamic buffer, carved below
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // --- carve shared memory ---
    float* q_sh = smem;                             // tile_q * head_dim  (this block's queries)
    float* k_sh = q_sh + tile_q * head_dim;         // tile_k * head_dim  (current key tile)
    float* v_sh = k_sh + tile_k * head_dim;         // tile_k * head_dim  (current value tile)
    float* s_sh = v_sh + tile_k * head_dim;         // tile_q * tile_k    (scores scratch)
    float* o_sh = s_sh + tile_q * tile_k;           // tile_q * head_dim  (output accumulator)
    float* m_sh = o_sh + tile_q * head_dim;         // tile_q  (running max per query row)
    float* l_sh = m_sh + tile_q;                    // tile_q  (running sum per query row)
    float* mt_sh = l_sh + tile_q;                   // tile_q  (this tile's max per row)
    float* ts_sh = mt_sh + tile_q;                  // tile_q  (this tile's sum per row)

    // --- which query rows does this block own? ---
    int q_start = blockIdx.x * tile_q;
    if (q_start >= seq_len) return;
    int q_eff = min(tile_q, seq_len - q_start);     // partial last query tile

    // --- load this block's Q tile once, init running state ---
    load_tile(q, q_sh, q_start, 0, seq_len, head_dim, q_eff, head_dim, tid, nthreads);
    for (int r = tid; r < q_eff; r += nthreads) { m_sh[r] = -FLT_MAX; l_sh[r] = 0.0f; }
    for (int t = tid; t < q_eff * head_dim; t += nthreads) o_sh[t] = 0.0f;
    __syncthreads();

    // --- sequential loop over KEY/VALUE tiles: count = ceil(seq_len / tile_k) ---
    int num_tiles = (seq_len + tile_k - 1) / tile_k;
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * tile_k;
        int k_eff = min(tile_k, seq_len - k_start);          // partial last key tile

        // 1) load K,V tiles
        load_tile(k, k_sh, k_start, 0, seq_len, head_dim, k_eff, head_dim, tid, nthreads);
        load_tile(v, v_sh, k_start, 0, seq_len, head_dim, k_eff, head_dim, tid, nthreads);
        __syncthreads();

        // 2) scores = scale * Q·Kᵀ   (per cell, width = k_eff)
        tile_scores(q_sh, k_sh, s_sh, q_eff, k_eff, head_dim, scale, tid, nthreads);
        __syncthreads();

        // 3) this tile's row max
        tile_rowmax(s_sh, mt_sh, q_eff, k_eff, tid, nthreads);
        __syncthreads();

        // 4+5) online max, correction, rescale accumulator & running sum  (per row)
        for (int r = tid; r < q_eff; r += nthreads) {
            float m_old = m_sh[r];
            float m_new = online_max(m_old, mt_sh[r]);        // fmaxf
            float corr  = expf(m_old - m_new);                // <= 1 (0 on first tile)
            l_sh[r] *= corr;                                  // fix running denominator
            rescale_output(o_sh + (size_t)r * head_dim, head_dim, corr);  // fix accumulator
            m_sh[r] = m_new;                                  // commit new running max
        }
        __syncthreads();

        // 6) p_tile = exp(s - m_new)   (per cell, in place on s_sh)
        tile_exp(s_sh, m_sh, q_eff, k_eff, tid, nthreads);
        __syncthreads();

        // 7) this tile's row sum, folded into running sum
        tile_rowsum(s_sh, ts_sh, q_eff, k_eff, tid, nthreads);
        __syncthreads();
        for (int r = tid; r < q_eff; r += nthreads) l_sh[r] += ts_sh[r];

        // 8) out_acc += p_tile · v_tile   (per cell)
        accumulate_pv(s_sh, v_sh, o_sh, q_eff, k_eff, head_dim, tid, nthreads);
        __syncthreads();
    }

    // --- finalize: divide accumulator by running sum, write to global ---
    for (int t = tid; t < q_eff * head_dim; t += nthreads) {
        int i = t / head_dim, d = t % head_dim;
        out[(size_t)(q_start + i) * head_dim + d] = o_sh[t] / l_sh[i];
    }


}

# Step 24 - flash_attention_launcher
void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    // TODO: configure grid/block/shared memory and launch flash_attention_kernel
    int block = 128;

    int grid = (seq_len + tile_q - 1) / tile_q;

    size_t shmem_floats = 
         (size_t)tile_q * head_dim        // q_sh
        + (size_t)tile_k * head_dim        // k_sh
        + (size_t)tile_k * head_dim        // v_sh
        + (size_t)tile_q * tile_k          // s_sh
        + (size_t)tile_q * head_dim        // o_sh
        + (size_t)4 * tile_q;              // m_sh, l_sh, mt_sh, ts_sh
    size_t shmem_bytes = shmem_floats * sizeof(float);

    float scale = 1.0f / sqrtf((float)head_dim);
    flash_attention_kernel<<<grid, block, shmem_bytes>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale);

}

# Step 25 - causal_mask
__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k, int thread_id, int num_threads) {
    // TODO: write -INFINITY into entries where the global key index exceeds the global query index.
    int total = tile_q * tile_k;
    
    for (int t = thread_id; t < total; t += num_threads) {
        int tr = t / tile_k;
        int tc = t % tile_k;
        
        int global_q = tr + q_row_start;
        int global_k = tc + k_col_start;

        if (global_k > global_q) {
            s_tile[(size_t)tr * tile_k + tc] = -INFINITY;
        }

    }
}

# Step 26 - flash_attention_causal_kernel
__global__ void flash_attention_causal_kernel(const float* q, const float* k, const float* v,
                                                float* out, int seq_len, int head_dim,
                                                int tile_q, int tile_k, float scale) {
    // TODO: tiled causal flash attention using shared memory and online softmax
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float* q_sh = smem;
    float* k_sh = q_sh + tile_q * head_dim;
    float* v_sh = k_sh + tile_k * head_dim;
    float* s_sh = v_sh + tile_k * head_dim;
    float* o_sh = s_sh + tile_q * tile_k;
    float* m_sh = o_sh + tile_q * head_dim;
    float* l_sh = m_sh + tile_q;
    float* mt_sh = l_sh + tile_q;
    float* ts_sh = mt_sh + tile_q;

    int q_start = blockIdx.x * tile_q;
    if (q_start >= seq_len) return;
    int q_end  = min(tile_q, seq_len - q_start);

    load_tile(q, q_sh, q_start, 0, seq_len, head_dim, q_end, head_dim, tid, nthreads);
    for (int r = tid; r < q_end; r += nthreads) {m_sh[r] = -FLT_MAX; l_sh[r] = 0.0f;}
    for (int e = tid; e < q_end * head_dim; e += nthreads) o_sh[e] =0.0f;
    __syncthreads();

    int last_q = q_start + q_end - 1;
    int max_tile = last_q / tile_k;
    for (int t = 0; t <= max_tile; t++) {
        int k_start = t * tile_k;
        int k_eff   = min(tile_k, seq_len - k_start);

        load_tile(k, k_sh, k_start, 0, seq_len, head_dim, k_eff, head_dim, tid, nthreads);
        load_tile(v, v_sh, k_start, 0, seq_len, head_dim, k_eff, head_dim, tid, nthreads);
        __syncthreads();

        tile_scores(q_sh, k_sh, s_sh, q_end, k_eff, head_dim, scale, tid, nthreads);
        __syncthreads();

        // causal mask BEFORE the max (so future keys can't win the max or leak into exp)
        causal_mask(s_sh, q_start, k_start, q_end, k_eff, tid, nthreads);
        __syncthreads();

        tile_rowmax(s_sh, mt_sh, q_end, k_eff, tid, nthreads);
        __syncthreads();

        // online max + correction: rescale accumulator and running sum, then commit new max
        for (int r = tid; r < q_end; r += nthreads) {
            float m_old = m_sh[r];
            float m_new = online_max(m_old, mt_sh[r]);
            float corr  = expf(m_old - m_new);
            l_sh[r] *= corr;
            rescale_output(o_sh + (size_t)r * head_dim, head_dim, corr);
            m_sh[r] = m_new;
        }
        __syncthreads();

        tile_exp(s_sh, m_sh, q_end, k_eff, tid, nthreads);        // p = exp(s - m_new)
        __syncthreads();

        tile_rowsum(s_sh, ts_sh, q_end, k_eff, tid, nthreads);
        __syncthreads();
        for (int r = tid; r < q_end; r += nthreads) l_sh[r] += ts_sh[r];

        accumulate_pv(s_sh, v_sh, o_sh, q_end, k_eff, head_dim, tid, nthreads);
        __syncthreads();
    }

    // finalize: divide accumulator by running sum, write to global out
    for (int e = tid; e < q_end * head_dim; e += nthreads) {
        int i = e / head_dim, d = e % head_dim;
        out[(size_t)(q_start + i) * head_dim + d] = o_sh[e] / l_sh[i];
    }

    }

