extern "C" __global__ void fused_min_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
  float placeholder_red_rf[1];
  __shared__ float red_buf0[1024];
  placeholder_red_rf[(0)] = 3.402823e+38f;
  for (int k0_outer = 0; k0_outer < 5; ++k0_outer) {
    if (((k0_outer * 1024) + ((int)threadIdx.x)) < 4507) {
      placeholder_red_rf[(0)] = min(placeholder_red_rf[(0)], placeholder[(((k0_outer * 1024) + ((int)threadIdx.x)))]);
    }
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = placeholder_red_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 512) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 512))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 256))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 128))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 64))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = min(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    placeholder_red[(0)] = ((volatile float*)red_buf0)[(0)];
  }
}

extern "C" __global__ void fused_zeros_kernel0(long* __restrict__ T_full, int any_dim, int stride) {
  if (((int)blockIdx.x) < (any_dim >> 9)) {
    T_full[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))] = (long)0;
  } else {
    if (((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) < any_dim) {
      T_full[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))] = (long)0;
    }
  }
}

extern "C" __global__ void fused_vision_non_max_suppression_kernel0(float* __restrict__ fetch_score, float* __restrict__ placeholder) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4507) {
    fetch_score[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 5120) + (((int)threadIdx.x) * 5)))];
  }
}

extern "C" __global__ void fused_subtract_add_expand_dims_cast_add_multiply_strided_slice_expand_dims_add_c_18397580542026144553__kernel0(float* __restrict__ T_expand_dims, float* __restrict__ placeholder, long* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ placeholder3, float* __restrict__ placeholder4) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 22535) {
    T_expand_dims[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((1 <= (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 5)) ? (placeholder[(((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 5) * 4) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 5)) - 1))] + (((float)placeholder1[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 5))]) * (placeholder2[(0)] + 1.000000e+00f))) : ((placeholder3[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 5))] - placeholder4[(0)]) + 1.000000e+00f));
  }
}

extern "C" __global__ void fused_cast_2_kernel0(long* __restrict__ T_cast, int* __restrict__ placeholder, int any_dim, int stride, int stride1) {
  if (((int)blockIdx.x) < (any_dim >> 9)) {
    T_cast[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride1))] = ((long)placeholder[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))]);
  } else {
    if (((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) < any_dim) {
      T_cast[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride1))] = ((long)placeholder[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))]);
    }
  }
}

extern "C" __global__ void fused_vision_non_max_suppression_kernel2(float* __restrict__ placeholder, int* __restrict__ placeholder1, float* __restrict__ topk_gpu_v0, int* __restrict__ placeholder2, int* __restrict__ nms_v1, int* __restrict__ placeholder3, int* __restrict__ topk_gpu_v1, int* __restrict__ nms_v2) {
  int num_valid_boxes_local[1];
  num_valid_boxes_local[(0)] = 0;
  if ((0.000000e+00f < placeholder[(0)]) && (0 < placeholder1[(0)])) {
    for (int i_0 = 0; i_0 < placeholder1[(0)]; ++i_0) {
      if (-1.000000e+00f < topk_gpu_v0[((i_0 * 5))]) {
        if (0 < placeholder2[(0)]) {
          if (num_valid_boxes_local[(0)] < placeholder2[(0)]) {
            nms_v1[(num_valid_boxes_local[(0)])] = placeholder3[(topk_gpu_v1[(i_0)])];
            num_valid_boxes_local[(0)] = (num_valid_boxes_local[(0)] + 1);
            for (int i_1 = 0; i_1 < (((placeholder1[(0)] + 1022) - i_0) >> 10); ++i_1) {
              if (((((i_1 * 1024) + i_0) + ((int)threadIdx.x)) < 4506) && (0.000000e+00f < topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 5))])) {
                if (placeholder[(0)] <= ((((((max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]) - min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))])) * (max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]) - min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]))) + ((max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))]) - min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) * (max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))]) - min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) - (max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])))))) <= 0.000000e+00f) ? 0.000000e+00f : ((max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))]))))) / ((((max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]) - min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))])) * (max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]) - min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]))) + ((max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))]) - min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) * (max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))]) - min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) - (max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), max(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), min(topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))]))))))))) {
                  topk_gpu_v0[(((((i_1 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 5))] = -1.000000e+00f;
                }
              }
              __syncthreads();
            }
          }
        } else {
          nms_v1[(num_valid_boxes_local[(0)])] = placeholder3[(topk_gpu_v1[(i_0)])];
          num_valid_boxes_local[(0)] = (num_valid_boxes_local[(0)] + 1);
          for (int i_2 = 0; i_2 < (((placeholder1[(0)] + 1022) - i_0) >> 10); ++i_2) {
            if (((((i_2 * 1024) + i_0) + ((int)threadIdx.x)) < 4506) && (0.000000e+00f < topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 5))])) {
              if (placeholder[(0)] <= ((((((max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]) - min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))])) * (max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]) - min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]))) + ((max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))]) - min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) * (max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))]) - min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) - (max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])))))) <= 0.000000e+00f) ? 0.000000e+00f : ((max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))]))))) / ((((max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]) - min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))])) * (max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]) - min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]))) + ((max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))]) - min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) * (max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))]) - min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) - (max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 2))], topk_gpu_v0[(((i_0 * 5) + 4))]), min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 7))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 9))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), max(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))])) - max(min(topk_gpu_v0[(((i_0 * 5) + 1))], topk_gpu_v0[(((i_0 * 5) + 3))]), min(topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 6))], topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 8))]))))))))) {
                topk_gpu_v0[(((((i_2 * 5120) + (i_0 * 5)) + (((int)threadIdx.x) * 5)) + 5))] = -1.000000e+00f;
              }
            }
            __syncthreads();
          }
        }
      }
    }
    nms_v2[(0)] = num_valid_boxes_local[(0)];
  } else {
    nms_v2[(0)] = 0;
  }
}

extern "C" __global__ void fused_squeeze_1_kernel0(int* __restrict__ T_squeeze, int* __restrict__ placeholder) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4507) {
    T_squeeze[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))];
  }
}

extern "C" __global__ void fused_squeeze_2_kernel0(int* __restrict__ T_squeeze, int* __restrict__ placeholder) {
  T_squeeze[(0)] = placeholder[(0)];
}

extern "C" __global__ void fused_vision_non_max_suppression_kernel1(float* __restrict__ placeholder, int* __restrict__ placeholder1, int* __restrict__ nms_v1, float* __restrict__ topk_gpu_v0, float* __restrict__ placeholder2, int* __restrict__ topk_gpu_v1) {
  if ((0.000000e+00f < placeholder[(0)]) && (0 < placeholder1[(0)])) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4507) {
      nms_v1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = -1;
    }
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < placeholder1[(0)]) {
      for (int i = 0; i < 5; ++i) {
        topk_gpu_v0[((((((int)blockIdx.x) * 5120) + (((int)threadIdx.x) * 5)) + i))] = placeholder2[(((topk_gpu_v1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] * 5) + i))];
      }
    } else {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4507) {
        for (int j = 0; j < 5; ++j) {
          topk_gpu_v0[((((((int)blockIdx.x) * 5120) + (((int)threadIdx.x) * 5)) + j))] = -1.000000e+00f;
        }
      }
    }
  } else {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < placeholder1[(0)]) {
      for (int k = 0; k < 5; ++k) {
        topk_gpu_v0[((((((int)blockIdx.x) * 5120) + (((int)threadIdx.x) * 5)) + k))] = placeholder2[((((((int)blockIdx.x) * 5120) + (((int)threadIdx.x) * 5)) + k))];
      }
      nms_v1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((((int)blockIdx.x) * 1024) + ((int)threadIdx.x));
    }
  }
}

extern "C" __global__ void fused_dyn_strided_slice_kernel0(int* __restrict__ T_strided_slice_dynamic, int* __restrict__ placeholder, int* __restrict__ placeholder1, int* __restrict__ placeholder2, int dim, int stride) {
  if (((int)blockIdx.x) < (dim >> 9)) {
    T_strided_slice_dynamic[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))] = placeholder[(((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * placeholder1[(0)]) + placeholder2[(0)]))];
  } else {
    if (((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) < dim) {
      T_strided_slice_dynamic[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))] = placeholder[(((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * placeholder1[(0)]) + placeholder2[(0)]))];
    }
  }
}

extern "C" __global__ void fused_strided_slice_kernel0(long* __restrict__ T_strided_slice_dynamic, long* __restrict__ placeholder, int dim, int stride, int stride1) {
  if (((int)blockIdx.x) < (dim >> 9)) {
    T_strided_slice_dynamic[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride1))] = placeholder[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))];
  } else {
    if (((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) < dim) {
      T_strided_slice_dynamic[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride1))] = placeholder[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))];
    }
  }
}

extern "C" __global__ void fused_max_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
  float placeholder_red_rf[1];
  __shared__ float red_buf0[1024];
  placeholder_red_rf[(0)] = -3.402823e+38f;
  for (int k0_k1_fused_outer = 0; k0_k1_fused_outer < 18; ++k0_k1_fused_outer) {
    if ((((k0_k1_fused_outer * 256) + (((int)threadIdx.x) >> 2)) < 4507) && (((k0_k1_fused_outer * 1024) + ((int)threadIdx.x)) < 18028)) {
      placeholder_red_rf[(0)] = max(placeholder_red_rf[(0)], placeholder[(((k0_k1_fused_outer * 1024) + ((int)threadIdx.x)))]);
    }
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = placeholder_red_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 512) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 512))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 256))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 128))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 64))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = max(((volatile float*)red_buf0)[(((int)threadIdx.x))], ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    placeholder_red[(0)] = ((volatile float*)red_buf0)[(0)];
  }
}
