WARNING:root:Untyped Tensor found, assume it is float32
WARNING:root:Untyped Tensor found, assume it is float32
WARNING:root:Untyped Tensor found, assume it is float32
WARNING:root:Untyped Tensor found, assume it is float32
WARNING:root:Untyped Tensor found, assume it is float32
WARNING:root:Untyped Tensor found, assume it is float32
WARNING:root:Untyped Tensor found, assume it is float32
WARNING:root:Untyped Tensor found, assume it is float32
WARNING:root:Untyped Tensor found, assume it is float32
WARNING:root:Untyped Tensor found, assume it is float32
[17:48:53] /home/masa/projects/dev/tvm/src/te/schedule/bound.cc:119: not in feed graph consumer = extern(topk_gpu, 0x56177154af20)
[17:48:53] /home/masa/projects/dev/tvm/src/runtime/cuda/cuda_module.cc:55: extern "C" __global__ void fused_squeeze_1_kernel0(int* __restrict__ T_squeeze, int* __restrict__ placeholder) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4507) {
    T_squeeze[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))];
  }
}

extern "C" __global__ void fused_squeeze_2_kernel0(int* __restrict__ T_squeeze, int* __restrict__ placeholder) {
  T_squeeze[(0)] = placeholder[(0)];
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

extern "C" __global__ void fused_cast_3_kernel0(long* __restrict__ T_cast, int* __restrict__ placeholder, int any_dim, int stride, int stride1) {
  if (((int)blockIdx.x) < (any_dim >> 9)) {
    T_cast[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride1))] = ((long)placeholder[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))]);
  } else {
    if (((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) < any_dim) {
      T_cast[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride1))] = ((long)placeholder[((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) * stride))]);
    }
  }
}

extern "C" __global__ void fused_vision_non_max_suppression_kernel1(float* __restrict__ placeholder, int* __restrict__ placeholder1, float* __restrict__ topk_gpu_v0, float* __restrict__ placeholder2, int* __restrict__ topk_gpu_v1, float* __restrict__ fetch_score, float* __restrict__ nms_v2, int* __restrict__ nms_v3) {
  if ((0.000000e+00f < placeholder[(0)]) && (0 < placeholder1[(0)])) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < placeholder1[(0)]) {
      topk_gpu_v0[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 4)))] = placeholder2[(((topk_gpu_v1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] * 6) + 2))];
      topk_gpu_v0[((((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 4)) + 1))] = placeholder2[(((topk_gpu_v1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] * 6) + 3))];
      topk_gpu_v0[((((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 4)) + 2))] = placeholder2[(((topk_gpu_v1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] * 6) + 4))];
      topk_gpu_v0[((((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 4)) + 3))] = placeholder2[(((topk_gpu_v1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] * 6) + 5))];
      fetch_score[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder2[(((topk_gpu_v1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] * 6) + 1))];
      nms_v2[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder2[((topk_gpu_v1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] * 6))];
    }
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4507) {
      nms_v3[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = -1;
    }
  } else {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < placeholder1[(0)]) {
      topk_gpu_v0[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 4)))] = placeholder2[((((((int)blockIdx.x) * 6144) + (((int)threadIdx.x) * 6)) + 2))];
      topk_gpu_v0[((((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 4)) + 1))] = placeholder2[((((((int)blockIdx.x) * 6144) + (((int)threadIdx.x) * 6)) + 3))];
      topk_gpu_v0[((((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 4)) + 2))] = placeholder2[((((((int)blockIdx.x) * 6144) + (((int)threadIdx.x) * 6)) + 4))];
      topk_gpu_v0[((((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 4)) + 3))] = placeholder2[((((((int)blockIdx.x) * 6144) + (((int)threadIdx.x) * 6)) + 5))];
      fetch_score[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder2[((((((int)blockIdx.x) * 6144) + (((int)threadIdx.x) * 6)) + 1))];
      nms_v2[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder2[(((((int)blockIdx.x) * 6144) + (((int)threadIdx.x) * 6)))];
      nms_v3[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((((int)blockIdx.x) * 1024) + ((int)threadIdx.x));
    }
  }
}

extern "C" __global__ void fused_vision_non_max_suppression_kernel2(float* __restrict__ placeholder, int* __restrict__ placeholder1, int* __restrict__ placeholder2, float* __restrict__ fetch_score, int* __restrict__ nms_v3, int* __restrict__ placeholder3, int* __restrict__ topk_gpu_v1, float* __restrict__ nms_v2, float* __restrict__ topk_gpu_v0, int* __restrict__ nms_v4) {
  int num_valid_boxes_local[1];
  num_valid_boxes_local[(0)] = 0;
  if ((0.000000e+00f < placeholder[(0)]) && (0 < placeholder1[(0)])) {
    if (0 < placeholder2[(0)]) {
      for (int k = 0; k < placeholder1[(0)] && ((num_valid_boxes_local[(0)] < placeholder2[(0)])); ++k) {
        if (-1.000000e+00f < fetch_score[(k)]) {
          if (((int)threadIdx.x) == 0) {
            nms_v3[(num_valid_boxes_local[(0)])] = placeholder3[(topk_gpu_v1[(k)])];
          }
          num_valid_boxes_local[(0)] = (num_valid_boxes_local[(0)] + 1);
          for (int i_0 = 0; i_0 < (((placeholder1[(0)] + 1022) - k) >> 10); ++i_0) {
            if (((((((i_0 * 1024) + k) + ((int)threadIdx.x)) + 1) < placeholder1[(0)]) && (0.000000e+00f < fetch_score[(((((i_0 * 1024) + k) + ((int)threadIdx.x)) + 1))])) && (nms_v2[(((((i_0 * 1024) + k) + ((int)threadIdx.x)) + 1))] == nms_v2[(k)])) {
              if (placeholder[(0)] <= ((((((max(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))]) - min(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))])) * (max(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]) - min(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]))) + ((max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))]) - min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))])) * (max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))]) - min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) - (max(0.000000e+00f, (min(max(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]), max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))])) - max(min(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]), min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))]), max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))])) - max(min(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))]), min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))])))))) <= 0.000000e+00f) ? 0.000000e+00f : ((max(0.000000e+00f, (min(max(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]), max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))])) - max(min(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]), min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))]), max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))])) - max(min(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))]), min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))]))))) / ((((max(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))]) - min(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))])) * (max(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]) - min(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]))) + ((max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))]) - min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))])) * (max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))]) - min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) - (max(0.000000e+00f, (min(max(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]), max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))])) - max(min(topk_gpu_v0[(((k * 4) + 1))], topk_gpu_v0[(((k * 4) + 3))]), min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))]), max(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))])) - max(min(topk_gpu_v0[((k * 4))], topk_gpu_v0[(((k * 4) + 2))]), min(topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_0 * 4096) + (k * 4)) + (((int)threadIdx.x) * 4)) + 6))]))))))))) {
                fetch_score[(((((i_0 * 1024) + k) + ((int)threadIdx.x)) + 1))] = -1.000000e+00f;
              }
            }
            __syncthreads();
          }
        }
      }
    } else {
      for (int i_1 = 0; i_1 < placeholder1[(0)]; ++i_1) {
        if (-1.000000e+00f < fetch_score[(i_1)]) {
          if (((int)threadIdx.x) == 0) {
            nms_v3[(num_valid_boxes_local[(0)])] = placeholder3[(topk_gpu_v1[(i_1)])];
          }
          num_valid_boxes_local[(0)] = (num_valid_boxes_local[(0)] + 1);
          for (int i_2 = 0; i_2 < (((placeholder1[(0)] + 1022) - i_1) >> 10); ++i_2) {
            if (((((((i_2 * 1024) + i_1) + ((int)threadIdx.x)) + 1) < placeholder1[(0)]) && (0.000000e+00f < fetch_score[(((((i_2 * 1024) + i_1) + ((int)threadIdx.x)) + 1))])) && (nms_v2[(((((i_2 * 1024) + i_1) + ((int)threadIdx.x)) + 1))] == nms_v2[(i_1)])) {
              if (placeholder[(0)] <= ((((((max(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))]) - min(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))])) * (max(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]) - min(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]))) + ((max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))]) - min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))])) * (max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))]) - min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) - (max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]), max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))])) - max(min(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]), min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))]), max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))])) - max(min(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))]), min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))])))))) <= 0.000000e+00f) ? 0.000000e+00f : ((max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]), max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))])) - max(min(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]), min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))]), max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))])) - max(min(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))]), min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))]))))) / ((((max(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))]) - min(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))])) * (max(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]) - min(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]))) + ((max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))]) - min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))])) * (max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))]) - min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) - (max(0.000000e+00f, (min(max(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]), max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))])) - max(min(topk_gpu_v0[(((i_1 * 4) + 1))], topk_gpu_v0[(((i_1 * 4) + 3))]), min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 5))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 7))])))) * max(0.000000e+00f, (min(max(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))]), max(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))])) - max(min(topk_gpu_v0[((i_1 * 4))], topk_gpu_v0[(((i_1 * 4) + 2))]), min(topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 4))], topk_gpu_v0[(((((i_2 * 4096) + (i_1 * 4)) + (((int)threadIdx.x) * 4)) + 6))]))))))))) {
                fetch_score[(((((i_2 * 1024) + i_1) + ((int)threadIdx.x)) + 1))] = -1.000000e+00f;
              }
            }
            __syncthreads();
          }
        }
      }
    }
    if (((int)threadIdx.x) == 0) {
      nms_v4[(0)] = num_valid_boxes_local[(0)];
    }
  } else {
    nms_v4[(0)] = 0;
  }
}

extern "C" __global__ void fused_expand_dims_cast_subtract_add_expand_dims_concatenate_expand_dims_kernel0(float* __restrict__ T_expand_dims, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, long* __restrict__ placeholder3) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 27042) {
    T_expand_dims[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((2 <= (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 6)) ? placeholder[(((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 6) * 4) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 6)) - 2))] : ((1 <= (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 6)) ? ((placeholder1[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 6))] - placeholder2[(0)]) + 1.000000e+00f) : ((float)placeholder3[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 6))])));
  }
}

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

extern "C" __global__ void fused_vision_non_max_suppression_kernel0(float* __restrict__ fetch_score, float* __restrict__ placeholder) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4507) {
    fetch_score[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[((((((int)blockIdx.x) * 6144) + (((int)threadIdx.x) * 6)) + 1))];
  }
}


matched <class 'int'>
(1000,) (1000,)
0
