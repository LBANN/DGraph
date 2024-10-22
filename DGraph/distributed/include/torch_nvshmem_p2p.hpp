#pragma once
#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

class NVSHMEMP2P {
  public:
    
    NVSHMEMP2P(){};
    static void init(int rank, int world_size);
    static void finalize();
    void  dist_put(torch::Tensor src, torch::Tensor dst, torch::Tensor indices);
    void  dist_get(torch::Tensor src, torch::Tensor dst, torch::Tensor indices);
    static int m_rank;
    static int m_world_size;
    static bool m_initialized;
};
  