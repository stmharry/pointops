#include <torch/extension.h>

using at::Tensor;

// CPU implementation of grouping forward
void grouping_forward_cpu(int m, int nsample, int c,
    Tensor input_tensor, Tensor idx_tensor, Tensor output_tensor) {
    auto input = input_tensor.data_ptr<float>();
    auto idx = idx_tensor.data_ptr<int>();
    auto output = output_tensor.data_ptr<float>();
    int total = m * nsample * c;
    for (int index = 0; index < total; ++index) {
        int c_idx = index % c;
        int ns_idx = (index / c) % nsample;
        int m_idx = index / (nsample * c);
        int in_idx = idx[m_idx * nsample + ns_idx] * c + c_idx;
        output[index] = input[in_idx];
    }
}

// CPU implementation of grouping backward
void grouping_backward_cpu(int m, int nsample, int c,
    Tensor grad_output_tensor, Tensor idx_tensor, Tensor grad_input_tensor) {
    auto grad_output = grad_output_tensor.data_ptr<float>();
    auto idx = idx_tensor.data_ptr<int>();
    auto grad_input = grad_input_tensor.data_ptr<float>();
    int total = m * nsample * c;
    for (int index = 0; index < total; ++index) {
        int c_idx = index % c;
        int ns_idx = (index / c) % nsample;
        int m_idx = index / (nsample * c);
        int in_idx = idx[m_idx * nsample + ns_idx] * c + c_idx;
        grad_input[in_idx] += grad_output[index];
    }
}