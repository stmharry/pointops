#include <torch/extension.h>

using at::Tensor;

// CPU implementation of interpolation forward
void interpolation_forward_cpu(int n, int c, int k,
    Tensor input_tensor, Tensor idx_tensor, Tensor weight_tensor,
    Tensor output_tensor) {
    auto input = input_tensor.data_ptr<float>();
    auto idx = idx_tensor.data_ptr<int>();
    auto weight = weight_tensor.data_ptr<float>();
    auto output = output_tensor.data_ptr<float>();
    int total = n * c;
    for (int index = 0; index < total; ++index) {
        int c_idx = index % c;
        int n_idx = index / c;
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            int idx_i = n_idx * k + i;
            int in_idx = idx[idx_i] * c + c_idx;
            sum += input[in_idx] * weight[idx_i];
        }
        output[index] = sum;
    }
}

// CPU implementation of interpolation backward
void interpolation_backward_cpu(int n, int c, int k,
    Tensor grad_output_tensor, Tensor idx_tensor,
    Tensor weight_tensor, Tensor grad_input_tensor) {
    auto grad_output = grad_output_tensor.data_ptr<float>();
    auto idx = idx_tensor.data_ptr<int>();
    auto weight = weight_tensor.data_ptr<float>();
    auto grad_input = grad_input_tensor.data_ptr<float>();
    int total = n * c;
    for (int index = 0; index < total; ++index) {
        int c_idx = index % c;
        int n_idx = index / c;
        float go = grad_output[index];
        for (int i = 0; i < k; ++i) {
            int idx_i = n_idx * k + i;
            int in_idx = idx[idx_i] * c + c_idx;
            grad_input[in_idx] += go * weight[idx_i];
        }
    }
}