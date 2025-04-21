#include <torch/extension.h>

using at::Tensor;

// CPU implementation of aggregation forward
void aggregation_forward_cpu(int n, int nsample, int c, int w_c,
    Tensor input_tensor, Tensor position_tensor,
    Tensor weight_tensor, Tensor idx_tensor,
    Tensor output_tensor) {
    auto input = input_tensor.data_ptr<float>();
    auto position = position_tensor.data_ptr<float>();
    auto weight = weight_tensor.data_ptr<float>();
    auto idx = idx_tensor.data_ptr<int>();
    auto output = output_tensor.data_ptr<float>();
    int total = n * c;
    for (int index = 0; index < total; ++index) {
        int c_idx = index % c;
        int n_idx = index / c;
        float sum = 0.0f;
        int w_c_idx = c_idx % w_c;
        for (int ns = 0; ns < nsample; ++ns) {
            int idx_i = n_idx * nsample + ns;
            int in_idx = idx[idx_i] * c + c_idx;
            int pos_idx = n_idx * nsample * c + ns * c + c_idx;
            int w_idx = n_idx * nsample * w_c + ns * w_c + w_c_idx;
            sum += (input[in_idx] + position[pos_idx]) * weight[w_idx];
        }
        output[index] = sum;
    }
}

// CPU implementation of aggregation backward
void aggregation_backward_cpu(int n, int nsample, int c, int w_c,
    Tensor input_tensor, Tensor position_tensor,
    Tensor weight_tensor, Tensor idx_tensor,
    Tensor grad_output_tensor, Tensor grad_input_tensor,
    Tensor grad_position_tensor, Tensor grad_weight_tensor) {
    auto input = input_tensor.data_ptr<float>();
    auto position = position_tensor.data_ptr<float>();
    auto weight = weight_tensor.data_ptr<float>();
    auto idx = idx_tensor.data_ptr<int>();
    auto grad_output = grad_output_tensor.data_ptr<float>();
    auto grad_input = grad_input_tensor.data_ptr<float>();
    auto grad_position = grad_position_tensor.data_ptr<float>();
    auto grad_weight = grad_weight_tensor.data_ptr<float>();
    int total = n * c;
    for (int index = 0; index < total; ++index) {
        int c_idx = index % c;
        int n_idx = index / c;
        float go = grad_output[index];
        int w_c_idx = c_idx % w_c;
        for (int ns = 0; ns < nsample; ++ns) {
            int idx_i = n_idx * nsample + ns;
            int in_idx = idx[idx_i] * c + c_idx;
            int pos_idx = n_idx * nsample * c + ns * c + c_idx;
            int w_idx = n_idx * nsample * w_c + ns * w_c + w_c_idx;
            grad_input[in_idx] += go * weight[w_idx];
            grad_position[pos_idx] = go * weight[w_idx];
            grad_weight[w_idx] += go * (input[in_idx] + position[pos_idx]);
        }
    }
}