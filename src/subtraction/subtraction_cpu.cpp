#include <torch/extension.h>

using at::Tensor;

// CPU implementation of subtraction forward
void subtraction_forward_cpu(int n, int nsample, int c,
    Tensor input1_tensor, Tensor input2_tensor,
    Tensor idx_tensor, Tensor output_tensor) {
    auto input1 = input1_tensor.data_ptr<float>();
    auto input2 = input2_tensor.data_ptr<float>();
    auto idx = idx_tensor.data_ptr<int>();
    auto output = output_tensor.data_ptr<float>();
    int total = n * nsample * c;
    for (int index = 0; index < total; ++index) {
        int c_idx = index % c;
        int ns_idx = (index / c) % nsample;
        int n_idx = index / (nsample * c);
        int in1 = n_idx * c + c_idx;
        int in2 = idx[n_idx * nsample + ns_idx] * c + c_idx;
        output[index] = input1[in1] - input2[in2];
    }
}

// CPU implementation of subtraction backward
void subtraction_backward_cpu(int n, int nsample, int c,
    Tensor idx_tensor, Tensor grad_output_tensor,
    Tensor grad_input1_tensor, Tensor grad_input2_tensor) {
    auto idx = idx_tensor.data_ptr<int>();
    auto grad_output = grad_output_tensor.data_ptr<float>();
    auto grad_input1 = grad_input1_tensor.data_ptr<float>();
    auto grad_input2 = grad_input2_tensor.data_ptr<float>();
    int total = n * nsample * c;
    for (int index = 0; index < total; ++index) {
        int c_idx = index % c;
        int ns_idx = (index / c) % nsample;
        int n_idx = index / (nsample * c);
        int in1 = n_idx * c + c_idx;
        int in2 = idx[n_idx * nsample + ns_idx] * c + c_idx;
        grad_input1[in1] += grad_output[index];
        grad_input2[in2] -= grad_output[index];
    }
}