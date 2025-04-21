#include <torch/extension.h>

// CPU implementations of point operations
// Declarations for CPU functions
void furthestsampling_cpu(int b, int n_max,
    at::Tensor xyz, at::Tensor offset, at::Tensor new_offset,
    at::Tensor tmp, at::Tensor idx);
void knnquery_cpu(int m, int nsample,
    at::Tensor xyz, at::Tensor new_xyz,
    at::Tensor offset, at::Tensor new_offset,
    at::Tensor idx, at::Tensor dist2);
void grouping_forward_cpu(int m, int nsample, int c,
    at::Tensor input, at::Tensor idx, at::Tensor output);
void grouping_backward_cpu(int m, int nsample, int c,
    at::Tensor grad_output, at::Tensor idx, at::Tensor grad_input);
void subtraction_forward_cpu(int n, int nsample, int c,
    at::Tensor input1, at::Tensor input2, at::Tensor idx, at::Tensor output);
void subtraction_backward_cpu(int n, int nsample, int c,
    at::Tensor idx, at::Tensor grad_output,
    at::Tensor grad_input1, at::Tensor grad_input2);
void interpolation_forward_cpu(int n, int c, int k,
    at::Tensor input, at::Tensor idx, at::Tensor weight,
    at::Tensor output);
void interpolation_backward_cpu(int n, int c, int k,
    at::Tensor grad_output, at::Tensor idx, at::Tensor weight,
    at::Tensor grad_input);
void aggregation_forward_cpu(int n, int nsample, int c, int w_c,
    at::Tensor input, at::Tensor position, at::Tensor weight,
    at::Tensor idx, at::Tensor output);
void aggregation_backward_cpu(int n, int nsample, int c, int w_c,
    at::Tensor input, at::Tensor position, at::Tensor weight,
    at::Tensor idx, at::Tensor grad_output,
    at::Tensor grad_input, at::Tensor grad_position,
    at::Tensor grad_weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("furthestsampling", &furthestsampling_cpu, "furthest point sampling (CPU)");
    m.def("knnquery", &knnquery_cpu, "knn query (CPU)");
    m.def("grouping_forward", &grouping_forward_cpu, "grouping forward (CPU)");
    m.def("grouping_backward", &grouping_backward_cpu, "grouping backward (CPU)");
    m.def("subtraction_forward", &subtraction_forward_cpu, "subtraction forward (CPU)");
    m.def("subtraction_backward", &subtraction_backward_cpu, "subtraction backward (CPU)");
    m.def("interpolation_forward", &interpolation_forward_cpu, "interpolation forward (CPU)");
    m.def("interpolation_backward", &interpolation_backward_cpu, "interpolation backward (CPU)");
    m.def("aggregation_forward", &aggregation_forward_cpu, "aggregation forward (CPU)");
    m.def("aggregation_backward", &aggregation_backward_cpu, "aggregation backward (CPU)");
}
