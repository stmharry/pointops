#include <torch/extension.h>

using at::Tensor;

// CPU implementation of furthest point sampling
void furthestsampling_cpu(int b, int /*n_max*/,
    Tensor xyz_tensor, Tensor offset_tensor, Tensor new_offset_tensor,
    Tensor tmp_tensor, Tensor idx_tensor) {
    auto xyz = xyz_tensor.data_ptr<float>();
    auto offset = offset_tensor.data_ptr<int>();
    auto new_offset = new_offset_tensor.data_ptr<int>();
    auto tmp = tmp_tensor.data_ptr<float>();
    auto idx = idx_tensor.data_ptr<int>();
    int B = offset_tensor.size(0);
    for (int i = 0; i < B; ++i) {
        int s_n = (i == 0) ? 0 : offset[i - 1];
        int e_n = offset[i];
        int s_m = (i == 0) ? 0 : new_offset[i - 1];
        int e_m = new_offset[i];
        if (s_m >= e_m) continue;
        idx[s_m] = s_n;
        int last = s_n;
        for (int j = s_m + 1; j < e_m; ++j) {
            float best = -1.0f;
            int besti = last;
            float x1 = xyz[last*3 + 0];
            float y1 = xyz[last*3 + 1];
            float z1 = xyz[last*3 + 2];
            for (int k = s_n; k < e_n; ++k) {
                float dx = xyz[k*3 + 0] - x1;
                float dy = xyz[k*3 + 1] - y1;
                float dz = xyz[k*3 + 2] - z1;
                float d = dx*dx + dy*dy + dz*dz;
                float d2 = d < tmp[k] ? d : tmp[k];
                tmp[k] = d2;
                if (d2 > best) {
                    best = d2;
                    besti = k;
                }
            }
            idx[j] = besti;
            last = besti;
        }
    }
}