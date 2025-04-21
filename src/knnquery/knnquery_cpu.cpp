#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <limits>

using at::Tensor;

// CPU implementation of KNN query
void knnquery_cpu(int /*m*/, int nsample,
    Tensor xyz_tensor, Tensor new_xyz_tensor,
    Tensor offset_tensor, Tensor new_offset_tensor,
    Tensor idx_tensor, Tensor dist2_tensor) {
    auto xyz = xyz_tensor.data_ptr<float>();
    auto new_xyz = new_xyz_tensor.data_ptr<float>();
    auto offset = offset_tensor.data_ptr<int>();
    auto new_offset = new_offset_tensor.data_ptr<int>();
    auto idx = idx_tensor.data_ptr<int>();
    auto dist2 = dist2_tensor.data_ptr<float>();
    int b = offset_tensor.size(0);
    for (int bi = 0; bi < b; ++bi) {
        int s_n = (bi == 0) ? 0 : offset[bi - 1];
        int e_n = offset[bi];
        int s_m = (bi == 0) ? 0 : new_offset[bi - 1];
        int e_m = new_offset[bi];
        for (int j = s_m; j < e_m; ++j) {
            std::vector<std::pair<float,int>> dist_idx;
            dist_idx.reserve(e_n - s_n);
            float x1 = new_xyz[j*3 + 0];
            float y1 = new_xyz[j*3 + 1];
            float z1 = new_xyz[j*3 + 2];
            for (int k = s_n; k < e_n; ++k) {
                float dx = xyz[k*3 + 0] - x1;
                float dy = xyz[k*3 + 1] - y1;
                float dz = xyz[k*3 + 2] - z1;
                float d = dx*dx + dy*dy + dz*dz;
                dist_idx.emplace_back(d, k);
            }
            if ((int)dist_idx.size() > nsample) {
                std::nth_element(dist_idx.begin(), dist_idx.begin() + nsample, dist_idx.end(),
                    [](auto &a, auto &b){ return a.first < b.first; });
                dist_idx.resize(nsample);
            }
            std::sort(dist_idx.begin(), dist_idx.end(),
                [](auto &a, auto &b){ return a.first < b.first; });
            for (int t = 0; t < nsample; ++t) {
                int o = j * nsample + t;
                if (t < (int)dist_idx.size()) {
                    idx[o] = dist_idx[t].second;
                    dist2[o] = dist_idx[t].first;
                } else {
                    idx[o] = s_n;
                    dist2[o] = std::numeric_limits<float>::infinity();
                }
            }
        }
    }
}