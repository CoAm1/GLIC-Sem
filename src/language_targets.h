#pragma once

#include <filesystem>
#include <string>

#include <torch/torch.h>

namespace language
{

struct PcaBasis
{
    torch::Tensor mean;       // [1, 512], float32 CPU
    torch::Tensor basis;      // [512, D], float32 CPU
    torch::Tensor basis_mean; // [D], float32 CPU
    double mean_norm_squared = 0.0;
    int dimension = 0;
};

struct PcaFrameTarget
{
    torch::Tensor region_ids; // [H, W], int64 CPU; 0 means unsupervised
    torch::Tensor basis_dot;  // [R, D], float32 CPU
    torch::Tensor mean_dot;   // [R], float32 CPU
    torch::Tensor confidence; // [R], float32 CPU
};

PcaBasis loadPcaBasis(const std::filesystem::path& root);

PcaFrameTarget loadPcaFrameTarget(const std::filesystem::path& root,
                                  const std::string& frame_stem,
                                  int expected_dimension);

} // namespace language
