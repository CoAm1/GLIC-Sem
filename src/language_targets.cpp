#include "language_targets.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace language
{
namespace
{

template <typename T>
T readScalar(std::ifstream& input, const fs::path& path)
{
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input)
    {
        throw std::runtime_error("Truncated binary file: " + path.string());
    }
    return value;
}

torch::Tensor loadFloat32Matrix(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("Cannot open matrix: " + path.string());
    }
    const std::uint32_t rows = readScalar<std::uint32_t>(input, path);
    const std::uint32_t cols = readScalar<std::uint32_t>(input, path);
    if (rows == 0 || cols == 0)
    {
        throw std::runtime_error("Empty matrix: " + path.string());
    }
    std::vector<float> values(static_cast<std::size_t>(rows) * cols);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!input)
    {
        throw std::runtime_error("Truncated matrix payload: " + path.string());
    }
    char extra = 0;
    if (input.read(&extra, 1))
    {
        throw std::runtime_error("Unexpected trailing matrix bytes: " + path.string());
    }
    return torch::from_blob(values.data(),
                            {static_cast<std::int64_t>(rows),
                             static_cast<std::int64_t>(cols)},
                            torch::kFloat32).clone();
}

torch::Tensor halfVectorToFloat(std::vector<at::Half>& values,
                                std::vector<std::int64_t> shape)
{
    return torch::from_blob(values.data(), std::move(shape), torch::kFloat16)
        .clone().to(torch::kFloat32);
}

} // namespace

PcaBasis loadPcaBasis(const fs::path& root)
{
    PcaBasis result;
    result.mean = loadFloat32Matrix(root / "mean.f32");
    result.basis = loadFloat32Matrix(root / "basis.f32");
    result.basis_mean = loadFloat32Matrix(root / "basis_mean.f32").squeeze(0);
    if (result.mean.size(0) != 1 || result.mean.size(1) != result.basis.size(0) ||
        result.basis_mean.size(0) != result.basis.size(1))
    {
        throw std::runtime_error("Inconsistent PCA basis shapes under " + root.string());
    }
    const YAML::Node constants = YAML::LoadFile((root / "constants.json").string());
    result.mean_norm_squared = constants["mean_norm_squared"].as<double>();
    result.dimension = static_cast<int>(result.basis.size(1));
    const double recomputed = result.mean.square().sum().item<double>();
    if (std::abs(recomputed - result.mean_norm_squared) > 1e-5)
    {
        throw std::runtime_error("PCA mean norm does not match constants.json");
    }
    return result;
}

PcaFrameTarget loadPcaFrameTarget(const fs::path& root,
                                  const std::string& frame_stem,
                                  int expected_dimension)
{
    const fs::path target_path = root / "targets" / (frame_stem + ".bin");
    std::ifstream input(target_path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("Cannot open PCA target: " + target_path.string());
    }
    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    const std::array<char, 8> expected{{'L', 'G', 'S', 'P', 'C', 'A', '\0', '\0'}};
    if (!input || magic != expected)
    {
        throw std::runtime_error("Invalid PCA target magic: " + target_path.string());
    }
    const std::uint32_t version = readScalar<std::uint32_t>(input, target_path);
    const std::uint32_t rows = readScalar<std::uint32_t>(input, target_path);
    const std::uint32_t dimension = readScalar<std::uint32_t>(input, target_path);
    if (version != 1 || rows == 0 || dimension != static_cast<std::uint32_t>(expected_dimension))
    {
        throw std::runtime_error("Unsupported PCA target header: " + target_path.string());
    }
    std::vector<at::Half> basis_dot(static_cast<std::size_t>(rows) * dimension);
    std::vector<at::Half> mean_dot(rows);
    std::vector<at::Half> confidence(rows);
    input.read(reinterpret_cast<char*>(basis_dot.data()),
               static_cast<std::streamsize>(basis_dot.size() * sizeof(at::Half)));
    input.read(reinterpret_cast<char*>(mean_dot.data()),
               static_cast<std::streamsize>(mean_dot.size() * sizeof(at::Half)));
    input.read(reinterpret_cast<char*>(confidence.data()),
               static_cast<std::streamsize>(confidence.size() * sizeof(at::Half)));
    if (!input)
    {
        throw std::runtime_error("Truncated PCA target payload: " + target_path.string());
    }
    char extra = 0;
    if (input.read(&extra, 1))
    {
        throw std::runtime_error("Unexpected trailing PCA target bytes: " + target_path.string());
    }

    const fs::path segmentation_path =
        root / "segmentation" / (frame_stem + ".png");
    cv::Mat segmentation = cv::imread(segmentation_path.string(), cv::IMREAD_UNCHANGED);
    if (segmentation.empty() || segmentation.type() != CV_16UC1)
    {
        throw std::runtime_error("Expected 16-bit region PNG: " + segmentation_path.string());
    }
    cv::Mat segmentation_i32;
    segmentation.convertTo(segmentation_i32, CV_32SC1);
    auto region_ids = torch::from_blob(
        segmentation_i32.data,
        {segmentation_i32.rows, segmentation_i32.cols},
        torch::TensorOptions().dtype(torch::kInt32)).clone().to(torch::kLong);
    if (region_ids.min().item<std::int64_t>() < 0 ||
        region_ids.max().item<std::int64_t>() > static_cast<std::int64_t>(rows))
    {
        throw std::runtime_error("Region PNG references an invalid PCA target row");
    }

    PcaFrameTarget result;
    result.region_ids = std::move(region_ids);
    result.basis_dot = halfVectorToFloat(
        basis_dot, {static_cast<std::int64_t>(rows),
                    static_cast<std::int64_t>(dimension)});
    result.mean_dot = halfVectorToFloat(
        mean_dot, {static_cast<std::int64_t>(rows)});
    result.confidence = halfVectorToFloat(
        confidence, {static_cast<std::int64_t>(rows)});
    if (!torch::isfinite(result.basis_dot).all().item<bool>() ||
        !torch::isfinite(result.mean_dot).all().item<bool>() ||
        !torch::isfinite(result.confidence).all().item<bool>())
    {
        throw std::runtime_error("PCA target contains non-finite values");
    }
    return result;
}

} // namespace language
