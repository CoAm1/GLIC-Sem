/*
 * Offline FAST-LIVO2-style incremental 3DGS loader.
 *
 * This executable does not subscribe to ROS topics. It reads a frame manifest
 * from disk, projects each frame's point cloud into its paired image, inserts
 * visible colored points, and reuses Gaussian-LIC's incremental optimizer.
 */

#include "gaussian.h"
#include "language_targets.h"

#include <Eigen/Eigen>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace fs = std::filesystem;

struct FastLivoFrame
{
    fs::path image_path;
    fs::path pcd_path;
    Eigen::Matrix3d R_wc = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_wc = Eigen::Vector3d::Zero();
    std::string image_name;
};

struct ProjectedPoint
{
    Eigen::Vector3d point_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d color = Eigen::Vector3d::Ones();
    float depth = 0.0f;
};

struct TemporalFilterConfig
{
    bool enabled = false;
    int radius = 3;
    double distance = 0.12;
    int min_support = 3;
    int min_gap = 2;
    bool require_bidirectional = true;
    int dynamic_max_support = 1;
};

struct TemporalFilterStats
{
    std::size_t input = 0;
    std::size_t kept = 0;
    std::size_t dynamic = 0;
};

std::string trim(const std::string& input)
{
    const auto begin = std::find_if_not(input.begin(), input.end(), [](unsigned char ch) {
        return std::isspace(ch);
    });
    const auto end = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char ch) {
        return std::isspace(ch);
    }).base();
    if (begin >= end) return "";
    return std::string(begin, end);
}

std::string getArg(int argc, char** argv, const std::string& key, const std::string& default_value = "")
{
    const std::string prefix = key + "=";
    const std::string ros_prefix = "_" + key + ":=";
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--" + key && i + 1 < argc)
        {
            return argv[i + 1];
        }
        if (arg.rfind("--" + prefix, 0) == 0)
        {
            return arg.substr(prefix.size() + 2);
        }
        if (arg.rfind(ros_prefix, 0) == 0)
        {
            return arg.substr(ros_prefix.size());
        }
    }
    return default_value;
}

int getIntArg(int argc, char** argv, const std::string& key, int default_value)
{
    const std::string value = getArg(argc, argv, key);
    return value.empty() ? default_value : std::stoi(value);
}

double getDoubleArg(int argc, char** argv, const std::string& key, double default_value)
{
    const std::string value = getArg(argc, argv, key);
    return value.empty() ? default_value : std::stod(value);
}

bool getBoolArg(int argc, char** argv, const std::string& key, bool default_value)
{
    std::string value = getArg(argc, argv, key);
    if (value.empty()) return default_value;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
    if (value == "0" || value == "false" || value == "no" || value == "off") return false;
    throw std::runtime_error("Invalid boolean value for --" + key + ": " + value);
}

bool isCommentOrEmpty(const std::string& line)
{
    const std::string stripped = trim(line);
    return stripped.empty() || stripped[0] == '#';
}

Eigen::Matrix3d quaternionToRotation(double qw, double qx, double qy, double qz)
{
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    return q.toRotationMatrix();
}

FastLivoFrame parseManifestLine(const std::string& line,
                                const fs::path& dataset_root,
                                const std::string& pose_type)
{
    std::istringstream iss(line);
    std::string image_rel;
    std::string pcd_rel;
    double qw = 0.0, qx = 0.0, qy = 0.0, qz = 0.0;
    double tx = 0.0, ty = 0.0, tz = 0.0;

    iss >> image_rel >> pcd_rel >> qw >> qx >> qy >> qz >> tx >> ty >> tz;
    if (!iss)
    {
        throw std::runtime_error("Invalid manifest line: " + line);
    }

    FastLivoFrame frame;
    frame.image_path = fs::path(image_rel).is_absolute() ? fs::path(image_rel) : dataset_root / image_rel;
    frame.pcd_path = fs::path(pcd_rel).is_absolute() ? fs::path(pcd_rel) : dataset_root / pcd_rel;
    frame.image_name = frame.image_path.filename().string();

    if (pose_type == "wc")
    {
        frame.R_wc = quaternionToRotation(qw, qx, qy, qz);
        frame.t_wc = Eigen::Vector3d(tx, ty, tz);
    }
    else if (pose_type == "cw" || pose_type == "colmap")
    {
        const Eigen::Matrix3d R_cw = quaternionToRotation(qw, qx, qy, qz);
        const Eigen::Vector3d t_cw(tx, ty, tz);
        frame.R_wc = R_cw.transpose();
        frame.t_wc = -frame.R_wc * t_cw;
    }
    else
    {
        throw std::runtime_error("Unsupported pose_type: " + pose_type + ". Use wc or cw.");
    }

    return frame;
}

std::vector<FastLivoFrame> loadManifest(const fs::path& manifest_path,
                                        const fs::path& dataset_root,
                                        const std::string& pose_type,
                                        int max_frames)
{
    std::ifstream input(manifest_path);
    if (!input)
    {
        throw std::runtime_error("Unable to open manifest: " + manifest_path.string());
    }

    std::vector<FastLivoFrame> frames;
    std::string line;
    while (std::getline(input, line))
    {
        if (isCommentOrEmpty(line)) continue;
        frames.push_back(parseManifestLine(line, dataset_root, pose_type));
        if (max_frames > 0 && static_cast<int>(frames.size()) >= max_frames) break;
    }

    if (frames.empty())
    {
        throw std::runtime_error("No frames were loaded from manifest.");
    }
    return frames;
}

bool loadRgbCloud(const fs::path& pcd_path, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
{
    pcl::PCLPointCloud2 blob;
    if (pcl::io::loadPCDFile(pcd_path.string(), blob) != 0)
    {
        return false;
    }

    const bool has_color = std::any_of(blob.fields.begin(), blob.fields.end(), [](const auto& field) {
        return field.name == "rgb" || field.name == "rgba";
    });

    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (has_color)
    {
        pcl::fromPCLPointCloud2(blob, *cloud);
        return true;
    }

    pcl::PointCloud<pcl::PointXYZ> xyz_cloud;
    pcl::fromPCLPointCloud2(blob, xyz_cloud);
    cloud->reserve(xyz_cloud.size());
    for (const auto& pt : xyz_cloud.points)
    {
        pcl::PointXYZRGB out;
        out.x = pt.x;
        out.y = pt.y;
        out.z = pt.z;
        out.r = 255;
        out.g = 255;
        out.b = 255;
        cloud->push_back(out);
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = xyz_cloud.is_dense;
    return true;
}

using Cloud = pcl::PointCloud<pcl::PointXYZRGB>;
using CloudPtr = Cloud::Ptr;
using KdTreePtr = std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZRGB>>;

std::vector<CloudPtr> loadClouds(const std::vector<FastLivoFrame>& frames)
{
    std::vector<CloudPtr> clouds;
    clouds.reserve(frames.size());
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        CloudPtr cloud;
        if (!loadRgbCloud(frames[i].pcd_path, cloud))
        {
            throw std::runtime_error("Unable to read PCD: " + frames[i].pcd_path.string());
        }
        clouds.push_back(std::move(cloud));
        if ((i + 1) % 50 == 0 || i + 1 == frames.size())
        {
            std::cout << "[FASTLIVO-INCR] loaded clouds " << i + 1 << "/" << frames.size() << std::endl;
        }
    }
    return clouds;
}

std::vector<KdTreePtr> buildKdTrees(const std::vector<CloudPtr>& clouds)
{
    std::vector<KdTreePtr> trees;
    trees.reserve(clouds.size());
    for (const auto& cloud : clouds)
    {
        auto tree = std::make_shared<pcl::KdTreeFLANN<pcl::PointXYZRGB>>();
        tree->setInputCloud(cloud);
        trees.push_back(std::move(tree));
    }
    return trees;
}

CloudPtr filterTemporalCloud(std::size_t frame_id,
                             const std::vector<CloudPtr>& clouds,
                             const std::vector<KdTreePtr>& trees,
                             const TemporalFilterConfig& config,
                             TemporalFilterStats& stats,
                             CloudPtr* dynamic_points = nullptr)
{
    const auto& source = clouds.at(frame_id);
    stats.input = source->size();

    CloudPtr filtered(new Cloud);
    filtered->reserve(source->size());
    CloudPtr dynamic(new Cloud);
    if (dynamic_points != nullptr) dynamic->reserve(source->size() / 4);

    const std::size_t begin = frame_id > static_cast<std::size_t>(config.radius)
                                  ? frame_id - static_cast<std::size_t>(config.radius)
                                  : 0;
    const std::size_t end = std::min(clouds.size() - 1,
                                     frame_id + static_cast<std::size_t>(config.radius));
    const int available_frames = static_cast<int>(end - begin + 1);
    const int required_support = std::min(config.min_support, available_frames);
    const bool has_far_past = frame_id >= begin + static_cast<std::size_t>(config.min_gap);
    const bool has_far_future = end >= frame_id + static_cast<std::size_t>(config.min_gap);
    const bool require_both_sides = config.require_bidirectional && has_far_past && has_far_future;

    std::vector<int> neighbors(1);
    std::vector<float> squared_distances(1);
    for (const auto& point : source->points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;

        int support = 1;
        bool supported_in_far_past = false;
        bool supported_in_far_future = false;
        for (std::size_t neighbor_id = begin; neighbor_id <= end; ++neighbor_id)
        {
            if (neighbor_id == frame_id) continue;
            neighbors.clear();
            squared_distances.clear();
            if (trees[neighbor_id]->radiusSearch(point, config.distance,
                                                 neighbors, squared_distances, 1) <= 0)
            {
                continue;
            }

            ++support;
            if (neighbor_id + static_cast<std::size_t>(config.min_gap) <= frame_id)
            {
                supported_in_far_past = true;
            }
            if (neighbor_id >= frame_id + static_cast<std::size_t>(config.min_gap))
            {
                supported_in_far_future = true;
            }
        }

        const bool support_ok = support >= required_support;
        const bool temporal_span_ok = !require_both_sides ||
                                      (supported_in_far_past && supported_in_far_future);
        if (support_ok && temporal_span_ok)
        {
            filtered->push_back(point);
        }
        else if (dynamic_points != nullptr && support <= config.dynamic_max_support)
        {
            // No spatial support in the temporal window is a high-confidence
            // dynamic seed. Other rejected points remain "uncertain" and are
            // not allowed to erase RGB supervision.
            dynamic->push_back(point);
        }
    }

    filtered->width = static_cast<std::uint32_t>(filtered->size());
    filtered->height = 1;
    filtered->is_dense = source->is_dense;
    stats.kept = filtered->size();
    if (dynamic_points != nullptr)
    {
        dynamic->width = static_cast<std::uint32_t>(dynamic->size());
        dynamic->height = 1;
        dynamic->is_dense = source->is_dense;
        stats.dynamic = dynamic->size();
        *dynamic_points = std::move(dynamic);
    }
    return filtered;
}

CloudPtr strideCloud(const CloudPtr& source, int stride)
{
    if (stride <= 1) return source;
    CloudPtr sampled(new Cloud);
    sampled->reserve((source->size() + static_cast<std::size_t>(stride) - 1) /
                     static_cast<std::size_t>(stride));
    for (std::size_t i = 0; i < source->size(); i += static_cast<std::size_t>(stride))
    {
        sampled->push_back(source->points[i]);
    }
    sampled->width = static_cast<std::uint32_t>(sampled->size());
    sampled->height = 1;
    sampled->is_dense = source->is_dense;
    return sampled;
}

cv::Mat makeStaticLossMask(const Cloud& dynamic_points,
                           const cv::Mat& image_bgr,
                           const Eigen::Matrix3d& R_wc,
                           const Eigen::Vector3d& t_wc,
                           const Params& params,
                           double max_depth,
                           int radius_pixels,
                           std::size_t& projected_dynamic_points)
{
    cv::Mat static_mask = cv::Mat::ones(image_bgr.rows, image_bgr.cols, CV_32FC1);
    const Eigen::Matrix3d R_cw = R_wc.transpose();
    const Eigen::Vector3d t_cw = -R_cw * t_wc;
    projected_dynamic_points = 0;

    for (const auto& point : dynamic_points.points)
    {
        const Eigen::Vector3d point_w(point.x, point.y, point.z);
        const Eigen::Vector3d point_c = R_cw * point_w + t_cw;
        const double z = point_c.z();
        if (z <= 0.01 || z > max_depth) continue;

        const int u = static_cast<int>(std::round(params.fx * point_c.x() / z + params.cx));
        const int v = static_cast<int>(std::round(params.fy * point_c.y() / z + params.cy));
        if (u < 0 || u >= image_bgr.cols || v < 0 || v >= image_bgr.rows) continue;

        cv::circle(static_mask, cv::Point(u, v), radius_pixels, cv::Scalar(0.0f), cv::FILLED);
        ++projected_dynamic_points;
    }
    return static_mask;
}

Eigen::Vector3d bilinearColor(const cv::Mat& image_rgb_float, double u, double v)
{
    const int width = image_rgb_float.cols;
    const int height = image_rgb_float.rows;
    const int u0 = std::clamp(static_cast<int>(std::floor(u)), 0, width - 1);
    const int v0 = std::clamp(static_cast<int>(std::floor(v)), 0, height - 1);
    const int u1 = std::min(u0 + 1, width - 1);
    const int v1 = std::min(v0 + 1, height - 1);
    const double du = u - u0;
    const double dv = v - v0;

    const cv::Vec3f c00 = image_rgb_float.at<cv::Vec3f>(v0, u0);
    const cv::Vec3f c10 = image_rgb_float.at<cv::Vec3f>(v0, u1);
    const cv::Vec3f c01 = image_rgb_float.at<cv::Vec3f>(v1, u0);
    const cv::Vec3f c11 = image_rgb_float.at<cv::Vec3f>(v1, u1);

    const cv::Vec3f color = (1.0 - du) * (1.0 - dv) * c00 +
                            du * (1.0 - dv) * c10 +
                            (1.0 - du) * dv * c01 +
                            du * dv * c11;
    return Eigen::Vector3d(color[0], color[1], color[2]);
}

std::vector<ProjectedPoint> projectCloudToFrame(const pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                                                const cv::Mat& image_bgr,
                                                const Eigen::Matrix3d& R_wc,
                                                const Eigen::Vector3d& t_wc,
                                                const Params& params,
                                                cv::Mat& depth_map,
                                                double max_depth)
{
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0 / 255.0);

    const int width = image_bgr.cols;
    const int height = image_bgr.rows;
    depth_map = cv::Mat::zeros(height, width, CV_32FC1);

    const Eigen::Matrix3d R_cw = R_wc.transpose();
    const Eigen::Vector3d t_cw = -R_cw * t_wc;

    std::vector<int> best_index(width * height, -1);
    std::vector<float> best_depth(width * height, std::numeric_limits<float>::max());
    std::vector<Eigen::Vector3d> points_w;
    std::vector<Eigen::Vector3d> colors;
    points_w.reserve(cloud.size());
    colors.reserve(cloud.size());

    for (const auto& pt : cloud.points)
    {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

        const Eigen::Vector3d point_w(pt.x, pt.y, pt.z);
        const Eigen::Vector3d point_c = R_cw * point_w + t_cw;
        const double z = point_c.z();
        if (z <= 0.01 || z > max_depth) continue;

        const double u = params.fx * point_c.x() / z + params.cx;
        const double v = params.fy * point_c.y() / z + params.cy;
        const int ui = static_cast<int>(std::round(u));
        const int vi = static_cast<int>(std::round(v));
        if (ui < 0 || ui >= width || vi < 0 || vi >= height) continue;

        const int pixel_index = vi * width + ui;
        if (z < best_depth[pixel_index])
        {
            best_depth[pixel_index] = static_cast<float>(z);
            best_index[pixel_index] = static_cast<int>(points_w.size());
        }

        points_w.push_back(point_w);
        colors.push_back(bilinearColor(image_rgb, u, v));
    }

    std::vector<ProjectedPoint> projected;
    projected.reserve(points_w.size());
    for (int pixel_index = 0; pixel_index < static_cast<int>(best_index.size()); ++pixel_index)
    {
        const int source_index = best_index[pixel_index];
        if (source_index < 0) continue;

        const int u = pixel_index % width;
        const int v = pixel_index / width;
        depth_map.at<float>(v, u) = best_depth[pixel_index];

        ProjectedPoint item;
        item.point_w = points_w[source_index];
        item.color = colors[source_index];
        item.depth = best_depth[pixel_index];
        projected.push_back(item);
    }

    return projected;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gaussian_lic_fastlivo_incremental", ros::init_options::AnonymousName);

    const std::string config_path = getArg(argc, argv, "config_path");
    const std::string manifest_path = getArg(argc, argv, "manifest_path");
    const std::string data_path = getArg(argc, argv, "data_path", ".");
    const std::string result_path = getArg(argc, argv, "result_path", "./result_fastlivo_incremental");
    const std::string lpips_path = getArg(argc, argv, "lpips_path");
    const std::string semantic_path = getArg(argc, argv, "semantic_path");
    const std::string pca_language_path = getArg(argc, argv, "pca_language_path");
    const std::string pca_query_path = getArg(argc, argv, "pca_query_path");
    const std::string pose_type = getArg(argc, argv, "pose_type", "wc");
    const int optimize_per_keyframe = getIntArg(argc, argv, "optimize_per_keyframe", 1);
    const int final_iterations = getIntArg(argc, argv, "final_iterations", 10);
    const int start_frame = getIntArg(argc, argv, "start_frame", 0);
    const int max_frames = getIntArg(argc, argv, "max_frames", 0);
    const int random_seed_override = getIntArg(argc, argv, "random_seed", -1);
    const bool filter_stats_only = getBoolArg(argc, argv, "filter_stats_only", false);
    const bool dynamic_loss_mask = getBoolArg(argc, argv, "dynamic_loss_mask", false);
    const int dynamic_mask_radius = getIntArg(argc, argv, "dynamic_mask_radius", 10);
    const std::string mask_output_path = getArg(argc, argv, "mask_output_path");
    const int point_stride = getIntArg(argc, argv, "point_stride", 1);
    const int depth_accum_frames = getIntArg(argc, argv, "depth_accum_frames", 1);
    const double projection_max_depth = getDoubleArg(argc, argv, "projection_max_depth", 80.0);
    const bool static_depth_fusion = getBoolArg(argc, argv, "static_depth_fusion", false);
    const int static_depth_radius = getIntArg(argc, argv, "static_depth_radius", 3);
    const bool strict_geometry_holdout =
        getBoolArg(argc, argv, "strict_geometry_holdout", false);

    TemporalFilterConfig temporal_filter;
    temporal_filter.enabled = getBoolArg(argc, argv, "temporal_filter", false);
    temporal_filter.radius = getIntArg(argc, argv, "temporal_radius", 3);
    temporal_filter.distance = getDoubleArg(argc, argv, "temporal_distance", 0.12);
    temporal_filter.min_support = getIntArg(argc, argv, "temporal_min_support", 3);
    temporal_filter.min_gap = getIntArg(argc, argv, "temporal_min_gap", 2);
    temporal_filter.require_bidirectional =
        getBoolArg(argc, argv, "temporal_require_bidirectional", true);
    temporal_filter.dynamic_max_support =
        getIntArg(argc, argv, "temporal_dynamic_max_support", 1);

    if (config_path.empty() || manifest_path.empty() || lpips_path.empty())
    {
        std::cerr << "Usage: gs_fastlivo_incremental --config_path CONFIG "
                  << "--manifest_path MANIFEST --lpips_path LPIPS "
                  << "[--data_path ROOT] [--result_path OUT] [--pose_type wc|cw] "
                  << "[--optimize_per_keyframe N] [--final_iterations N] "
                  << "[--start_frame N] [--max_frames N] [--random_seed N] "
                  << "[--temporal_filter true|false] [--temporal_radius 3] "
                  << "[--temporal_distance 0.12] [--temporal_min_support 3] "
                  << "[--temporal_min_gap 2] [--temporal_require_bidirectional true|false] "
                  << "[--temporal_dynamic_max_support 1] [--dynamic_loss_mask true|false] "
                  << "[--dynamic_mask_radius 10] [--mask_output_path DIR] "
                  << "[--point_stride 1] [--depth_accum_frames 1] [--projection_max_depth 80] "
                  << "[--static_depth_fusion true|false] [--static_depth_radius 3] "
                  << "[--strict_geometry_holdout true|false] "
                  << "[--semantic_path ROOT_WITH_LABELS_AND_CONFIDENCE] "
                  << "[--pca_language_path ROOT_WITH_PCA_TARGETS] "
                  << "[--pca_query_path LANGSPLAT_QUERY_JSON] "
                  << "[--filter_stats_only true|false]"
                  << std::endl;
        return 1;
    }

    if (temporal_filter.radius < 1 || temporal_filter.distance <= 0.0 ||
        temporal_filter.min_support < 1 || temporal_filter.min_gap < 1 ||
        temporal_filter.dynamic_max_support < 1 || dynamic_mask_radius < 0 ||
        point_stride < 1 || depth_accum_frames < 1 || projection_max_depth <= 0.0 ||
        static_depth_radius < 1)
    {
        std::cerr << "[FASTLIVO-INCR] invalid temporal filter parameters." << std::endl;
        return 1;
    }
    if (start_frame < 0 || max_frames < 0 || random_seed_override < -1)
    {
        std::cerr << "[FASTLIVO-INCR] start_frame and max_frames must be "
                  << "non-negative; random_seed must be non-negative when set."
                  << std::endl;
        return 1;
    }
    if (start_frame > 0 && temporal_filter.enabled)
    {
        std::cerr << "[FASTLIVO-INCR] start_frame > 0 is not yet supported with "
                  << "the temporal filter because preceding context would be truncated."
                  << std::endl;
        return 1;
    }
    if (filter_stats_only && !temporal_filter.enabled)
    {
        std::cerr << "[FASTLIVO-INCR] --filter_stats_only requires --temporal_filter true." << std::endl;
        return 1;
    }
    if (dynamic_loss_mask && !temporal_filter.enabled)
    {
        std::cerr << "[FASTLIVO-INCR] --dynamic_loss_mask requires --temporal_filter true." << std::endl;
        return 1;
    }
    if (static_depth_fusion && !temporal_filter.enabled)
    {
        std::cerr << "[FASTLIVO-INCR] --static_depth_fusion requires --temporal_filter true."
                  << std::endl;
        return 1;
    }
    if (strict_geometry_holdout &&
        (temporal_filter.enabled || static_depth_fusion || depth_accum_frames != 1))
    {
        std::cerr << "[FASTLIVO-INCR] --strict_geometry_holdout currently requires "
                  << "--temporal_filter false, --static_depth_fusion false, and "
                  << "--depth_accum_frames 1 so held-out LiDAR cannot leak through "
                  << "neighboring-frame fusion."
                  << std::endl;
        return 1;
    }

    try
    {
        const auto total_start = std::chrono::steady_clock::now();
        YAML::Node config_node = YAML::LoadFile(config_path);
        Params params(config_node);
        if (random_seed_override >= 0)
        {
            params.random_seed = random_seed_override;
        }
        if (params.semantic_training && semantic_path.empty())
        {
            throw std::runtime_error(
                "semantic_training=true requires --semantic_path with labels/ and confidence/.");
        }
        if (params.pca_language_training && pca_language_path.empty())
        {
            throw std::runtime_error(
                "pca_language_training=true requires --pca_language_path.");
        }
        auto dataset = std::make_shared<Dataset>(params);
        auto gaussians = std::make_shared<GaussianModel>(params);
        if (params.pca_language_training)
        {
            const auto pca_basis = language::loadPcaBasis(pca_language_path);
            if (pca_basis.dimension != params.semantic_class_count)
            {
                throw std::runtime_error(
                    "PCA target dimension does not match semantic_class_count.");
            }
            gaussians->setPcaLanguageBasis(
                pca_basis.basis_mean, pca_basis.mean_norm_squared);
        }

        auto frames = loadManifest(manifest_path, data_path, pose_type, 0);
        if (static_cast<std::size_t>(start_frame) >= frames.size())
        {
            throw std::runtime_error(
                "start_frame is outside the manifest: " +
                std::to_string(start_frame) + " >= " +
                std::to_string(frames.size()));
        }
        frames.erase(frames.begin(), frames.begin() + start_frame);
        const std::size_t process_frame_count = max_frames > 0
            ? std::min(frames.size(), static_cast<std::size_t>(max_frames))
            : frames.size();
        const std::size_t context_frame_count = temporal_filter.enabled
            ? std::min(frames.size(), process_frame_count +
                                      static_cast<std::size_t>(temporal_filter.radius))
            : process_frame_count;
        frames.resize(context_frame_count);
        dataset->all_frame_num_ = start_frame;
        const auto mapping_start = std::chrono::steady_clock::now();
        std::vector<CloudPtr> temporal_clouds;
        std::vector<KdTreePtr> temporal_trees;
        if (temporal_filter.enabled || depth_accum_frames > 1)
        {
            temporal_clouds = loadClouds(frames);
        }
        if (temporal_filter.enabled)
        {
            std::cout << "[FASTLIVO-INCR] temporal filter: window="
                      << 2 * temporal_filter.radius + 1
                      << ", distance=" << temporal_filter.distance
                      << " m, min_support=" << temporal_filter.min_support
                      << ", min_gap=" << temporal_filter.min_gap
                      << ", bidirectional=" << std::boolalpha
                      << temporal_filter.require_bidirectional << std::noboolalpha
                      << ", dynamic_max_support=" << temporal_filter.dynamic_max_support
                      << std::endl;
            temporal_trees = buildKdTrees(temporal_clouds);
        }
        if (dynamic_loss_mask && !mask_output_path.empty())
        {
            fs::create_directories(mask_output_path);
        }

        if (filter_stats_only)
        {
            std::size_t total_input = 0;
            std::size_t total_kept = 0;
            for (std::size_t frame_id = 0; frame_id < process_frame_count; ++frame_id)
            {
                TemporalFilterStats stats;
                const auto filtered = filterTemporalCloud(frame_id, temporal_clouds,
                                                          temporal_trees, temporal_filter, stats);
                (void)filtered;
                total_input += stats.input;
                total_kept += stats.kept;
                const double ratio = stats.input == 0 ? 0.0 :
                    100.0 * static_cast<double>(stats.kept) / static_cast<double>(stats.input);
                std::cout << "[TEMPORAL-FILTER] frame " << frame_id + 1 << "/" << process_frame_count
                          << ", input=" << stats.input << ", kept=" << stats.kept
                          << ", ratio=" << std::fixed << std::setprecision(2) << ratio << "%"
                          << std::endl;
            }
            const double total_ratio = total_input == 0 ? 0.0 :
                100.0 * static_cast<double>(total_kept) / static_cast<double>(total_input);
            std::cout << "[TEMPORAL-FILTER] total input=" << total_input
                      << ", kept=" << total_kept
                      << ", ratio=" << std::fixed << std::setprecision(2) << total_ratio << "%"
                      << std::endl;
            return 0;
        }

        // Offline symmetric depth fusion needs the same temporally confirmed
        // static cloud for both Gaussian seeding and depth supervision.  Cache
        // it once per frame so a 7-frame fusion window does not repeat seven
        // expensive KD-tree support tests for every output image.
        std::vector<CloudPtr> static_clouds;
        std::vector<CloudPtr> dynamic_clouds;
        std::vector<TemporalFilterStats> cached_filter_stats;
        if (static_depth_fusion)
        {
            static_clouds.reserve(frames.size());
            if (dynamic_loss_mask) dynamic_clouds.reserve(frames.size());
            cached_filter_stats.resize(frames.size());
            for (std::size_t context_id = 0; context_id < frames.size(); ++context_id)
            {
                CloudPtr dynamic_cloud;
                auto static_cloud = filterTemporalCloud(
                    context_id, temporal_clouds, temporal_trees, temporal_filter,
                    cached_filter_stats[context_id],
                    dynamic_loss_mask ? &dynamic_cloud : nullptr);
                static_clouds.push_back(std::move(static_cloud));
                if (dynamic_loss_mask) dynamic_clouds.push_back(std::move(dynamic_cloud));
            }
            std::cout << "[FASTLIVO-INCR] cached temporally confirmed static clouds for "
                      << static_clouds.size() << " frames; symmetric depth radius="
                      << static_depth_radius << std::endl;
        }

        bool initialized = false;

        std::cout << "[FASTLIVO-INCR] frames: " << process_frame_count
                  << ", start frame: " << start_frame
                  << ", temporal context frames: " << frames.size()
                  << ", random seed: " << params.random_seed
                  << ", semantic geometry gradients: " << std::boolalpha
                  << params.semantic_geometry_gradients << std::noboolalpha
                  << std::endl;
        if (params.pca_language_training)
        {
            std::cout << "[FASTLIVO-INCR] pca_language_training=true"
                      << std::endl;
            std::cout << "[FASTLIVO-INCR] PCA language gradient route="
                      << (params.semantic_geometry_gradients
                              ? "joint"
                              : "head-only")
                      << std::endl;
        }
        for (size_t frame_id = 0; frame_id < process_frame_count; ++frame_id)
        {
            const auto& frame = frames[frame_id];
            const bool will_be_keyframe =
                (dataset->all_frame_num_ + 1) % dataset->select_every_k_frame_ == 0;
            cv::Mat image_bgr = cv::imread(frame.image_path.string(), cv::IMREAD_COLOR);
            if (image_bgr.empty())
            {
                throw std::runtime_error("Unable to read image: " + frame.image_path.string());
            }
            if (image_bgr.cols != params.width || image_bgr.rows != params.height)
            {
                cv::resize(image_bgr, image_bgr, cv::Size(params.width, params.height),
                           0.0, 0.0, cv::INTER_AREA);
            }

            CloudPtr cloud;
            CloudPtr dynamic_cloud;
            TemporalFilterStats filter_stats;
            if (temporal_filter.enabled)
            {
                if (static_depth_fusion)
                {
                    cloud = static_clouds.at(frame_id);
                    filter_stats = cached_filter_stats.at(frame_id);
                    if (dynamic_loss_mask) dynamic_cloud = dynamic_clouds.at(frame_id);
                }
                else
                {
                    cloud = filterTemporalCloud(frame_id, temporal_clouds, temporal_trees,
                                                temporal_filter, filter_stats,
                                                dynamic_loss_mask ? &dynamic_cloud : nullptr);
                }
            }
            else if (!loadRgbCloud(frame.pcd_path, cloud))
            {
                throw std::runtime_error("Unable to read PCD: " + frame.pcd_path.string());
            }

            cv::Mat depth_map;
            std::vector<ProjectedPoint> visible_points;
            if (point_stride == 1)
            {
                visible_points = projectCloudToFrame(
                    *cloud, image_bgr, frame.R_wc, frame.t_wc, params, depth_map,
                    projection_max_depth);
            }
            else
            {
                // Point stride controls only the Gaussian seeding budget.  Keep
                // the complete z-buffer for depth supervision and evaluation so
                // A/B runs with different seeding strides are measured on the
                // same LiDAR pixels.
                const auto depth_points = projectCloudToFrame(
                    *cloud, image_bgr, frame.R_wc, frame.t_wc, params, depth_map,
                    projection_max_depth);
                (void)depth_points;
                const CloudPtr seed_cloud = strideCloud(cloud, point_stride);
                cv::Mat ignored_seed_depth;
                visible_points = projectCloudToFrame(
                    *seed_cloud, image_bgr, frame.R_wc, frame.t_wc, params,
                    ignored_seed_depth, projection_max_depth);
            }

            if (static_depth_fusion)
            {
                Cloud accumulated_static_depth_cloud;
                const std::size_t depth_begin = frame_id > static_cast<std::size_t>(static_depth_radius)
                    ? frame_id - static_cast<std::size_t>(static_depth_radius)
                    : 0;
                const std::size_t depth_end = std::min(
                    static_clouds.size() - 1,
                    frame_id + static_cast<std::size_t>(static_depth_radius));
                for (std::size_t depth_frame = depth_begin; depth_frame <= depth_end; ++depth_frame)
                {
                    accumulated_static_depth_cloud += *static_clouds.at(depth_frame);
                }
                cv::Mat accumulated_static_depth;
                const auto ignored_points = projectCloudToFrame(
                    accumulated_static_depth_cloud, image_bgr, frame.R_wc, frame.t_wc,
                    params, accumulated_static_depth, projection_max_depth);
                (void)ignored_points;
                depth_map = std::move(accumulated_static_depth);
            }
            else if (depth_accum_frames > 1)
            {
                Cloud accumulated_depth_cloud;
                const std::size_t depth_begin = frame_id + 1 > static_cast<std::size_t>(depth_accum_frames)
                    ? frame_id + 1 - static_cast<std::size_t>(depth_accum_frames)
                    : 0;
                for (std::size_t depth_frame = depth_begin; depth_frame <= frame_id; ++depth_frame)
                {
                    accumulated_depth_cloud += *temporal_clouds.at(depth_frame);
                }
                cv::Mat accumulated_depth;
                const auto ignored_points = projectCloudToFrame(
                    accumulated_depth_cloud, image_bgr, frame.R_wc, frame.t_wc,
                    params, accumulated_depth, projection_max_depth);
                (void)ignored_points;
                depth_map = std::move(accumulated_depth);
            }

            cv::Mat static_loss_mask;
            cv::Mat semantic_labels;
            cv::Mat semantic_confidence;
            std::optional<language::PcaFrameTarget> pca_language_target;
            std::size_t projected_dynamic_points = 0;
            std::size_t masked_pixels = 0;
            if (dynamic_loss_mask)
            {
                static_loss_mask = makeStaticLossMask(*dynamic_cloud, image_bgr,
                                                      frame.R_wc, frame.t_wc,
                                                      params, projection_max_depth,
                                                      dynamic_mask_radius,
                                                      projected_dynamic_points);
                masked_pixels = static_loss_mask.total() -
                                static_cast<std::size_t>(cv::countNonZero(static_loss_mask));
                if (!mask_output_path.empty())
                {
                    cv::Mat mask_u8;
                    static_loss_mask.convertTo(mask_u8, CV_8UC1, 255.0);
                    const fs::path output_name = fs::path(mask_output_path) /
                                                 (frame.image_path.stem().string() + ".png");
                    cv::imwrite(output_name.string(), mask_u8);
                }
            }

            if (params.semantic_training && will_be_keyframe)
            {
                const std::string semantic_name = frame.image_path.stem().string() + ".png";
                const fs::path label_path = fs::path(semantic_path) / "labels" / semantic_name;
                const fs::path confidence_path =
                    fs::path(semantic_path) / "confidence" / semantic_name;
                semantic_labels = cv::imread(label_path.string(), cv::IMREAD_GRAYSCALE);
                semantic_confidence = cv::imread(confidence_path.string(), cv::IMREAD_GRAYSCALE);
                if (semantic_labels.empty() || semantic_confidence.empty())
                {
                    throw std::runtime_error(
                        "Missing semantic label/confidence for keyframe: " + semantic_name);
                }
                if (semantic_labels.size() != image_bgr.size())
                {
                    cv::resize(semantic_labels, semantic_labels, image_bgr.size(),
                               0.0, 0.0, cv::INTER_NEAREST);
                }
                if (semantic_confidence.size() != image_bgr.size())
                {
                    cv::resize(semantic_confidence, semantic_confidence, image_bgr.size(),
                               0.0, 0.0, cv::INTER_LINEAR);
                }
            }
            if (params.pca_language_training)
            {
                const std::string language_stem = frame.image_path.stem().string();
                const fs::path target_file = fs::path(pca_language_path) /
                                             "targets" / (language_stem + ".bin");
                const fs::path segmentation_file = fs::path(pca_language_path) /
                                                   "segmentation" /
                                                   (language_stem + ".png");
                const bool language_target_exists =
                    fs::exists(target_file) && fs::exists(segmentation_file);
                if (will_be_keyframe && !language_target_exists)
                {
                    throw std::runtime_error(
                        "Missing PCA language target for training keyframe: " +
                        language_stem);
                }
                if (language_target_exists)
                {
                    pca_language_target = language::loadPcaFrameTarget(
                        pca_language_path, language_stem,
                        params.semantic_class_count);
                    if (pca_language_target->region_ids.size(0) != image_bgr.rows ||
                        pca_language_target->region_ids.size(1) != image_bgr.cols)
                    {
                        throw std::runtime_error(
                            "PCA language region map size does not match input image.");
                    }
                }
            }

            if (!strict_geometry_holdout || will_be_keyframe)
            {
                for (const auto& point : visible_points)
                {
                    dataset->addInitialPoint(point.point_w, point.color, point.depth);
                }
            }

            dataset->addOfflineFrame(image_bgr, depth_map, frame.R_wc, frame.t_wc,
                                     params.fx, params.fy, params.cx, params.cy,
                                     frame.image_name, static_loss_mask,
                                     semantic_labels, semantic_confidence);
            if (pca_language_target)
            {
                auto& camera_collection = will_be_keyframe
                    ? dataset->train_cameras_
                    : dataset->test_cameras_;
                if (camera_collection.empty())
                {
                    throw std::runtime_error(
                        "PCA language target camera collection is unexpectedly empty.");
                }
                auto& camera = camera_collection.back();
                camera->language_region_ids_ =
                    std::move(pca_language_target->region_ids);
                camera->language_basis_dot_ =
                    std::move(pca_language_target->basis_dot);
                camera->language_mean_dot_ =
                    std::move(pca_language_target->mean_dot);
                camera->language_confidence_ =
                    std::move(pca_language_target->confidence);
            }

            const int depth_pixels = cv::countNonZero(depth_map > 0.0f);

            if (dataset->is_keyframe_current_ != will_be_keyframe)
            {
                throw std::runtime_error("Keyframe prediction disagrees with Dataset split.");
            }

            std::cout << "[FASTLIVO-INCR] frame " << frame_id + 1 << "/" << process_frame_count
                      << ", visible points " << visible_points.size()
                      << (temporal_filter.enabled ? ", temporal input " : "")
                      << (temporal_filter.enabled ? std::to_string(filter_stats.input) : "")
                      << (temporal_filter.enabled ? ", temporal kept " : "")
                      << (temporal_filter.enabled ? std::to_string(filter_stats.kept) : "")
                      << (dynamic_loss_mask ? ", dynamic seeds " : "")
                      << (dynamic_loss_mask ? std::to_string(filter_stats.dynamic) : "")
                      << (dynamic_loss_mask ? ", projected dynamic " : "")
                      << (dynamic_loss_mask ? std::to_string(projected_dynamic_points) : "")
                      << (dynamic_loss_mask ? ", masked pixels " : "")
                      << (dynamic_loss_mask ? std::to_string(masked_pixels) : "")
                      << ", point stride " << point_stride
                      << ", depth pixels " << depth_pixels
                      << ", depth accum " << depth_accum_frames
                      << (static_depth_fusion ? ", static symmetric depth radius " : "")
                      << (static_depth_fusion ? std::to_string(static_depth_radius) : "")
                      << ", projection max depth " << projection_max_depth
                      << (strict_geometry_holdout ? ", strict holdout " : "")
                      << (strict_geometry_holdout ? (will_be_keyframe ? "train" : "test") : "")
                      << ", train views " << dataset->train_cameras_.size()
                      << ", test views " << dataset->test_cameras_.size()
                      << std::endl;

            if (!dataset->is_keyframe_current_) continue;

            if (!initialized)
            {
                if (dataset->pointcloud_.empty())
                {
                    std::cout << "[FASTLIVO-INCR] skip init: no visible points in keyframe." << std::endl;
                    continue;
                }
                gaussians->initialize(dataset);
                gaussians->trainingSetup();
                initialized = true;
                dataset->pointcloud_.clear();
                dataset->pointcolor_.clear();
                dataset->pointdepth_.clear();
                std::cout << "[FASTLIVO-INCR] Gaussian model initialized." << std::endl;
            }
            else
            {
                if (!dataset->pointcloud_.empty())
                {
                    extend(dataset, gaussians);
                }
            }

            for (int i = 0; i < optimize_per_keyframe; ++i)
            {
                const double updated_num = optimize(dataset, gaussians);
                std::cout << "[FASTLIVO-INCR] optimize " << i + 1 << "/"
                          << optimize_per_keyframe << ", update "
                          << std::fixed << std::setprecision(2)
                          << updated_num / 10000 << "w GS"
                          << (gaussians->last_semantic_loss_ >= 0.0
                                  ? (params.pca_language_training
                                         ? ", PCA language cosine "
                                         : ", semantic CE ")
                                  : "")
                          << (gaussians->last_semantic_loss_ >= 0.0
                                  ? std::to_string(gaussians->last_semantic_loss_)
                                  : std::string())
                          << (gaussians->last_semantic_region_loss_ >= 0.0
                                  ? (params.pca_language_training
                                         ? ", PCA coefficient L1 "
                                         : ", semantic region ")
                                  : "")
                          << (gaussians->last_semantic_region_loss_ >= 0.0
                                  ? std::to_string(gaussians->last_semantic_region_loss_)
                                  : std::string())
                          << std::endl;
            }
        }

        if (!initialized)
        {
            throw std::runtime_error("No Gaussian model was initialized. Check poses, point clouds, and projection.");
        }

        if (dataset->depth_completion_)
        {
            std::cout << "[DEPTH-COMPLETION] summary attempts="
                      << dataset->depth_completion_attempts_
                      << ", accepted=" << dataset->depth_completion_accepted_
                      << ", rejected=" << dataset->depth_completion_rejected_
                      << ", candidates=" << dataset->depth_completion_candidates_
                        << ", proximity_rejected="
                        << dataset->depth_completion_proximity_rejected_
                        << ", support_rejected="
                        << dataset->depth_completion_support_rejected_
                        << ", added_points=" << dataset->depth_completion_added_points_
                      << std::endl;
        }

        for (int i = 0; i < final_iterations; ++i)
        {
            const double updated_num = optimize(dataset, gaussians);
            std::cout << "[FASTLIVO-INCR] final optimize " << i + 1 << "/"
                      << final_iterations << ", update "
                      << std::fixed << std::setprecision(2)
                      << updated_num / 10000 << "w GS"
                      << (gaussians->last_semantic_loss_ >= 0.0
                              ? (params.pca_language_training
                                     ? ", PCA language cosine "
                                     : ", semantic CE ")
                              : "")
                      << (gaussians->last_semantic_loss_ >= 0.0
                              ? std::to_string(gaussians->last_semantic_loss_)
                              : std::string())
                      << (gaussians->last_semantic_region_loss_ >= 0.0
                              ? (params.pca_language_training
                                     ? ", PCA coefficient L1 "
                                     : ", semantic region ")
                              : "")
                      << (gaussians->last_semantic_region_loss_ >= 0.0
                              ? std::to_string(gaussians->last_semantic_region_loss_)
                              : std::string())
                      << std::endl;
        }

        torch::cuda::synchronize();
        const auto mapping_end = std::chrono::steady_clock::now();
        torch::NoGradGuard no_grad;
        const auto evaluation_start = std::chrono::steady_clock::now();
        evaluateVisualQuality(
            dataset, gaussians, result_path, lpips_path, pca_query_path);
        torch::cuda::synchronize();
        const auto evaluation_end = std::chrono::steady_clock::now();
        const auto save_start = std::chrono::steady_clock::now();
        gaussians->saveMap(result_path);
        const auto save_end = std::chrono::steady_clock::now();
        const auto seconds = [](const auto& begin, const auto& end) {
            return std::chrono::duration_cast<std::chrono::duration<double>>(
                end - begin).count();
        };
        std::cout << std::fixed << std::setprecision(3)
                  << "[TIMING] preparation_s=" << seconds(total_start, mapping_start)
                  << " mapping_s=" << seconds(mapping_start, mapping_end)
                  << " evaluation_s=" << seconds(evaluation_start, evaluation_end)
                  << " save_ply_s=" << seconds(save_start, save_end)
                  << " total_s=" << seconds(total_start, save_end)
                  << " optimize_select_s=" << gaussians->t_optlist_
                  << " cpu_to_gpu_s=" << gaussians->t_tocuda_
                  << " forward_s=" << gaussians->t_forward_
                  << " backward_s=" << gaussians->t_backward_
                  << " optimizer_step_s=" << gaussians->t_step_
                  << std::endl;
        std::cout << "[FASTLIVO-INCR] Done. Results saved to " << result_path << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FASTLIVO-INCR] failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
