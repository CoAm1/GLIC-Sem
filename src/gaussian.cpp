/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "gaussian.h"
#include <array>
#include "tensor_utils.h"
#include "loss_utils.h"

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <torch/script.h>
#include <memory>

namespace fs = std::filesystem;

struct PixelPosition 
{
    int u, v;
};

struct SemanticNormalWeight
{
    torch::Tensor weight;
    torch::Tensor reliable;
};

SemanticNormalWeight computeSemanticNormalWeight(
    const torch::Tensor& rendered_depth,
    const std::shared_ptr<Camera>& camera,
    double power,
    double minimum_weight,
    double absolute_depth_tolerance,
    double relative_depth_tolerance)
{
    auto depth = rendered_depth.detach();
    const int64_t height = depth.size(0);
    const int64_t width = depth.size(1);
    auto weights = torch::full_like(depth, minimum_weight);
    auto reliable_full = torch::zeros_like(depth, torch::kBool);
    if (height < 3 || width < 3)
    {
        return {weights, reliable_full};
    }

    using torch::indexing::Slice;
    auto pixel_u = torch::arange(width, depth.options())
                       .view({1, width}).expand({height, width});
    auto pixel_v = torch::arange(height, depth.options())
                       .view({height, 1}).expand({height, width});
    auto points_camera = torch::stack(
        {(pixel_u - static_cast<float>(camera->cx_)) * depth /
             static_cast<float>(camera->fx_),
         (pixel_v - static_cast<float>(camera->cy_)) * depth /
             static_cast<float>(camera->fy_),
         depth},
        2);

    auto tangent_u =
        points_camera.index({Slice(1, height - 1), Slice(2, width), Slice()}) -
        points_camera.index({Slice(1, height - 1), Slice(0, width - 2), Slice()});
    auto tangent_v =
        points_camera.index({Slice(2, height), Slice(1, width - 1), Slice()}) -
        points_camera.index({Slice(0, height - 2), Slice(1, width - 1), Slice()});
    auto normal = torch::cross(tangent_u, tangent_v, 2);
    auto center_point =
        points_camera.index({Slice(1, height - 1), Slice(1, width - 1), Slice()});
    auto normal_norm = torch::sqrt(torch::sum(normal * normal, 2));
    auto view_norm = torch::sqrt(torch::sum(center_point * center_point, 2));
    auto cosine = torch::abs(
        torch::sum(normal * (-center_point), 2) /
        (normal_norm * view_norm).clamp_min(1e-8)).clamp(0.0, 1.0);

    auto center_depth = depth.index({Slice(1, height - 1), Slice(1, width - 1)});
    auto left_depth = depth.index({Slice(1, height - 1), Slice(0, width - 2)});
    auto right_depth = depth.index({Slice(1, height - 1), Slice(2, width)});
    auto top_depth = depth.index({Slice(0, height - 2), Slice(1, width - 1)});
    auto bottom_depth = depth.index({Slice(2, height), Slice(1, width - 1)});
    auto max_depth_delta = torch::stack(
        {(left_depth - center_depth).abs(),
         (right_depth - center_depth).abs(),
         (top_depth - center_depth).abs(),
         (bottom_depth - center_depth).abs()},
        0).amax(0);
    auto depth_tolerance = torch::maximum(
        torch::full_like(center_depth, absolute_depth_tolerance),
        center_depth.abs() * relative_depth_tolerance);
    auto reliable =
        (center_depth > 0.0) & (left_depth > 0.0) & (right_depth > 0.0) &
        (top_depth > 0.0) & (bottom_depth > 0.0) &
        (normal_norm > 1e-8) & (view_norm > 1e-8) &
        (max_depth_delta <= depth_tolerance);

    auto incidence_weight = minimum_weight +
        (1.0 - minimum_weight) * torch::pow(cosine, power);
    auto interior_weight = torch::where(
        reliable, incidence_weight, torch::full_like(incidence_weight, minimum_weight));
    weights.index_put_(
        {Slice(1, height - 1), Slice(1, width - 1)}, interior_weight);
    reliable_full.index_put_(
        {Slice(1, height - 1), Slice(1, width - 1)}, reliable);
    return {weights, reliable_full};
}

std::vector<PixelPosition> selectFromDepthCompletion(const cv::Mat& depth_A, const cv::Mat& depth_B, int patch_size = 20) 
{
    CV_Assert(depth_A.size() == depth_B.size());
    CV_Assert(depth_A.type() == depth_B.type());
    
    int H = depth_A.rows;
    int W = depth_A.cols;
    std::vector<PixelPosition> result;
    result.reserve((H / patch_size) * (W / patch_size));

    for (int i = 0; i < H; i += patch_size) 
    {
        for (int j = 0; j < W; j += patch_size) 
        {
            int h_end = std::min(i + patch_size, H);
            int w_end = std::min(j + patch_size, W);
            
            bool has_valid_A = false;
            bool has_valid_B = false;
            float min_val = std::numeric_limits<float>::max();
            PixelPosition min_pos;
            
            for (int y = i; y < h_end; ++y) 
            {
                const float* ptr_A = depth_A.ptr<float>(y);
                const float* ptr_B = depth_B.ptr<float>(y);
                
                for (int x = j; x < w_end; ++x) 
                {
                    if (ptr_A[x] > 0) 
                    {
                        has_valid_A = true;
                        y = h_end;
                        break;
                    }
                    
                    if (ptr_B[x] > 0) 
                    {
                        has_valid_B = true;
                        if (ptr_B[x] < min_val) 
                        {
                            min_val = ptr_B[x];
                            min_pos = {x, y};
                        }
                    }
                }
            }
            
            if (has_valid_A || !has_valid_B) 
            {
                continue;
            }
            
            result.push_back(min_pos);
        }
    }
    
    return result;
}

bool hasNearbyDepthSupport(const cv::Mat& lidar_depth,
                           int center_u,
                           int center_v,
                           float predicted_depth,
                           int radius_px,
                           double absolute_tolerance_m,
                           double relative_tolerance)
{
    if (lidar_depth.empty() || predicted_depth <= 0.0f || radius_px < 0) return false;

    const int u_begin = std::max(0, center_u - radius_px);
    const int u_end = std::min(lidar_depth.cols - 1, center_u + radius_px);
    const int v_begin = std::max(0, center_v - radius_px);
    const int v_end = std::min(lidar_depth.rows - 1, center_v + radius_px);
    const int radius_squared = radius_px * radius_px;
    const float tolerance = static_cast<float>(std::max(
        absolute_tolerance_m,
        relative_tolerance * static_cast<double>(predicted_depth)));

    for (int v = v_begin; v <= v_end; ++v)
    {
        const float* row = lidar_depth.ptr<float>(v);
        for (int u = u_begin; u <= u_end; ++u)
        {
            const int du = u - center_u;
            const int dv = v - center_v;
            if (du * du + dv * dv > radius_squared) continue;
            const float measured_depth = row[u];
            if (measured_depth > 0.0f &&
                std::abs(measured_depth - predicted_depth) <= tolerance)
            {
                return true;
            }
        }
    }
    return false;
}

void Dataset::addFrame(Frame& cur_frame)
{
    /// image
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(cur_frame.image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_bgr = cv_ptr->image;
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);  // 0-255
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0f / 255.0f);  // 0-1

    /// depth
    cv_bridge::CvImagePtr dp_ptr;
    dp_ptr = cv_bridge::toCvCopy(cur_frame.depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    cv::Mat depth_map = dp_ptr->image;  // metric float32

    /// pose
    Eigen::Quaterniond q_wc;
    Eigen::Vector3d t_wc;
    tf::quaternionMsgToEigen(cur_frame.pose_msg->pose.orientation, q_wc);
    tf::pointMsgToEigen(cur_frame.pose_msg->pose.position, t_wc);
    R_wc_.push_back(q_wc.toRotationMatrix());
    t_wc_.push_back(t_wc);

    /// point
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cur_frame.point_msg, *cloud);
    for (const auto& pt : cloud->points)
    {
        pointcloud_.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
        pointcolor_.emplace_back(Eigen::Vector3d(pt.r, pt.g, pt.b) / 255.0);
        Eigen::Matrix3d R_cw = q_wc.toRotationMatrix().transpose();
        Eigen::Vector3d t_cw = - R_cw * t_wc;
        Eigen::Vector3d pt_c = R_cw * pointcloud_.back() + t_cw;
        assert(pt_c(2) > 0);
        pointdepth_.push_back(static_cast<float>(pt_c(2)));
    }

    /// train & test
    int width = image_rgb.cols, height = image_rgb.rows;
    if ((all_frame_num_ + 1) % select_every_k_frame_ == 0)
    {
        is_keyframe_current_ = true;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        if (depth_completion_)
        {
            cv::Mat completed_depth;  // metric float32
            completed_depth = depth_completer_->complete(image_rgb, depth_map);

            cv::Mat mask_known = depth_map > 0;  // 0/255 uint8
            cv::Mat completed_depth_known;
            completed_depth.copyTo(completed_depth_known, mask_known);
            cv::Mat depth_difference = completed_depth_known - depth_map;
            double mean_depth_difference = cv::mean(depth_difference, mask_known)[0];

            if (std::abs(mean_depth_difference) < 0.1)
            {
                // wanted_depth：non-edge && positive
                cv::Mat depth_gradient_x, depth_gradient_y;
                cv::Sobel(completed_depth, depth_gradient_x, CV_32F, 1, 0, 3);
                cv::Sobel(completed_depth, depth_gradient_y, CV_32F, 0, 1, 3);
                cv::Mat depth_edges;
                cv::magnitude(depth_gradient_x, depth_gradient_y, depth_edges);
                double edge_threshold = 0.1;
                cv::Mat mask_not_edges = depth_edges < edge_threshold;  // 0/255 uint8
                completed_depth -= mean_depth_difference;
                cv::Mat mask = (completed_depth > 0) & mask_not_edges;  // 0/255 uint8
                cv::Mat wanted_depth;
                completed_depth.copyTo(wanted_depth, mask);

                // select
                std::vector<PixelPosition> new_positions = selectFromDepthCompletion(depth_map, wanted_depth, patch_size_);
                for (const auto& pt : new_positions) 
                {
                    int u = pt.u, v = pt.v;
                    float depth = wanted_depth.at<float>(v, u);
                    assert(depth > 0);
                    if (depth > max_depth_) continue;

                    cv::Vec3f color = image_rgb.at<cv::Vec3f>(v, u);
                    Eigen::Vector3d eigen_color(color[0], color[1], color[2]);

                    Eigen::Vector3d cam_point((u - cx_) * depth / fx_, 
                                            (v - cy_) * depth / fy_, 
                                            depth);
                    Eigen::Vector3d world_point = q_wc * cam_point + t_wc;

                    pointcloud_.emplace_back(world_point);
                    pointcolor_.emplace_back(eigen_color);
                    pointdepth_.emplace_back(static_cast<float>(depth));
                }
            }
            else
            {
                // std::cout << "[bef vs aft diff]: " << mean_depth_difference << " m" << std::endl;
            }
        }

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU, true);
        cam->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(depth_map, torch::kCPU, true);
        cv::Mat static_mask = cv::Mat::ones(image_rgb.rows, image_rgb.cols, CV_32FC1);
        cam->original_mask_ = tensor_utils::cvMat2TorchTensor_Float32(static_mask, torch::kCPU, true);
        
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "train_" + formatted_str + ".jpg";

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        train_cameras_.emplace_back(cam);
    }
    else
    {
        is_keyframe_current_ = false;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU);
        cam->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(depth_map, torch::kCPU);
        cv::Mat static_mask = cv::Mat::ones(image_rgb.rows, image_rgb.cols, CV_32FC1);
        cam->original_mask_ = tensor_utils::cvMat2TorchTensor_Float32(static_mask, torch::kCPU);

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "test_" + formatted_str + ".jpg";

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        test_cameras_.emplace_back(cam);
    }

    all_frame_num_ += 1;
}

void Dataset::addOfflineFrame(const cv::Mat& image_bgr,
                              const cv::Mat& depth_map,
                              const Eigen::Matrix3d& R_wc,
                              const Eigen::Vector3d& t_wc,
                              double fx,
                              double fy,
                              double cx,
                              double cy,
                              const std::string& image_name,
                              const cv::Mat& static_mask,
                              const cv::Mat& semantic_labels,
                              const cv::Mat& semantic_confidence)
{
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0f / 255.0f);

    cv::Mat depth_float;
    if (depth_map.empty())
    {
        depth_float = cv::Mat::zeros(image_rgb.rows, image_rgb.cols, CV_32FC1);
    }
    else
    {
        depth_map.convertTo(depth_float, CV_32FC1);
    }

    cv::Mat mask_float;
    if (static_mask.empty())
    {
        mask_float = cv::Mat::ones(image_rgb.rows, image_rgb.cols, CV_32FC1);
    }
    else
    {
        if (static_mask.rows != image_rgb.rows || static_mask.cols != image_rgb.cols)
        {
            throw std::runtime_error("Static mask size does not match image size.");
        }
        static_mask.convertTo(mask_float, CV_32FC1);
        cv::threshold(mask_float, mask_float, 0.5, 1.0, cv::THRESH_BINARY);
    }

    const bool will_be_keyframe = (all_frame_num_ + 1) % select_every_k_frame_ == 0;
    if (will_be_keyframe && depth_completion_)
    {
        ++depth_completion_attempts_;
        cv::Mat completed_depth = depth_completer_->complete(image_rgb, depth_float);
        // Optional, default-off diagnostic export for reports and regression
        // tests. The 16-bit PNG stores millimetres and therefore preserves the
        // actual SPNet prediction instead of only a display-normalized image.
        // This output is never consumed by mapping.
        if (const char* debug_depth_dir = std::getenv("GAUSSIAN_LIC_SPNet_DEPTH_DIR"))
        {
            if (*debug_depth_dir != '\0')
            {
                fs::create_directories(debug_depth_dir);
                std::ostringstream frame_name;
                frame_name << std::setw(6) << std::setfill('0') << all_frame_num_;
                cv::Mat completed_depth_mm;
                completed_depth.convertTo(completed_depth_mm, CV_16UC1, 1000.0);
                cv::imwrite(
                    (fs::path(debug_depth_dir) /
                     (frame_name.str() + "_spnet_mm.png")).string(),
                    completed_depth_mm);
            }
        }
        cv::Mat mask_known = depth_float > 0;
        const int known_pixels = cv::countNonZero(mask_known);
        double mean_depth_difference = std::numeric_limits<double>::infinity();
        std::size_t added_this_frame = 0;
        std::vector<PixelPosition> new_positions;
        std::size_t proximity_rejected_this_frame = 0;
        std::size_t support_rejected_this_frame = 0;

        if (known_pixels > 0)
        {
            cv::Mat completed_depth_known;
            completed_depth.copyTo(completed_depth_known, mask_known);
            cv::Mat depth_difference = completed_depth_known - depth_float;
            mean_depth_difference = cv::mean(depth_difference, mask_known)[0];
        }

        if (std::isfinite(mean_depth_difference) && std::abs(mean_depth_difference) < 0.1)
        {
            ++depth_completion_accepted_;
            cv::Mat depth_gradient_x, depth_gradient_y;
            cv::Sobel(completed_depth, depth_gradient_x, CV_32F, 1, 0, 3);
            cv::Sobel(completed_depth, depth_gradient_y, CV_32F, 0, 1, 3);
            cv::Mat depth_edges;
            cv::magnitude(depth_gradient_x, depth_gradient_y, depth_edges);
            cv::Mat mask_not_edges = depth_edges < 0.1;
            completed_depth -= mean_depth_difference;
            cv::Mat wanted_mask = (completed_depth > 0) & mask_not_edges;
            cv::Mat wanted_depth;
            completed_depth.copyTo(wanted_depth, wanted_mask);

            cv::Mat distance_to_lidar;
            if (depth_completion_max_lidar_distance_px_ > 0.0)
            {
                // distanceTransform measures distance to a zero pixel.  Mark
                // known LiDAR pixels as zero and missing pixels as non-zero so
                // every SPNet candidate receives a distance to the nearest
                // real LiDAR anchor in the current keyframe.
                cv::Mat missing_mask = depth_float <= 0.0f;
                cv::distanceTransform(missing_mask, distance_to_lidar,
                                      cv::DIST_L2, cv::DIST_MASK_PRECISE);
            }

            new_positions = selectFromDepthCompletion(depth_float, wanted_depth, patch_size_);
            depth_completion_candidates_ += new_positions.size();
            for (const auto& position : new_positions)
            {
                const int u = position.u;
                const int v = position.v;
                if (mask_float.at<float>(v, u) <= 0.5f) continue;
                if (!distance_to_lidar.empty() &&
                    distance_to_lidar.at<float>(v, u) >
                        static_cast<float>(depth_completion_max_lidar_distance_px_))
                {
                    ++proximity_rejected_this_frame;
                    continue;
                }
                const float depth = wanted_depth.at<float>(v, u);
                if (depth <= 0.0f || depth > max_depth_) continue;

                const Eigen::Vector3d cam_point((u - cx) * depth / fx,
                                                (v - cy) * depth / fy,
                                                depth);
                const Eigen::Vector3d world_point = R_wc * cam_point + t_wc;

                if (depth_completion_temporal_min_support_ > 0)
                {
                    int support_count = 0;
                    const int current_radius = depth_completion_max_lidar_distance_px_ > 0.0
                        ? static_cast<int>(std::ceil(depth_completion_max_lidar_distance_px_))
                        : depth_completion_temporal_radius_px_;
                    if (hasNearbyDepthSupport(depth_float, u, v, depth, current_radius,
                                              depth_completion_depth_tolerance_m_,
                                              depth_completion_depth_tolerance_ratio_))
                    {
                        ++support_count;
                    }

                    for (auto support_it = depth_completion_support_frames_.rbegin();
                         support_it != depth_completion_support_frames_.rend() &&
                         support_count < depth_completion_temporal_min_support_;
                         ++support_it)
                    {
                        const Eigen::Vector3d support_point_c =
                            support_it->R_wc.transpose() * (world_point - support_it->t_wc);
                        const double support_z = support_point_c.z();
                        if (support_z <= 0.01 || support_z > max_depth_) continue;
                        const int support_u = static_cast<int>(std::round(
                            support_it->fx * support_point_c.x() / support_z + support_it->cx));
                        const int support_v = static_cast<int>(std::round(
                            support_it->fy * support_point_c.y() / support_z + support_it->cy));
                        if (support_u < 0 || support_u >= support_it->depth.cols ||
                            support_v < 0 || support_v >= support_it->depth.rows)
                        {
                            continue;
                        }
                        if (hasNearbyDepthSupport(
                                support_it->depth, support_u, support_v,
                                static_cast<float>(support_z),
                                depth_completion_temporal_radius_px_,
                                depth_completion_depth_tolerance_m_,
                                depth_completion_depth_tolerance_ratio_))
                        {
                            ++support_count;
                        }
                    }

                    if (support_count < depth_completion_temporal_min_support_)
                    {
                        ++support_rejected_this_frame;
                        continue;
                    }
                }

                const cv::Vec3f color = image_rgb.at<cv::Vec3f>(v, u);
                pointcloud_.emplace_back(world_point);
                pointcolor_.emplace_back(color[0], color[1], color[2]);
                pointdepth_.emplace_back(depth);
                ++added_this_frame;
            }
            depth_completion_proximity_rejected_ += proximity_rejected_this_frame;
            depth_completion_support_rejected_ += support_rejected_this_frame;
            depth_completion_added_points_ += added_this_frame;
        }
        else
        {
            ++depth_completion_rejected_;
        }

        std::cout << "[DEPTH-COMPLETION] frame=" << all_frame_num_
                  << " known=" << known_pixels
                  << " mean_error=" << mean_depth_difference
                  << " accepted=" << (added_this_frame > 0 ||
                                         (std::isfinite(mean_depth_difference) &&
                                          std::abs(mean_depth_difference) < 0.1))
                  << " candidates=" << new_positions.size()
                  << " proximity_rejected=" << proximity_rejected_this_frame
                  << " support_rejected=" << support_rejected_this_frame
                  << " added=" << added_this_frame << std::endl;
    }

    if (will_be_keyframe && depth_completion_temporal_window_keyframes_ > 0)
    {
        OfflineDepthSupportFrame support_frame;
        support_frame.depth = depth_float.clone();
        support_frame.R_wc = R_wc;
        support_frame.t_wc = t_wc;
        support_frame.fx = fx;
        support_frame.fy = fy;
        support_frame.cx = cx;
        support_frame.cy = cy;
        depth_completion_support_frames_.push_back(std::move(support_frame));
        while (depth_completion_support_frames_.size() >
               static_cast<std::size_t>(depth_completion_temporal_window_keyframes_))
        {
            depth_completion_support_frames_.pop_front();
        }
    }

    R_wc_.push_back(R_wc);
    t_wc_.push_back(t_wc);

    std::shared_ptr<Camera> cam = std::make_shared<Camera>();
    cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU, true);
    cam->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(depth_float, torch::kCPU, true);
    cam->original_mask_ = tensor_utils::cvMat2TorchTensor_Float32(mask_float, torch::kCPU, true);

    if (!semantic_labels.empty())
    {
        if (semantic_labels.rows != image_rgb.rows || semantic_labels.cols != image_rgb.cols)
        {
            throw std::runtime_error("Semantic label size does not match image size.");
        }
        cv::Mat labels_u8;
        semantic_labels.convertTo(labels_u8, CV_8UC1);
        cv::Mat confidence_float;
        if (semantic_confidence.empty())
        {
            confidence_float = cv::Mat::ones(image_rgb.rows, image_rgb.cols, CV_32FC1);
        }
        else
        {
            if (semantic_confidence.rows != image_rgb.rows ||
                semantic_confidence.cols != image_rgb.cols)
            {
                throw std::runtime_error("Semantic confidence size does not match image size.");
            }
            semantic_confidence.convertTo(confidence_float, CV_32FC1, 1.0 / 255.0);
        }

        cv::Mat labels_float;
        labels_u8.convertTo(labels_float, CV_32FC1);
        cv::Mat semantic_weight = cv::Mat::zeros(image_rgb.rows, image_rgb.cols, CV_32FC1);
        for (int v = 0; v < labels_u8.rows; ++v)
        {
            const auto* label_row = labels_u8.ptr<std::uint8_t>(v);
            const auto* confidence_row = confidence_float.ptr<float>(v);
            auto* weight_row = semantic_weight.ptr<float>(v);
            for (int u = 0; u < labels_u8.cols; ++u)
            {
                const int label = static_cast<int>(label_row[u]);
                if (label <= 0) continue;
                weight_row[u] = std::clamp(confidence_row[u], 0.0f, 1.0f);
            }
        }
        cam->original_semantic_ =
            tensor_utils::cvMat2TorchTensor_Float32(labels_float, torch::kCPU, true);
        cam->semantic_weight_ =
            tensor_utils::cvMat2TorchTensor_Float32(semantic_weight, torch::kCPU, true);
    }

    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << all_frame_num_;
    cam->image_name_ = image_name.empty() ? ("colmap_" + ss.str() + ".jpg") : image_name;
    cam->setIntrinsic(image_rgb.cols, image_rgb.rows, fx, fy, cx, cy);
    cam->setPose(R_wc, t_wc);

    if (will_be_keyframe)
    {
        is_keyframe_current_ = true;
        train_cameras_.emplace_back(cam);
    }
    else
    {
        is_keyframe_current_ = false;
        test_cameras_.emplace_back(cam);
    }

    all_frame_num_ += 1;
}

void Dataset::addInitialPoint(const Eigen::Vector3d& point,
                              const Eigen::Vector3d& color,
                              double depth)
{
    pointcloud_.emplace_back(point);
    pointcolor_.emplace_back(color);
    pointdepth_.emplace_back(static_cast<float>(std::max(depth, 0.01)));
}

GaussianModel::GaussianModel(const Params& prm)
{
    sh_degree_ = prm.sh_degree;
    white_background_ = prm.white_background;
    random_background_ = prm.random_background;
    convert_SHs_python_ = prm.convert_SHs_python;
    compute_cov3D_python_ = prm.compute_cov3D_python;
    lambda_erank_ = prm.lambda_erank;
    scaling_scale_ = prm.scaling_scale;

    position_lr_ = prm.position_lr;
    feature_lr_ = prm.feature_lr;
    opacity_lr_ = prm.opacity_lr;
    scaling_lr_ = prm.scaling_lr;
    rotation_lr_ = prm.rotation_lr;
    lambda_dssim_ = prm.lambda_dssim;
    optimize_depth_ = prm.optimize_depth;
    lambda_depth_ = prm.lambda_depth;
    semantic_training_ = prm.semantic_training;
    semantic_streaming_ce_ = prm.semantic_streaming_ce;
    semantic_geometry_gradients_ = prm.semantic_geometry_gradients;
    pca_language_training_ = prm.pca_language_training;
    semantic_lr_ = prm.semantic_lr;
    lambda_semantic_ = prm.lambda_semantic;
    lambda_pca_language_ = prm.lambda_pca_language;
    lambda_pca_coefficient_ = prm.lambda_pca_coefficient;
    pca_max_coefficient_norm_ = prm.pca_max_coefficient_norm;
    lambda_semantic_region_ = prm.lambda_semantic_region;
    semantic_region_stride_ = prm.semantic_region_stride;
    semantic_balance_power_ = prm.semantic_balance_power;
    semantic_balance_max_ = prm.semantic_balance_max;
    semantic_normal_weighting_ = prm.semantic_normal_weighting;
    semantic_normal_power_ = prm.semantic_normal_power;
    semantic_normal_min_weight_ = prm.semantic_normal_min_weight;
    semantic_normal_depth_tolerance_m_ =
        prm.semantic_normal_depth_tolerance_m;
    semantic_normal_depth_tolerance_ratio_ =
        prm.semantic_normal_depth_tolerance_ratio;
    semantic_normal_weight_classes_ = prm.semantic_normal_weight_classes;
    semantic_init_logit_scale_ = prm.semantic_init_logit_scale;
    semantic_keyframe_window_ = prm.semantic_keyframe_window;
    semantic_observation_ema_ = prm.semantic_observation_ema;
    semantic_observation_depth_tolerance_m_ =
        prm.semantic_observation_depth_tolerance_m;
    semantic_observation_depth_tolerance_ratio_ =
        prm.semantic_observation_depth_tolerance_ratio;
    semantic_observation_switch_support_ =
        prm.semantic_observation_switch_support;
    semantic_observation_cumulative_ =
        prm.semantic_observation_cumulative;
    semantic_class_count_ = prm.semantic_class_count;
    semantic_min_support_ = prm.semantic_min_support;
    semantic_min_confidence_ = prm.semantic_min_confidence;
    random_seed_ = prm.random_seed;
    extend_alpha_threshold_ = prm.extend_alpha_threshold;
    max_gaussian_scale_ = prm.max_gaussian_scale;
    iteration_decay_ = prm.iteration_decay;

    apply_exposure_ = prm.apply_exposure;
    exposure_lr_ = prm.exposure_lr;
    skybox_points_num_ = prm.skybox_points_num;
    skybox_radius_ = prm.skybox_radius;

    auto device_type = torch::kCUDA;
    GAUSSIAN_MODEL_INIT_TENSORS(device_type)

    is_init_ = false;

    t_forward_ = 0;
    t_backward_ = 0;
    t_step_ = 0;
    t_optlist_ = 0;
    t_tocuda_ = 0;
}

torch::Tensor GaussianModel::getScaling()
{
    return torch::exp(scaling_);
}

torch::Tensor GaussianModel::getRotation()
{
    return torch::nn::functional::normalize(rotation_);
}

torch::Tensor GaussianModel::getXYZ()
{
    return xyz_;
}

torch::Tensor GaussianModel::getFeaturesDc()
{
    return features_dc_;
}

torch::Tensor GaussianModel::getFeaturesRest()
{
    return features_rest_;
}

torch::Tensor GaussianModel::getSemanticFeatures()
{
    return semantic_feature_;
}

void GaussianModel::setPcaLanguageBasis(const torch::Tensor& basis_mean,
                                        double mean_norm_squared)
{
    if (!pca_language_training_)
    {
        throw std::runtime_error("PCA language basis supplied while mode is disabled.");
    }
    if (basis_mean.dim() != 1 || basis_mean.size(0) != semantic_class_count_ ||
        !torch::isfinite(basis_mean).all().item<bool>() ||
        !std::isfinite(mean_norm_squared) || mean_norm_squared <= 0.0)
    {
        throw std::runtime_error("Invalid PCA language basis constants.");
    }
    pca_basis_mean_ = basis_mean.to(torch::kCUDA).to(torch::kFloat32).contiguous();
    pca_mean_norm_squared_ = mean_norm_squared;
}

torch::Tensor GaussianModel::getOpacity()
{
    return torch::sigmoid(opacity_);
}

torch::Tensor GaussianModel::getCovariance(int scaling_modifier)
{
    // build_rotation
    auto r = this->rotation_;
    auto R = general_utils::build_rotation(r);

    // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
    auto s = scaling_modifier * this->getScaling();
    auto L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
    L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
    L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
    L = R.matmul(L); // L = R @ L

    // build_covariance_from_scaling_rotation
    auto actual_covariance = L.matmul(L.transpose(1, 2));
    // strip_symmetric
    // strip_lowerdiag
    auto symm_uncertainty = torch::zeros({actual_covariance.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    symm_uncertainty.select(1, 0).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 0}));
    symm_uncertainty.select(1, 1).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 1}));
    symm_uncertainty.select(1, 2).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 2}));
    symm_uncertainty.select(1, 3).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 1}));
    symm_uncertainty.select(1, 4).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 2}));
    symm_uncertainty.select(1, 5).copy_(actual_covariance.index({torch::indexing::Slice(), 2, 2}));

    return symm_uncertainty;
}

torch::Tensor GaussianModel::getExposure()
{
    return exposure_;
}

void GaussianModel::initialize(const std::shared_ptr<Dataset>& dataset)
{
    /// foreground
    int num = static_cast<int>(dataset->pointcloud_.size());
    assert(num > 0);
    torch::Tensor fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();  // (n, 3)
    int deg_2 = (sh_degree_ + 1) * (sh_degree_ + 1);
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    torch::Tensor scales = torch::zeros({num}, torch::kFloat32).cuda();

    double f = (dataset->fx_ + dataset->fy_) / 2;
    for (int i = 0; i < num; ++i) 
    {
        auto& pt_w = dataset->pointcloud_[i];
        auto& color = dataset->pointcolor_[i];
        fused_point_cloud.index({i, 0}) = pt_w.x();
        fused_point_cloud.index({i, 1}) = pt_w.y();
        fused_point_cloud.index({i, 2}) = pt_w.z();
        features.index({i, 0, 0}) = RGB2SH(color.x());
        features.index({i, 1, 0}) = RGB2SH(color.y());
        features.index({i, 2, 0}) = RGB2SH(color.z());

        double d = dataset->pointdepth_[i];
        scales.index({i}) = std::log(scaling_scale_ * d / f);
    }
    scales = scales.unsqueeze(1).repeat({1, 3});  // (n, 3)
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)

    /// sky
    if (skybox_points_num_ > 0)
    {
        int num = skybox_points_num_;
        double radius = skybox_radius_;
        torch::Tensor pi = torch::acos(torch::tensor(-1.0, torch::kFloat32).cuda());
        torch::Tensor theta = 2.0 * pi * torch::rand({num}, torch::kFloat32).cuda();
        torch::Tensor phi = torch::acos(1.0 - 1.4 * torch::rand({num}, torch::kFloat32).cuda());
        torch::Tensor sky_fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();
        sky_fused_point_cloud.index({torch::indexing::Slice(), 0}) = radius * 10 * torch::cos(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 1}) = radius * 10 * torch::sin(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 2}) = radius * 10 * torch::cos(phi);

        torch::Tensor sky_features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();
        sky_features.index({torch::indexing::Slice(), 0, 0}) = 0.7;
        sky_features.index({torch::indexing::Slice(), 1, 0}) = 0.8;
        sky_features.index({torch::indexing::Slice(), 2, 0}) = 0.95;

        torch::Tensor point_cloud_copy = sky_fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor sky_scales = torch::log(torch::sqrt(dist2));
        sky_scales = sky_scales.unsqueeze(1).repeat({1, 3});
        torch::Tensor sky_rots = torch::zeros({num, 4}, torch::kFloat32).cuda();
        sky_rots.index({torch::indexing::Slice(), 0}) = 1;
        torch::Tensor sky_opacities = general_utils::inverse_sigmoid(0.7f * torch::ones({num, 1}, torch::kFloat32).cuda());

        fused_point_cloud = torch::cat({sky_fused_point_cloud, fused_point_cloud}, 0);
        features = torch::cat({sky_features, features}, 0);
        scales = torch::cat({sky_scales, scales}, 0);
        rots = torch::cat({sky_rots, rots}, 0);
        opacities = torch::cat({sky_opacities, opacities}, 0);
    }

    this->xyz_ = fused_point_cloud.requires_grad_();  // (n, 3)
    // this->xyz_ = fused_point_cloud.requires_grad_(false);  // fix xyz
    this->features_dc_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().requires_grad_();  // (n, 1, 3)
    this->features_rest_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous().requires_grad_();  // (n, 15, 3)
    this->semantic_feature_ = torch::zeros(
        {fused_point_cloud.size(0), semantic_class_count_},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).requires_grad_();
    this->semantic_support_ = torch::zeros(
        {fused_point_cloud.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    this->semantic_evidence_ = torch::zeros(
        {fused_point_cloud.size(0), semantic_class_count_},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    this->semantic_pending_label_ = torch::zeros(
        {fused_point_cloud.size(0)},
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    this->semantic_pending_support_ = torch::zeros(
        {fused_point_cloud.size(0)},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    this->scaling_ = scales.requires_grad_();  // (n, 3)
    this->rotation_ = rots.requires_grad_();  // (n, 4)
    this->opacity_ = opacities.requires_grad_();  // (n, 1)

    if (apply_exposure_)
    {
        torch::Tensor exposure = torch::eye(3, torch::kFloat32).cuda();
        exposure = torch::cat({exposure, torch::zeros({3, 1}, torch::kFloat32).cuda()}, 1);
        this->exposure_ = exposure.requires_grad_();  // (3, 4)
    }

    GAUSSIAN_MODEL_TENSORS_TO_VEC
    
    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;37m Init Map with " 
              << double(fused_point_cloud.size(0)) / 10000 << "w GS" 
              << ",\033[0m";

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

void GaussianModel::saveMap(const std::string& result_path)
{
    std::string pc_path = result_path + "/point_cloud.ply";

    torch::Tensor xyz = this->xyz_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    // torch::Tensor normals = torch::zeros_like(xyz);
    torch::Tensor f_dc = this->features_dc_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor f_rest = this->features_rest_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor opacities = this->opacity_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor scale = this->scaling_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor rotation = this->rotation_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    auto semantic_logits = this->semantic_feature_.index(
        {torch::indexing::Slice(skybox_points_num_)}).detach();
    torch::Tensor semantic_latent;
    torch::Tensor semantic_prob;
    torch::Tensor semantic_support = this->semantic_support_.index(
        {torch::indexing::Slice(skybox_points_num_)}).detach().contiguous().cpu();
    torch::Tensor semantic_confidence;
    torch::Tensor semantic_label;
    if (pca_language_training_)
    {
        semantic_latent = semantic_logits.contiguous().cpu();
    }
    else
    {
        semantic_prob = torch::softmax(semantic_logits, 1).contiguous().cpu();
        auto semantic_max = semantic_prob.max(1);
        semantic_confidence = std::get<0>(semantic_max);
        torch::Tensor semantic_label_i64 = std::get<1>(semantic_max);
        auto supported = (semantic_support.squeeze(1) >= semantic_min_support_) &
                         (semantic_confidence >= semantic_min_confidence_);
        semantic_label_i64 = torch::where(
            supported, semantic_label_i64, torch::zeros_like(semantic_label_i64));
        semantic_confidence = (semantic_confidence *
            semantic_support.squeeze(1).clamp(0.0, 1.0)).contiguous();
        semantic_label = semantic_label_i64.to(torch::kInt32).contiguous();
    }

    std::filebuf fb_binary;
    fb_binary.open(pc_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);

    tinyply::PlyFile result_file;

    // xyz
    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // // normals
    // result_file.add_properties_to_element(
    //     "vertex", {"nx", "ny", "nz"},
    //     tinyply::Type::FLOAT32, normals.size(0),
    //     reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
    //     tinyply::Type::INVALID, 0);

    // f_dc
    std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
    std::vector<std::string> property_names_f_dc(n_f_dc);
    for (int i = 0; i < n_f_dc; ++i)
        property_names_f_dc[i] = "f_dc_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_dc,
        tinyply::Type::FLOAT32, this->features_dc_.size(0),
        reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // f_rest
    std::size_t n_f_rest = this->features_rest_.size(1) * this->features_rest_.size(2);
    std::vector<std::string> property_names_f_rest(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
        property_names_f_rest[i] = "f_rest_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_rest,
        tinyply::Type::FLOAT32, this->features_rest_.size(0),
        reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // opacities
    result_file.add_properties_to_element(
        "vertex", {"opacity"},
        tinyply::Type::FLOAT32, opacities.size(0),
        reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // scale
    std::size_t n_scale = scale.size(1);
    std::vector<std::string> property_names_scale(n_scale);
    for (int i = 0; i < n_scale; ++i)
        property_names_scale[i] = "scale_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_scale,
        tinyply::Type::FLOAT32, scale.size(0),
        reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // rotation
    std::size_t n_rotation = rotation.size(1);
    std::vector<std::string> property_names_rotation(n_rotation);
    for (int i = 0; i < n_rotation; ++i)
        property_names_rotation[i] = "rot_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_rotation,
        tinyply::Type::FLOAT32, rotation.size(0),
        reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    if (pca_language_training_)
    {
        std::vector<std::string> property_names_language(semantic_class_count_);
        for (int i = 0; i < semantic_class_count_; ++i)
            property_names_language[i] = "language_pca_" + std::to_string(i);
        result_file.add_properties_to_element(
            "vertex", property_names_language,
            tinyply::Type::FLOAT32, semantic_latent.size(0),
            reinterpret_cast<uint8_t*>(semantic_latent.data_ptr<float>()),
            tinyply::Type::INVALID, 0);
    }
    else
    {
        std::vector<std::string> property_names_semantic(semantic_class_count_);
        for (int i = 0; i < semantic_class_count_; ++i)
            property_names_semantic[i] = "semantic_prob_" + std::to_string(i);
        result_file.add_properties_to_element(
            "vertex", property_names_semantic,
            tinyply::Type::FLOAT32, semantic_prob.size(0),
            reinterpret_cast<uint8_t*>(semantic_prob.data_ptr<float>()),
            tinyply::Type::INVALID, 0);

        result_file.add_properties_to_element(
            "vertex", {"semantic_label"},
            tinyply::Type::INT32, semantic_label.size(0),
            reinterpret_cast<uint8_t*>(semantic_label.data_ptr<int>()),
            tinyply::Type::INVALID, 0);

        result_file.add_properties_to_element(
            "vertex", {"semantic_confidence"},
            tinyply::Type::FLOAT32, semantic_confidence.size(0),
            reinterpret_cast<uint8_t*>(semantic_confidence.data_ptr<float>()),
            tinyply::Type::INVALID, 0);

        result_file.add_properties_to_element(
            "vertex", {"semantic_support"},
            tinyply::Type::FLOAT32, semantic_support.size(0),
            reinterpret_cast<uint8_t*>(semantic_support.data_ptr<float>()),
            tinyply::Type::INVALID, 0);
    }

    // Write the file
    result_file.write(outstream_binary, true);

    fb_binary.close();
}

void GaussianModel::trainingSetup()
{
    this->sparse_optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, 0.0, 1e-15));
    sparse_optimizer_->param_groups()[0].options().set_lr(position_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_dc_);
    sparse_optimizer_->param_groups()[1].options().set_lr(feature_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_rest_);
    sparse_optimizer_->param_groups()[2].options().set_lr(feature_lr_ / 20.0);

    sparse_optimizer_->add_param_group(Tensor_vec_opacity_);
    sparse_optimizer_->param_groups()[3].options().set_lr(opacity_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_scaling_);
    sparse_optimizer_->param_groups()[4].options().set_lr(scaling_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_rotation_);
    sparse_optimizer_->param_groups()[5].options().set_lr(rotation_lr_);

    // Keep the original six geometry/appearance groups byte-for-byte in the
    // main optimizer.  Semantic features have an independent sparse Adam so
    // enabling semantics cannot change group ordering or learning rates.
    if (semantic_training_ || pca_language_training_)
    {
        this->semantic_optimizer_.reset(
            new SparseGaussianAdam(Tensor_vec_semantic_feature_, semantic_lr_, 1e-15));
    }

    if (apply_exposure_)
    {
        this->exposure_optimizer_.reset(new torch::optim::Adam(Tensor_vec_exposure_, {}));
        exposure_optimizer_->param_groups()[0].options().set_lr(exposure_lr_);
    }
}

void GaussianModel::densificationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_semantic_features,
    torch::Tensor& new_semantic_support,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation)
{
    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = 
    {
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    };
    auto& param_groups = this->sparse_optimizer_->param_groups();
    auto& optimizer_state = this->sparse_optimizer_->get_state();

    for (int group_idx = 0; group_idx < 6; ++group_idx) 
    {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];

        auto old_param_impl = param.unsafeGetTensorImpl();

        param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
        // if (group_idx == 0) param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_(false);  // fix xyz
        // else param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();  // fix xyz
        group.params()[0] = param;

        auto new_param_impl = param.unsafeGetTensorImpl();

        auto state_it = optimizer_state.find(old_param_impl);
        if (state_it != optimizer_state.end()) 
        {
            auto stored_state = state_it->second;

            stored_state.exp_avg = torch::cat({stored_state.exp_avg.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);
            stored_state.exp_avg_sq = torch::cat({stored_state.exp_avg_sq.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);

            optimizer_state.erase(state_it);

            optimizer_state[new_param_impl] = stored_state;
        }
        else 
        {
            State new_state;
            new_state.step = 0;
            new_state.exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.initialized = true;

            optimizer_state[new_param_impl] = new_state;
        }

        optimizable_tensors[group_idx] = param;
    }

    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];

    if (semantic_optimizer_)
    {
        auto& semantic_group = semantic_optimizer_->param_groups()[0];
        auto& semantic_param = semantic_group.params()[0];
        auto& semantic_state = semantic_optimizer_->get_state();
        auto old_param_impl = semantic_param.unsafeGetTensorImpl();
        semantic_param = torch::cat({semantic_param, new_semantic_features}, 0).requires_grad_();
        semantic_group.params()[0] = semantic_param;
        auto new_param_impl = semantic_param.unsafeGetTensorImpl();
        auto state_it = semantic_state.find(old_param_impl);
        if (state_it != semantic_state.end())
        {
            auto stored_state = state_it->second;
            stored_state.exp_avg = torch::cat(
                {stored_state.exp_avg.clone(), torch::zeros_like(new_semantic_features)}, 0);
            stored_state.exp_avg_sq = torch::cat(
                {stored_state.exp_avg_sq.clone(), torch::zeros_like(new_semantic_features)}, 0);
            semantic_state.erase(state_it);
            semantic_state[new_param_impl] = stored_state;
        }
        this->semantic_feature_ = semantic_param;
    }
    else
    {
        this->semantic_feature_ = torch::cat(
            {this->semantic_feature_, new_semantic_features}, 0).requires_grad_();
    }
    this->semantic_support_ = torch::cat(
        {this->semantic_support_, new_semantic_support},
        0);
    auto new_semantic_evidence = torch::zeros_like(new_semantic_features);
    if (this->semantic_init_logit_scale_ > 0.0)
    {
        new_semantic_evidence = new_semantic_features.detach() /
            this->semantic_init_logit_scale_;
    }
    this->semantic_evidence_ = torch::cat(
        {this->semantic_evidence_, new_semantic_evidence}, 0);
    this->semantic_pending_label_ = torch::cat(
        {this->semantic_pending_label_,
         torch::zeros({new_semantic_features.size(0)},
                      this->semantic_pending_label_.options())},
        0);
    this->semantic_pending_support_ = torch::cat(
        {this->semantic_pending_support_,
         torch::zeros({new_semantic_features.size(0)},
                      this->semantic_pending_support_.options())},
        0);

    GAUSSIAN_MODEL_TENSORS_TO_VEC
}

void extend(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc)
{
    torch::NoGradGuard no_grad;
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    std::shared_ptr<Camera> viewpoint_cam = dataset->train_cameras_.back();
    auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_, true);
    auto rendered_alpha = 1 - std::get<2>(render_pkg).squeeze(0);

    int n = dataset->pointcloud_.size();
    std::vector<float> float_point(n * 3);
    std::vector<float> float_color(n * 3);
    for (size_t i = 0; i < n; ++i) 
    {
        float_point[3 * i + 0] = static_cast<float>(dataset->pointcloud_[i][0]);
        float_point[3 * i + 1] = static_cast<float>(dataset->pointcloud_[i][1]);
        float_point[3 * i + 2] = static_cast<float>(dataset->pointcloud_[i][2]);
        float_color[3 * i + 0] = static_cast<float>(dataset->pointcolor_[i][0]);
        float_color[3 * i + 1] = static_cast<float>(dataset->pointcolor_[i][1]);
        float_color[3 * i + 2] = static_cast<float>(dataset->pointcolor_[i][2]);
    }
    torch::Tensor points = torch::from_blob(float_point.data(), {n, 3}).to(torch::kFloat32).cuda();
    torch::Tensor colors = torch::from_blob(float_color.data(), {n, 3}).to(torch::kFloat32).cuda();
    torch::Tensor depths_in_rsp_frame = torch::from_blob(dataset->pointdepth_.data(), {n}).to(torch::kFloat32).cuda();

    /// filter
    auto R_wc = dataset->R_wc_.back();
    auto t_wc = dataset->t_wc_.back();
    auto R_cw = R_wc.transpose();
    auto t_cw = - R_cw * t_wc;
    std::vector<float> float_R_cw(3 * 3);
    std::vector<float> float_t_cw(3);
    for (size_t i = 0; i < 3; ++i)
    {
        float_R_cw[3 * i + 0] = static_cast<float>(R_cw(i, 0));
        float_R_cw[3 * i + 1] = static_cast<float>(R_cw(i, 1));
        float_R_cw[3 * i + 2] = static_cast<float>(R_cw(i, 2));
        float_t_cw[i] = static_cast<float>(t_cw[i]);
    }
    torch::Tensor R_cw_tensor = torch::from_blob(float_R_cw.data(), {3, 3}).to(torch::kFloat32).cuda();
    torch::Tensor t_cw_tensor = torch::from_blob(float_t_cw.data(), {3, 1}).to(torch::kFloat32).cuda();

    // One recursive semantic update per arriving keyframe. Unlike the CE
    // replay below, this can revise an already-created Gaussian when a newer
    // view provides a different confident label. Geometry and appearance are
    // deliberately untouched.
    if (pc->semantic_training_ &&
        (pc->semantic_observation_ema_ > 0.0 ||
         pc->semantic_observation_cumulative_) &&
        viewpoint_cam->original_semantic_.defined() &&
        viewpoint_cam->semantic_weight_.defined())
    {
        auto existing_xyz = pc->getXYZ().detach();
        auto existing_h = torch::cat(
            {existing_xyz, torch::ones({existing_xyz.size(0), 1}, existing_xyz.options())},
            1);
        auto existing_camera_h = torch::matmul(
            existing_h, viewpoint_cam->world_view_transform_);
        auto existing_camera = existing_camera_h.slice(1, 0, 3);
        auto existing_z = existing_camera.index({torch::indexing::Slice(), 2});
        auto safe_existing_z = torch::where(
            existing_z.abs() > 1e-6, existing_z, torch::ones_like(existing_z));
        auto existing_x = ((existing_camera.index({torch::indexing::Slice(), 0}) *
                            static_cast<float>(viewpoint_cam->fx_)) /
                           safe_existing_z + static_cast<float>(viewpoint_cam->cx_))
                              .round().to(torch::kLong);
        auto existing_y = ((existing_camera.index({torch::indexing::Slice(), 1}) *
                            static_cast<float>(viewpoint_cam->fy_)) /
                           safe_existing_z + static_cast<float>(viewpoint_cam->cy_))
                              .round().to(torch::kLong);
        auto inside = (existing_z > 0.0) & (existing_x >= 0) &
                      (existing_x < viewpoint_cam->image_width_) &
                      (existing_y >= 0) &
                      (existing_y < viewpoint_cam->image_height_);
        auto safe_x = existing_x.clamp(0, viewpoint_cam->image_width_ - 1);
        auto safe_y = existing_y.clamp(0, viewpoint_cam->image_height_ - 1);
        auto target_image = viewpoint_cam->original_semantic_.to(
            torch::kCUDA, /*non_blocking=*/true).to(torch::kLong);
        auto confidence_image = viewpoint_cam->semantic_weight_.to(
            torch::kCUDA, /*non_blocking=*/true);
        auto observed_target = target_image.index({safe_y, safe_x});
        auto observed_confidence = confidence_image.index({safe_y, safe_x})
                                       .clamp(0.0, 1.0);
        // The no-color render used by extend() intentionally returns a zero
        // depth image. Request one normal render only for the optional
        // recursive semantic update so the z-buffer gate is meaningful.
        auto semantic_depth_pkg = render(
            viewpoint_cam, pc, bg, pc->apply_exposure_, false);
        auto visible_existing = std::get<4>(semantic_depth_pkg);
        auto rendered_depth_image = std::get<1>(semantic_depth_pkg);
        auto rendered_surface_depth = rendered_depth_image.index({safe_y, safe_x});
        auto center_depth_error = (existing_z - rendered_surface_depth).abs();
        auto depth_tolerance = torch::maximum(
            torch::full_like(rendered_surface_depth,
                             pc->semantic_observation_depth_tolerance_m_),
            rendered_surface_depth.abs() *
                pc->semantic_observation_depth_tolerance_ratio_);
        auto depth_consistent = (rendered_surface_depth > 0.0) &
            ((existing_z - rendered_surface_depth).abs() <= depth_tolerance);
        auto valid_observation = inside & visible_existing &
            depth_consistent &
            (observed_target > 0) &
            (observed_target < pc->semantic_class_count_) &
            (observed_confidence > 0.0);
        auto valid_indices = torch::nonzero(valid_observation).squeeze(1);
        if (valid_indices.numel() > 0)
        {
            auto target_rows = observed_target.index_select(0, valid_indices);
            auto confidence_rows = observed_confidence.index_select(0, valid_indices);
            auto desired_logits = torch::zeros(
                {valid_indices.size(0), pc->semantic_class_count_},
                pc->semantic_feature_.options());
            desired_logits.scatter_(
                1, target_rows.unsqueeze(1),
                (confidence_rows * pc->semantic_init_logit_scale_).unsqueeze(1));
            auto old_logits = pc->semantic_feature_.index_select(0, valid_indices);
            auto updated_logits = old_logits.clone();
            if (pc->semantic_observation_cumulative_)
            {
                auto evidence_rows = pc->semantic_evidence_.index_select(
                    0, valid_indices);
                auto evidence_increment = torch::zeros_like(evidence_rows);
                evidence_increment.scatter_(
                    1, target_rows.unsqueeze(1), confidence_rows.unsqueeze(1));
                evidence_rows = evidence_rows + evidence_increment;
                auto evidence_sum = evidence_rows.sum(1, true).clamp_min(1e-6);
                updated_logits = pc->semantic_init_logit_scale_ *
                    evidence_rows / evidence_sum;
                pc->semantic_evidence_.index_put_(
                    {valid_indices}, evidence_rows);
            }
            else if (pc->semantic_observation_switch_support_ > 0.0)
            {
                auto current_max = old_logits.max(1);
                auto current_strength = std::get<0>(current_max);
                auto current_label = std::get<1>(current_max);
                auto same_class = current_label == target_rows;
                auto unknown_class = current_strength.abs() <= 1e-6;
                auto direct_update = same_class | unknown_class;
                auto ema_logits =
                    (1.0 - pc->semantic_observation_ema_) * old_logits +
                    pc->semantic_observation_ema_ * desired_logits;
                updated_logits.index_put_(
                    {direct_update}, ema_logits.index({direct_update}));

                auto pending_label = pc->semantic_pending_label_.index_select(
                    0, valid_indices);
                auto pending_support = pc->semantic_pending_support_.index_select(
                    0, valid_indices);
                auto conflict = ~direct_update;
                auto same_pending = pending_label == target_rows;
                auto next_pending_support = torch::where(
                    same_pending, pending_support + confidence_rows,
                    confidence_rows);
                auto should_switch = conflict &
                    (next_pending_support >=
                     pc->semantic_observation_switch_support_);
                updated_logits.index_put_(
                    {should_switch}, desired_logits.index({should_switch}));

                auto keep_pending = conflict & ~should_switch;
                auto zero_label = torch::zeros_like(target_rows);
                auto zero_support = torch::zeros_like(confidence_rows);
                auto updated_pending_label = torch::where(
                    keep_pending, target_rows, zero_label);
                auto updated_pending_support = torch::where(
                    keep_pending, next_pending_support, zero_support);
                pc->semantic_pending_label_.index_put_(
                    {valid_indices}, updated_pending_label);
                pc->semantic_pending_support_.index_put_(
                    {valid_indices}, updated_pending_support);
            }
            else
            {
                updated_logits =
                    (1.0 - pc->semantic_observation_ema_) * old_logits +
                    pc->semantic_observation_ema_ * desired_logits;
            }
            pc->semantic_feature_.index_put_({valid_indices}, updated_logits);
            auto old_support = pc->semantic_support_.index_select(0, valid_indices);
            auto updated_support = torch::maximum(
                old_support, confidence_rows.unsqueeze(1));
            pc->semantic_support_.index_put_({valid_indices}, updated_support);
        }
    }

    auto points_camera = torch::matmul(points, R_cw_tensor.t()) + t_cw_tensor.view({1, 3});  // (n, 3)
    auto depths = points_camera.index({torch::indexing::Slice(), 2});  // (n)
    float fx = static_cast<float>(viewpoint_cam->fx_);
    float fy = static_cast<float>(viewpoint_cam->fy_);
    float cx = static_cast<float>(viewpoint_cam->cx_);
    float cy = static_cast<float>(viewpoint_cam->cy_);
    float focal = (fx + fy) / 2.0;
    torch::Tensor x_pixel = (points_camera.index({torch::indexing::Slice(), 0}) * fx) / depths + cx;
    torch::Tensor y_pixel = (points_camera.index({torch::indexing::Slice(), 1}) * fy) / depths + cy;
    auto pixels = torch::stack({x_pixel, y_pixel}, 1);  // (n, 2)
    pixels = pixels.floor().to(torch::kInt32);

    auto pixels_float = pixels.to(torch::kFloat32);
    auto pixels_with_depth = torch::cat({pixels_float, depths.unsqueeze(1)}, 1).to(torch::kCPU);
    auto pixels_depth_a = pixels_with_depth.accessor<float, 2>();

    std::unordered_map<std::string, std::pair<int, float>> pixel_depth_map;
    for (int i = 0; i < pixels_with_depth.size(0); ++i) {
        int x = static_cast<int>(pixels_depth_a[i][0]);
        int y = static_cast<int>(pixels_depth_a[i][1]);
        float depth = pixels_depth_a[i][2];
        
        std::string key = std::to_string(x) + "_" + std::to_string(y);
        if (!pixel_depth_map.count(key) || depth < pixel_depth_map[key].second) {
            pixel_depth_map[key] = {i, depth};
        }
    }

    std::vector<int64_t> keep_indices;
    for (const auto& item : pixel_depth_map) {
        keep_indices.push_back(item.second.first);
    }

    auto keep_indices_tensor = torch::from_blob(
        keep_indices.data(), 
        {static_cast<int64_t>(keep_indices.size())}, 
        torch::kInt64
    ).to(points.device());
    auto filtered_points = points.index_select(0, keep_indices_tensor);
    auto filtered_colors = colors.index_select(0, keep_indices_tensor);
    auto filtered_depths_in_rsp_frame = depths_in_rsp_frame.index_select(0, keep_indices_tensor);
    auto filtered_pixels = pixels.index_select(0, keep_indices_tensor);

    int H = viewpoint_cam->image_height_, W = viewpoint_cam->image_width_;
    const float extend_alpha_threshold = static_cast<float>(pc->extend_alpha_threshold_);
    auto filter = [H, W, &rendered_alpha, extend_alpha_threshold](const torch::Tensor& points, 
                                        const torch::Tensor& colors, 
                                        const torch::Tensor& depths_in_rsp_frame, 
                                        const torch::Tensor& pixels) 
    {
        auto in_image = (pixels.index({torch::indexing::Slice(), 0}) >= 0) & 
                        (pixels.index({torch::indexing::Slice(), 0}) < W) &
                        (pixels.index({torch::indexing::Slice(), 1}) >= 0) & 
                        (pixels.index({torch::indexing::Slice(), 1}) < H);  // (n) bool
        
        auto positive_depth = depths_in_rsp_frame > 0;

        auto x_coords = pixels.index({torch::indexing::Slice(), 0}).clamp(0, W - 1);
        auto y_coords = pixels.index({torch::indexing::Slice(), 1}).clamp(0, H - 1);
        auto opaque = rendered_alpha.index({y_coords, x_coords}) < extend_alpha_threshold;  // (n) bool

        auto valid_flag = torch::logical_and(torch::logical_and(in_image, positive_depth), opaque);
        auto filtered_points = points.index({valid_flag, torch::indexing::Slice()});
        auto filtered_colors = colors.index({valid_flag, torch::indexing::Slice()});
        auto filtered_depths = depths_in_rsp_frame.index({valid_flag});
        auto filtered_pixels = pixels.index({valid_flag, torch::indexing::Slice()});
        return std::make_tuple(filtered_points, filtered_colors, filtered_depths,
                               filtered_pixels);
    };

    // auto filtered_pkg = filter(points, colors, depths_in_rsp_frame, pixels);
    auto filtered_pkg = filter(filtered_points, filtered_colors, filtered_depths_in_rsp_frame, filtered_pixels);
    
    /// densification
    torch::Tensor fused_point_cloud = std::get<0>(filtered_pkg);  // (n, 3)
    torch::Tensor fused_color = RGB2SH(std::get<1>(filtered_pkg));
    int num = fused_point_cloud.size(0);
    int deg_2 = (pc->sh_degree_ + 1) * (pc->sh_degree_ + 1);
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) = fused_color;
    torch::Tensor features_dc = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();  // (n, 1, 3)
    torch::Tensor features_rest = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous();  // (n, 15, 3)
    torch::Tensor scales = torch::log(pc->scaling_scale_ * std::get<2>(filtered_pkg) / focal).unsqueeze(1).repeat({1, 3});  // (n, 3)
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)
    torch::Tensor semantic_features = torch::zeros(
        {num, pc->semantic_class_count_}, torch::kFloat32).cuda();
    torch::Tensor semantic_support = torch::zeros({num, 1}, torch::kFloat32).cuda();
    if (pc->semantic_training_ && pc->semantic_init_logit_scale_ > 0.0 &&
        viewpoint_cam->original_semantic_.defined() &&
        viewpoint_cam->semantic_weight_.defined() && num > 0)
    {
        auto semantic_pixels = std::get<3>(filtered_pkg).to(torch::kLong);
        auto semantic_x = semantic_pixels.index({torch::indexing::Slice(), 0});
        auto semantic_y = semantic_pixels.index({torch::indexing::Slice(), 1});
        auto semantic_target = viewpoint_cam->original_semantic_.to(
            torch::kCUDA, /*non_blocking=*/true).to(torch::kLong)
            .index({semantic_y, semantic_x});
        auto semantic_confidence = viewpoint_cam->semantic_weight_.to(
            torch::kCUDA, /*non_blocking=*/true)
            .index({semantic_y, semantic_x}).clamp(0.0, 1.0);
        auto semantic_valid = (semantic_target > 0) &
                              (semantic_target < pc->semantic_class_count_) &
                              (semantic_confidence > 0.0);
        auto safe_target = semantic_target.clamp(0, pc->semantic_class_count_ - 1);
        auto initial_strength = semantic_confidence *
            semantic_valid.to(semantic_confidence.scalar_type()) *
            pc->semantic_init_logit_scale_;
        semantic_features.scatter_(1, safe_target.unsqueeze(1),
                                   initial_strength.unsqueeze(1));
        semantic_support.copy_((semantic_confidence *
            semantic_valid.to(semantic_confidence.scalar_type())).unsqueeze(1));
    }

    pc->densificationPostfix(fused_point_cloud, features_dc, features_rest,
                             semantic_features, semantic_support,
                             opacities, scales, rots);

    const int nearest_z_points = static_cast<int>(keep_indices.size());
    const int alpha_rejected = nearest_z_points - num;
    std::cout << "[EXTEND-STATS] input=" << n
              << ", nearest_z=" << nearest_z_points
              << ", alpha_rejected=" << alpha_rejected
              << ", inserted=" << num << std::endl;

    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;32m Insert " << double(fused_point_cloud.size(0)) / 1000 
              << "k GS" << ",\033[0m";

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

void decayOptList(int max_iters, const int train_camera_num, 
                  const std::shared_ptr<Dataset>& dataset, const std::vector<int>& all_list, std::vector<int>& opt_list)
{
    Eigen::Vector3d t0 = dataset->t_wc_[0];
    double dist = (dataset->t_wc_.back() - t0).norm();
    if (dist > 120)
    {
        max_iters /= 2;
        opt_list.clear();
        std::random_device rd;
        std::mt19937 gen(rd());
        int split = train_camera_num * 2 / 3;
        int half = max_iters / 2;
        std::sample(all_list.begin(), all_list.begin() + split,
                    std::back_inserter(opt_list), std::min(half, split), gen);
        std::sample(all_list.begin() + split, all_list.end(),
                    std::back_inserter(opt_list), std::min(half, train_camera_num - split), gen);
    }
}

double optimize(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc)
{
    pc->t_start_ = std::chrono::steady_clock::now();
    int updated_num = 0;
    std::vector<int> opt_list;
    int max_iters = 100;

    int train_camera_num = dataset->train_cameras_.size();
    std::vector<int> all_list(train_camera_num);
    std::iota(all_list.begin(), all_list.end(), 0);

    // A/B experiments must see the same camera-order sequence.  The call
    // counter preserves variation across optimization calls while making an
    // entire run reproducible for a fixed seed.
    std::mt19937 gen(static_cast<uint32_t>(
        pc->random_seed_ + pc->optimize_call_count_++));
    if (train_camera_num <= max_iters) 
    {
        opt_list = all_list;
    }
    else
    {
        std::sample(all_list.begin(), all_list.end(), 
                    std::back_inserter(opt_list), max_iters, gen);
    } 
    if (pc->iteration_decay_) decayOptList(max_iters, train_camera_num, dataset, all_list, opt_list);
    std::shuffle(opt_list.begin(), opt_list.end(), gen);
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_optlist_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

    pc->t_start_ = std::chrono::steady_clock::now();
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    double semantic_loss_sum = 0.0;
    int semantic_loss_count = 0;
    double semantic_region_loss_sum = 0.0;
    int semantic_region_loss_count = 0;
    bool semantic_normal_stats_logged = false;
    for (int idx : opt_list)
    {
        pc->t_start_ = std::chrono::steady_clock::now();
        const std::shared_ptr<Camera>& viewpoint_cam = dataset->train_cameras_[idx];
        auto gt_image = viewpoint_cam->original_image_.to(torch::kCUDA, /*non_blocking=*/true);
        auto gt_depth = viewpoint_cam->original_depth_.to(torch::kCUDA, /*non_blocking=*/true);
        auto gt_static_mask = viewpoint_cam->original_mask_.defined()
                                  ? viewpoint_cam->original_mask_.to(torch::kCUDA, /*non_blocking=*/true)
                                  : torch::ones_like(gt_depth);
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
        pc->t_start_ = std::chrono::steady_clock::now();
        auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_);
        auto visible = std::get<4>(render_pkg);
        auto rendered_image = std::get<0>(render_pkg);
        auto rendered_depth = std::get<1>(render_pkg);
        auto static_mask = gt_static_mask > 0.5;
        auto depth_mask = (gt_depth > 0) & (rendered_depth > 0) & static_mask;
        auto rgb_mask = static_mask.unsqueeze(0).expand_as(rendered_image);
        auto Ll1 = torch::abs(rendered_image.masked_select(rgb_mask) -
                              gt_image.masked_select(rgb_mask)).mean();
        float lambda_dssim = pc->lambda_dssim_;
        float lambda_depth = pc->lambda_depth_;
        torch::Tensor ssim_value;
        auto rgb_mask_float = rgb_mask.to(rendered_image.scalar_type());
        auto rendered_image_for_ssim = rendered_image * rgb_mask_float +
                                       gt_image * (1.0 - rgb_mask_float);
        torch::Tensor rendered_image_unsq = rendered_image_for_ssim.unsqueeze(0);
        torch::Tensor gt_image_unsq = gt_image.unsqueeze(0);
        ssim_value = loss_utils::fused_ssim(rendered_image_unsq, gt_image_unsq);
        auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value);
        torch::Tensor pca_head_loss;
        torch::Tensor semantic_support_increment;
        if (pc->optimize_depth_ && depth_mask.any().item<bool>())
        {
            auto Ll1_depth = torch::abs(rendered_depth.masked_select(depth_mask) -
                                        gt_depth.masked_select(depth_mask)).mean();
            loss += lambda_depth * Ll1_depth;
        }
        if (pc->pca_language_training_ &&
            viewpoint_cam->language_region_ids_.defined() &&
            viewpoint_cam->language_basis_dot_.defined() &&
            viewpoint_cam->language_mean_dot_.defined() &&
            viewpoint_cam->language_confidence_.defined())
        {
            auto semantic_bg = torch::zeros({3}, bg.options());
            std::vector<torch::Tensor> language_chunks;
            torch::Tensor language_alpha;
            for (int offset = 0; offset < pc->semantic_class_count_; offset += 3)
            {
                auto language_pkg = renderSemanticChunk(
                    viewpoint_cam, pc, semantic_bg, offset,
                    pc->semantic_geometry_gradients_);
                if (!language_alpha.defined())
                {
                    language_alpha =
                        (1.0 - std::get<2>(language_pkg)).clamp(0.0, 1.0);
                }
                language_chunks.push_back(
                    std::get<0>(language_pkg) /
                    language_alpha.clamp_min(0.05).unsqueeze(0));
            }
            auto rendered_language = torch::cat(language_chunks, 0).slice(
                0, 0, pc->semantic_class_count_);
            auto region_ids = viewpoint_cam->language_region_ids_.to(
                torch::kCUDA, /*non_blocking=*/true).to(torch::kLong);
            auto valid_language = (region_ids > 0) & (language_alpha > 0.05) &
                                  (gt_static_mask > 0.5);
            auto valid_flat = torch::nonzero(valid_language.flatten()).squeeze(1);
            if (valid_flat.numel() > 0)
            {
                auto region_rows = region_ids.flatten().index_select(
                    0, valid_flat) - 1;
                auto rendered_samples = rendered_language.flatten(1).transpose(0, 1)
                    .index_select(0, valid_flat);
                auto basis_dot = viewpoint_cam->language_basis_dot_.to(
                    torch::kCUDA, /*non_blocking=*/true).index_select(0, region_rows);
                auto mean_dot = viewpoint_cam->language_mean_dot_.to(
                    torch::kCUDA, /*non_blocking=*/true).index_select(0, region_rows);
                auto confidence = viewpoint_cam->language_confidence_.to(
                    torch::kCUDA, /*non_blocking=*/true).index_select(0, region_rows);
                auto numerator = mean_dot +
                    (rendered_samples * basis_dot).sum(1);
                auto denominator_squared =
                    pc->pca_mean_norm_squared_ +
                    2.0 * (rendered_samples * pc->pca_basis_mean_).sum(1) +
                    rendered_samples.square().sum(1);
                auto cosine = numerator /
                    torch::sqrt(denominator_squared.clamp_min(1e-12));
                auto confidence_sum = confidence.sum().clamp_min(1e-6);
                auto language_loss =
                    ((1.0 - cosine.clamp(-1.0, 1.0)) * confidence).sum() /
                    confidence_sum;
                auto weighted_language_loss =
                    pc->lambda_pca_language_ * language_loss;
                if (pc->semantic_geometry_gradients_)
                {
                    loss = loss + weighted_language_loss;
                }
                else
                {
                    pca_head_loss = weighted_language_loss;
                }
                semantic_loss_sum += language_loss.detach().item<double>();
                ++semantic_loss_count;
                if (pc->lambda_pca_coefficient_ > 0.0)
                {
                    // For an orthonormal PCA basis, the exact teacher
                    // coefficient is (teacher - mean) @ basis. basis_dot is
                    // teacher @ basis and pca_basis_mean_ is mean @ basis.
                    // This term removes the scale ambiguity left by cosine
                    // supervision and prevents per-Gaussian latent blow-up.
                    auto target_coefficients =
                        basis_dot - pc->pca_basis_mean_.unsqueeze(0);
                    auto coefficient_per_pixel = torch::abs(
                        rendered_samples - target_coefficients).mean(1);
                    auto coefficient_loss =
                        (coefficient_per_pixel * confidence).sum() /
                        confidence_sum;
                    auto weighted_coefficient_loss =
                        pc->lambda_pca_coefficient_ * coefficient_loss;
                    if (pc->semantic_geometry_gradients_)
                    {
                        loss = loss + weighted_coefficient_loss;
                    }
                    else
                    {
                        pca_head_loss = pca_head_loss.defined()
                            ? pca_head_loss + weighted_coefficient_loss
                            : weighted_coefficient_loss;
                    }
                    semantic_region_loss_sum +=
                        coefficient_loss.detach().item<double>();
                    ++semantic_region_loss_count;
                }
            }
        }
        else if (pc->semantic_training_ && pc->semantic_streaming_ce_ &&
            (pc->semantic_keyframe_window_ <= 0 ||
             idx >= std::max(0, train_camera_num - pc->semantic_keyframe_window_)) &&
            viewpoint_cam->original_semantic_.defined() &&
            viewpoint_cam->semantic_weight_.defined())
        {
            auto semantic_target = viewpoint_cam->original_semantic_.to(
                torch::kCUDA, /*non_blocking=*/true).to(torch::kLong);
            auto semantic_weight = viewpoint_cam->semantic_weight_.to(
                torch::kCUDA, /*non_blocking=*/true);
            auto semantic_bg = torch::zeros({3}, bg.options());
            torch::Tensor semantic_alpha;
            torch::Tensor semantic_gradient;
            torch::Tensor valid_semantic_weight;

            // Pass 1 has no autograd graph. It computes the exact global
            // softmax probabilities and d(CE)/d(logit) while keeping only the
            // small CxHxW logits tensor, not one rasterizer graph per chunk.
            {
                torch::NoGradGuard no_grad;
                std::vector<torch::Tensor> detached_chunks;
                for (int offset = 0; offset < pc->semantic_class_count_; offset += 3)
                {
                    auto semantic_pkg = renderSemanticChunk(
                        viewpoint_cam, pc, semantic_bg, offset);
                    if (!semantic_alpha.defined())
                    {
                        semantic_alpha =
                            (1.0 - std::get<2>(semantic_pkg)).clamp(0.0, 1.0);
                    }
                    detached_chunks.push_back(
                        std::get<0>(semantic_pkg) /
                        semantic_alpha.clamp_min(0.05).unsqueeze(0));
                }
                auto detached_logits = torch::cat(detached_chunks, 0).slice(
                    0, 0, pc->semantic_class_count_);
                auto valid_target = (semantic_target > 0) &
                                    (semantic_target < pc->semantic_class_count_);
                valid_semantic_weight =
                    semantic_weight * gt_static_mask *
                    valid_target.to(semantic_weight.scalar_type()) *
                    (semantic_alpha > 0.05).to(semantic_weight.scalar_type());
                auto semantic_weight_sum = valid_semantic_weight.sum();
                if (semantic_weight_sum.item<float>() > 0.0f)
                {
                    auto safe_target = semantic_target.clamp(
                        0, pc->semantic_class_count_ - 1);
                    auto probabilities = torch::softmax(detached_logits, 0);
                    auto target_one_hot = torch::zeros_like(probabilities);
                    target_one_hot.scatter_(
                        0, safe_target.unsqueeze(0), 1.0);
                    semantic_gradient =
                        (probabilities - target_one_hot) *
                        (valid_semantic_weight / semantic_weight_sum).unsqueeze(0);
                    auto per_pixel_semantic_ce = -torch::log_softmax(
                        detached_logits, 0).gather(
                            0, safe_target.unsqueeze(0)).squeeze(0);
                    auto semantic_loss =
                        (per_pixel_semantic_ce * valid_semantic_weight).sum() /
                        semantic_weight_sum;
                    semantic_loss_sum += semantic_loss.item<double>();
                    ++semantic_loss_count;
                }
            }

            if (semantic_gradient.defined())
            {
                // Pass 2 recreates one three-channel render at a time and
                // immediately backpropagates its exact softmax-CE gradient.
                // Each rasterizer graph is released before the next chunk.
                for (int offset = 0; offset < pc->semantic_class_count_; offset += 3)
                {
                    auto semantic_pkg = renderSemanticChunk(
                        viewpoint_cam, pc, semantic_bg, offset);
                    const int channels =
                        std::min(3, pc->semantic_class_count_ - offset);
                    auto normalized_chunk =
                        (std::get<0>(semantic_pkg) /
                         semantic_alpha.clamp_min(0.05).unsqueeze(0))
                            .slice(0, 0, channels);
                    auto gradient_chunk = semantic_gradient.slice(
                        0, offset, offset + channels);
                    normalized_chunk.backward(
                        pc->lambda_semantic_ * gradient_chunk);
                }

                // Head-only ablation: retain gradients on semantic logits but
                // remove the semantic contribution to shared position, SH,
                // opacity, scale, and rotation parameters. The ordinary
                // RGB/depth loss below then repopulates geometry gradients.
                // The semantic optimizer is independent, so zeroing the main
                // sparse optimizer does not discard semantic-feature updates.
                if (!pc->semantic_geometry_gradients_)
                {
                    pc->sparse_optimizer_->zero_grad(true);
                }

                // Keep the existing support accounting unchanged.
                auto xyz = pc->getXYZ().detach();
                auto points_h = torch::cat(
                    {xyz, torch::ones({xyz.size(0), 1}, xyz.options())}, 1);
                auto points_camera_h = torch::matmul(
                    points_h, viewpoint_cam->world_view_transform_);
                auto points_camera = points_camera_h.slice(1, 0, 3);
                auto z = points_camera.index({torch::indexing::Slice(), 2});
                auto safe_z = torch::where(
                    z.abs() > 1e-6, z, torch::ones_like(z));
                auto pixel_x =
                    (points_camera.index({torch::indexing::Slice(), 0}) *
                     static_cast<float>(viewpoint_cam->fx_) / safe_z +
                     static_cast<float>(viewpoint_cam->cx_))
                        .round().to(torch::kLong);
                auto pixel_y =
                    (points_camera.index({torch::indexing::Slice(), 1}) *
                     static_cast<float>(viewpoint_cam->fy_) / safe_z +
                     static_cast<float>(viewpoint_cam->cy_))
                        .round().to(torch::kLong);
                auto inside = (z > 0.0) & (pixel_x >= 0) &
                              (pixel_x < viewpoint_cam->image_width_) &
                              (pixel_y >= 0) &
                              (pixel_y < viewpoint_cam->image_height_);
                auto safe_x =
                    pixel_x.clamp(0, viewpoint_cam->image_width_ - 1);
                auto safe_y =
                    pixel_y.clamp(0, viewpoint_cam->image_height_ - 1);
                auto sampled_weight = semantic_weight.index({safe_y, safe_x});
                auto sampled_label = semantic_target.index({safe_y, safe_x});
                auto sampled_static = gt_static_mask.index({safe_y, safe_x});
                auto supported_observation = inside & visible &
                    (sampled_label > 0) &
                    (sampled_label < pc->semantic_class_count_) &
                    (sampled_static > 0.5);
                semantic_support_increment = sampled_weight *
                    supported_observation.to(sampled_weight.scalar_type());
            }
        }
        else if (pc->semantic_training_ && !pc->semantic_streaming_ce_ &&
            (pc->semantic_keyframe_window_ <= 0 ||
             idx >= std::max(0, train_camera_num - pc->semantic_keyframe_window_)) &&
            viewpoint_cam->original_semantic_.defined() &&
            viewpoint_cam->semantic_weight_.defined())
        {
            auto semantic_target = viewpoint_cam->original_semantic_.to(
                torch::kCUDA, /*non_blocking=*/true).to(torch::kLong);
            auto semantic_weight = viewpoint_cam->semantic_weight_.to(
                torch::kCUDA, /*non_blocking=*/true);
            auto semantic_bg = torch::zeros({3}, bg.options());
            std::vector<torch::Tensor> semantic_chunks;
            torch::Tensor semantic_alpha;
            for (int offset = 0; offset < pc->semantic_class_count_; offset += 3)
            {
                auto semantic_pkg = renderSemanticChunk(
                    viewpoint_cam, pc, semantic_bg, offset);
                if (!semantic_alpha.defined())
                {
                    semantic_alpha = (1.0 - std::get<2>(semantic_pkg)).clamp(0.0, 1.0);
                }
                semantic_chunks.push_back(
                    std::get<0>(semantic_pkg) /
                    semantic_alpha.clamp_min(0.05).unsqueeze(0));
            }
            auto rendered_semantic_logits = torch::cat(semantic_chunks, 0).slice(
                0, 0, pc->semantic_class_count_);
            auto valid_target = (semantic_target > 0) &
                                (semantic_target < pc->semantic_class_count_);
            auto valid_semantic_weight = semantic_weight * gt_static_mask *
                                         valid_target.to(semantic_weight.scalar_type()) *
                                         (semantic_alpha > 0.05).to(semantic_weight.scalar_type());
            if (pc->semantic_normal_weighting_)
            {
                const auto normal_weight = computeSemanticNormalWeight(
                    rendered_depth,
                    viewpoint_cam,
                    pc->semantic_normal_power_,
                    pc->semantic_normal_min_weight_,
                    pc->semantic_normal_depth_tolerance_m_,
                    pc->semantic_normal_depth_tolerance_ratio_);
                auto normal_apply_mask = valid_target;
                if (!pc->semantic_normal_weight_classes_.empty())
                {
                    normal_apply_mask = torch::zeros_like(valid_target);
                    for (const int label : pc->semantic_normal_weight_classes_)
                    {
                        normal_apply_mask =
                            normal_apply_mask | (semantic_target == label);
                    }
                }
                const auto effective_normal_weight = torch::where(
                    normal_apply_mask,
                    normal_weight.weight,
                    torch::ones_like(normal_weight.weight));
                valid_semantic_weight =
                    valid_semantic_weight * effective_normal_weight;
                if (!semantic_normal_stats_logged &&
                    (pc->optimize_call_count_ == 1 ||
                     pc->optimize_call_count_ % 20 == 0))
                {
                    const auto trusted_pixels =
                        valid_target &
                        (semantic_weight > 0.0) &
                        (gt_static_mask > 0.5) &
                        (semantic_alpha > 0.05);
                    const auto trusted_count = trusted_pixels.sum().item<int64_t>();
                    const auto applied_pixels =
                        trusted_pixels & normal_apply_mask;
                    const auto applied_count =
                        applied_pixels.sum().item<int64_t>();
                    double mean_weight = 0.0;
                    double reliable_fraction = 0.0;
                    if (applied_count > 0)
                    {
                        mean_weight = normal_weight.weight.index({applied_pixels})
                                          .mean().item<double>();
                        reliable_fraction =
                            normal_weight.reliable.index({applied_pixels})
                                .to(torch::kFloat32).mean().item<double>();
                    }
                    std::cout << "[SEMANTIC-NORMAL] optimize_call="
                              << pc->optimize_call_count_
                              << " mean_weight=" << mean_weight
                              << " reliable_fraction=" << reliable_fraction
                              << " applied_pixels=" << applied_count
                              << " trusted_pixels=" << trusted_count << std::endl;
                    semantic_normal_stats_logged = true;
                }
            }
            auto loss_semantic_weight = valid_semantic_weight;
            if (pc->semantic_balance_power_ > 0.0)
            {
                auto safe_target_for_histogram = semantic_target.clamp(
                    0, pc->semantic_class_count_ - 1);
                auto class_histogram = torch::zeros(
                    {pc->semantic_class_count_}, valid_semantic_weight.options());
                class_histogram.scatter_add_(
                    0, safe_target_for_histogram.flatten(),
                    valid_semantic_weight.flatten());
                auto present = class_histogram > 0.0;
                auto present_count = present.sum().to(
                    class_histogram.scalar_type()).clamp_min(1.0);
                auto mean_present_count = class_histogram.sum() / present_count;
                auto class_balance = torch::pow(
                    mean_present_count / class_histogram.clamp_min(1e-6),
                    pc->semantic_balance_power_).clamp_max(
                        pc->semantic_balance_max_);
                class_balance = torch::where(
                    present, class_balance, torch::zeros_like(class_balance));
                loss_semantic_weight = valid_semantic_weight *
                    class_balance.index({safe_target_for_histogram});
            }
            auto semantic_weight_sum = loss_semantic_weight.sum();
            if (semantic_weight_sum.item<float>() > 0.0f)
            {
                auto safe_target = semantic_target.clamp(0, pc->semantic_class_count_ - 1);
                auto per_pixel_semantic_ce = -torch::log_softmax(
                    rendered_semantic_logits, 0).gather(
                        0, safe_target.unsqueeze(0)).squeeze(0);
                auto semantic_loss =
                    (per_pixel_semantic_ce * loss_semantic_weight).sum() /
                    semantic_weight_sum;
                loss += pc->lambda_semantic_ * semantic_loss;
                semantic_loss_sum += semantic_loss.detach().item<double>();
                ++semantic_loss_count;

                // Boundary-aware local coherence: only neighbouring pixels with
                // the same trusted teacher label are encouraged to agree.  This
                // avoids smoothing across semantic boundaries and does not
                // require unreliable pseudo instance IDs.
                if (pc->lambda_semantic_region_ > 0.0)
                {
                    const int stride = pc->semantic_region_stride_;
                    auto semantic_prob = torch::softmax(rendered_semantic_logits, 0);
                    auto horizontal_pair =
                        (semantic_target.index({torch::indexing::Slice(),
                                                torch::indexing::Slice(stride, torch::indexing::None)}) ==
                         semantic_target.index({torch::indexing::Slice(),
                                                torch::indexing::Slice(torch::indexing::None, -stride)}));
                    auto horizontal_weight = torch::minimum(
                        valid_semantic_weight.index({torch::indexing::Slice(),
                                                     torch::indexing::Slice(stride, torch::indexing::None)}),
                        valid_semantic_weight.index({torch::indexing::Slice(),
                                                     torch::indexing::Slice(torch::indexing::None, -stride)})) *
                        horizontal_pair.to(valid_semantic_weight.scalar_type());
                    auto horizontal_difference = torch::abs(
                        semantic_prob.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                             torch::indexing::Slice(stride, torch::indexing::None)}) -
                        semantic_prob.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                             torch::indexing::Slice(torch::indexing::None, -stride)})).mean(0);

                    auto vertical_pair =
                        (semantic_target.index({torch::indexing::Slice(stride, torch::indexing::None),
                                                torch::indexing::Slice()}) ==
                         semantic_target.index({torch::indexing::Slice(torch::indexing::None, -stride),
                                                torch::indexing::Slice()}));
                    auto vertical_weight = torch::minimum(
                        valid_semantic_weight.index({torch::indexing::Slice(stride, torch::indexing::None),
                                                     torch::indexing::Slice()}),
                        valid_semantic_weight.index({torch::indexing::Slice(torch::indexing::None, -stride),
                                                     torch::indexing::Slice()})) *
                        vertical_pair.to(valid_semantic_weight.scalar_type());
                    auto vertical_difference = torch::abs(
                        semantic_prob.index({torch::indexing::Slice(),
                                             torch::indexing::Slice(stride, torch::indexing::None),
                                             torch::indexing::Slice()}) -
                        semantic_prob.index({torch::indexing::Slice(),
                                             torch::indexing::Slice(torch::indexing::None, -stride),
                                             torch::indexing::Slice()})).mean(0);

                    auto pair_weight_sum = horizontal_weight.sum() + vertical_weight.sum();
                    if (pair_weight_sum.item<float>() > 0.0f)
                    {
                        auto semantic_region_loss =
                            ((horizontal_difference * horizontal_weight).sum() +
                             (vertical_difference * vertical_weight).sum()) /
                            pair_weight_sum;
                        loss += pc->lambda_semantic_region_ * semantic_region_loss;
                        semantic_region_loss_sum +=
                            semantic_region_loss.detach().item<double>();
                        ++semantic_region_loss_count;
                    }
                }

                // Record whether a Gaussian center is repeatedly observed on
                // a trusted labelled pixel. This lets unsupported Gaussians be
                // exported as unknown instead of forcing a confident class.
                auto xyz = pc->getXYZ().detach();
                auto points_h = torch::cat(
                    {xyz, torch::ones({xyz.size(0), 1}, xyz.options())}, 1);
                auto points_camera_h = torch::matmul(
                    points_h, viewpoint_cam->world_view_transform_);
                auto points_camera = points_camera_h.slice(1, 0, 3);
                auto z = points_camera.index({torch::indexing::Slice(), 2});
                auto safe_z = torch::where(
                    z.abs() > 1e-6, z, torch::ones_like(z));
                auto pixel_x = (points_camera.index({torch::indexing::Slice(), 0}) *
                                static_cast<float>(viewpoint_cam->fx_) / safe_z +
                                static_cast<float>(viewpoint_cam->cx_)).round().to(torch::kLong);
                auto pixel_y = (points_camera.index({torch::indexing::Slice(), 1}) *
                                static_cast<float>(viewpoint_cam->fy_) / safe_z +
                                static_cast<float>(viewpoint_cam->cy_)).round().to(torch::kLong);
                auto inside = (z > 0.0) & (pixel_x >= 0) &
                              (pixel_x < viewpoint_cam->image_width_) &
                              (pixel_y >= 0) &
                              (pixel_y < viewpoint_cam->image_height_);
                auto safe_x = pixel_x.clamp(0, viewpoint_cam->image_width_ - 1);
                auto safe_y = pixel_y.clamp(0, viewpoint_cam->image_height_ - 1);
                auto sampled_weight = semantic_weight.index({safe_y, safe_x});
                auto sampled_label = semantic_target.index({safe_y, safe_x});
                auto sampled_static = gt_static_mask.index({safe_y, safe_x});
                auto supported_observation = inside & visible &
                    (sampled_label > 0) &
                    (sampled_label < pc->semantic_class_count_) &
                    (sampled_static > 0.5);
                semantic_support_increment = sampled_weight *
                    supported_observation.to(sampled_weight.scalar_type());
            }
        }
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_forward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
        
        pc->t_start_ = std::chrono::steady_clock::now();
        if (pca_head_loss.defined())
        {
            if (!pc->semantic_optimizer_)
            {
                throw std::runtime_error(
                    "PCA head-only loss requires the semantic optimizer.");
            }
            pca_head_loss.backward();
            // Preserve language-feature gradients in semantic_optimizer_, but
            // discard every PCA contribution to shared geometry/appearance.
            // RGB/depth backward below repopulates the sparse optimizer.
            pc->sparse_optimizer_->zero_grad(true);
        }
        loss.backward();
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_backward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

        pc->t_start_ = std::chrono::steady_clock::now();
        updated_num += visible.sum().item<int>();
        if (semantic_support_increment.defined())
        {
            torch::NoGradGuard no_grad;
            pc->semantic_support_.add_(semantic_support_increment.unsqueeze(1));
        }
        pc->sparse_optimizer_->set_visibility_and_N(visible, pc->getXYZ().size(0));
        if (pc->semantic_optimizer_)
        {
            pc->semantic_optimizer_->set_visibility_and_N(visible, pc->getXYZ().size(0));
        }
        pc->sparse_optimizer_->step();
        if (pc->semantic_optimizer_)
        {
            pc->semantic_optimizer_->step();
        }
        if (pc->pca_language_training_ && pc->pca_max_coefficient_norm_ > 0.0)
        {
            torch::NoGradGuard no_grad;
            auto coefficient_norm = pc->semantic_feature_.norm(2, 1, true);
            auto coefficient_scale =
                (pc->pca_max_coefficient_norm_ /
                 coefficient_norm.clamp_min(1e-12)).clamp_max(1.0);
            pc->semantic_feature_.mul_(coefficient_scale);
        }
        if (pc->max_gaussian_scale_ > 0.0)
        {
            torch::NoGradGuard no_grad;
            const double max_log_scale = std::log(pc->max_gaussian_scale_);
            pc->scaling_.clamp_max_(max_log_scale);
        }
        pc->sparse_optimizer_->zero_grad(true);
        if (pc->semantic_optimizer_)
        {
            pc->semantic_optimizer_->zero_grad(true);
        }
        if (pc->apply_exposure_)
        {
            pc->exposure_optimizer_->step();
            pc->exposure_optimizer_->zero_grad(true);
        }
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_step_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    }

    pc->last_semantic_loss_ = semantic_loss_count > 0
                                  ? semantic_loss_sum / semantic_loss_count
                                  : -1.0;
    pc->last_semantic_region_loss_ = semantic_region_loss_count > 0
                                         ? semantic_region_loss_sum /
                                               semantic_region_loss_count
                                         : -1.0;

    // Long incremental sequences exercise progressively larger rasterizer
    // workspaces.  The CUDA caching allocator otherwise keeps every historic
    // high-water block reserved, which can exhaust a 24 GB device even though
    // the live tensors fit.  Releasing unused cached blocks after a streaming
    // semantic optimization call does not change tensors or gradients.
    if (pc->semantic_streaming_ce_)
    {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    return updated_num / opt_list.size();
}

namespace
{
struct PcaLanguageQuerySet
{
    std::vector<std::string> labels;
    torch::Tensor positive_basis;
    torch::Tensor positive_mean;
    torch::Tensor negative_basis;
    torch::Tensor negative_mean;

    bool available() const
    {
        return positive_basis.defined() && positive_basis.numel() > 0 &&
               negative_basis.defined() && negative_basis.numel() > 0;
    }
};

std::string safeFileLabel(std::string label)
{
    for (char& character : label)
    {
        const unsigned char value = static_cast<unsigned char>(character);
        if (!std::isalnum(value) && character != '-' && character != '_')
        {
            character = '_';
        }
    }
    return label.empty() ? "query" : label;
}

PcaLanguageQuerySet loadPcaLanguageQueries(const std::string& path, int dimension)
{
    PcaLanguageQuerySet result;
    if (path.empty()) return result;

    const YAML::Node root = YAML::LoadFile(path);
    const YAML::Node positives = root["queries"];
    const YAML::Node negatives = root["negatives"];
    if (!positives || !positives.IsSequence() || positives.size() == 0 ||
        !negatives || !negatives.IsSequence() || negatives.size() == 0)
    {
        throw std::runtime_error(
            "PCA query JSON requires non-empty queries and negatives arrays.");
    }

    std::vector<float> positive_basis;
    std::vector<float> positive_mean;
    std::vector<float> negative_basis;
    std::vector<float> negative_mean;
    const auto append_record =
        [&](const YAML::Node& record,
            std::vector<float>& basis_values,
            std::vector<float>& mean_values)
    {
        const YAML::Node basis = record["basis_dot"];
        if (!basis || !basis.IsSequence() ||
            static_cast<int>(basis.size()) != dimension)
        {
            throw std::runtime_error(
                "PCA query basis dimension does not match semantic_class_count.");
        }
        for (const auto& value : basis)
        {
            basis_values.push_back(value.as<float>());
        }
        mean_values.push_back(record["mean_dot"].as<float>());
    };

    for (const auto& record : positives)
    {
        result.labels.push_back(record["label"].as<std::string>());
        append_record(record, positive_basis, positive_mean);
    }
    for (const auto& record : negatives)
    {
        append_record(record, negative_basis, negative_mean);
    }

    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    result.positive_basis = torch::from_blob(
        positive_basis.data(),
        {static_cast<int64_t>(result.labels.size()), dimension}, options).clone().cuda();
    result.positive_mean = torch::from_blob(
        positive_mean.data(),
        {static_cast<int64_t>(result.labels.size())}, options).clone().cuda();
    result.negative_basis = torch::from_blob(
        negative_basis.data(),
        {static_cast<int64_t>(negatives.size()), dimension}, options).clone().cuda();
    result.negative_mean = torch::from_blob(
        negative_mean.data(),
        {static_cast<int64_t>(negatives.size())}, options).clone().cuda();
    return result;
}

int extractFrameIndex(const std::string& image_name)
{
    const auto dot = image_name.find_last_of('.');
    const size_t end = dot == std::string::npos ? image_name.size() : dot;
    if (end == 0)
    {
        return -1;
    }

    size_t begin = end;
    while (begin > 0 && std::isdigit(static_cast<unsigned char>(image_name[begin - 1])))
    {
        --begin;
    }
    if (begin == end)
    {
        return -1;
    }
    try
    {
        return std::stoi(image_name.substr(begin, end - begin));
    }
    catch (...)
    {
        return -1;
    }
}

std::vector<std::shared_ptr<Camera>> collectOrderedCameras(const std::shared_ptr<Dataset>& dataset)
{
    std::vector<std::shared_ptr<Camera>> ordered;
    ordered.reserve(dataset->train_cameras_.size() + dataset->test_cameras_.size());
    ordered.insert(ordered.end(), dataset->train_cameras_.begin(), dataset->train_cameras_.end());
    ordered.insert(ordered.end(), dataset->test_cameras_.begin(), dataset->test_cameras_.end());
    std::sort(ordered.begin(), ordered.end(),
              [](const std::shared_ptr<Camera>& lhs, const std::shared_ptr<Camera>& rhs)
              {
                  return extractFrameIndex(lhs->image_name_) < extractFrameIndex(rhs->image_name_);
              });
    return ordered;
}

std::shared_ptr<Camera> interpolateCamera(const std::shared_ptr<Camera>& lhs,
                                          const std::shared_ptr<Camera>& rhs,
                                          double alpha,
                                          int frame_idx)
{
    auto camera = std::make_shared<Camera>();
    camera->setIntrinsic(lhs->image_width_, lhs->image_height_, lhs->fx_, lhs->fy_, lhs->cx_, lhs->cy_);

    const Eigen::Matrix3d R0_wc = lhs->R_cw_.transpose();
    const Eigen::Matrix3d R1_wc = rhs->R_cw_.transpose();
    const Eigen::Vector3d t0_wc = -R0_wc * lhs->t_cw_;
    const Eigen::Vector3d t1_wc = -R1_wc * rhs->t_cw_;

    Eigen::Quaterniond q0(R0_wc);
    Eigen::Quaterniond q1(R1_wc);
    if (q0.dot(q1) < 0.0)
    {
        q1.coeffs() *= -1.0;
    }

    const Eigen::Quaterniond q_wc = q0.slerp(alpha, q1);
    const Eigen::Vector3d t_wc = (1.0 - alpha) * t0_wc + alpha * t1_wc;
    camera->setPose(q_wc.toRotationMatrix(), t_wc);

    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << frame_idx;
    camera->image_name_ = "traj_" + ss.str() + ".jpg";
    return camera;
}

void exportTrajectoryRender(const std::shared_ptr<Dataset>& dataset,
                            std::shared_ptr<GaussianModel>& pc,
                            torch::Tensor bg,
                            const std::string& result_path)
{
    if (dataset->trajectory_render_frames_ <= 0)
    {
        return;
    }

    const auto ordered = collectOrderedCameras(dataset);
    if (ordered.empty())
    {
        std::cout << "        [Trajectory Render] skipped: no camera poses available." << std::endl;
        return;
    }

    const int output_frames = std::max(dataset->trajectory_render_frames_, static_cast<int>(ordered.size()));
    const fs::path frame_dir = fs::path(result_path) / "trajectory_render";
    fs::create_directories(frame_dir);

    const int width = ordered.front()->image_width_;
    const int height = ordered.front()->image_height_;
    const double fps = 10.0;

    fs::path video_path = fs::path(result_path) / "trajectory_render.mp4";
    cv::VideoWriter writer(video_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    if (!writer.isOpened())
    {
        video_path = fs::path(result_path) / "trajectory_render.avi";
        writer.open(video_path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
    }

    for (int i = 0; i < output_frames; ++i)
    {
        std::shared_ptr<Camera> render_camera;
        if (output_frames == 1 || ordered.size() == 1)
        {
            render_camera = ordered.front();
        }
        else
        {
            const double position = static_cast<double>(i) * static_cast<double>(ordered.size() - 1) /
                                    static_cast<double>(output_frames - 1);
            const int left = static_cast<int>(std::floor(position));
            const int right = std::min(left + 1, static_cast<int>(ordered.size()) - 1);
            const double alpha = position - static_cast<double>(left);
            if (left == right || alpha < 1e-6)
            {
                render_camera = ordered[left];
            }
            else
            {
                render_camera = interpolateCamera(ordered[left], ordered[right], alpha, i);
            }
        }

        auto render_pkg = render(render_camera, pc, bg, pc->apply_exposure_);
        auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
        torch::Tensor rgb_cpu = rendered_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
        rgb_cpu = rgb_cpu.mul(255).clamp(0, 255).to(torch::kU8);
        cv::Mat bgr_image(height, width, CV_8UC3, rgb_cpu.data_ptr<uint8_t>());
        cv::cvtColor(bgr_image, bgr_image, cv::COLOR_RGB2BGR);
        cv::Mat output = bgr_image.clone();

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << i;
        cv::imwrite((frame_dir / ("traj_" + ss.str() + ".jpg")).string(), output);
        if (writer.isOpened())
        {
            writer.write(output);
        }
    }

    if (writer.isOpened())
    {
        writer.release();
        std::cout << "        [Trajectory Render Video] " << video_path.string() << std::endl;
    }
    std::cout << "        [Trajectory Render Frames] " << output_frames << std::endl;
}
}  // namespace

void evaluateVisualQuality(const std::shared_ptr<Dataset>& dataset, 
                           std::shared_ptr<GaussianModel>& pc,
                           const std::string& result_path,
                           const std::string& lpips_path,
                           const std::string& pca_query_path)
{
    std::cout << "\n     🎉 Evaluate Visual Quality 🎉\n";
    std::cout << "\n        [Number of Final Gaussians] " << pc->getXYZ().size(0) << std::endl;

    if (fs::exists(result_path))
    {
        throw std::runtime_error(
            "Refusing to overwrite existing result path: " + result_path);
    }
    fs::create_directories(result_path);

    std::string render_dir_path = result_path + "/render";
    fs::create_directories(render_dir_path);
    std::string render_depth_dir_path = result_path + "/render_depth";
    fs::create_directories(render_depth_dir_path);
    std::string gt_dir_path = result_path + "/gt";
    fs::create_directories(gt_dir_path);
    std::string render_semantic_dir_path = result_path + "/render_semantic";
    std::string render_semantic_label_dir_path = result_path + "/render_semantic_label";
    std::string render_semantic_confidence_dir_path =
        result_path + "/render_semantic_confidence";
    std::string render_semantic_overlay_dir_path = result_path + "/render_semantic_overlay";
    if (pc->semantic_training_)
    {
        fs::create_directories(render_semantic_dir_path);
        fs::create_directories(render_semantic_label_dir_path);
        fs::create_directories(render_semantic_confidence_dir_path);
        fs::create_directories(render_semantic_overlay_dir_path);
    }
    const auto pca_queries = loadPcaLanguageQueries(
        pca_query_path, pc->semantic_class_count_);
    const fs::path language_query_dir =
        fs::path(result_path) / "render_language_query";
    const fs::path language_query_overlay_dir =
        fs::path(result_path) / "render_language_query_overlay";
    const fs::path language_query_absolute_dir =
        fs::path(result_path) / "render_language_query_absolute";
    const fs::path language_query_score16_raw_dir =
        fs::path(result_path) / "render_language_query_score16_raw";
    const fs::path language_query_score16_smooth_dir =
        fs::path(result_path) / "render_language_query_score16_smooth";
    const fs::path language_alpha_dir =
        fs::path(result_path) / "render_language_alpha";
    std::ofstream language_query_stats;
    if (pca_queries.available())
    {
        if (!pc->pca_language_training_)
        {
            throw std::runtime_error(
                "pca_query_path requires pca_language_training=true.");
        }
        for (const auto& label : pca_queries.labels)
        {
            fs::create_directories(language_query_dir / safeFileLabel(label));
            fs::create_directories(language_query_overlay_dir / safeFileLabel(label));
            fs::create_directories(language_query_absolute_dir / safeFileLabel(label));
            fs::create_directories(
                language_query_score16_raw_dir / safeFileLabel(label));
            fs::create_directories(
                language_query_score16_smooth_dir / safeFileLabel(label));
        }
        fs::create_directories(language_alpha_dir);
        language_query_stats.open(
            (fs::path(result_path) / "language_query_stats.csv").string());
        if (!language_query_stats)
        {
            throw std::runtime_error("Unable to create language_query_stats.csv.");
        }
        language_query_stats
            << "camera,query,valid_pixels,absolute_min,absolute_mean,"
               "absolute_max,absolute_range\n";
        std::cout << "        [PCA Language Queries] "
                  << pca_queries.labels.size()
                  << " positives with " << pca_queries.negative_basis.size(0)
                  << " negatives" << std::endl;
    }

    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::jit::script::Module m_lpips;
    try 
    {
        m_lpips = torch::jit::load(lpips_path + "/lpips_alex.pt");
        m_lpips.to(torch::kCUDA);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "lpips model loading failed: " << e.what() << std::endl;
    }

    const auto save_semantic_render =
        [&](const std::shared_ptr<Camera>& camera, const cv::Mat& rgb_bgr)
    {
        if (!pc->semantic_training_ || !pc->getSemanticFeatures().defined() ||
            pc->getSemanticFeatures().numel() == 0)
        {
            return;
        }

        auto semantic_bg = torch::zeros(
            {3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        std::vector<torch::Tensor> semantic_chunks;
        torch::Tensor semantic_alpha;
        for (int offset = 0; offset < pc->semantic_class_count_; offset += 3)
        {
            auto semantic_pkg = renderSemanticChunk(camera, pc, semantic_bg, offset);
            if (!semantic_alpha.defined())
            {
                semantic_alpha = (1.0 - std::get<2>(semantic_pkg)).clamp(0.0, 1.0);
            }
            semantic_chunks.push_back(
                std::get<0>(semantic_pkg) /
                semantic_alpha.clamp_min(0.05).unsqueeze(0));
        }
        auto logits = torch::cat(semantic_chunks, 0).slice(
            0, 0, pc->semantic_class_count_);
        auto probabilities = torch::softmax(logits, 0);
        auto maximum = probabilities.max(0);
        auto confidence = std::get<0>(maximum);
        auto labels = std::get<1>(maximum).to(torch::kLong);
        auto valid = (semantic_alpha > 0.05) &
                     (confidence >= pc->semantic_min_confidence_);
        labels = torch::where(valid, labels, torch::zeros_like(labels));

        // Preserve the historical 12-class colors, then extend the palette
        // deterministically for datasets with more classes (for example MCD).
        // The previous fixed 12x3 tensor caused an out-of-range CUDA
        // index_select when semantic_class_count_ exceeded 12.
        std::vector<std::uint8_t> palette_values = {
            96, 96, 96, 230, 25, 75, 60, 180, 75, 255, 225, 25,
            0, 130, 200, 245, 130, 48, 145, 30, 180, 70, 240, 240,
            240, 50, 230, 210, 245, 60, 250, 190, 212, 0, 128, 128};
        for (int class_id = 12; class_id < pc->semantic_class_count_; ++class_id)
        {
            // Pascal-VOC-style bit palette: stable, distinct, and independent
            // of frame order or random seeds.
            std::uint8_t red = 0;
            std::uint8_t green = 0;
            std::uint8_t blue = 0;
            int value = class_id;
            for (int bit = 0; bit < 8; ++bit)
            {
                red |= static_cast<std::uint8_t>((value & 1) << (7 - bit));
                green |= static_cast<std::uint8_t>(((value >> 1) & 1) << (7 - bit));
                blue |= static_cast<std::uint8_t>(((value >> 2) & 1) << (7 - bit));
                value >>= 3;
            }
            palette_values.push_back(red);
            palette_values.push_back(green);
            palette_values.push_back(blue);
        }
        auto palette = torch::from_blob(
                           palette_values.data(),
                           {pc->semantic_class_count_, 3},
                           torch::TensorOptions().dtype(torch::kUInt8))
                           .clone()
                           .to(torch::kCUDA);
        auto semantic_rgb = palette.index_select(0, labels.reshape({-1})).reshape(
            {camera->image_height_, camera->image_width_, 3});
        semantic_rgb = torch::where(
            valid.unsqueeze(2), semantic_rgb, torch::zeros_like(semantic_rgb));

        auto semantic_cpu = semantic_rgb.to(torch::kCPU).contiguous();
        cv::Mat semantic_bgr(
            camera->image_height_, camera->image_width_, CV_8UC3,
            semantic_cpu.data_ptr<uint8_t>());
        cv::cvtColor(semantic_bgr, semantic_bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(
            render_semantic_dir_path + "/" + camera->image_name_, semantic_bgr);

        auto label_cpu = labels.to(torch::kCPU).to(torch::kUInt8).contiguous();
        cv::Mat label_image(
            camera->image_height_, camera->image_width_, CV_8UC1,
            label_cpu.data_ptr<uint8_t>());
        cv::imwrite(
            render_semantic_label_dir_path + "/" + camera->image_name_, label_image);

        auto confidence_u8 = (confidence * valid.to(confidence.scalar_type()))
            .mul(255.0).clamp(0.0, 255.0).to(torch::kUInt8)
            .to(torch::kCPU).contiguous();
        cv::Mat confidence_image(
            camera->image_height_, camera->image_width_, CV_8UC1,
            confidence_u8.data_ptr<uint8_t>());
        cv::Mat confidence_color;
        cv::applyColorMap(confidence_image, confidence_color, cv::COLORMAP_TURBO);
        auto valid_cpu = valid.to(torch::kCPU).to(torch::kUInt8).mul(255).contiguous();
        cv::Mat valid_mask(
            camera->image_height_, camera->image_width_, CV_8UC1,
            valid_cpu.data_ptr<uint8_t>());
        confidence_color.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);
        cv::imwrite(
            render_semantic_confidence_dir_path + "/" + camera->image_name_,
            confidence_color);

        cv::Mat overlay;
        cv::addWeighted(rgb_bgr, 0.45, semantic_bgr, 0.55, 0.0, overlay);
        rgb_bgr.copyTo(overlay, valid_mask == 0);
        cv::imwrite(
            render_semantic_overlay_dir_path + "/" + camera->image_name_, overlay);
    };

    // Returns confidence-weighted cosine sum, confidence sum, rendered-valid
    // pixels, and teacher-labelled static pixels. A missing target returns
    // zero counts and is excluded from the aggregate rather than treated as a
    // failure; this allows sparse, strictly held-out teacher evaluation.
    const auto evaluate_pca_language =
        [&](const std::shared_ptr<Camera>& camera, const cv::Mat& rgb_bgr)
            -> std::tuple<double, double, std::int64_t, std::int64_t>
    {
        if (!pc->pca_language_training_ ||
            !camera->language_region_ids_.defined() ||
            !camera->language_basis_dot_.defined() ||
            !camera->language_mean_dot_.defined() ||
            !camera->language_confidence_.defined())
        {
            return {0.0, 0.0, 0, 0};
        }

        torch::NoGradGuard no_grad;
        auto language_bg = torch::zeros(
            {3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        std::vector<torch::Tensor> language_chunks;
        torch::Tensor language_alpha;
        for (int offset = 0; offset < pc->semantic_class_count_; offset += 3)
        {
            auto language_pkg = renderSemanticChunk(camera, pc, language_bg, offset);
            if (!language_alpha.defined())
            {
                language_alpha =
                    (1.0 - std::get<2>(language_pkg)).clamp(0.0, 1.0);
            }
            language_chunks.push_back(
                std::get<0>(language_pkg) /
                language_alpha.clamp_min(0.05).unsqueeze(0));
        }
        auto rendered_language = torch::cat(language_chunks, 0).slice(
            0, 0, pc->semantic_class_count_);
        if (pca_queries.available())
        {
            const int height = camera->image_height_;
            const int width = camera->image_width_;
            auto flat = rendered_language.flatten(1).transpose(0, 1);
            auto denominator_squared = pc->pca_mean_norm_squared_ +
                2.0 * (flat * pc->pca_basis_mean_).sum(1) +
                flat.square().sum(1);
            auto denominator = torch::sqrt(
                denominator_squared.clamp_min(1e-12)).unsqueeze(1);
            auto positive_cosine =
                (torch::matmul(flat, pca_queries.positive_basis.transpose(0, 1)) +
                 pca_queries.positive_mean.unsqueeze(0)) / denominator;
            auto negative_cosine =
                (torch::matmul(flat, pca_queries.negative_basis.transpose(0, 1)) +
                 pca_queries.negative_mean.unsqueeze(0)) / denominator;
            auto delta = positive_cosine.unsqueeze(2) - negative_cosine.unsqueeze(1);
            auto relevance = std::get<0>(torch::sigmoid(10.0 * delta).min(2));
            auto alpha_valid = language_alpha > 0.05;
            const std::string score_name =
                fs::path(camera->image_name_).stem().string() + ".png";
            auto alpha_cpu = alpha_valid.to(torch::kCPU).to(torch::kUInt8)
                .mul(255).contiguous();
            cv::Mat alpha_mask(
                height, width, CV_8UC1, alpha_cpu.data_ptr<uint8_t>());
            if (!cv::imwrite(
                    (language_alpha_dir / score_name).string(), alpha_mask))
            {
                throw std::runtime_error(
                    "Failed to write PCA language alpha mask: " + score_name);
            }

            const auto save_score16 =
                [&](const torch::Tensor& response, const fs::path& output_path)
            {
                auto score_i32 = (
                    response.clamp(0.0, 1.0) *
                    alpha_valid.to(response.scalar_type()))
                    .mul(65535.0)
                    .round()
                    .clamp(0.0, 65535.0)
                    .to(torch::kInt32)
                    .to(torch::kCPU)
                    .contiguous();
                cv::Mat score_i32_image(
                    height, width, CV_32SC1, score_i32.data_ptr<int32_t>());
                cv::Mat score_u16;
                score_i32_image.convertTo(score_u16, CV_16UC1);
                if (!cv::imwrite(output_path.string(), score_u16))
                {
                    throw std::runtime_error(
                        "Failed to write uint16 PCA language score: " +
                        output_path.string());
                }
            };

            for (int64_t query_index = 0;
                 query_index < relevance.size(1); ++query_index)
            {
                auto raw_response =
                    relevance.select(1, query_index).reshape({height, width});
                auto pooled_response = torch::avg_pool2d(
                    raw_response.unsqueeze(0).unsqueeze(0),
                    {29, 29}, {1, 1}, {14, 14}).squeeze();
                auto smooth_response =
                    0.5 * (raw_response + pooled_response);
                const std::string safe_label = safeFileLabel(
                    pca_queries.labels.at(static_cast<std::size_t>(query_index)));
                save_score16(
                    raw_response,
                    language_query_score16_raw_dir / safe_label / score_name);
                save_score16(
                    smooth_response,
                    language_query_score16_smooth_dir / safe_label / score_name);

                // Preserve the historical display and CSV behavior while the
                // new score16 files retain absolute, cross-frame values.
                auto response = smooth_response;
                auto valid_values = response.masked_select(alpha_valid);
                if (valid_values.numel() == 0) continue;
                const auto response_min = valid_values.min();
                const auto response_max = valid_values.max();
                const auto response_mean = valid_values.mean();
                language_query_stats
                    << camera->image_name_ << ',' << safe_label << ','
                    << valid_values.numel() << ','
                    << response_min.item<float>() << ','
                    << response_mean.item<float>() << ','
                    << response_max.item<float>() << ','
                    << (response_max - response_min).item<float>() << '\n';

                auto absolute_u8 = (
                    response * alpha_valid.to(response.scalar_type()))
                    .mul(255.0).clamp(0.0, 255.0).to(torch::kUInt8)
                    .to(torch::kCPU).contiguous();
                cv::Mat absolute_gray(
                    height, width, CV_8UC1, absolute_u8.data_ptr<uint8_t>());
                cv::Mat absolute_color;
                cv::applyColorMap(
                    absolute_gray, absolute_color, cv::COLORMAP_TURBO);
                absolute_color.setTo(cv::Scalar(0, 0, 0), alpha_mask == 0);
                cv::imwrite(
                    (language_query_absolute_dir / safe_label /
                     camera->image_name_).string(), absolute_color);

                auto normalized = (
                    2.0 * (response - response_min) /
                    (response_max - response_min + 1e-9) - 1.0).clamp(0.0, 1.0);
                normalized = normalized * alpha_valid.to(normalized.scalar_type());
                auto heat_u8 = normalized.mul(255.0).to(torch::kUInt8)
                    .to(torch::kCPU).contiguous();
                cv::Mat heat_gray(height, width, CV_8UC1, heat_u8.data_ptr<uint8_t>());
                cv::Mat heat_color;
                cv::applyColorMap(heat_gray, heat_color, cv::COLORMAP_TURBO);
                heat_color.setTo(cv::Scalar(0, 0, 0), alpha_mask == 0);
                cv::imwrite(
                    (language_query_dir / safe_label / camera->image_name_).string(),
                    heat_color);
                cv::Mat overlay;
                cv::addWeighted(rgb_bgr, 0.45, heat_color, 0.55, 0.0, overlay);
                rgb_bgr.copyTo(overlay, alpha_mask == 0);
                cv::imwrite(
                    (language_query_overlay_dir / safe_label /
                     camera->image_name_).string(), overlay);
            }
        }
        auto region_ids = camera->language_region_ids_.to(
            torch::kCUDA, /*non_blocking=*/true).to(torch::kLong);
        auto static_mask = camera->original_mask_.defined()
            ? camera->original_mask_.to(torch::kCUDA, /*non_blocking=*/true)
            : torch::ones_like(
                  region_ids, torch::TensorOptions().dtype(torch::kFloat32));
        auto labelled_static = (region_ids > 0) & (static_mask > 0.5);
        auto valid = labelled_static & (language_alpha > 0.05);
        const std::int64_t target_pixels =
            labelled_static.sum().item<std::int64_t>();
        const std::int64_t valid_pixels = valid.sum().item<std::int64_t>();
        if (valid_pixels == 0)
        {
            return {0.0, 0.0, 0, target_pixels};
        }

        auto valid_flat = torch::nonzero(valid.flatten()).squeeze(1);
        auto region_rows = region_ids.flatten().index_select(0, valid_flat) - 1;
        auto rendered_samples = rendered_language.flatten(1).transpose(0, 1)
            .index_select(0, valid_flat);
        auto basis_dot = camera->language_basis_dot_.to(
            torch::kCUDA, /*non_blocking=*/true).index_select(0, region_rows);
        auto mean_dot = camera->language_mean_dot_.to(
            torch::kCUDA, /*non_blocking=*/true).index_select(0, region_rows);
        auto confidence = camera->language_confidence_.to(
            torch::kCUDA, /*non_blocking=*/true).index_select(0, region_rows);
        auto numerator = mean_dot + (rendered_samples * basis_dot).sum(1);
        auto denominator_squared = pc->pca_mean_norm_squared_ +
            2.0 * (rendered_samples * pc->pca_basis_mean_).sum(1) +
            rendered_samples.square().sum(1);
        auto cosine = numerator /
            torch::sqrt(denominator_squared.clamp_min(1e-12));
        return {
            (cosine.clamp(-1.0, 1.0) * confidence).sum().item<double>(),
            confidence.sum().item<double>(), valid_pixels, target_pixels};
    };

    {
        double psnrs = 0;
        double ssims = 0;
        double lpipss = 0;
        double depth_l1_sum = 0;
        std::int64_t depth_valid_count = 0;
        double language_cosine_sum = 0.0;
        double language_confidence_sum = 0.0;
        std::int64_t language_valid_pixels = 0;
        std::int64_t language_target_pixels = 0;
        std::int64_t language_target_views = 0;
        for (const auto& train_camera : dataset->train_cameras_)
        {
            auto render_pkg = render(train_camera, pc, bg, pc->apply_exposure_);
            auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
            auto rendered_depth = std::get<1>(render_pkg);
            auto gt_image = train_camera->original_image_.cuda().clamp(0, 1);
            auto gt_depth = train_camera->original_depth_.cuda();
            double psnr = loss_utils::psnr(rendered_image, gt_image).mean().item<double>();
            double ssim = loss_utils::ssim(rendered_image, gt_image).item<double>();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(rendered_image.unsqueeze(0));
            inputs.push_back(gt_image.unsqueeze(0));
            double lpips = m_lpips.forward(inputs).toTensor().item<double>();
            psnrs += psnr;
            ssims += ssim;
            lpipss += lpips;

            auto depth_mask = (gt_depth > 0) & (rendered_depth > 0);
            const std::int64_t valid_depth = depth_mask.sum().item<std::int64_t>();
            if (valid_depth > 0)
            {
                depth_l1_sum += torch::abs(
                    rendered_depth.masked_select(depth_mask) - gt_depth.masked_select(depth_mask)
                ).sum().item<double>();
                depth_valid_count += valid_depth;
            }

            int H = rendered_image.size(1), W = rendered_image.size(2);

            torch::Tensor a_cpu = rendered_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            a_cpu = a_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat a_img(H, W, CV_8UC3, a_cpu.data_ptr<uint8_t>());
            cv::cvtColor(a_img, a_img, cv::COLOR_RGB2BGR);
            cv::imwrite(render_dir_path + "/" + train_camera->image_name_, a_img);
            save_semantic_render(train_camera, a_img);
            auto language_metric = evaluate_pca_language(train_camera, a_img);
            if (std::get<3>(language_metric) > 0)
            {
                language_cosine_sum += std::get<0>(language_metric);
                language_confidence_sum += std::get<1>(language_metric);
                language_valid_pixels += std::get<2>(language_metric);
                language_target_pixels += std::get<3>(language_metric);
                ++language_target_views;
            }

            torch::Tensor b_cpu = gt_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            b_cpu = b_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat b_img(H, W, CV_8UC3, b_cpu.data_ptr<uint8_t>());
            cv::cvtColor(b_img, b_img, cv::COLOR_RGB2BGR);
            cv::imwrite(gt_dir_path + "/" + train_camera->image_name_, b_img);

            torch::Tensor depth_map_normalized = (rendered_depth - rendered_depth.min()) / 
                                                     (rendered_depth.max() - rendered_depth.min()) * 255;
            torch::Tensor c_cpu = depth_map_normalized.to(torch::kCPU);
            cv::Mat c_img(H, W, CV_32FC1, c_cpu.data_ptr<float>());
            c_img.convertTo(c_img, CV_8UC1);
            cv::applyColorMap(c_img, c_img, cv::COLORMAP_JET);
            cv::imwrite(render_depth_dir_path + "/" + train_camera->image_name_, c_img);
        }
        psnrs /= dataset->train_cameras_.size();
        ssims /= dataset->train_cameras_.size();
        lpipss /= dataset->train_cameras_.size();
        std::cout << std::fixed << std::setprecision(2) << "        [Training View PSNR] " << psnrs << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [Training View SSIM] " << ssims << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [Training View LPIPS] " << lpipss << std::endl;
        if (depth_valid_count > 0)
        {
            std::cout << std::fixed << std::setprecision(4)
                      << "        [Training View Depth-L1] "
                      << depth_l1_sum / static_cast<double>(depth_valid_count)
                      << " m over " << depth_valid_count << " valid LiDAR pixels"
                      << std::endl;
        }
        if (language_confidence_sum > 0.0)
        {
            std::cout << std::fixed << std::setprecision(6)
                      << "        [Training View PCA-Language Cosine] "
                      << language_cosine_sum / language_confidence_sum
                      << " over " << language_target_views << " target views, "
                      << language_valid_pixels << "/" << language_target_pixels
                      << " rendered/teacher pixels" << std::endl;
        }
    }
    {
        double psnrs = 0;
        double ssims = 0;
        double lpipss = 0;
        double depth_l1_sum = 0;
        std::int64_t depth_valid_count = 0;
        double language_cosine_sum = 0.0;
        double language_confidence_sum = 0.0;
        std::int64_t language_valid_pixels = 0;
        std::int64_t language_target_pixels = 0;
        std::int64_t language_target_views = 0;
        for (const auto& test_camera : dataset->test_cameras_)
        {
            auto render_pkg = render(test_camera, pc, bg, pc->apply_exposure_);
            auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
            auto rendered_depth = std::get<1>(render_pkg);
            auto gt_image = test_camera->original_image_.cuda().clamp(0, 1);
            auto gt_depth = test_camera->original_depth_.cuda();
            double psnr = loss_utils::psnr(rendered_image, gt_image).mean().item<double>();
            double ssim = loss_utils::ssim(rendered_image, gt_image).item<double>();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(rendered_image.unsqueeze(0));
            inputs.push_back(gt_image.unsqueeze(0));
            double lpips = m_lpips.forward(inputs).toTensor().item<double>();
            psnrs += psnr;
            ssims += ssim;
            lpipss += lpips;

            auto depth_mask = (gt_depth > 0) & (rendered_depth > 0);
            const std::int64_t valid_depth = depth_mask.sum().item<std::int64_t>();
            if (valid_depth > 0)
            {
                depth_l1_sum += torch::abs(
                    rendered_depth.masked_select(depth_mask) - gt_depth.masked_select(depth_mask)
                ).sum().item<double>();
                depth_valid_count += valid_depth;
            }

            int H = rendered_image.size(1), W = rendered_image.size(2);

            torch::Tensor a_cpu = rendered_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            a_cpu = a_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat a_img(H, W, CV_8UC3, a_cpu.data_ptr<uint8_t>());
            cv::cvtColor(a_img, a_img, cv::COLOR_RGB2BGR);
            cv::imwrite(render_dir_path + "/" + test_camera->image_name_, a_img);
            save_semantic_render(test_camera, a_img);
            auto language_metric = evaluate_pca_language(test_camera, a_img);
            if (std::get<3>(language_metric) > 0)
            {
                language_cosine_sum += std::get<0>(language_metric);
                language_confidence_sum += std::get<1>(language_metric);
                language_valid_pixels += std::get<2>(language_metric);
                language_target_pixels += std::get<3>(language_metric);
                ++language_target_views;
            }

            torch::Tensor b_cpu = gt_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            b_cpu = b_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat b_img(H, W, CV_8UC3, b_cpu.data_ptr<uint8_t>());
            cv::cvtColor(b_img, b_img, cv::COLOR_RGB2BGR);
            cv::imwrite(gt_dir_path + "/" + test_camera->image_name_, b_img);

            torch::Tensor depth_map_normalized = (rendered_depth - rendered_depth.min()) / 
                                                     (rendered_depth.max() - rendered_depth.min()) * 255;
            torch::Tensor c_cpu = depth_map_normalized.to(torch::kCPU);
            cv::Mat c_img(H, W, CV_32FC1, c_cpu.data_ptr<float>());
            c_img.convertTo(c_img, CV_8UC1);
            cv::applyColorMap(c_img, c_img, cv::COLORMAP_JET);
            cv::imwrite(render_depth_dir_path + "/" + test_camera->image_name_, c_img);
        }
        psnrs /= dataset->test_cameras_.size();
        ssims /= dataset->test_cameras_.size();
        lpipss /= dataset->test_cameras_.size();
        std::cout << std::fixed << std::setprecision(2) << "        [In-Sequence Novel View PSNR] " << psnrs << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [In-Sequence Novel View SSIM] " << ssims << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [In-Sequence Novel View LPIPS] " << lpipss << std::endl;
        if (depth_valid_count > 0)
        {
            std::cout << std::fixed << std::setprecision(4)
                      << "        [In-Sequence Novel View Depth-L1] "
                      << depth_l1_sum / static_cast<double>(depth_valid_count)
                      << " m over " << depth_valid_count << " valid LiDAR pixels"
                      << std::endl;
        }
        if (language_confidence_sum > 0.0)
        {
            std::cout << std::fixed << std::setprecision(6)
                      << "        [Held-Out View PCA-Language Cosine] "
                      << language_cosine_sum / language_confidence_sum
                      << " over " << language_target_views << " target views, "
                      << language_valid_pixels << "/" << language_target_pixels
                      << " rendered/teacher pixels" << std::endl;
        }
    }

    exportTrajectoryRender(dataset, pc, bg, result_path);
}
