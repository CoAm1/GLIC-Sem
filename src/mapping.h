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

#pragma once

#include "yaml_utils.h"

#include <chrono>
#include <deque>
#include <queue>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

class Params
{
public:
    Params(const YAML::Node &node)
    {
        height = node["height"].as<int>();
        width = node["width"].as<int>();
        fx = node["fx"].as<double>();
        fy = node["fy"].as<double>();
        cx = node["cx"].as<double>();
        cy = node["cy"].as<double>();

        select_every_k_frame = node["select_every_k_frame"].as<int>();
        depth_completion = node["depth_completion"].as<bool>();
        patch_size = node["patch_size"].as<int>();
        depth_completion_max_lidar_distance_px =
            node["depth_completion_max_lidar_distance_px"]
                ? node["depth_completion_max_lidar_distance_px"].as<double>()
                : 0.0;
        depth_completion_temporal_window_keyframes =
            node["depth_completion_temporal_window_keyframes"]
                ? node["depth_completion_temporal_window_keyframes"].as<int>()
                : 0;
        depth_completion_temporal_radius_px =
            node["depth_completion_temporal_radius_px"]
                ? node["depth_completion_temporal_radius_px"].as<int>()
                : 5;
        depth_completion_temporal_min_support =
            node["depth_completion_temporal_min_support"]
                ? node["depth_completion_temporal_min_support"].as<int>()
                : 0;
        depth_completion_depth_tolerance_m =
            node["depth_completion_depth_tolerance_m"]
                ? node["depth_completion_depth_tolerance_m"].as<double>()
                : 0.15;
        depth_completion_depth_tolerance_ratio =
            node["depth_completion_depth_tolerance_ratio"]
                ? node["depth_completion_depth_tolerance_ratio"].as<double>()
                : 0.01;
        max_depth = node["max_depth"].as<double>();
        trajectory_render_frames = node["trajectory_render_frames"] ? node["trajectory_render_frames"].as<int>() : 0;
        std::string pkg_path = ros::package::getPath("gaussian_lic");
        if (height == 512 && width == 640) engine_path = pkg_path + "/ckpt/spnet_512_640.engine";
        if (height == 480 && width == 640) engine_path = pkg_path + "/ckpt/spnet_480_640.engine";
        // The native camera is 1440x1080, while the high-resolution SPNet
        // engine is padded to 1440x1088 to satisfy its encoder stride.
        if (height == 1080 && width == 1440) engine_path = pkg_path + "/ckpt/spnet_1088_1440.engine";
        if (depth_completion && engine_path.empty())
        {
            throw std::runtime_error("No SPNet TensorRT engine for configured image size " +
                                     std::to_string(width) + "x" + std::to_string(height));
        }

        sh_degree = node["sh_degree"].as<int>();
        white_background = node["white_background"].as<bool>();
        random_background = node["random_background"].as<bool>();
        convert_SHs_python = node["convert_SHs_python"].as<bool>();
        compute_cov3D_python = node["compute_cov3D_python"].as<bool>();
        lambda_erank = node["lambda_erank"].as<double>();
        scaling_scale = node["scaling_scale"].as<double>();

        position_lr = node["position_lr"].as<double>();
        feature_lr = node["feature_lr"].as<double>();
        opacity_lr = node["opacity_lr"].as<double>();
        scaling_lr = node["scaling_lr"].as<double>();
        rotation_lr = node["rotation_lr"].as<double>();
        lambda_dssim = node["lambda_dssim"].as<double>();
        optimize_depth = node["optimize_depth"].as<bool>();
        lambda_depth = node["lambda_depth"].as<double>();
        random_seed = node["random_seed"] ? node["random_seed"].as<int>() : 3407;
        semantic_training = node["semantic_training"] ? node["semantic_training"].as<bool>() : false;
        semantic_streaming_ce = node["semantic_streaming_ce"]
                                    ? node["semantic_streaming_ce"].as<bool>()
                                    : false;
        semantic_geometry_gradients = node["semantic_geometry_gradients"]
                                          ? node["semantic_geometry_gradients"].as<bool>()
                                          : true;
        pca_language_training = node["pca_language_training"]
                                    ? node["pca_language_training"].as<bool>()
                                    : false;
        semantic_lr = node["semantic_lr"] ? node["semantic_lr"].as<double>() : 0.01;
        lambda_semantic = node["lambda_semantic"] ? node["lambda_semantic"].as<double>() : 0.0;
        lambda_pca_language = node["lambda_pca_language"]
                                  ? node["lambda_pca_language"].as<double>()
                                  : 0.0;
        lambda_pca_coefficient = node["lambda_pca_coefficient"]
                                     ? node["lambda_pca_coefficient"].as<double>()
                                     : 0.0;
        pca_max_coefficient_norm = node["pca_max_coefficient_norm"]
                                       ? node["pca_max_coefficient_norm"].as<double>()
                                       : 0.0;
        lambda_semantic_region = node["lambda_semantic_region"]
                                     ? node["lambda_semantic_region"].as<double>()
                                     : 0.0;
        semantic_region_stride = node["semantic_region_stride"]
                                     ? node["semantic_region_stride"].as<int>()
                                     : 1;
        semantic_balance_power = node["semantic_balance_power"]
                                     ? node["semantic_balance_power"].as<double>()
                                     : 0.0;
        semantic_balance_max = node["semantic_balance_max"]
                                   ? node["semantic_balance_max"].as<double>()
                                   : 4.0;
        semantic_normal_weighting = node["semantic_normal_weighting"]
                                        ? node["semantic_normal_weighting"].as<bool>()
                                        : false;
        semantic_normal_power = node["semantic_normal_power"]
                                    ? node["semantic_normal_power"].as<double>()
                                    : 2.0;
        semantic_normal_min_weight = node["semantic_normal_min_weight"]
                                         ? node["semantic_normal_min_weight"].as<double>()
                                         : 0.25;
        semantic_normal_depth_tolerance_m =
            node["semantic_normal_depth_tolerance_m"]
                ? node["semantic_normal_depth_tolerance_m"].as<double>()
                : 0.10;
        semantic_normal_depth_tolerance_ratio =
            node["semantic_normal_depth_tolerance_ratio"]
                ? node["semantic_normal_depth_tolerance_ratio"].as<double>()
                : 0.02;
        semantic_normal_weight_classes.clear();
        if (node["semantic_normal_weight_classes"])
        {
            if (!node["semantic_normal_weight_classes"].IsSequence())
            {
                throw std::runtime_error(
                    "semantic_normal_weight_classes must be a YAML sequence.");
            }
            for (const auto &value : node["semantic_normal_weight_classes"])
            {
                semantic_normal_weight_classes.push_back(value.as<int>());
            }
        }
        semantic_init_logit_scale = node["semantic_init_logit_scale"]
                                        ? node["semantic_init_logit_scale"].as<double>()
                                        : 0.0;
        semantic_keyframe_window = node["semantic_keyframe_window"]
                                       ? node["semantic_keyframe_window"].as<int>()
                                       : 0;
        semantic_observation_ema = node["semantic_observation_ema"]
                                       ? node["semantic_observation_ema"].as<double>()
                                       : 0.0;
        semantic_observation_depth_tolerance_m =
            node["semantic_observation_depth_tolerance_m"]
                ? node["semantic_observation_depth_tolerance_m"].as<double>()
                : 0.1;
        semantic_observation_depth_tolerance_ratio =
            node["semantic_observation_depth_tolerance_ratio"]
                ? node["semantic_observation_depth_tolerance_ratio"].as<double>()
                : 0.02;
        semantic_observation_switch_support =
            node["semantic_observation_switch_support"]
                ? node["semantic_observation_switch_support"].as<double>()
                : 0.0;
        semantic_observation_cumulative =
            node["semantic_observation_cumulative"]
                ? node["semantic_observation_cumulative"].as<bool>()
                : false;
        semantic_class_count = node["semantic_class_count"] ? node["semantic_class_count"].as<int>() : 12;
        semantic_min_support = node["semantic_min_support"] ? node["semantic_min_support"].as<double>() : 0.5;
        semantic_min_confidence = node["semantic_min_confidence"] ? node["semantic_min_confidence"].as<double>() : 0.35;
        if (semantic_class_count < 2)
        {
            throw std::runtime_error("semantic_class_count must include unknown and at least one class.");
        }
        if (semantic_training && pca_language_training)
        {
            throw std::runtime_error(
                "Closed-set semantics and PCA language training are mutually exclusive.");
        }
        if (semantic_streaming_ce &&
            (pca_language_training || lambda_semantic_region > 0.0 ||
             semantic_balance_power > 0.0 || semantic_normal_weighting))
        {
            throw std::runtime_error(
                "semantic_streaming_ce currently requires plain closed-set CE "
                "without region, balance, normal, or PCA terms.");
        }
        const bool closed_set_head_only_supported =
            semantic_training && semantic_streaming_ce;
        const bool pca_head_only_supported = pca_language_training;
        if (!semantic_geometry_gradients &&
            !closed_set_head_only_supported &&
            !pca_head_only_supported)
        {
            throw std::runtime_error(
                "semantic_geometry_gradients=false requires either "
                "closed-set streaming CE or PCA language training.");
        }
        if (pca_language_training && lambda_pca_language <= 0.0)
        {
            throw std::runtime_error(
                "pca_language_training requires lambda_pca_language > 0.");
        }
        if (lambda_pca_coefficient < 0.0 || pca_max_coefficient_norm < 0.0)
        {
            throw std::runtime_error(
                "PCA coefficient loss and norm bound must be non-negative.");
        }
        if (semantic_min_support < 0.0 || semantic_min_confidence < 0.0 ||
            semantic_min_confidence > 1.0)
        {
            throw std::runtime_error("Invalid semantic support/confidence threshold.");
        }
        if (lambda_semantic_region < 0.0 || semantic_region_stride < 1)
        {
            throw std::runtime_error("Invalid semantic region-loss configuration.");
        }
        if (semantic_balance_power < 0.0 || semantic_balance_power > 1.0 ||
            semantic_balance_max < 1.0)
        {
            throw std::runtime_error("Invalid semantic class-balance configuration.");
        }
        if (semantic_normal_power < 0.0 ||
            semantic_normal_min_weight < 0.0 ||
            semantic_normal_min_weight > 1.0 ||
            semantic_normal_depth_tolerance_m < 0.0 ||
            semantic_normal_depth_tolerance_ratio < 0.0)
        {
            throw std::runtime_error("Invalid semantic normal-weight configuration.");
        }
        for (const int label : semantic_normal_weight_classes)
        {
            if (label <= 0 || label >= semantic_class_count)
            {
                throw std::runtime_error(
                    "semantic_normal_weight_classes contains an invalid class label.");
            }
        }
        if (semantic_init_logit_scale < 0.0)
        {
            throw std::runtime_error("semantic_init_logit_scale must be non-negative.");
        }
        if (semantic_keyframe_window < 0)
        {
            throw std::runtime_error("semantic_keyframe_window must be non-negative.");
        }
        if (semantic_observation_ema < 0.0 || semantic_observation_ema > 1.0)
        {
            throw std::runtime_error("semantic_observation_ema must be in [0, 1].");
        }
        if (semantic_observation_depth_tolerance_m < 0.0 ||
            semantic_observation_depth_tolerance_ratio < 0.0 ||
            semantic_observation_switch_support < 0.0)
        {
                throw std::runtime_error("Semantic observation depth tolerances must be non-negative.");
        }
        if (semantic_observation_cumulative && semantic_init_logit_scale <= 0.0)
        {
            throw std::runtime_error(
                "Cumulative semantic observations require semantic_init_logit_scale > 0.");
        }
        extend_alpha_threshold = node["extend_alpha_threshold"] ? node["extend_alpha_threshold"].as<double>() : 0.99;
        max_gaussian_scale = node["max_gaussian_scale"] ? node["max_gaussian_scale"].as<double>() : 0.0;
        iteration_decay = node["iteration_decay"].as<bool>();

        apply_exposure = node["apply_exposure"].as<bool>();
        exposure_lr = node["exposure_lr"].as<double>();
        skybox_points_num = node["skybox_points_num"].as<int>();
        skybox_radius = node["skybox_radius"].as<int>();
    }

    /// dataset
    int height;
    int width;
    double fx;
    double fy;
    double cx;
    double cy;

    int select_every_k_frame;
    bool depth_completion;
    int patch_size;
    double depth_completion_max_lidar_distance_px;
    int depth_completion_temporal_window_keyframes;
    int depth_completion_temporal_radius_px;
    int depth_completion_temporal_min_support;
    double depth_completion_depth_tolerance_m;
    double depth_completion_depth_tolerance_ratio;
    double max_depth;
    int trajectory_render_frames;
    std::string engine_path;

    /// gaussian
    int sh_degree;
    bool white_background;
    bool random_background;
    bool convert_SHs_python;
    bool compute_cov3D_python;
    float lambda_erank;
    double scaling_scale;

    double position_lr;
    double feature_lr;
    double opacity_lr;
    double scaling_lr;
    double rotation_lr;
    double lambda_dssim;
    bool optimize_depth;
    double lambda_depth;
    int random_seed;
    bool semantic_training;
    bool semantic_streaming_ce;
    bool semantic_geometry_gradients;
    bool pca_language_training;
    double semantic_lr;
    double lambda_semantic;
    double lambda_pca_language;
    double lambda_pca_coefficient;
    double pca_max_coefficient_norm;
    double lambda_semantic_region;
    int semantic_region_stride;
    double semantic_balance_power;
    double semantic_balance_max;
    bool semantic_normal_weighting;
    double semantic_normal_power;
    double semantic_normal_min_weight;
    double semantic_normal_depth_tolerance_m;
    double semantic_normal_depth_tolerance_ratio;
    std::vector<int> semantic_normal_weight_classes;
    double semantic_init_logit_scale;
    int semantic_keyframe_window;
    double semantic_observation_ema;
    double semantic_observation_depth_tolerance_m;
    double semantic_observation_depth_tolerance_ratio;
    double semantic_observation_switch_support;
    bool semantic_observation_cumulative;
    int semantic_class_count;
    double semantic_min_support;
    double semantic_min_confidence;
    double extend_alpha_threshold;
    double max_gaussian_scale;
    bool iteration_decay;

    bool apply_exposure;
    double exposure_lr;
    int skybox_points_num;
    int skybox_radius;
};

struct Frame 
{
    sensor_msgs::PointCloud2ConstPtr point_msg;
    geometry_msgs::PoseStampedConstPtr pose_msg;
    sensor_msgs::ImageConstPtr image_msg;
    sensor_msgs::ImageConstPtr depth_msg;
};
