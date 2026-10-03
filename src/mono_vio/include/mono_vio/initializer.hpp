#pragma once
#include "mono_vio/imu_integrator.hpp"
#include "mono_vio/feature_tracker.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <map>
#include <string>

namespace mono_vio {

struct InitFrame {
  double          timestamp;
  Eigen::Matrix3d R_wc;
  Eigen::Vector3d t_wc;
  ImuIntegrator   pim;
  Eigen::Vector3d velocity;
};

struct InitResult {
  double          scale      {1.0};
  Eigen::Vector3d gravity_w  {0, 0, -9.81};
  std::vector<Eigen::Vector3d> velocities;
  Eigen::Vector3d ba         {Eigen::Vector3d::Zero()};
  Eigen::Vector3d bg         {Eigen::Vector3d::Zero()};
};

struct SfmFeature {
  bool used {false};
  std::vector<std::pair<int, Eigen::Vector2d>> obs;
  Eigen::Vector3d position;
};

class Initializer {
public:
  // Camera intrinsics — set before tryInitialize()
  double fx{458.654}, fy{457.296}, cx{367.215}, cy{248.375};

  // Camera-from-IMU rotation (R_ci = T_cam_imu.topLeftCorner<3,3>()).
  // Used in VI alignment to rotate delta_p from body to world frame.
  // MUST be set by the node before calling tryInitialize().
  Eigen::Matrix3d R_ci {Eigen::Matrix3d::Identity()};

  Initializer() = default;

  void addFrame(const InitFrame& frame);
  void addSfmFeatures(const FeatureFrame& ff, int frame_idx);

  int  frameCount() const { return (int)frames_.size(); }
  bool isExcited(double threshold = 0.5) const;

  bool tryInitialize(InitResult& result, std::string* fail_reason = nullptr);

  void reset() {
    frames_.clear(); raw_acc_.clear();
    sfm_map_.clear(); sfm_R_.clear(); sfm_t_.clear();
  }

  const std::vector<InitFrame>&      frames() const { return frames_; }
  const std::vector<Eigen::Matrix3d>& sfmR()  const { return sfm_R_; }
  const std::vector<Eigen::Vector3d>& sfmT()  const { return sfm_t_; }
  /// Read-only access to the per-landmark observation map built during SfM.
  /// Used by the node to seed window observations after initialization.
  const std::map<int, SfmFeature>&   sfmMap() const { return sfm_map_; }

private:
  std::vector<InitFrame>       frames_;
  std::vector<Eigen::Vector3d> raw_acc_;
  std::map<int, SfmFeature>    sfm_map_;
  std::vector<Eigen::Matrix3d> sfm_R_;
  std::vector<Eigen::Vector3d> sfm_t_;

  bool solveSfM(std::vector<Eigen::Matrix3d>& R_wc,
                std::vector<Eigen::Vector3d>& t_wc,
                std::string* reason = nullptr);

  bool visualInertialAlign(const std::vector<Eigen::Matrix3d>& R_wc,
                           const std::vector<Eigen::Vector3d>& t_wc,
                           InitResult& result,
                           std::string* reason = nullptr);

  double computeParallax(int fi, int fj) const;

  void matchPoints(int fi, int fj,
                   std::vector<cv::Point2f>& pts_i,
                   std::vector<cv::Point2f>& pts_j,
                   std::vector<int>& lm_ids) const;

  bool pnpLocalize(int fi,
                   std::vector<Eigen::Matrix3d>& R_wc,
                   std::vector<Eigen::Vector3d>& t_wc);

  void triangulateSfm(int fi, int fj,
                      const std::vector<Eigen::Matrix3d>& R_wc,
                      const std::vector<Eigen::Vector3d>& t_wc);

  static Eigen::Matrix3d gravityAlignRotation(const Eigen::Vector3d& g_raw);
};

}  // namespace mono_vio