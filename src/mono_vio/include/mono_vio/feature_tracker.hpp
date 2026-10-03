#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <map>

namespace mono_vio {

struct CameraIntrinsics;

// ── Per-feature data ──────────────────────────────────────────────────────────
struct TrackedFeature {
  int             landmark_id {-1};
  int             track_count {0};
  cv::Point2f     px;           ///< raw distorted pixel (u,v)
  Eigen::Vector3d uv_norm;      ///< undistorted normalised [x, y, 1]
};

// ── Output of one track() call ────────────────────────────────────────────────
struct FeatureFrame {
  double                       timestamp {0.0};
  std::vector<TrackedFeature>  features;
};

// ── FeatureTracker ────────────────────────────────────────────────────────────
/**
 * Monocular feature tracker — VINS-Mono pipeline:
 *   1. IMU-prediction warm-start for KLT (optional, via R_hint)
 *   2. Forward KLT optical flow
 *   3. Reverse (back-tracking) verification to cull bad tracks
 *   4. Fundamental-matrix RANSAC to remove dynamic-object outliers
 *   5. Grid-based FAST detection to fill up to max_features
 *   6. Undistortion to normalised bearing vectors
 */
class FeatureTracker {
public:
  struct Params {
    int    max_features             {150};
    int    fast_threshold           {20};
    int    min_dist                 {30};
    int    klt_win_size             {21};
    int    klt_max_level            {3};
    double ransac_reproj_threshold  {1.0};
    float  backtrack_thr            {1.5f};  ///< max round-trip px error
    int    grid_rows                {4};     ///< image rows for detection grid
    int    grid_cols                {6};     ///< image cols for detection grid
  };

  explicit FeatureTracker(const CameraIntrinsics& K, const Params& params);

  /**
   * Track features from the previous frame into @p img.
   * @param img        Grayscale image (CV_8U)
   * @param timestamp  Sensor timestamp [s]
   * @param R_hint     Optional IMU-integrated rotation dq (body i → body j).
   *                   If provided, features are pre-positioned in KLT using
   *                   the predicted camera rotation.
   */
  FeatureFrame track(const cv::Mat& img, double timestamp,
                     const Eigen::Quaterniond* R_hint = nullptr);

private:
  double fx_, fy_, cx_, cy_;
  double k1_, k2_, p1_, p2_;
  int    width_, height_;
  cv::Mat cv_K_, cv_dist_;

  Params params_;

  cv::Mat prev_img_;
  std::vector<TrackedFeature> features_;
  int next_id_{0};

  void undistortFeatures();
  void rejectOutliersFundamental();
  void detectNew(const cv::Mat& img);
  int  countFeaturesInRoi(const cv::Rect& roi) const;
};

}  // namespace mono_vio