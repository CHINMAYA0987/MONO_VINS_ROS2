#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <optional>
#include <vector>
#include <deque>

namespace mono_vio {

struct LoopConstraint {
  int             cur_kf_id;
  int             loop_kf_id;
  Eigen::Vector3d t_loop_cur;
  Eigen::Matrix3d R_loop_cur;
  int             n_inliers{0};
};

class LoopDetector {
public:
  struct Params {
    int    max_db_size      {300};
    int    min_temporal_gap {10};
    int    min_inliers      {25};
    float  ratio_thresh     {0.80f};
    double ransac_thresh    {1.5};
    int    orb_features     {800};
  };

  explicit LoopDetector(const Params& p);

  void addKeyframe(int kf_id, const cv::Mat& gray,
                   const Eigen::Matrix3d& R_wc, const Eigen::Vector3d& t_wc);

  std::optional<LoopConstraint> detect(int kf_id, const cv::Mat& gray,
                                       const Eigen::Matrix3d& R_wc,
                                       const Eigen::Vector3d& t_wc);

  void reset();

  // ── Pose accessors for loop correction in the node ───────────────────────
  /// Returns the world-frame rotation stored for kf_id (identity if not found)
  Eigen::Matrix3d getR(int kf_id) const {
    for (auto& f : db_)
      if (f.kf_id == kf_id) return f.R_wc;
    return Eigen::Matrix3d::Identity();
  }
  /// Returns the world-frame position stored for kf_id (zero if not found)
  Eigen::Vector3d getT(int kf_id) const {
    for (auto& f : db_)
      if (f.kf_id == kf_id) return f.t_wc;
    return Eigen::Vector3d::Zero();
  }

private:
  struct DBFrame {
    int                       kf_id;
    cv::Mat                   descriptors;
    std::vector<cv::KeyPoint> keypoints;
    Eigen::Matrix3d           R_wc;
    Eigen::Vector3d           t_wc;
  };

  Params                 params_;
  cv::Ptr<cv::ORB>       orb_;
  cv::Ptr<cv::BFMatcher> matcher_;
  std::deque<DBFrame>    db_;

  int verifyGeometry(const std::vector<cv::Point2f>& pts1,
                     const std::vector<cv::Point2f>& pts2,
                     Eigen::Matrix3d& R_out, Eigen::Vector3d& t_out) const;
};

}  // namespace mono_vio