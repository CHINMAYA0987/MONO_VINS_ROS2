#include "mono_vio/loop_detector.hpp"
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <optional>

namespace mono_vio {

LoopDetector::LoopDetector(const Params& p) : params_(p) {
  orb_     = cv::ORB::create(params_.orb_features);
  matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);
}

void LoopDetector::reset() { db_.clear(); }

void LoopDetector::addKeyframe(int kf_id, const cv::Mat& gray,
                                const Eigen::Matrix3d& R_wc,
                                const Eigen::Vector3d& t_wc) {
  DBFrame f;
  f.kf_id = kf_id;
  f.R_wc  = R_wc;
  f.t_wc  = t_wc;
  orb_->detectAndCompute(gray, cv::noArray(), f.keypoints, f.descriptors);
  db_.push_back(std::move(f));
  if ((int)db_.size() > params_.max_db_size) db_.pop_front();
}

std::optional<LoopConstraint> LoopDetector::detect(
    int kf_id, const cv::Mat& gray,
    const Eigen::Matrix3d& R_wc, const Eigen::Vector3d& t_wc) {
  if ((int)db_.size() < params_.min_temporal_gap + 1) return std::nullopt;

  // Extract current keyframe descriptors
  std::vector<cv::KeyPoint> kps_cur;
  cv::Mat desc_cur;
  orb_->detectAndCompute(gray, cv::noArray(), kps_cur, desc_cur);
  if (kps_cur.empty() || desc_cur.empty()) return std::nullopt;

  LoopConstraint best;
  best.n_inliers = 0;

  // Search all database frames with sufficient temporal gap
  for (int di = 0; di < (int)db_.size() - params_.min_temporal_gap; ++di) {
    const DBFrame& db_f = db_[di];
    if (db_f.descriptors.empty()) continue;

    // Lowe ratio test matching
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(desc_cur, db_f.descriptors, knn_matches, 2);

    std::vector<cv::Point2f> pts_cur, pts_db;
    for (auto& m : knn_matches) {
      if (m.size() < 2) continue;
      if (m[0].distance < params_.ratio_thresh * m[1].distance) {
        pts_cur.push_back(kps_cur[m[0].queryIdx].pt);
        pts_db.push_back(db_f.keypoints[m[0].trainIdx].pt);
      }
    }
    if ((int)pts_cur.size() < params_.min_inliers) continue;

    Eigen::Matrix3d R_rel;
    Eigen::Vector3d t_rel;
    int inliers = verifyGeometry(pts_cur, pts_db, R_rel, t_rel);
    if (inliers > best.n_inliers) {
      best.cur_kf_id   = kf_id;
      best.loop_kf_id  = db_f.kf_id;
      best.R_loop_cur  = R_rel;
      best.t_loop_cur  = t_rel;
      best.n_inliers   = inliers;
    }
  }

  if (best.n_inliers >= params_.min_inliers) return best;
  return std::nullopt;
}

int LoopDetector::verifyGeometry(const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2,
                                  Eigen::Matrix3d& R_out,
                                  Eigen::Vector3d& t_out) const {
  if ((int)pts1.size() < 8) return 0;
  cv::Mat inlier_mask;
  cv::Mat E = cv::findEssentialMat(
      pts1, pts2, 1.0, cv::Point2d(0,0),
      cv::RANSAC, 0.999, params_.ransac_thresh, inlier_mask);
  if (E.empty()) return 0;

  cv::Mat R_cv, t_cv;
  int inliers = cv::recoverPose(
      E, pts1, pts2, R_cv, t_cv, 1.0, cv::Point2d(0,0), inlier_mask);
  if (inliers < 8) return 0;

  R_out << R_cv.at<double>(0,0), R_cv.at<double>(0,1), R_cv.at<double>(0,2),
           R_cv.at<double>(1,0), R_cv.at<double>(1,1), R_cv.at<double>(1,2),
           R_cv.at<double>(2,0), R_cv.at<double>(2,1), R_cv.at<double>(2,2);
  t_out << t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2);
  return inliers;
}

}  // namespace mono_vio