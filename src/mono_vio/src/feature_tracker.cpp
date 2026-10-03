#include "mono_vio/feature_tracker.hpp"
#include "mono_vio/optimizer.hpp"   // CameraIntrinsics definition

#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>

namespace mono_vio {

FeatureTracker::FeatureTracker(const CameraIntrinsics& K, const Params& params)
  : fx_(K.fx), fy_(K.fy), cx_(K.cx), cy_(K.cy),
    k1_(K.k1), k2_(K.k2), p1_(K.p1), p2_(K.p2),
    width_(K.width), height_(K.height),
    params_(params)
{
  cv_K_    = (cv::Mat_<double>(3,3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
  cv_dist_ = (cv::Mat_<double>(1,4) << k1_, k2_, p1_, p2_);
}

FeatureFrame FeatureTracker::track(const cv::Mat& img, double timestamp,
                                    const Eigen::Quaterniond* R_hint) {
  FeatureFrame frame;
  frame.timestamp = timestamp;

  if (prev_img_.empty()) {
    prev_img_ = img.clone();
    detectNew(img);
    undistortFeatures();
    frame.features = features_;
    return frame;
  }

  // ── 1. KLT optical flow ───────────────────────────────────────────────────
  if (!features_.empty()) {
    std::vector<cv::Point2f> prev_pts, next_pts;
    prev_pts.reserve(features_.size());
    for (auto& f : features_) prev_pts.push_back(f.px);

    // IMU rotation prediction: project bearing through dR to get better
    // initial positions for KLT — reduces search area and improves accuracy.
    next_pts = prev_pts;
    if (R_hint) {
      // dR_cam = R_ci * dR_body * R_ci^T  (rotation of camera between frames)
      // We stored R_ci as R_hint already converted by caller.
      const Eigen::Matrix3d dR = R_hint->normalized().toRotationMatrix();
      for (int i = 0; i < (int)features_.size(); ++i) {
        Eigen::Vector3d b = features_[i].uv_norm;  // already z=1
        Eigen::Vector3d b2 = dR * b;
        if (b2.z() < 1e-4) continue;
        const float u = (float)(b2.x() / b2.z() * fx_ + cx_);
        const float v = (float)(b2.y() / b2.z() * fy_ + cy_);
        if (u > 1 && u < width_-1 && v > 1 && v < height_-1)
          next_pts[i] = cv::Point2f(u, v);
      }
    }

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        prev_img_, img, prev_pts, next_pts, status, err,
        cv::Size(params_.klt_win_size, params_.klt_win_size),
        params_.klt_max_level,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // ── 2. Back-tracking verification (reverse KLT) ──────────────────────────
    std::vector<cv::Point2f> back_pts = prev_pts;
    std::vector<uchar>  back_status;
    std::vector<float>  back_err;
    cv::calcOpticalFlowPyrLK(
        img, prev_img_, next_pts, back_pts, back_status, back_err,
        cv::Size(params_.klt_win_size, params_.klt_win_size),
        params_.klt_max_level,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<TrackedFeature> survived;
    survived.reserve(features_.size());
    for (int i = 0; i < (int)features_.size(); ++i) {
      if (!status[i] || !back_status[i]) continue;
      // Round-trip consistency
      const float dx = prev_pts[i].x - back_pts[i].x;
      const float dy = prev_pts[i].y - back_pts[i].y;
      if (dx*dx + dy*dy > params_.backtrack_thr * params_.backtrack_thr) continue;
      const cv::Point2f& p = next_pts[i];
      if (p.x < 2 || p.y < 2 || p.x > width_-2 || p.y > height_-2) continue;
      TrackedFeature f = features_[i];
      f.px = p;
      f.track_count++;
      survived.push_back(f);
    }
    features_ = std::move(survived);
  }

  // ── 3. Fundamental matrix RANSAC ──────────────────────────────────────────
  rejectOutliersFundamental();

  // ── 4. Grid-based FAST detection ─────────────────────────────────────────
  detectNew(img);

  // ── 5. Undistort ─────────────────────────────────────────────────────────
  undistortFeatures();

  prev_img_ = img.clone();
  frame.features = features_;
  return frame;
}

// ── Grid-based detection ──────────────────────────────────────────────────────
int FeatureTracker::countFeaturesInRoi(const cv::Rect& roi) const {
  int cnt = 0;
  for (auto& f : features_) {
    if (roi.contains(cv::Point((int)f.px.x, (int)f.px.y))) ++cnt;
  }
  return cnt;
}

void FeatureTracker::detectNew(const cv::Mat& img) {
  const int total_needed = params_.max_features - (int)features_.size();
  if (total_needed <= 0) return;

  // Build global exclusion mask
  cv::Mat mask(img.size(), CV_8U, cv::Scalar(255));
  for (auto& f : features_)
    cv::circle(mask, f.px, params_.min_dist, cv::Scalar(0), -1);

  const int cell_w    = width_  / params_.grid_cols;
  const int cell_h    = height_ / params_.grid_rows;
  const int cell_quota = std::max(1,
      params_.max_features / (params_.grid_rows * params_.grid_cols));

  for (int gr = 0; gr < params_.grid_rows; ++gr) {
    for (int gc = 0; gc < params_.grid_cols; ++gc) {
      cv::Rect roi(gc * cell_w, gr * cell_h, cell_w, cell_h);
      roi &= cv::Rect(0, 0, width_, height_);  // clamp to image

      int in_cell = countFeaturesInRoi(roi);
      int needed  = cell_quota - in_cell;
      if (needed <= 0) continue;

      // Detect FAST in this cell
      cv::Mat cell_img  = img(roi);
      cv::Mat cell_mask = mask(roi);
      std::vector<cv::KeyPoint> kps;
      cv::FAST(cell_img, kps, params_.fast_threshold, true);
      std::sort(kps.begin(), kps.end(),
                [](const cv::KeyPoint& a, const cv::KeyPoint& b){
                  return a.response > b.response; });

      for (auto& kp : kps) {
        if (needed <= 0) break;
        cv::Point pt_cell(static_cast<int>(kp.pt.x),
                          static_cast<int>(kp.pt.y));
        cv::Point pt_full(pt_cell.x + roi.x, pt_cell.y + roi.y);
        if (pt_full.x < 2 || pt_full.y < 2 ||
            pt_full.x > width_-2 || pt_full.y > height_-2) continue;
        if (cell_mask.at<uchar>(pt_cell) == 0) continue;

        TrackedFeature f;
        f.landmark_id = next_id_++;
        f.track_count = 0;
        f.px          = cv::Point2f((float)pt_full.x, (float)pt_full.y);
        features_.push_back(f);

        // Update global mask
        cv::circle(mask, pt_full, params_.min_dist, cv::Scalar(0), -1);
        --needed;
      }
    }
  }
}

// ── Undistort ─────────────────────────────────────────────────────────────────
void FeatureTracker::undistortFeatures() {
  if (features_.empty()) return;
  std::vector<cv::Point2f> pts;
  pts.reserve(features_.size());
  for (auto& f : features_) pts.push_back(f.px);

  std::vector<cv::Point2f> undist;
  cv::undistortPoints(pts, undist, cv_K_, cv_dist_);  // normalised (z=1)

  for (int i = 0; i < (int)features_.size(); ++i)
    features_[i].uv_norm = Eigen::Vector3d(undist[i].x, undist[i].y, 1.0);
}

// ── Fundamental RANSAC ────────────────────────────────────────────────────────
void FeatureTracker::rejectOutliersFundamental() {
  if (features_.size() < 8) return;

  // std::vector<cv::Point2f> pts1, pts2;
  std::vector<cv::Point2f> pts1, pts2_raw;
  std::vector<int> indices;
  pts1.reserve(features_.size());
  // pts2.reserve(features_.size());
  pts2_raw.reserve(features_.size());
  indices.reserve(features_.size());

  for (int i = 0; i < (int)features_.size(); ++i) {
    if (features_[i].track_count < 1) continue;
    pts1.emplace_back((float)features_[i].uv_norm.x(),
                      (float)features_[i].uv_norm.y());
    // pts2.emplace_back(features_[i].px.x / (float)fx_ - (float)(cx_ / fx_),
                      // features_[i].px.y / (float)fy_ - (float)(cy_ / fy_));
    pts2_raw.emplace_back(features_[i].px);                  
    indices.push_back(i);
  }
  if ((int)pts1.size() < 8) return;
  std::vector<cv::Point2f> pts2;
  cv::undistortPoints(pts2_raw, pts2, cv_K_, cv_dist_);

  std::vector<uchar> mask;
  cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                          params_.ransac_reproj_threshold / fx_, 0.99, mask);

  std::vector<TrackedFeature> kept;
  kept.reserve(features_.size());
  int mi = 0;
  for (int i = 0; i < (int)features_.size(); ++i) {
    if (features_[i].track_count < 1) {
      kept.push_back(features_[i]);
    } else {
      if (mask[mi]) kept.push_back(features_[i]);
      ++mi;
    }
  }
  features_ = std::move(kept);
}

}  // namespace mono_vio