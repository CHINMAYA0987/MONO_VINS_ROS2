#include "mono_vio/initializer.hpp"
#include "mono_vio/triangulator.hpp"
#include <Eigen/SVD>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>

namespace mono_vio {

void Initializer::addFrame(const InitFrame& frame_in) {
  InitFrame frame = frame_in;
  if (frames_.empty()) {
    frame.R_wc = Eigen::Matrix3d::Identity();
    frame.t_wc = Eigen::Vector3d::Zero();
  } else {
    const InitFrame& prev = frames_.back();
    frame.R_wc = prev.R_wc * prev.pim.delta_q.toRotationMatrix();
    frame.t_wc = prev.t_wc + prev.R_wc * prev.pim.delta_p;
  }
  for (auto& m : frame.pim.measurements()) raw_acc_.push_back(m.acc);
  frames_.push_back(frame);
}

void Initializer::addSfmFeatures(const FeatureFrame& ff, int frame_idx) {
  for (const auto& f : ff.features) {
    SfmFeature& sf = sfm_map_[f.landmark_id];
    sf.obs.push_back({frame_idx, f.uv_norm.head<2>()});
  }
}

bool Initializer::isExcited(double threshold) const {
  if (raw_acc_.size() < 5) return false;
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (auto& a : raw_acc_) mean += a;
  mean /= static_cast<double>(raw_acc_.size());
  double var = 0.0;
  for (auto& a : raw_acc_) var += (a - mean).squaredNorm();
  return std::sqrt(var / static_cast<double>(raw_acc_.size())) > threshold;
}

bool Initializer::tryInitialize(InitResult& result, std::string* fail_reason) {
  auto set_reason = [&](const std::string& r){ if (fail_reason) *fail_reason = r; };
  if ((int)frames_.size() < 5) { set_reason("too_few_frames"); return false; }

  std::string sfm_reason;
  if (!solveSfM(sfm_R_, sfm_t_, &sfm_reason)) {
    set_reason("sfm:" + sfm_reason); return false;
  }

  std::string vi_reason;
  if (!visualInertialAlign(sfm_R_, sfm_t_, result, &vi_reason)) {
    set_reason("vi_align:" + vi_reason); return false;
  }

  Eigen::Matrix3d R_align = gravityAlignRotation(result.gravity_w);
  // Reject only clearly inverted gravity (flipped by > 150°)
  Eigen::AngleAxisd aa_check(R_align);
  RCLCPP_INFO(rclcpp::get_logger("vi_align"), "  gravity_align angle=%.1f deg", aa_check.angle() * 57.3);
  if (std::abs(aa_check.angle()) > 1.57) {   // > 90°
    RCLCPP_WARN(rclcpp::get_logger("vi_align"),
        "  Large gravity rotation (%.1f deg) — initial ATE will be large, "
        "loop closure will correct", aa_check.angle() * 57.3);
  }
  if (std::abs(aa_check.angle()) > 2.62) {  // > 150 degrees
      if (fail_reason) *fail_reason = "gravity_flip_"
          + std::to_string(aa_check.angle() * 57.3) + "deg";
      return false;
  }
  
  result.gravity_w = Eigen::Vector3d(0.0, 0.0, -9.81);

  const int N = (int)sfm_R_.size();
  for (int i = 0; i < N; ++i) {
    sfm_R_[i] = R_align * sfm_R_[i];
    sfm_t_[i] = R_align * sfm_t_[i] * result.scale;
    if (i < (int)result.velocities.size())
      result.velocities[i] = R_align * result.velocities[i];
  }
  for (auto& [id, sf] : sfm_map_) {
    if (sf.used)
      sf.position = R_align * sf.position * result.scale;  // ← ADD THIS
  }

  return true;
}

double Initializer::computeParallax(int fi, int fj) const {
  double sum = 0.0; int cnt = 0;
  for (auto& [id, sf] : sfm_map_) {
    const Eigen::Vector2d* pi = nullptr; const Eigen::Vector2d* pj = nullptr;
    for (auto& [idx, uv] : sf.obs) {
      if (idx == fi) pi = &uv;
      if (idx == fj) pj = &uv;
    }
    if (!pi || !pj) continue;
    double dx = (pi->x() - pj->x()) * fx;
    double dy = (pi->y() - pj->y()) * fy;
    sum += std::sqrt(dx*dx + dy*dy); ++cnt;
  }
  return cnt > 0 ? sum / cnt : 0.0;
}

void Initializer::matchPoints(int fi, int fj,
                               std::vector<cv::Point2f>& pts_i,
                               std::vector<cv::Point2f>& pts_j,
                               std::vector<int>& lm_ids) const {
  pts_i.clear(); pts_j.clear(); lm_ids.clear();
  for (auto& [id, sf] : sfm_map_) {
    const Eigen::Vector2d* pi = nullptr; const Eigen::Vector2d* pj = nullptr;
    for (auto& [idx, uv] : sf.obs) {
      if (idx == fi) pi = &uv;
      if (idx == fj) pj = &uv;
    }
    if (!pi || !pj) continue;
    pts_i.push_back(cv::Point2f((float)pi->x(), (float)pi->y()));
    pts_j.push_back(cv::Point2f((float)pj->x(), (float)pj->y()));
    lm_ids.push_back(id);
  }
}

void Initializer::triangulateSfm(int fi, int fj,
                                   const std::vector<Eigen::Matrix3d>& R_wc,
                                   const std::vector<Eigen::Vector3d>& t_wc) {
  Eigen::Matrix3d Rcw_i = R_wc[fi].transpose();
  Eigen::Vector3d tcw_i = -Rcw_i * t_wc[fi];
  Eigen::Matrix3d Rcw_j = R_wc[fj].transpose();
  Eigen::Vector3d tcw_j = -Rcw_j * t_wc[fj];
  Eigen::Matrix<double,3,4> P_i, P_j;
  P_i.leftCols<3>() = Rcw_i; P_i.rightCols<1>() = tcw_i;
  P_j.leftCols<3>() = Rcw_j; P_j.rightCols<1>() = tcw_j;
  for (auto& [id, sf] : sfm_map_) {
    if (sf.used) continue;
    const Eigen::Vector2d* pi = nullptr; const Eigen::Vector2d* pj = nullptr;
    for (auto& [idx, uv] : sf.obs) {
      if (idx == fi) pi = &uv;
      if (idx == fj) pj = &uv;
    }
    if (!pi || !pj) continue;
    Eigen::Vector3d b0(pi->x(), pi->y(), 1.0);
    Eigen::Vector3d b1(pj->x(), pj->y(), 1.0);
    Eigen::Vector3d pw;
    if (Triangulator::triangulate2(P_i, P_j, b0.normalized(), b1.normalized(), pw)) {
      sf.position = pw; sf.used = true;
    }
  }
}

bool Initializer::pnpLocalize(int fi,
                                std::vector<Eigen::Matrix3d>& R_wc,
                                std::vector<Eigen::Vector3d>& t_wc) {
  std::vector<cv::Point3f> pts3d; std::vector<cv::Point2f> pts2d;
  for (auto& [id, sf] : sfm_map_) {
    if (!sf.used) continue;
    for (auto& [idx, uv] : sf.obs) {
      if (idx != fi) continue;
      pts3d.push_back(cv::Point3f((float)sf.position.x(),(float)sf.position.y(),(float)sf.position.z()));
      pts2d.push_back(cv::Point2f((float)uv.x(), (float)uv.y()));
    }
  }
  if ((int)pts3d.size() < 6) return false;
  // Normalised coords → identity camera matrix, tight threshold
  cv::Mat K_norm = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat rvec, tvec, inliers;
  bool ok = cv::solvePnPRansac(pts3d, pts2d, K_norm, cv::Mat(),
                                 rvec, tvec, false, 100, 0.01f, 0.99, inliers);
  if (!ok || inliers.rows < 6) return false;
  cv::Mat R_cv; cv::Rodrigues(rvec, R_cv);
  Eigen::Matrix3d Rcw;
  Rcw << R_cv.at<double>(0,0),R_cv.at<double>(0,1),R_cv.at<double>(0,2),
         R_cv.at<double>(1,0),R_cv.at<double>(1,1),R_cv.at<double>(1,2),
         R_cv.at<double>(2,0),R_cv.at<double>(2,1),R_cv.at<double>(2,2);
  Eigen::Vector3d tcw(tvec.at<double>(0),tvec.at<double>(1),tvec.at<double>(2));
  R_wc[fi] = Rcw.transpose();
  t_wc[fi] = -Rcw.transpose() * tcw;
  return true;
}

bool Initializer::solveSfM(std::vector<Eigen::Matrix3d>& R_wc,
                             std::vector<Eigen::Vector3d>& t_wc,
                             std::string* reason) {
  auto fail = [&](const std::string& r) -> bool { if (reason) *reason=r; return false; };
  const int N = (int)frames_.size();

  // ── Assign FIRST (was after gravity block — overwrote R_wc[0]) ────────────
  R_wc.assign(N, Eigen::Matrix3d::Identity());
  t_wc.assign(N, Eigen::Vector3d::Zero());

  // ── Gravity pre-alignment: seed R_wc[0] so world = gravity-aligned frame ──
  // Uses only near-static acc samples from the first frame's IMU buffer.
  // This makes all downstream SfM and VI-align live in a consistent frame,
  // reducing the gravity rotation from ~130° to near 0° before tryInitialize.
  {
    Eigen::Vector3d g_imu = Eigen::Vector3d::Zero();
    int gc = 0;
    for (const auto& m : frames_[0].pim.measurements()) {
      if (std::abs(m.acc.norm() - 9.81) < 1.0) {
        g_imu += m.acc; ++gc;
      }
    }
    if (gc >= 3) {
      g_imu /= gc;
      const Eigen::Vector3d g_cam = R_ci * g_imu;
      const Eigen::Matrix3d R_gravity_align =
          Eigen::Quaterniond::FromTwoVectors(g_cam, Eigen::Vector3d(0, 0, 9.81))
              .toRotationMatrix();
      R_wc[0] = R_gravity_align;
    }
    // If gc < 3 (highly excited first frame), R_wc[0] stays Identity —
    // fallback to old behaviour, gravityAlignRotation in tryInitialize corrects it.
  }

  // ── Pivot selection (unchanged) ───────────────────────────────────────────
  int pivot=-1; double best_par=0.0;
  for (int i = 1; i < N; ++i) {
    int shared = 0;
    for (auto& [id,sf] : sfm_map_) {
      bool has0=false, hasi=false;
      for (auto& [idx,uv] : sf.obs) { if(idx==0) has0=true; if(idx==i) hasi=true; }
      if (has0 && hasi) ++shared;
    }
    if (shared < 8) continue;
    double par = computeParallax(0, i);
    if (par > best_par) { best_par=par; pivot=i; }
  }

  if (pivot < 0) return fail("no_pivot_with_8_shared");
  if (best_par < 10.0)
    return fail("low_parallax_" + std::to_string((int)best_par) + "px_need_10");

  std::vector<cv::Point2f> pts0, pts_p; std::vector<int> shared_ids;
  matchPoints(0, pivot, pts0, pts_p, shared_ids);
  if ((int)pts0.size() < 8)
    return fail("few_matches_" + std::to_string(pts0.size()));

  cv::Mat inlier_mask;
  cv::Mat E = cv::findEssentialMat(pts0, pts_p, 1.0, cv::Point2d(0.0,0.0),
                                    cv::RANSAC, 0.999, 1.0/fx, inlier_mask);
  if (E.empty()) return fail("essential_failed");

  cv::Mat R_cv, t_cv;
  int inliers = cv::recoverPose(E, pts0, pts_p, R_cv, t_cv,
                                 1.0, cv::Point2d(0.0,0.0), inlier_mask);
  if (inliers < 8)
    return fail("recover_pose_" + std::to_string(inliers) + "_inliers");

  Eigen::Matrix3d Rcw_p;
  Rcw_p << R_cv.at<double>(0,0),R_cv.at<double>(0,1),R_cv.at<double>(0,2),
           R_cv.at<double>(1,0),R_cv.at<double>(1,1),R_cv.at<double>(1,2),
           R_cv.at<double>(2,0),R_cv.at<double>(2,1),R_cv.at<double>(2,2);
  Eigen::Vector3d tcw_p(t_cv.at<double>(0),t_cv.at<double>(1),t_cv.at<double>(2));

  // recoverPose gives pivot pose relative to camera 0's OWN frame.
  // Rotate into gravity-aligned world frame by prepending R_wc[0].
  // If R_wc[0] = I (fallback), these are identical to the old lines.
  R_wc[pivot] = R_wc[0] * Rcw_p.transpose();                    // ← was: Rcw_p.transpose()
  t_wc[pivot] = R_wc[0] * (-Rcw_p.transpose() * tcw_p);         // ← was: -Rcw_p.transpose()*tcw_p

  triangulateSfm(0, pivot, R_wc, t_wc);

  for (int i = 1; i < N; ++i) {
    if (i == pivot) continue;
    if (!pnpLocalize(i, R_wc, t_wc)) {
      // Interpolation fallback: slerp from R_wc[0] (not Identity) to R_wc[pivot]
      double alpha = (double)i / std::max(pivot, 1);
      R_wc[i] = Eigen::Quaterniond(R_wc[0])                     // ← was: Identity()
                  .slerp(alpha, Eigen::Quaterniond(R_wc[pivot]))
                  .toRotationMatrix();
      t_wc[i] = t_wc[pivot] * alpha;
    }
    triangulateSfm(0, i, R_wc, t_wc);
    triangulateSfm(pivot, i, R_wc, t_wc);
  }

  int n_tri = 0;
  for (auto& [id,sf] : sfm_map_) if (sf.used) ++n_tri;
  if (n_tri < 8)
    return fail("few_tri_" + std::to_string(n_tri));

  double spread = 0.0;
  for (int i = 1; i < N; ++i) spread += (t_wc[i]-t_wc[i-1]).norm();
  if (spread < 0.005)
    return fail("zero_spread_" + std::to_string(spread));

  return true;
}

bool Initializer::visualInertialAlign(
    const std::vector<Eigen::Matrix3d>& R_wc,
    const std::vector<Eigen::Vector3d>& t_wc,
    InitResult& result, std::string* reason)
{
  auto fail = [&](const std::string& r) -> bool {
    if (reason) *reason = r;
    return false;
  };
  const int N = (int)frames_.size();
  if (N < 3) return fail("too_few_frames");

  // ── Build IMU-chained R_wb (replaces noisy SfM R_wc throughout) ───────────
  // World frame = first camera frame → R_wc[0] = I → R_wb[0] = I * R_ci = R_ci.
  // Each subsequent frame chains the IMU delta_q from the PREVIOUS frame's pim,
  // matching the same convention used in Initializer::addFrame().
  // This is far more reliable than SfM R_wc during the low-parallax hover phase.
  std::vector<Eigen::Matrix3d> R_wb_imu(N);
  R_wb_imu[0] = R_wc[0] * R_ci;
  for (int i = 1; i < N; ++i)
    R_wb_imu[i] = R_wb_imu[i-1] * frames_[i-1].pim.delta_q.toRotationMatrix();

  // ── Stage 1: Gravity from near-static IMU samples ─────────────────────────
  // Filter: acc norm ≈ 9.81 AND low gyro (not rotating).
  // Three-tier fallback so excited sequences still produce a gravity estimate.
  auto estimateGravity = [&](double acc_tol, double gyr_tol, bool use_gyr) {
    Eigen::Vector3d gs = Eigen::Vector3d::Zero(); int gc = 0;
    for (int i = 0; i < N; ++i) {
      for (const auto& m : frames_[i].pim.measurements()) {
        if (std::abs(m.acc.norm() - 9.81) > acc_tol) continue;
        if (use_gyr && m.gyr.norm() > gyr_tol)        continue;
        gs += -(R_wb_imu[i] * m.acc);
        ++gc;
      }
    }
    return std::make_pair(gs, gc);
  };

  Eigen::Vector3d g_sum; int g_count;

  // Tier 1: tight acc + gyro filter (most reliable)
  std::tie(g_sum, g_count) = estimateGravity(1.0, 0.3, true);

  if (g_count < 10) {
    // Tier 2: relax acc tolerance, keep gyro filter
    std::tie(g_sum, g_count) = estimateGravity(1.5, 0.3, true);
    if (g_count >= 10)
      RCLCPP_WARN(rclcpp::get_logger("vi_align"), "  gravity: fell back to tier2 (relaxed acc)");
  }

  if (g_count < 10) {
    // Tier 3: drop gyro filter entirely (highly excited sequence)
    std::tie(g_sum, g_count) = estimateGravity(2.0, 0.0, false);
    if (g_count >= 10)
      RCLCPP_WARN(rclcpp::get_logger("vi_align"), "  gravity: fell back to tier3 (no gyro filter)");
  }

  if (g_count < 10) return fail("too_few_acc");

  const Eigen::Vector3d g_raw = g_sum / g_count;
  if (g_raw.norm() < 1.0) return fail("gravity_direction_degenerate");
  result.gravity_w = g_raw.normalized() * 9.81;
  const Eigen::Vector3d& g = result.gravity_w;

  // ── Stage 2: Velocities (gravity known) ───────────────────────────────────
  // Use R_wb_imu instead of R_wc[i]*R_ci
  {
    Eigen::MatrixXd Av = Eigen::MatrixXd::Zero(3*(N-1), 3*N);
    Eigen::VectorXd bv = Eigen::VectorXd::Zero(3*(N-1));
    for (int i = 0; i < N-1; ++i) {
      const ImuIntegrator& pim = frames_[i+1].pim;
      const double dt = pim.sum_dt;
      if (dt < 1e-6) return fail("zero_dt_vel_" + std::to_string(i));
      Av.block<3,3>(3*i, 3*i)     = -Eigen::Matrix3d::Identity();
      Av.block<3,3>(3*i, 3*(i+1)) =  Eigen::Matrix3d::Identity();
      bv.segment<3>(3*i)           =  R_wb_imu[i] * pim.delta_v + g * dt;  // ← R_wb_imu
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_v(Av,
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd xv = svd_v.solve(bv);
    result.velocities.resize(N);
    for (int i = 0; i < N; ++i) result.velocities[i] = xv.segment<3>(3*i);
  }

  // ── Stage 3: Scale (robust median with IQR outlier rejection) ─────────────
  std::vector<double> sfm_disps(N-1);
  for (int i = 0; i < N-1; ++i)
    sfm_disps[i] = (t_wc[i] - t_wc[i+1]).norm();

  std::vector<double> sorted_d = sfm_disps;
  std::sort(sorted_d.begin(), sorted_d.end());
  const double med_d = sorted_d[sorted_d.size()/2];
  const double min_disp = std::max(0.001, 0.02 * med_d);

  std::vector<double> scale_ests;
  scale_ests.reserve(N-1);
  for (int i = 0; i < N-1; ++i) {
    if (sfm_disps[i] < min_disp)       continue;
    if (sfm_disps[i] > 50.0 * med_d)   continue;

    const ImuIntegrator& pim = frames_[i+1].pim;
    const double dt  = pim.sum_dt;
    const double dt2 = dt * dt;
    if (dt < 1e-6) continue;

    const Eigen::Vector3d col  = t_wc[i] - t_wc[i+1];
    const double          col2 = col.squaredNorm();
    if (col2 < 1e-10) continue;

    const Eigen::Vector3d rhs =
        -(R_wb_imu[i] * pim.delta_p)                        // ← R_wb_imu
        + (result.velocities[i] - result.velocities[i+1]) * dt
        + 0.5 * g * dt2;

    const double s_i = col.dot(rhs) / col2;
    if (std::isfinite(s_i) && s_i > 0.0)
      scale_ests.push_back(s_i);
  }

  // IQR outlier rejection — removes the 46.862-type spikes before taking median
  if ((int)scale_ests.size() >= 4) {
    std::sort(scale_ests.begin(), scale_ests.end());
    const double q1  = scale_ests[scale_ests.size() / 4];
    const double q3  = scale_ests[3 * scale_ests.size() / 4];
    const double iqr = q3 - q1;
    scale_ests.erase(
      std::remove_if(scale_ests.begin(), scale_ests.end(),
        [q1, q3, iqr](double s){
          return s < q1 - 1.5 * iqr || s > q3 + 1.5 * iqr;
        }),
      scale_ests.end());
    std::sort(scale_ests.begin(), scale_ests.end());
  }

  std::sort(scale_ests.begin(), scale_ests.end());

  RCLCPP_INFO(rclcpp::get_logger("vi_align"),
    "  g=[%.2f,%.2f,%.2f] n_scale_valid=%d/%d med_disp=%.4f",
    g.x(), g.y(), g.z(), (int)scale_ests.size(), N-1, med_d);

  const int min_needed = std::max(2, (N-1) / 7);
  if ((int)scale_ests.size() < min_needed)
    return fail("too_few_scale_ests_" + std::to_string(scale_ests.size()));

  result.scale = scale_ests[scale_ests.size() / 2];

  RCLCPP_INFO(rclcpp::get_logger("vi_align"),
    "  scale estimates: min=%.3f median=%.3f max=%.3f",
    scale_ests.front(), result.scale, scale_ests.back());

  if (result.scale < 0.2 || result.scale > 3.0)
    return fail("bad_scale_" + std::to_string(result.scale));

  result.ba = Eigen::Vector3d::Zero();
  result.bg = Eigen::Vector3d::Zero();
  return true;
}

Eigen::Matrix3d Initializer::gravityAlignRotation(const Eigen::Vector3d& g_raw) {
  const Eigen::Vector3d g_hat    = g_raw.normalized();
  const Eigen::Vector3d g_target = Eigen::Vector3d(0.0, 0.0, -1.0);
  const Eigen::Vector3d axis     = g_hat.cross(g_target);
  const double sin_a = axis.norm();
  const double cos_a = g_hat.dot(g_target);

  RCLCPP_INFO(rclcpp::get_logger("vi_align"),
    "  gravity_align: g_raw=[%.2f,%.2f,%.2f] angle=%.1f deg",
    g_raw.x(), g_raw.y(), g_raw.z(),
    std::atan2(sin_a, cos_a) * 57.3);

  if (sin_a < 1e-8) {
    if (cos_a > 0) return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
  }
  return Eigen::AngleAxisd(std::atan2(sin_a,cos_a), axis/sin_a).toRotationMatrix();
}

} // namespace mono_vio