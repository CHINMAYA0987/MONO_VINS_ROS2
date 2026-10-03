#include "mono_vio/optimizer.hpp"
#include <Eigen/Geometry>
#include <cmath>

namespace mono_vio {

// ── PoseLocalParameterization ─────────────────────────────────────────────────
bool PoseLocalParameterization::Plus(
    const double* x, const double* delta, double* xpd) const {
  xpd[0] = x[0] + delta[0];
  xpd[1] = x[1] + delta[1];
  xpd[2] = x[2] + delta[2];

  const Eigen::Map<const Eigen::Quaterniond> q(x + 3);
  const Eigen::Vector3d dtheta(delta[3], delta[4], delta[5]);
  const double angle = dtheta.norm();
  Eigen::Quaterniond dq;
  if (angle < 1e-8)
    dq = Eigen::Quaterniond(1.0, 0.5*dtheta.x(), 0.5*dtheta.y(), 0.5*dtheta.z());
  else
    dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, dtheta / angle));

  Eigen::Quaterniond q_new = (q * dq).normalized();
  xpd[3] = q_new.x(); xpd[4] = q_new.y();
  xpd[5] = q_new.z(); xpd[6] = q_new.w();
  return true;
}

bool PoseLocalParameterization::ComputeJacobian(
    const double* /*x*/, double* jacobian) const {
  Eigen::Map<Eigen::Matrix<double,7,6,Eigen::RowMajor>> J(jacobian);
  J.setZero();
  J.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
  J(3,3) = 0.5; J(4,4) = 0.5; J(5,5) = 0.5;
  return true;
}

// ── ReprojectionFactor ────────────────────────────────────────────────────────
ReprojectionFactor::ReprojectionFactor(const Eigen::Vector2d& obs,
                                       const CameraIntrinsics& K,
                                       const Eigen::Matrix4d& T_cam_imu)
    : obs_(obs) {
  (void)K;  // intrinsics not needed for normalised-coord residual
  R_ci_ = T_cam_imu.topLeftCorner<3,3>();
  t_ci_ = T_cam_imu.topRightCorner<3,1>();
}

bool ReprojectionFactor::Evaluate(double const* const* p,
                                   double* res, double** jac) const {
  const Eigen::Map<const Eigen::Vector3d>    t_wb(p[0]);
  const Eigen::Map<const Eigen::Quaterniond> q_wb(p[0] + 3);
  const Eigen::Matrix3d R_wb = q_wb.normalized().toRotationMatrix();
  const Eigen::Map<const Eigen::Vector3d> pw(p[1]);

  const Eigen::Vector3d pb = R_wb.transpose() * (pw - t_wb);
  const Eigen::Vector3d pc = R_ci_ * pb + t_ci_;
  const double x = pc.x(), y = pc.y(), z = pc.z();

  if (z < 0.05) {
    res[0] = res[1] = 0.0;
    if (jac) {
      if (jac[0]) Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>>(jac[0]).setZero();
      if (jac[1]) Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>>(jac[1]).setZero();
    }
    return true;
  }

  // Residual in NORMALISED camera coordinates: res = [x/z - obs_x, y/z - obs_y]
  // obs_ holds uv_norm (undistorted normalised bearing), NOT pixels.
  // Huber loss threshold must be set in normalised units (~1px / focal ≈ 0.003).
  const double iz = 1.0/z, iz2 = iz*iz;
  res[0] = x * iz - obs_.x();
  res[1] = y * iz - obs_.y();

  if (!jac) return true;

  Eigen::Matrix<double,2,3> Jp;
  Jp << iz,  0, -x*iz2,
         0, iz, -y*iz2;
  const Eigen::Matrix<double,2,3> Jpc = Jp * R_ci_;

  if (jac[0]) {
    Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> J0(jac[0]);
    J0.setZero();
    J0.block<2,3>(0,0) = Jpc * (-R_wb.transpose());
    J0.block<2,3>(0,3) = Jpc * skewSymmetric(pb);
  }
  if (jac[1]) {
    Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>> J1(jac[1]);
    J1 = Jpc * R_wb.transpose();
  }
  return true;
}

// ── ImuFactor ─────────────────────────────────────────────────────────────────
ImuFactor::ImuFactor(const ImuIntegrator& preint,
                     const Eigen::Vector3d& gravity_world)
    : preint_(preint), gravity_(gravity_world) {}

bool ImuFactor::Evaluate(double const* const* p,
                          double* res, double** jac) const {
  const Eigen::Map<const Eigen::Vector3d>    pi(p[0]);
  const Eigen::Map<const Eigen::Quaterniond> qi(p[0]+3);
  const Eigen::Map<const Eigen::Vector3d>    vi(p[1]);
  const Eigen::Map<const Eigen::Vector3d>    bai(p[2]);
  const Eigen::Map<const Eigen::Vector3d>    bgi(p[2]+3);
  const Eigen::Map<const Eigen::Vector3d>    pj(p[3]);
  const Eigen::Map<const Eigen::Quaterniond> qj(p[3]+3);
  const Eigen::Map<const Eigen::Vector3d>    vj(p[4]);
  const Eigen::Map<const Eigen::Vector3d>    baj(p[5]);
  const Eigen::Map<const Eigen::Vector3d>    bgj(p[5]+3);

  ImuIntegrator pim = preint_;
  pim.correctBias(bai, bgi);

  const double dt  = preint_.sum_dt;
  const double dt2 = dt * dt;
  const Eigen::Matrix3d Ri  = qi.normalized().toRotationMatrix();
  const Eigen::Matrix3d Rj  = qj.normalized().toRotationMatrix();
  const Eigen::Matrix3d RiT = Ri.transpose();

  const Eigen::Vector3d rp =
      RiT * (pj - pi - vi*dt - 0.5*gravity_*dt2) - pim.delta_p;
  const Eigen::Vector3d rv =
      RiT * (vj - vi - gravity_*dt) - pim.delta_v;

  const Eigen::Matrix3d dR = pim.delta_q.toRotationMatrix();
  Eigen::AngleAxisd aa(dR.transpose() * RiT * Rj);
  const Eigen::Vector3d rq = aa.axis() * aa.angle();

  const Eigen::Vector3d rba = baj - bai;
  const Eigen::Vector3d rbg = bgj - bgi;

  Eigen::Map<Eigen::Matrix<double,15,1>> r(res);
  r.segment<3>(0)  = rp;
  r.segment<3>(3)  = rv;
  r.segment<3>(6)  = rq;
  r.segment<3>(9)  = rba;
  r.segment<3>(12) = rbg;

  if (!jac) return true;

  const Eigen::Vector3d tmp_p = RiT*(pj-pi-vi*dt-0.5*gravity_*dt2);
  const Eigen::Vector3d tmp_v = RiT*(vj-vi-gravity_*dt);

  if (jac[0]) {
    Eigen::Map<Eigen::Matrix<double,15,7,Eigen::RowMajor>> J(jac[0]); J.setZero();
    J.block<3,3>(0,0) = -RiT;
    J.block<3,3>(0,3) =  skewSymmetric(tmp_p);
    J.block<3,3>(3,3) =  skewSymmetric(tmp_v);
    J.block<3,3>(6,3) = -Eigen::Matrix3d::Identity();
  }
  if (jac[1]) {
    Eigen::Map<Eigen::Matrix<double,15,3,Eigen::RowMajor>> J(jac[1]); J.setZero();
    J.block<3,3>(0,0) = -RiT * dt;
    J.block<3,3>(3,0) = -RiT;
  }
  if (jac[2]) {
    Eigen::Map<Eigen::Matrix<double,15,6,Eigen::RowMajor>> J(jac[2]); J.setZero();
    J.block<3,3>(0,0) = -preint_.dp_dba;
    J.block<3,3>(0,3) = -preint_.dp_dbg;
    J.block<3,3>(3,0) = -preint_.dv_dba;
    J.block<3,3>(3,3) = -preint_.dv_dbg;
    J.block<3,3>(6,3) = -preint_.dq_dbg;
    J.block<3,3>(9,0) = -Eigen::Matrix3d::Identity();
    J.block<3,3>(12,3)= -Eigen::Matrix3d::Identity();
  }
  if (jac[3]) {
    Eigen::Map<Eigen::Matrix<double,15,7,Eigen::RowMajor>> J(jac[3]); J.setZero();
    J.block<3,3>(0,0) = RiT;
    J.block<3,3>(6,3) = Eigen::Matrix3d::Identity();
  }
  if (jac[4]) {
    Eigen::Map<Eigen::Matrix<double,15,3,Eigen::RowMajor>> J(jac[4]); J.setZero();
    J.block<3,3>(3,0) = RiT;
  }
  if (jac[5]) {
    Eigen::Map<Eigen::Matrix<double,15,6,Eigen::RowMajor>> J(jac[5]); J.setZero();
    J.block<3,3>(9,0)  = Eigen::Matrix3d::Identity();
    J.block<3,3>(12,3) = Eigen::Matrix3d::Identity();
  }
  return true;
}

// ── Optimizer ─────────────────────────────────────────────────────────────────
Optimizer::Optimizer(const Params& p, const CameraIntrinsics& K,
                     const Eigen::Matrix4d& T_cam_imu,
                     const Eigen::Vector3d& gravity)
    : params_(p), K_(K), T_cam_imu_(T_cam_imu), gravity_(gravity) {}

bool Optimizer::optimize(SlidingWindow& window,
                          std::shared_ptr<MarginalizationFactor> mf,
                          const std::vector<double*>& mp) {
  if (window.size() < 2) return false;

  ceres::Problem problem;
  auto* pose_param = new PoseLocalParameterization();

  auto& frames = window.frames();
  auto& lms    = window.landmarks();
  auto& obs    = window.observations();

  // When use_imu_factors=false (visual-only BA): only add pose blocks.
  // Velocities and biases are not Ceres variables — they're computed kinematically.
  // When use_imu_factors=true: add all blocks and fix oldest frame completely.
  for (auto& kf : frames) {
    problem.AddParameterBlock(kf.pose_params, 7, pose_param);
    if (params_.use_imu_factors) {
      problem.AddParameterBlock(kf.vel_params,  3);
      problem.AddParameterBlock(kf.bias_params, 6);
    }
  }
  if (!frames.empty()) {
    problem.SetParameterBlockConstant(frames[0].pose_params);
    if (params_.use_imu_factors && params_.fix_first_vel_bias) {
      problem.SetParameterBlockConstant(frames[0].vel_params);
      problem.SetParameterBlockConstant(frames[0].bias_params);
    }
  }

  // Landmark blocks
  for (auto& [id, lm] : lms)
    if (lm.initialized) problem.AddParameterBlock(lm.params, 3);

  // ── Marginalization prior (wired in) ────────────────────────────────────
  if (mf && !mp.empty()) {
    // Ceres takes raw non-owning pointer; shared_ptr keeps it alive
    problem.AddResidualBlock(mf.get(), nullptr, mp);
  }

  // Reprojection residuals
  auto* huber = new ceres::HuberLoss(params_.huber_loss_scale);
  for (auto& o : obs) {
    if (o.frame_idx < 0 || o.frame_idx >= (int)frames.size()) continue;
    auto it = lms.find(o.landmark_id);
    if (it == lms.end() || !it->second.initialized) continue;
    problem.AddResidualBlock(
        new ReprojectionFactor(o.pixel, K_, T_cam_imu_),
        huber,
        frames[o.frame_idx].pose_params,
        it->second.params);
  }

  // IMU residuals with ScaledLoss to balance against visual cost.
  // Visual residuals: ~0.003 (normalised), IMU residuals: ~0.1 m/s.
  // ScaledLoss(w) multiplies cost by w. With w=0.001: IMU cost ≈ visual cost.
  // Without this, v→0 collapse: visual-only BA has a degenerate fixed-point
  // where stationary robot + moved landmarks gives zero reprojection error.
  if (params_.use_imu_factors) {
    for (int i = 0; i+1 < (int)frames.size(); ++i) {
      if (!frames[i+1].preint) continue;
      auto* scaled = new ceres::ScaledLoss(
          nullptr, params_.imu_weight, ceres::TAKE_OWNERSHIP);
      problem.AddResidualBlock(
          new ImuFactor(*frames[i+1].preint, gravity_), scaled,
          frames[i].pose_params,   frames[i].vel_params,   frames[i].bias_params,
          frames[i+1].pose_params, frames[i+1].vel_params, frames[i+1].bias_params);
    }
  }

  ceres::Solver::Options opts;
  opts.linear_solver_type           = ceres::DENSE_SCHUR;
  opts.trust_region_strategy_type   = ceres::LEVENBERG_MARQUARDT;
  opts.max_num_iterations           = params_.max_iterations;
  opts.num_threads                  = params_.num_threads;
  opts.max_solver_time_in_seconds   = params_.max_solver_time;
  opts.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);
  return summary.termination_type == ceres::CONVERGENCE ||
         summary.termination_type == ceres::USER_SUCCESS;
}

}  // namespace mono_vio