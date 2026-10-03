#pragma once
#include "mono_vio/imu_integrator.hpp"
#include "mono_vio/sliding_window.hpp"
#include "mono_vio/marginalization.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <memory>
#include <vector>
#include <cmath>

namespace mono_vio {

// ── Camera intrinsics ─────────────────────────────────────────────────────────
struct CameraIntrinsics {
  double fx{0}, fy{0}, cx{0}, cy{0};
  double k1{0}, k2{0}, p1{0}, p2{0};
  int    width{752}, height{480};
};

// ── Free helpers ──────────────────────────────────────────────────────────────
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S <<     0, -v.z(),  v.y(),
       v.z(),      0, -v.x(),
      -v.y(),  v.x(),      0;
  return S;
}

// ── PoseLocalParameterization ─────────────────────────────────────────────────
/**
 * Local parameterisation for the 7-element pose block [tx,ty,tz, qx,qy,qz,qw].
 *
 * GlobalSize = 7  (tx,ty,tz, qx,qy,qz,qw)
 * LocalSize  = 6  (delta_t ∈ ℝ³,  delta_theta ∈ ℝ³)
 *
 * Update rule:
 *   t_new = t + delta_t
 *   q_new = q * Exp(delta_theta)          (right perturbation)
 *
 * Quaternion storage: [qx, qy, qz, qw]   (Eigen Map convention)
 */
class PoseLocalParameterization : public ceres::LocalParameterization {
public:
  bool Plus(const double* x, const double* delta,
            double* x_plus_delta) const override;
  bool ComputeJacobian(const double* x, double* jacobian) const override;
  int  GlobalSize() const override { return 7; }
  int  LocalSize()  const override { return 6; }
};

// ── ReprojectionFactor ────────────────────────────────────────────────────────
/**
 * 2-D reprojection residual with analytic Jacobians.
 * params[0] = pose      [tx,ty,tz, qx,qy,qz,qw]  world←body  (7)
 * params[1] = landmark  [x,y,z] world frame                   (3)
 */
class ReprojectionFactor : public ceres::SizedCostFunction<2, 7, 3> {
public:
  /**
   * @param observed_uv_norm  Undistorted normalised bearing [x/z, y/z] (NOT pixels).
   * @param K                 Camera intrinsics (used only for T_cam_imu extraction).
   * @param T_cam_imu         Extrinsic transform camera←IMU (4×4).
   *
   * Residual is in NORMALISED camera coordinates.
   * Set HuberLoss scale ≈ 1.5/focal_length (e.g. 0.003 for f≈460).
   */
  ReprojectionFactor(const Eigen::Vector2d&  observed_uv_norm,
                     const CameraIntrinsics& K,
                     const Eigen::Matrix4d&  T_cam_imu);

  bool Evaluate(double const* const* params,
                double* residuals, double** jacobians) const override;
private:
  Eigen::Vector2d obs_;   ///< normalised camera coords [x/z, y/z]
  Eigen::Matrix3d R_ci_;
  Eigen::Vector3d t_ci_;
};

// ── ImuFactor ─────────────────────────────────────────────────────────────────
/**
 * 15-D IMU preintegration residual.
 * params[0]=pose_i(7)  params[1]=vel_i(3)  params[2]=bias_i(6)
 * params[3]=pose_j(7)  params[4]=vel_j(3)  params[5]=bias_j(6)
 * Residual: [r_p(3), r_v(3), r_q(3), r_ba(3), r_bg(3)]
 */
class ImuFactor : public ceres::SizedCostFunction<15, 7, 3, 6, 7, 3, 6> {
public:
  ImuFactor(const ImuIntegrator& preint, const Eigen::Vector3d& gravity_world);
  bool Evaluate(double const* const* params,
                double* residuals, double** jacobians) const override;
private:
  ImuIntegrator   preint_;
  Eigen::Vector3d gravity_;
};

// ── Optimizer ─────────────────────────────────────────────────────────────────
class Optimizer {
public:
  struct Params {
    int    max_iterations  {8};
    int    num_threads     {4};
    double max_solver_time {0.04};
    double huber_loss_scale{0.003};  ///< normalised units: ~1.5px / 460px focal
    bool   use_imu_factors    {true};  ///< add IMU preintegration cost to problem
    bool   fix_first_vel_bias {true};  ///< fix vel+bias of oldest frame (needs use_imu_factors)
    double imu_weight         {0.001}; ///< ScaledLoss weight balancing IMU vs visual cost
  };

  Optimizer(const Params&          params,
            const CameraIntrinsics& K,
            const Eigen::Matrix4d&  T_cam_imu,
            const Eigen::Vector3d&  gravity_world);

  bool optimize(SlidingWindow&                         window,
                std::shared_ptr<MarginalizationFactor> marg_factor,
                const std::vector<double*>&            marg_params);

private:
  Params           params_;
  CameraIntrinsics K_;
  Eigen::Matrix4d  T_cam_imu_;
  Eigen::Vector3d  gravity_;
};

}  // namespace mono_vio