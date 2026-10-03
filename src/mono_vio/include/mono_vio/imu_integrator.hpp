#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace mono_vio {

struct ImuMeasurement {
  double timestamp;
  Eigen::Vector3d acc;
  Eigen::Vector3d gyr;
};

/**
 * ImuIntegrator — preintegrates IMU samples and holds the full result.
 *
 * All preintegrated state is stored as public members so that ImuFactor
 * can copy the integrator by value and access fields directly without
 * a separate PreintegratedImu wrapper.
 *
 * Field naming matches VINS-Mono convention used in ImuFactor:
 *   sum_dt          — total integrated time [s]
 *   delta_p/v/q     — integrated position/velocity/rotation increments
 *   linearized_ba/bg— bias at the linearisation point
 *   dp_dba, dp_dbg  — ∂Δp/∂ba,  ∂Δp/∂bg  (3×3 each)
 *   dv_dba, dv_dbg  — ∂Δv/∂ba,  ∂Δv/∂bg
 *   dq_dbg          — ∂Δq/∂bg   (tangent space, 3×3)
 *   covariance      — 15×15 full covariance [θ,p,v,ba,bg]
 */
class ImuIntegrator {
public:
  // ── Preintegrated state ─────────────────────────────────────────────────
  Eigen::Vector3d    delta_p   { Eigen::Vector3d::Zero()         };
  Eigen::Vector3d    delta_v   { Eigen::Vector3d::Zero()         };
  Eigen::Quaterniond delta_q   { Eigen::Quaterniond::Identity()  };
  double             sum_dt    { 0.0                             };

  // Linearisation-point biases (set by reset())
  Eigen::Vector3d linearized_ba { Eigen::Vector3d::Zero() };
  Eigen::Vector3d linearized_bg { Eigen::Vector3d::Zero() };

  // ── Bias Jacobians (3×3) ────────────────────────────────────────────────
  Eigen::Matrix3d dp_dba { Eigen::Matrix3d::Zero() };
  Eigen::Matrix3d dp_dbg { Eigen::Matrix3d::Zero() };
  Eigen::Matrix3d dv_dba { Eigen::Matrix3d::Zero() };
  Eigen::Matrix3d dv_dbg { Eigen::Matrix3d::Zero() };
  Eigen::Matrix3d dq_dbg { Eigen::Matrix3d::Zero() };

  // ── Full covariance (15×15) ─────────────────────────────────────────────
  Eigen::Matrix<double,15,15> covariance {
      Eigen::Matrix<double,15,15>::Zero() };

  // ── Constructor ─────────────────────────────────────────────────────────
  /** Default constructor — noise params set to zero; call reset() before use. */
  ImuIntegrator() : ImuIntegrator(0.01, 0.001, 0.0001, 0.00001) {}

  ImuIntegrator(double acc_noise_density,
                double gyr_noise_density,
                double acc_random_walk,
                double gyr_random_walk);

  /** Reset accumulated state to identity and set new linearisation biases. */
  void reset(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg);

  /** Feed one IMU sample; dt is the integration step [s]. */
  void integrate(const Eigen::Vector3d& acc_raw,
                 const Eigen::Vector3d& gyr_raw,
                 double dt);

  /**
   * First-order bias correction after optimiser updates biases.
   * Updates delta_p, delta_v, delta_q in-place.
   */
  void correctBias(const Eigen::Vector3d& ba_new,
                   const Eigen::Vector3d& bg_new);

  /** Raw measurements (for IMU excitation check). */
  const std::vector<ImuMeasurement>& measurements() const { return meas_; }

private:
  double acc_n_, gyr_n_, acc_w_, gyr_w_;
  std::vector<ImuMeasurement> meas_;

  // 6×6 continuous-time noise matrix Q = diag(acc_n², gyr_n²)
  Eigen::Matrix<double,6,6> Q_;

  static Eigen::Matrix3d skew(const Eigen::Vector3d& v);
  static Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& phi);
};

// Backward-compat alias so Initializer / SlidingWindow keep compiling.
using PreintegratedImu = ImuIntegrator;

}  // namespace mono_vio