#include "mono_vio/imu_integrator.hpp"
#include <cmath>

namespace mono_vio {

// ── helpers ──────────────────────────────────────────────────────────────────

Eigen::Matrix3d ImuIntegrator::skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S <<    0, -v(2),  v(1),
       v(2),     0, -v(0),
      -v(1),  v(0),     0;
  return S;
}

// Right Jacobian of SO(3): J_r(phi) such that Exp(phi+dphi)≈Exp(phi)*Exp(J_r*dphi)
Eigen::Matrix3d ImuIntegrator::rightJacobian(const Eigen::Vector3d& phi) {
  double angle = phi.norm();
  if (angle < 1e-8) return Eigen::Matrix3d::Identity();
  Eigen::Matrix3d K = skew(phi) / angle;
  return Eigen::Matrix3d::Identity()
       - (1.0 - std::cos(angle)) / angle * K
       + (1.0 - std::sin(angle) / angle) * K * K;
}

// ── ImuIntegrator ─────────────────────────────────────────────────────────────

ImuIntegrator::ImuIntegrator(double acc_n, double gyr_n,
                               double acc_w, double gyr_w)
  : acc_n_(acc_n), gyr_n_(gyr_n), acc_w_(acc_w), gyr_w_(gyr_w)
{
  Q_.setZero();
  Q_.block<3,3>(0,0) = acc_n_ * acc_n_ * Eigen::Matrix3d::Identity();
  Q_.block<3,3>(3,3) = gyr_n_ * gyr_n_ * Eigen::Matrix3d::Identity();
}

void ImuIntegrator::reset(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg) {
  delta_p.setZero();
  delta_v.setZero();
  delta_q = Eigen::Quaterniond::Identity();
  sum_dt  = 0.0;

  linearized_ba = ba;
  linearized_bg = bg;

  dp_dba.setZero();  dp_dbg.setZero();
  dv_dba.setZero();  dv_dbg.setZero();
  dq_dbg.setZero();

  covariance.setZero();
  meas_.clear();
}

void ImuIntegrator::integrate(const Eigen::Vector3d& acc_raw,
                               const Eigen::Vector3d& gyr_raw,
                               double dt) {
  meas_.push_back({sum_dt, acc_raw, gyr_raw});

  // De-bias
  const Eigen::Vector3d acc = acc_raw - linearized_ba;
  const Eigen::Vector3d gyr = gyr_raw - linearized_bg;

  // Current rotation matrix
  const Eigen::Matrix3d R = delta_q.toRotationMatrix();

  // Propagate rotation (midpoint: use gyr as constant over dt)
  const Eigen::Vector3d dtheta = gyr * dt;
  const double          angle  = dtheta.norm();
  Eigen::Quaterniond dq;
  if (angle < 1e-8)
    dq = Eigen::Quaterniond(1.0, 0.5*dtheta.x(), 0.5*dtheta.y(), 0.5*dtheta.z());
  else
    dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, dtheta/angle));

  const Eigen::Quaterniond q_new = (delta_q * dq).normalized();
  const Eigen::Matrix3d    R_new = q_new.toRotationMatrix();
  const Eigen::Matrix3d    R_mid = 0.5 * (R + R_new);

  // Propagate position and velocity
  delta_p += delta_v * dt + 0.5 * (R_mid * acc) * dt * dt;
  delta_v += (R_mid * acc) * dt;
  delta_q  = q_new;
  sum_dt  += dt;

  // ── Jacobian propagation ─────────────────────────────────────────────────
  // F: 9×9 discrete state transition (θ, p, v)
  // G: 9×6 noise input matrix
  // States ordered as [θ(3), p(3), v(3)]

  const Eigen::Matrix3d Rdt = R_mid * dt;
  const Eigen::Matrix3d acc_skew = skew(acc);

  // ∂Δθ/∂bg update: dq_dbg = (I - skew(dtheta)) * dq_dbg - Jr*dt
  dq_dbg = (Eigen::Matrix3d::Identity() - skew(dtheta)) * dq_dbg
           - rightJacobian(dtheta) * dt;

  // ∂Δp/∂ba and ∂Δp/∂bg
  dp_dba += dv_dba * dt - 0.5 * Rdt * dt;
  dp_dbg += dv_dbg * dt + 0.5 * R_mid * acc_skew * dq_dbg * dt * dt;

  // ∂Δv/∂ba and ∂Δv/∂bg
  dv_dba += -Rdt;
  dv_dbg += R_mid * acc_skew * dq_dbg * dt;

  // ── Covariance propagation (15×15) ───────────────────────────────────────
  // Discrete F_d (15×15) — simplified first-order form
  Eigen::Matrix<double,15,15> F = Eigen::Matrix<double,15,15>::Identity();
  // θ block: F[0:3, 0:3] = I - skew(dtheta)
  F.block<3,3>(0,0) = Eigen::Matrix3d::Identity() - skew(dtheta);
  // p block: F[3:6, 6:9] = R_mid*dt
  F.block<3,3>(3,6) = R_mid * dt;
  // v block: F[6:9, 0:3] = -R_mid * skew(acc) * dt
  F.block<3,3>(6,0) = -R_mid * acc_skew * dt;

  // Discrete G_d (15×6)
  Eigen::Matrix<double,15,6> G = Eigen::Matrix<double,15,6>::Zero();
  G.block<3,3>(0,3) = -rightJacobian(dtheta) * dt;
  G.block<3,3>(6,0) = -R_mid * dt;
  // Bias random walk
  G.block<3,3>(9,0)  = Eigen::Matrix3d::Identity() * dt;
  G.block<3,3>(12,3) = Eigen::Matrix3d::Identity() * dt;

  // Full 6×6 Q with bias walk noise
  Eigen::Matrix<double,6,6> Q_full = Eigen::Matrix<double,6,6>::Zero();
  Q_full.block<3,3>(0,0) = Q_.block<3,3>(0,0);
  Q_full.block<3,3>(3,3) = Q_.block<3,3>(3,3);

  covariance = F * covariance * F.transpose()
             + G * Q_full * G.transpose();
}

void ImuIntegrator::correctBias(const Eigen::Vector3d& ba_new,
                                 const Eigen::Vector3d& bg_new) {
  const Eigen::Vector3d dba = ba_new - linearized_ba;
  const Eigen::Vector3d dbg = bg_new - linearized_bg;

  delta_p += dp_dba * dba + dp_dbg * dbg;
  delta_v += dv_dba * dba + dv_dbg * dbg;

  const Eigen::Vector3d dtheta = dq_dbg * dbg;
  const double angle = dtheta.norm();
  Eigen::Quaterniond dq;
  if (angle < 1e-8)
    dq = Eigen::Quaterniond(1.0, 0.5*dtheta.x(), 0.5*dtheta.y(), 0.5*dtheta.z());
  else
    dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, dtheta/angle));
  delta_q = (delta_q * dq).normalized();
}

}  // namespace mono_vio