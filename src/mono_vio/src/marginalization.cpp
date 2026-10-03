#include "mono_vio/marginalization.hpp"
#include <Eigen/SVD>
#include <stdexcept>

namespace mono_vio {

// ── MarginalizationInfo ───────────────────────────────────────────────────────

void MarginalizationInfo::addBlock(const Eigen::VectorXd& residual,
                                    const Eigen::MatrixXd& J_marg,
                                    const Eigen::MatrixXd& J_remain) {
  int res_dim    = (int)residual.size();
  int marg_dim   = (int)J_marg.cols();
  int remain_dim = (int)J_remain.cols();

  if (J_full_.cols() == 0) {
    // First call: allocate.
    J_full_ = Eigen::MatrixXd::Zero(0, marg_dim + remain_dim);
    r_full_ = Eigen::VectorXd::Zero(0);
  }

  // Stack vertically.
  int old_rows = (int)J_full_.rows();
  J_full_.conservativeResize(old_rows + res_dim, Eigen::NoChange);
  r_full_.conservativeResize(old_rows + res_dim);

  J_full_.block(old_rows, 0,        res_dim, marg_dim)   = J_marg;
  J_full_.block(old_rows, marg_dim, res_dim, remain_dim) = J_remain;
  r_full_.segment(old_rows, res_dim) = residual;
}

void MarginalizationInfo::marginalise() {
  if (marginalised_) throw std::logic_error("already marginalised");
  if (J_full_.rows() == 0) return;

  // Determine split.
  // We assume columns [0..marg_dim) belong to the marginalised block;
  // the split must have been set up by the caller via addBlock().
  // Here we just apply the Schur complement symbolically.

  int full_cols = (int)J_full_.cols();
  // We can't know the marg/remain split without external info.
  // For simplicity, use the stored lin_point_ size as the remain_dim.
  int remain_dim = (int)lin_point_.size();
  int marg_dim   = full_cols - remain_dim;
  if (marg_dim <= 0 || remain_dim <= 0) {
    // Degenerate; skip prior.
    J_prior_.resize(0, remain_dim);
    r_prior_.resize(0);
    marginalised_ = true;
    return;
  }

  Eigen::MatrixXd Jm = J_full_.leftCols(marg_dim);
  Eigen::MatrixXd Jr = J_full_.rightCols(remain_dim);

  // Schur complement: eliminate Jm using its pseudo-inverse.
  // J_prior = (I - Jm * Jm^+) * Jr  (projection onto null space of Jm)
  // r_prior = (I - Jm * Jm^+) * r_full
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jm,
    Eigen::ComputeThinU | Eigen::ComputeThinV);

  // Threshold small singular values.
  double eps = 1e-8 * svd.singularValues()(0);
  Eigen::VectorXd sv_inv = svd.singularValues().unaryExpr(
    [eps](double s){ return s > eps ? 1.0/s : 0.0; });

  // Pseudo-inverse of Jm: V * diag(sv_inv) * U^T
  Eigen::MatrixXd Jm_pinv = svd.matrixV()
                           * sv_inv.asDiagonal()
                           * svd.matrixU().transpose();

  Eigen::MatrixXd P = Eigen::MatrixXd::Identity(J_full_.rows(), J_full_.rows())
                    - Jm * Jm_pinv;

  J_prior_ = P * Jr;
  r_prior_ = P * r_full_;

  // Thin the prior: keep only rows with significant information.
  // (optional: SVD on J_prior_ to remove rank-deficient rows)

  marginalised_ = true;
}

// ── MarginalizationFactor ─────────────────────────────────────────────────────

MarginalizationFactor::MarginalizationFactor(const MarginalizationInfo& info)
  : info_(info) {
  // Set cost function sizes.
  set_num_residuals(info.residualDim());
  mutable_parameter_block_sizes()->push_back(info.remainDim());
}

bool MarginalizationFactor::Evaluate(double const* const* parameters,
                                      double* residuals,
                                      double** jacobians) const {
  int r_dim   = info_.residualDim();
  int rem_dim = info_.remainDim();

  Eigen::Map<const Eigen::VectorXd> x_cur(parameters[0], rem_dim);
  const Eigen::VectorXd& x_lin = info_.linPoint();

  Eigen::VectorXd dx = x_cur - x_lin;
  Eigen::Map<Eigen::VectorXd> res(residuals, r_dim);
  res = info_.J_prior() * dx + info_.r_prior();

  if (jacobians && jacobians[0]) {
    Eigen::Map<Eigen::MatrixXd> J(jacobians[0], r_dim, rem_dim);
    J = info_.J_prior();
  }
  return true;
}

}  // namespace mono_vio