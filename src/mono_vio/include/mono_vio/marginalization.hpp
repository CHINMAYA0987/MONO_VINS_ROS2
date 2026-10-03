#pragma once
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <vector>

namespace mono_vio {

/**
 * Marginalization prior (Schur complement).
 *
 * When the oldest keyframe is dropped from the sliding window, all residuals
 * touching that frame are linearised at the current estimate and compressed
 * into a dense prior factor on the remaining variables.  This ensures that
 * information from discarded frames is not lost.
 *
 * Mathematical background:
 *   Given a linearised cost: 0.5 * ||J * dx - r||^2
 *   Partition into [Jm | Jr] where m = marginalised, r = remaining.
 *   After Schur complement: prior on r is 0.5 * ||Jr_bar * dxr - r_bar||^2
 *   where  Jr_bar = Jr - Jm * Jm^+ * Jr  (using pseudo-inverse).
 *
 * This class accumulates Jacobian blocks from each residual that touches the
 * marginalised frame, then performs the Schur complement and exposes a
 * ceres::CostFunction (MarginalizationFactor) ready to be plugged into the
 * next window optimisation.
 *
 * Reference: Leutenegger et al., IJRR 2015, Section III-B.
 */
class MarginalizationInfo {
public:
  MarginalizationInfo() = default;

  /**
   * Register a residual that involves the marginalised frame.
   * @param residual     evaluated residual vector
   * @param J_marg       Jacobian block w.r.t. marginalised variables
   * @param J_remain     Jacobian block w.r.t. remaining variables
   */
  void addBlock(const Eigen::VectorXd& residual,
                const Eigen::MatrixXd& J_marg,
                const Eigen::MatrixXd& J_remain);

  /**
   * Perform the Schur complement and store the linearised prior.
   * Must be called once, after all addBlock() calls.
   */
  void marginalise();

  /** Linearisation point for the remaining variables (copied at marg time). */
  const Eigen::VectorXd& linPoint() const { return lin_point_; }
  void setLinPoint(const Eigen::VectorXd& lp) { lin_point_ = lp; }

  /** Dimensions. */
  int remainDim() const { return static_cast<int>(J_prior_.cols()); }
  int residualDim() const { return static_cast<int>(J_prior_.rows()); }

  const Eigen::MatrixXd& J_prior() const { return J_prior_; }
  const Eigen::VectorXd& r_prior() const { return r_prior_; }

private:
  // Accumulated full Jacobian and residual before Schur complement.
  Eigen::MatrixXd J_full_;   ///< [J_marg | J_remain]
  Eigen::VectorXd r_full_;

  // Result after Schur complement.
  Eigen::MatrixXd J_prior_;
  Eigen::VectorXd r_prior_;

  Eigen::VectorXd lin_point_;

  bool marginalised_{false};
};

/**
 * Ceres cost function encoding the marginalization prior.
 *
 * Computes: residual = J_prior * (x - x_lin) - r_prior
 * where x_lin is the linearisation point stored in MarginalizationInfo.
 */
class MarginalizationFactor : public ceres::CostFunction {
public:
  explicit MarginalizationFactor(const MarginalizationInfo& info);

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override;

private:
  const MarginalizationInfo& info_;
};

}  // namespace mono_vio