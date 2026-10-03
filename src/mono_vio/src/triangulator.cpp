#include "mono_vio/triangulator.hpp"
#include <Eigen/SVD>

namespace mono_vio {

bool Triangulator::triangulate(
    const std::vector<Eigen::Matrix<double,3,4>>& T_c_w,
    const std::vector<Eigen::Vector3d>&            bearing,
    Eigen::Vector3d&                               p_w_out)
{
  const int N = (int)T_c_w.size();
  if (N < 2) return false;

  // Build 2N x 4 DLT system  A * X = 0
  Eigen::MatrixXd A(2 * N, 4);
  for (int i = 0; i < N; ++i) {
    const auto& P  = T_c_w[i];          // 3x4 projection matrix
    const Eigen::Vector3d& b = bearing[i];
    // rows:  x * P[2] - P[0],   y * P[2] - P[1]
    A.row(2*i)   = b.x() * P.row(2) - P.row(0);
    A.row(2*i+1) = b.y() * P.row(2) - P.row(1);
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  Eigen::Vector4d X = svd.matrixV().col(3);
  if (std::abs(X(3)) < 1e-10) return false;
  X /= X(3);

  p_w_out = X.head<3>();

  // Check positive depth in all views.
  for (int i = 0; i < N; ++i) {
    Eigen::Vector3d p_c = T_c_w[i] * X;
    if (p_c.z() < 0.1) return false;
  }
  return true;
}

bool Triangulator::triangulate2(
    const Eigen::Matrix<double,3,4>& T_c0_w,
    const Eigen::Matrix<double,3,4>& T_c1_w,
    const Eigen::Vector3d&           b0,
    const Eigen::Vector3d&           b1,
    Eigen::Vector3d&                 p_w_out)
{
  return triangulate({T_c0_w, T_c1_w}, {b0, b1}, p_w_out);
}

}  // namespace mono_vio