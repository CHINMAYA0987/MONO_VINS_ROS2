#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace mono_vio {

/**
 * Triangulator — DLT triangulation from N≥2 views.
 *
 * All inputs are in the CAMERA frame (not body/world).
 * The caller is responsible for transforming observations appropriately.
 *
 * Reference: Hartley & Zisserman, MVG, Section 12.2
 */
class Triangulator {
public:
  /**
   * Triangulate a 3-D point from bearing observations across multiple frames.
   *
   * @param T_c_w   Vector of camera-from-world transforms (one per view).
   *                T_c_w[i] maps a world point into camera i.
   * @param bearing Unit bearing vectors (normalised, z=1 plane) per view.
   * @param p_w_out Output: triangulated world-frame point.
   * @return true if triangulation succeeded (depth positive in all views).
   */
  static bool triangulate(
      const std::vector<Eigen::Matrix<double,3,4>>& T_c_w,
      const std::vector<Eigen::Vector3d>&            bearing,
      Eigen::Vector3d&                               p_w_out);

  /**
   * Two-view triangulation (fast path, exactly 2 views).
   * Returns false if the point is behind either camera or rays are nearly parallel.
   */
  static bool triangulate2(
      const Eigen::Matrix<double,3,4>& T_c0_w,
      const Eigen::Matrix<double,3,4>& T_c1_w,
      const Eigen::Vector3d&           b0,
      const Eigen::Vector3d&           b1,
      Eigen::Vector3d&                 p_w_out);
};

}  // namespace mono_vio