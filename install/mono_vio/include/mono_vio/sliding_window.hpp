#pragma once
#include "mono_vio/imu_integrator.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <map>
#include <memory>

namespace mono_vio {

static constexpr int POSE_BLOCK = 7;  ///< [tx,ty,tz, qx,qy,qz,qw]
static constexpr int VEL_BLOCK  = 3;
static constexpr int BIAS_BLOCK = 6;  ///< [ba(3), bg(3)]
static constexpr int LM_BLOCK   = 3;

/**
 * WindowFrame — one keyframe in the sliding window.
 *
 * State is stored as raw double arrays so Ceres can hold direct pointers.
 * Layout:
 *   pose_params[7]  = [tx,ty,tz, qx,qy,qz,qw]
 *   vel_params[3]   = [vx,vy,vz]
 *   bias_params[6]  = [ba_x,ba_y,ba_z, bg_x,bg_y,bg_z]
 *
 * syncToParams() / syncFromParams() convert between Eigen and raw arrays.
 */
struct WindowFrame {
  int    id        {-1};
  double timestamp {0.0};

  // Raw Ceres parameter arrays
  double pose_params[POSE_BLOCK] {0,0,0, 0,0,0,1};  // identity
  double vel_params [VEL_BLOCK]  {0,0,0};
  double bias_params[BIAS_BLOCK] {0,0,0, 0,0,0};

  // Preintegrated IMU from the previous frame to this one.
  std::shared_ptr<ImuIntegrator> preint;

  // ── Eigen accessors ──────────────────────────────────────────────────────
  Eigen::Vector3d    position()   const { return Eigen::Map<const Eigen::Vector3d>(pose_params); }
  Eigen::Quaterniond rotation()   const {
    // storage: [tx,ty,tz, qx,qy,qz,qw]
    return Eigen::Quaterniond(pose_params[6], pose_params[3],
                               pose_params[4], pose_params[5]);
  }
  Eigen::Vector3d velocity()      const { return Eigen::Map<const Eigen::Vector3d>(vel_params); }
  Eigen::Vector3d acc_bias()      const { return Eigen::Map<const Eigen::Vector3d>(bias_params); }
  Eigen::Vector3d gyr_bias()      const { return Eigen::Map<const Eigen::Vector3d>(bias_params+3); }

  void setPosition(const Eigen::Vector3d& p) {
    pose_params[0]=p(0); pose_params[1]=p(1); pose_params[2]=p(2); }
  void setRotation(const Eigen::Quaterniond& q) {
    Eigen::Quaterniond qn = q.normalized();
    pose_params[3]=qn.x(); pose_params[4]=qn.y();
    pose_params[5]=qn.z(); pose_params[6]=qn.w(); }
  void setVelocity(const Eigen::Vector3d& v) {
    vel_params[0]=v(0); vel_params[1]=v(1); vel_params[2]=v(2); }
};

// Legacy alias so mono_vio_node still compiles.
using Keyframe = WindowFrame;

// ── Landmark ─────────────────────────────────────────────────────────────────
struct Landmark {
  int    id          {-1};
  double params[3]   {0,0,0};    ///< [x,y,z] world frame (Ceres parameter)
  bool   initialized {false};

  Eigen::Vector3d position() const { return Eigen::Map<const Eigen::Vector3d>(params); }
  void setPosition(const Eigen::Vector3d& p) {
    params[0]=p(0); params[1]=p(1); params[2]=p(2); initialized=true; }
};

// ── Observation ───────────────────────────────────────────────────────────────
struct Observation {
  int             landmark_id {-1};
  int             frame_idx   {-1};  ///< index into SlidingWindow::frames_
  Eigen::Vector2d pixel;             ///< undistorted normalised coordinates
};

// ── SlidingWindow ─────────────────────────────────────────────────────────────
/**
 * Bounded ring-buffer of WindowFrames, landmarks, and observations.
 *
 * syncToParams() / syncFromParams() are no-ops here because state IS the
 * raw arrays — they exist for API symmetry with direct-array Ceres usage.
 */
class SlidingWindow {
public:
  explicit SlidingWindow(int max_size = 10);

  // ── Keyframe management ──────────────────────────────────────────────────
  /** Add a new keyframe; returns a reference to the inserted frame. */
  WindowFrame& addKeyframe(double timestamp,
                            std::shared_ptr<ImuIntegrator> preint = nullptr);

  bool isFull()  const { return (int)frames_.size() >= max_size_; }
  int  size()    const { return (int)frames_.size(); }

  /** Remove oldest keyframe (after marginalisation). */
  void removeOldest();

  // ── Landmark / Observation management ───────────────────────────────────
  Landmark& addLandmark(int lm_id);
  void      addObservation(int lm_id, int frame_idx,
                            const Eigen::Vector2d& pixel);
  void      removeLandmark(int lm_id);
  void      removeObservationsOfFrame(int frame_idx);

  // ── Accessors ────────────────────────────────────────────────────────────
  std::vector<WindowFrame>&           frames()       { return frames_; }
  const std::vector<WindowFrame>&     frames() const { return frames_; }
  std::map<int,Landmark>&             landmarks()       { return landmarks_; }
  const std::map<int,Landmark>&       landmarks() const { return landmarks_; }
  std::vector<Observation>&           observations()       { return obs_; }
  const std::vector<Observation>&     observations() const { return obs_; }

  /**
   * No-op stubs — state lives directly in raw arrays, no copy needed.
   * Present so Optimizer::optimize() can call them for future extension.
   */
  void syncToParams()   {}
  void syncFromParams() {}

private:
  int max_size_;
  int next_id_{0};

  std::vector<WindowFrame>    frames_;
  std::map<int,Landmark>      landmarks_;
  std::vector<Observation>    obs_;
};

}  // namespace mono_vio