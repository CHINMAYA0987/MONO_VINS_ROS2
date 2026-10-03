#include "mono_vio/sliding_window.hpp"
#include <algorithm>
#include <stdexcept>

namespace mono_vio {

SlidingWindow::SlidingWindow(int max_size) : max_size_(max_size) {}

WindowFrame& SlidingWindow::addKeyframe(double timestamp,
                                         std::shared_ptr<ImuIntegrator> preint) {
  WindowFrame kf;
  kf.id        = next_id_++;
  kf.timestamp = timestamp;
  kf.preint    = preint;
  // pose_params already initialised to identity in struct definition
  frames_.push_back(kf);
  return frames_.back();
}

void SlidingWindow::removeOldest() {
  if (frames_.empty()) return;

  // 1. Drop all observations that reference the frame being removed.
  removeObservationsOfFrame(0);

  // 2. Remove the frame itself.
  frames_.erase(frames_.begin());

  // 3. Decrement every remaining observation's frame_idx by 1.
  //    This is mandatory: erasing frames_[0] shifts all subsequent frames
  //    down by one position, so frame_idx=1 now means frame_idx=0, etc.
  //    Without this, every reprojection factor references the wrong camera pose.
  for (auto& o : obs_) {
    --o.frame_idx;
  }
}

Landmark& SlidingWindow::addLandmark(int lm_id) {
  auto it = landmarks_.find(lm_id);
  if (it != landmarks_.end()) return it->second;
  Landmark lm;
  lm.id = lm_id;
  landmarks_[lm_id] = lm;
  return landmarks_[lm_id];
}

void SlidingWindow::addObservation(int lm_id, int frame_idx,
                                    const Eigen::Vector2d& pixel) {
  Observation obs;
  obs.landmark_id = lm_id;
  obs.frame_idx   = frame_idx;
  obs.pixel       = pixel;
  obs_.push_back(obs);
}

void SlidingWindow::removeObservationsOfFrame(int frame_idx) {
  obs_.erase(
    std::remove_if(obs_.begin(), obs_.end(),
                   [frame_idx](const Observation& o){
                     return o.frame_idx == frame_idx; }),
    obs_.end());
}

void SlidingWindow::removeLandmark(int lm_id) {
  landmarks_.erase(lm_id);
  obs_.erase(
    std::remove_if(obs_.begin(), obs_.end(),
                   [lm_id](const Observation& o){
                     return o.landmark_id == lm_id; }),
    obs_.end());
}

}  // namespace mono_vio