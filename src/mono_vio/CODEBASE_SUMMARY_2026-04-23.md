# mono_vio Codebase Summary (as of 2026-04-23)

## 1) High-level architecture

`mono_vio` is a ROS2 monocular VIO package with a VINS-like structure:

1. Frontend tracking (`FeatureTracker`)
2. IMU preintegration (`ImuIntegrator`)
3. Initialization (`Initializer`)
4. Sliding-window state container (`SlidingWindow`)
5. Nonlinear optimization with Ceres (`Optimizer`)
6. Marginalization prior (`MarginalizationInfo/Factor`)
7. Triangulation (`Triangulator`)
8. Optional loop closure (`LoopDetector` + 4-DOF correction in node)

Main orchestration is centralized in `src/mono_vio_node.cpp`.

---

## 2) File/module inventory

- `src/mono_vio_node.cpp`: ROS node, state machine (INIT/RUN), full pipeline orchestration.
- `src/feature_tracker.cpp`: KLT + F-matrix RANSAC + FAST refill + undistortion.
- `src/imu_integrator.cpp`: IMU preintegration, Jacobians, covariance propagation.
- `src/initializer.cpp`: gyro-bias calibration, visual SfM chain, scale/gravity/velocity solve.
- `src/sliding_window.cpp`: keyframe/landmark/observation container operations.
- `src/triangulator.cpp`: DLT triangulation from multi-view bearings.
- `src/optimizer.cpp`: reprojection + IMU residuals, Ceres solve, landmark filtering, prior build.
- `src/marginalization.cpp`: Schur-style linear prior construction and Ceres factor.
- `src/loop_detector.cpp`: ORB-BF matching + essential geometry verification.
- `config/params.yaml`: runtime parameters.
- `launch/mono_vio.launch.py`: node + RViz launch wiring.

---

## 3) Runtime control flow (what runs where)

### Startup
- `MonoVioNode::MonoVioNode()`
  - calls `loadParams()`, `setupSubscribers()`, `setupPublishers()`.

### Sensor callbacks
- `imuCb(...)`
  - appends IMU sample to `imu_buf_`.
- `imageCb(...)`
  - tracks features (`tracker_->track(...)`)
  - keyframe decision via `avgParallax(...)` + max keyframe time gap
  - integrates IMU up to image timestamp (`integrateUpTo(...)`)
  - branches:
    - INIT state: `runInit(...)`
    - RUN state: `runVIO(...)`
  - resets preintegrator with latest biases.

### INIT state path
- `runInit(...)`
  - creates `InitFrame` and calls `initializer_.addFrame(...)`
  - checks excitation (`initializer_.isExcited(...)`)
  - attempts solve (`initializer_.solve(...)`)
  - on success: `enterRun(...)`, `seedWindow(...)`
  - on repeated failure: fallback to fixed-gravity RUN.

### RUN state path
- `runVIO(...)`
  1. add keyframe in `SlidingWindow`
  2. IMU propagate current state from previous keyframe
  3. add feature observations + landmark slots
  4. `triangulateAll()`
  5. if full window: build prior (`optimizer_->buildMarginalizationPrior(...)`), shift obs, remove oldest
  6. optimize (`optimizer_->optimize(...)`)
  7. remove bad landmarks (`optimizer_->removeBadLandmarks(...)`)
  8. optional loop closure detect/apply (`loop_detector_->detect(...)`, `applyLoopCorrection4DOF(...)`)
  9. publish odom/path/tf (`publish(...)`).

---

## 4) Method-to-callsite map (core methods)

## 4.1 `ImuIntegrator`
- `reset(...)`
  - called in `MonoVioNode::loadParams()` and after each keyframe in `imageCb(...)`.
- `integrate(...)`
  - called from `MonoVioNode::integrateUpTo(...)` (midpoint interpolation).
- `correctBias(...)`
  - called only inside `ImuFactor::Evaluate(...)`.
- `measurements()`
  - consumed by `Initializer::addFrame(...)` for excitation statistics.

## 4.2 `FeatureTracker`
- `track(...)`
  - called in `MonoVioNode::imageCb(...)` for every image.
- internals:
  - `rejectOutliersFundamental()`, `detectNew(...)`, `undistortFeatures()`
  - called only from `track(...)`.

## 4.3 `Initializer`
- `setCamera(...)`
  - called in `MonoVioNode::loadParams()`.
- `addFrame(...)`
  - called in `runInit(...)`.
- `isExcited(...)`
  - called in `runInit(...)` for readiness checks.
- `solve(...)`
  - called in `runInit(...)` once enough init keyframes exist.
- `reset()`
  - called in `runInit(...)` after failed solve retry.
- internal substeps (only called from `solve(...)`):
  - `calibrateGyroBias(...)`
  - `buildVisualSfM(...)`
  - `solveScaleGravityVelocity(...)`
  - `refineGravity(...)`
  - `recoverRelativeRotation(...)`.

## 4.4 `SlidingWindow`
- `addKeyframe(...)`
  - called in `seedWindow(...)` and `runVIO(...)`.
- `isFull()`, `removeOldest()`
  - used in `runVIO(...)`.
- `addLandmark(...)`, `addObservation(...)`
  - used in `runVIO(...)` when adding current frame tracks.
- `removeLandmark(...)`
  - used in `Optimizer::removeBadLandmarks(...)`.
- `frames()/landmarks()/observations()`
  - heavily used by node + optimizer + triangulation.

## 4.5 `Triangulator`
- `triangulate(...)`
  - called in `MonoVioNode::triangulateAll()`.
- `triangulate2(...)`
  - present but not currently used in this code path.

## 4.6 `Optimizer`
- ctor
  - called in `loadParams()` and `enterRun(...)`.
- `optimize(...)`
  - called in `runVIO(...)`.
- `removeBadLandmarks(...)`
  - called in `runVIO(...)` after optimize.
- `buildMarginalizationPrior(...)`
  - called in `runVIO(...)` before removing oldest frame.

### Internal optimizer factor usage
- `ReprojectionFactor`
  - created in `Optimizer::optimize(...)` and `buildMarginalizationPrior(...)`.
- `ImuFactor`
  - created in `Optimizer::optimize(...)` and `buildMarginalizationPrior(...)`.
- `PoseLocalParameterization`
  - used in `Optimizer::optimize(...)` for pose blocks.

## 4.7 Marginalization classes
- `MarginalizationInfo::addBlock(...)`
  - called in `Optimizer::buildMarginalizationPrior(...)`.
- `MarginalizationInfo::marginalise()`
  - called in `Optimizer::buildMarginalizationPrior(...)`.
- `MarginalizationFactor`
  - instantiated in `Optimizer::optimize(...)` when prior is valid.

## 4.8 `LoopDetector`
- `addKeyframe(...)`
  - called in `runVIO(...)` for latest keyframe.
- `detect(...)`
  - called in `runVIO(...)` each keyframe (after start threshold).
- `reset()`
  - called in `enterRun(...)`.
- `verifyGeometry(...)`
  - internal helper called only by `detect(...)`.

---

## 5) What your current logs indicate

Observed from your run:
- INIT prints `spread=0.000` repeatedly.
- INIT still succeeds with tiny `scale=0.027`.
- During RUN, speed repeatedly prints exactly `|v|=5.00`.
- Position drifts rapidly (especially z).

Interpretation:
- The repeated exact speed means your estimator is hitting the hard velocity clamp (`max_vel_m_s=5.0`) frequently, i.e., unconstrained solution tends to exceed it.
- Tiny initialization scale plus immediate velocity clamping is a strong sign of poor metric alignment / insufficiently constrained optimization.
- `spread=0.000` in your INIT log is expected from current implementation (diagnostic is based on `InitFrame::t_wc` which is not updated during frame collection).

---

## 6) Probable causes (most important first)

### A) Marginalization prior is not actually tied to live state blocks
In `Optimizer::optimize(...)`, the prior residual is added on a single concatenated pointer `marg_param` (vector copy), not on the current window frame parameter blocks. That means prior constraints do not directly regularize the active Ceres variables for frame states.

Effect: older information is not properly carried forward after slide-out, causing drift and instability.

### B) Initialization frame consistency issue
`enterRun(...)` forcibly sets gravity to `[0,0,-9.81]` regardless of estimated gravity direction (`g_raw`). If world frame is not perfectly gravity-aligned at this stage, this introduces model inconsistency.

### C) Simplified/approximate Jacobians and model shortcuts
IMU factor Jacobians are simplified; prior construction for landmark reprojection terms ignores landmark Jacobians by treating them as fully marginalized with zero remain contribution. This can underconstrain and bias updates.

### D) Aggressive velocity clamping masks divergence
Hard-clamping every keyframe velocity to max speed keeps output bounded but can hide estimator inconsistency and produce non-physical trajectories.

### E) Potential feature geometry fragility
Frontend uses normalized-space fundamental-matrix rejection with approximate previous-point usage; if outlier rejection becomes weak, bad tracks leak into triangulation and BA.

---

## 7) Why build warnings are not the main runtime failure

1. `initializer.cpp: static skew(...) defined but not used`
   - harmless dead-code warning.

2. `svd.singularValues()(2) may be used uninitialized`
   - typical compiler false-positive around Eigen templates in optimized builds.
   - in this context, `svd` is constructed from a fully initialized matrix `A`.

These are not likely responsible for your drift behavior.

---

## 8) Gap vs VINS-Mono / OpenVINS

Your code is currently a **VINS-inspired simplified implementation**, not yet feature-equivalent to VINS-Mono/OpenVINS:

Missing or weaker pieces compared to reference systems:
- Proper FEJ-consistent marginalization over real parameter blocks.
- Mature initialization with robust alignment checks and gravity/orientation handling.
- Time-offset/extrinsic online calibration options.
- Better feature lifecycle and consistency checks (track management, robust weighting).
- Loop closure integrated with global pose-graph optimization (not just local correction).
- More complete IMU noise/discrete model handling and consistency-preserving Jacobians.

---

## 9) Practical roadmap to make behavior VINS/OpenVINS-like

### Phase 1 (highest ROI, do first)
1. Rework marginalization prior parameterization so prior residual connects directly to current frame parameter blocks in Ceres.
2. Keep gravity from initialization (or rotate world frame consistently once), instead of forcing fixed world-z gravity unconditionally.
3. Temporarily disable hard velocity clamp during debugging (or raise significantly) and inspect raw residual behavior.

### Phase 2
4. Improve IMU factor Jacobians and bias handling consistency.
5. Improve feature outlier handling (geometric verification and track-quality gating).
6. Add initialization quality gates (min triangulation angle, inlier ratio, scale sanity, gravity consistency).

### Phase 3
7. Introduce a dedicated pose-graph backend for loop closure instead of one-shot local correction.
8. Add detailed residual diagnostics (reprojection RMSE, IMU residual norms, prior residual norm) per keyframe.

---

## 10) RViz plugin error in your launch

The RViz error shown is unrelated to VIO math:
- `rviz_default_plugins/Image` panel plugin not available in your current RViz plugin set/config.

This affects visualization panel loading, not estimator core computation.

---

## 11) Notes on diagnostics in current node

- `[INIT] spread=...` currently does not reflect visual SfM trajectory spread (it uses `InitFrame::t_wc` collected before solve).
- For meaningful init diagnostics, log spread from solved `ts` in initializer solve path, or from seeded window states after `seedWindow(...)`.

---

## 12) Immediate actionable checks for your next run

1. Set `use_loop_closure: false` temporarily to isolate local VIO behavior.
2. Raise `max_vel_m_s` to a high value (e.g., 50) for debugging to detect true unconstrained dynamics.
3. Log per-keyframe:
   - IMU factor residual norm
   - mean reprojection error
   - prior residual norm
   - number of removed landmarks
4. Verify time synchronization and units of IMU topic and camera timestamps.

If you want, the next implementation step should be the prior rewire in `Optimizer::optimize(...)` + `buildMarginalizationPrior(...)` (this is the most likely root-cause fix).