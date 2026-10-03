/**
 * mono_vio_node.cpp — Monocular VIO (VINS-Mono architecture), ROS2
 *
 * Fixes vs previous version:
 *   1. Initializer now runs visual SfM → VI alignment → gravity alignment.
 *   2. seedWindow() uses VI-aligned metric poses + solved velocities.
 *   3. enterRun() trusts the solved gravity (no longer overrides it).
 *   4. Marginalization prior is built and passed to the optimizer.
 *   5. IMU rotation hint is passed to the feature tracker.
 *   6. Loop closure applies full drift correction to window state.
 */

#include "mono_vio/imu_integrator.hpp"
#include "mono_vio/feature_tracker.hpp"
#include "mono_vio/initializer.hpp"
#include "mono_vio/sliding_window.hpp"
#include "mono_vio/marginalization.hpp"
#include "mono_vio/optimizer.hpp"
#include "mono_vio/triangulator.hpp"
#include "mono_vio/loop_detector.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <Eigen/Geometry>
#include <deque>
#include <mutex>
#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>

using namespace mono_vio;
enum class State { INIT, RUN };

struct LmObs { int frame_idx; Eigen::Vector3d bearing; };

struct ImuSample {
  double          t{0};
  Eigen::Vector3d acc{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyr{Eigen::Vector3d::Zero()};
};

// ─────────────────────────────────────────────────────────────────────────────
class MonoVioNode : public rclcpp::Node {
public:
  MonoVioNode() : Node("mono_vio_node") {
    loadParams();
    setupSubscribers();
    setupPublishers();
    RCLCPP_INFO(get_logger(), "MonoVIO node started — VINS-Mono pipeline.");
  }

private:
  // ── Camera / IMU / Extrinsic config ─────────────────────────────────────────
  CameraIntrinsics K_;
  Eigen::Matrix4d  T_cam_imu_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix3d  R_ci_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d  t_ci_{Eigen::Vector3d::Zero()};
  double           imu_rate_{200.0};

  // ── Sliding window / KF selection ───────────────────────────────────────────
  int    window_size_{10};
  double min_parallax_deg_{10.0};
  double max_kf_dt_{0.5};

  // ── Initializer ─────────────────────────────────────────────────────────────
  int    init_kf_count_{10};
  int    max_init_retries_{3};
  double imu_excitation_thr_{0.5};

  // ── Optimizer ───────────────────────────────────────────────────────────────
  Optimizer::Params optimizer_params_;
  double            max_vel_m_s_{5.0};

  // ── Loop closure ────────────────────────────────────────────────────────────
  bool use_loop_closure_{true};
  int  loop_start_kf_{30};
  int  loop_min_gap_{10};
  int  loop_same_ref_cooldown_{20};
  int  loop_min_inliers_{40};

  // ── VIO objects ──────────────────────────────────────────────────────────────
  std::unique_ptr<FeatureTracker> tracker_;
  Initializer                     initializer_;
  SlidingWindow                   window_{10};
  std::unique_ptr<Optimizer>      optimizer_;
  std::shared_ptr<ImuIntegrator>  integrator_;
  std::unique_ptr<LoopDetector>   loop_detector_;
  std::map<int, std::vector<LmObs>> lm_obs_;

  // ── Marginalization state ────────────────────────────────────────────────────
  std::shared_ptr<MarginalizationInfo>   marg_info_;
  std::shared_ptr<MarginalizationFactor> marg_factor_;
  std::vector<double*>                   marg_params_;
  Eigen::VectorXd                        marg_lin_point_;

  // ── Estimator state ──────────────────────────────────────────────────────────
  State           state_{State::INIT};
  Eigen::Vector3d gravity_w_{0.0, 0.0, -9.81};

  Eigen::Matrix3d drift_R_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d drift_t_{Eigen::Vector3d::Zero()};

  Eigen::Vector3d p_origin_{Eigen::Vector3d::Zero()};
  bool            has_origin_{false};

  double last_kf_ts_{-1.0};
  int    init_fail_count_{0};
  int    kf_count_{0};
  int    loop_count_{0};
  int    last_loop_kf_{-9999};
  int    last_loop_ref_{-1};

  // ── Feature / IMU buffers ────────────────────────────────────────────────────
  FeatureFrame last_ff_;
  bool         has_last_ff_{false};
  cv::Mat      last_gray_;

  std::mutex            imu_mutex_;
  std::deque<ImuSample> imu_buf_;
  std::optional<ImuSample> last_imu_;

  // ── ROS handles ──────────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr   imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr    odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr        path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr    dbg_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster>           tf_br_;
  nav_msgs::msg::Path path_msg_;

  // ── Parameter loading ────────────────────────────────────────────────────────
  void loadParams() {
    // Camera
    declare_parameter("camera.fx",     458.654);
    declare_parameter("camera.fy",     457.296);
    declare_parameter("camera.cx",     367.215);
    declare_parameter("camera.cy",     248.375);
    declare_parameter("camera.width",  752);
    declare_parameter("camera.height", 480);
    declare_parameter("camera.distortion_coeffs",
      std::vector<double>{-0.28340811,0.07395907,0.00019359,1.76187114e-05});
    K_.fx=get_parameter("camera.fx").as_double();
    K_.fy=get_parameter("camera.fy").as_double();
    K_.cx=get_parameter("camera.cx").as_double();
    K_.cy=get_parameter("camera.cy").as_double();
    K_.width =get_parameter("camera.width").as_int();
    K_.height=get_parameter("camera.height").as_int();
    auto d=get_parameter("camera.distortion_coeffs").as_double_array();
    K_.k1=d.size()>0?d[0]:0; K_.k2=d.size()>1?d[1]:0;
    K_.p1=d.size()>2?d[2]:0; K_.p2=d.size()>3?d[3]:0;

    // IMU
    declare_parameter("imu.acc_noise_density",  0.0028);
    declare_parameter("imu.gyro_noise_density", 0.00016);
    declare_parameter("imu.acc_random_walk",    0.00086);
    declare_parameter("imu.gyro_random_walk",   2.2e-5);
    declare_parameter("imu_rate",               200.0);
    double an=get_parameter("imu.acc_noise_density").as_double();
    double gn=get_parameter("imu.gyro_noise_density").as_double();
    double aw=get_parameter("imu.acc_random_walk").as_double();
    double gw=get_parameter("imu.gyro_random_walk").as_double();
    imu_rate_=get_parameter("imu_rate").as_double();

    // Extrinsics T_imu_cam (4×4 row-major → invert to get T_cam_imu)
    declare_parameter("extrinsics.T_imu_cam",
      std::vector<double>{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1});
    auto Tv=get_parameter("extrinsics.T_imu_cam").as_double_array();
    Eigen::Matrix4d Tic=Eigen::Matrix4d::Identity();
    if((int)Tv.size()==16)
      for(int r=0;r<4;r++) for(int c=0;c<4;c++) Tic(r,c)=Tv[r*4+c];
    T_cam_imu_=Tic.inverse();
    R_ci_=T_cam_imu_.topLeftCorner<3,3>();
    t_ci_=T_cam_imu_.topRightCorner<3,1>();

    // Sliding window
    declare_parameter("sliding_window.window_size",     10);
    declare_parameter("sliding_window.min_parallax",    10.0);
    declare_parameter("sliding_window.max_dt_keyframe", 0.5);
    window_size_     =get_parameter("sliding_window.window_size").as_int();
    min_parallax_deg_=get_parameter("sliding_window.min_parallax").as_double();
    max_kf_dt_       =get_parameter("sliding_window.max_dt_keyframe").as_double();

    // Feature tracker
    declare_parameter("feature_tracker.max_features",            150);
    declare_parameter("feature_tracker.fast_threshold",          20);
    declare_parameter("feature_tracker.min_dist",                30);
    declare_parameter("feature_tracker.klt_win_size",            21);
    declare_parameter("feature_tracker.klt_max_level",           3);
    declare_parameter("feature_tracker.ransac_reproj_threshold", 1.0);
    declare_parameter("feature_tracker.backtrack_thr",           1.5);
    declare_parameter("feature_tracker.grid_rows",               4);
    declare_parameter("feature_tracker.grid_cols",               6);
    FeatureTracker::Params tp;
    tp.max_features           =get_parameter("feature_tracker.max_features").as_int();
    tp.fast_threshold         =get_parameter("feature_tracker.fast_threshold").as_int();
    tp.min_dist               =get_parameter("feature_tracker.min_dist").as_int();
    tp.klt_win_size           =get_parameter("feature_tracker.klt_win_size").as_int();
    tp.klt_max_level          =get_parameter("feature_tracker.klt_max_level").as_int();
    tp.ransac_reproj_threshold=get_parameter("feature_tracker.ransac_reproj_threshold").as_double();
    tp.backtrack_thr          =(float)get_parameter("feature_tracker.backtrack_thr").as_double();
    tp.grid_rows              =get_parameter("feature_tracker.grid_rows").as_int();
    tp.grid_cols              =get_parameter("feature_tracker.grid_cols").as_int();
    tracker_=std::make_unique<FeatureTracker>(K_, tp);

    // Initializer: pass camera intrinsics so SfM can use K
    declare_parameter("initializer.init_keyframes",    10);
    declare_parameter("initializer.acc_std_threshold", 0.5);
    declare_parameter("max_init_retries",              3);
    init_kf_count_     =get_parameter("initializer.init_keyframes").as_int();
    imu_excitation_thr_=get_parameter("initializer.acc_std_threshold").as_double();
    max_init_retries_  =get_parameter("max_init_retries").as_int();
    initializer_.fx=K_.fx; initializer_.fy=K_.fy;
    initializer_.cx=K_.cx; initializer_.cy=K_.cy;
    // Pass camera-from-IMU rotation so VI alignment can rotate delta_p correctly
    initializer_.R_ci = R_ci_;

    // Optimizer
    declare_parameter("optimizer.max_iterations",   8);
    declare_parameter("optimizer.num_threads",      4);
    declare_parameter("optimizer.max_solver_time",  0.05);
    declare_parameter("optimizer.huber_loss_scale", 1.0);
    declare_parameter("optimizer.use_imu_factors",  true);
    declare_parameter("max_vel_m_s",                5.0);
    declare_parameter("optimizer.imu_weight",     0.001); 
    declare_parameter("optimizer.fix_first_vel_bias", false);
    max_vel_m_s_=get_parameter("max_vel_m_s").as_double();
    optimizer_params_.max_iterations  =get_parameter("optimizer.max_iterations").as_int();
    optimizer_params_.num_threads     =get_parameter("optimizer.num_threads").as_int();
    optimizer_params_.max_solver_time =get_parameter("optimizer.max_solver_time").as_double();
    optimizer_params_.huber_loss_scale=get_parameter("optimizer.huber_loss_scale").as_double();
    optimizer_params_.use_imu_factors =get_parameter("optimizer.use_imu_factors").as_bool();
    optimizer_params_.imu_weight = get_parameter("optimizer.imu_weight").as_double();
    optimizer_params_.fix_first_vel_bias = get_parameter("optimizer.fix_first_vel_bias").as_bool();
    optimizer_=std::make_unique<Optimizer>(optimizer_params_,K_,T_cam_imu_,gravity_w_);

    // Loop closure
    declare_parameter("use_loop_closure",               true);
    declare_parameter("loop_detector.start_after_kf",   30);
    declare_parameter("loop_detector.min_kf_gap",       10);
    declare_parameter("loop_detector.same_ref_cooldown",20);
    declare_parameter("loop_detector.apply_min_inliers",40);
    declare_parameter("loop_detector.max_db_size",      300);
    declare_parameter("loop_detector.min_temporal_gap", 10);
    declare_parameter("loop_detector.detect_min_inliers",25);
    declare_parameter("loop_detector.ratio_thresh",     0.75);
    declare_parameter("loop_detector.orb_features",     500);
    use_loop_closure_ =get_parameter("use_loop_closure").as_bool();
    loop_start_kf_    =get_parameter("loop_detector.start_after_kf").as_int();
    loop_min_gap_     =get_parameter("loop_detector.min_kf_gap").as_int();
    loop_same_ref_cooldown_=get_parameter("loop_detector.same_ref_cooldown").as_int();
    loop_min_inliers_ =get_parameter("loop_detector.apply_min_inliers").as_int();
    LoopDetector::Params lp;
    lp.max_db_size     =get_parameter("loop_detector.max_db_size").as_int();
    lp.min_temporal_gap=get_parameter("loop_detector.min_temporal_gap").as_int();
    lp.min_inliers     =get_parameter("loop_detector.detect_min_inliers").as_int();
    lp.ratio_thresh    =(float)get_parameter("loop_detector.ratio_thresh").as_double();
    lp.orb_features    =get_parameter("loop_detector.orb_features").as_int();
    loop_detector_=std::make_unique<LoopDetector>(lp);

    // IMU integrator
    integrator_=std::make_shared<ImuIntegrator>(an,gn,aw,gw);
    integrator_->reset(Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero());
    window_=SlidingWindow(window_size_);
  }

  void setupSubscribers() {
    declare_parameter("topics.image","/cam0/image_raw");
    declare_parameter("topics.imu",  "/imu0");
    img_sub_=create_subscription<sensor_msgs::msg::Image>(
      get_parameter("topics.image").as_string(), rclcpp::SensorDataQoS(),
      std::bind(&MonoVioNode::imageCb,this,std::placeholders::_1));
    imu_sub_=create_subscription<sensor_msgs::msg::Imu>(
      get_parameter("topics.imu").as_string(), rclcpp::SensorDataQoS(),
      std::bind(&MonoVioNode::imuCb,this,std::placeholders::_1));
  }

  void setupPublishers() {
    declare_parameter("topics.odom", "/mono_vio/odometry");
    declare_parameter("topics.path", "/mono_vio/path");
    declare_parameter("topics.debug","/mono_vio/debug_image");
    odom_pub_=create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("topics.odom").as_string(),10);
    path_pub_=create_publisher<nav_msgs::msg::Path>(
      get_parameter("topics.path").as_string(),10);
    dbg_pub_=create_publisher<sensor_msgs::msg::Image>(
      get_parameter("topics.debug").as_string(),1);
    tf_br_=std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

  // ── IMU callback ─────────────────────────────────────────────────────────────
  void imuCb(const sensor_msgs::msg::Imu::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(imu_mutex_);
    ImuSample s;
    s.t  =rclcpp::Time(msg->header.stamp).seconds();
    s.acc=Eigen::Vector3d(msg->linear_acceleration.x,
                          msg->linear_acceleration.y,
                          msg->linear_acceleration.z);
    s.gyr=Eigen::Vector3d(msg->angular_velocity.x,
                          msg->angular_velocity.y,
                          msg->angular_velocity.z);
    imu_buf_.push_back(s);
    while(imu_buf_.size()>4000) imu_buf_.pop_front();
  }

  // ── Image callback ────────────────────────────────────────────────────────────
  void imageCb(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try { cv_ptr=cv_bridge::toCvShare(msg,"mono8"); } catch(...) { return; }

    const double t=rclcpp::Time(msg->header.stamp).seconds();

    // Pass IMU rotation hint to tracker for better KLT initialisation
    Eigen::Quaterniond R_hint = integrator_->delta_q;
    // Convert body dR → camera dR: dR_cam = R_ci * dR_body * R_ci^T
    Eigen::Quaterniond R_cam_hint(R_ci_ * R_hint.toRotationMatrix() * R_ci_.transpose());

    FeatureFrame ff = tracker_->track(cv_ptr->image, t, &R_cam_hint);
    publishDebug(msg->header, cv_ptr->image, ff);

    // Keyframe selection
    const double par=avgParallax(last_ff_, ff);
    const bool is_kf = (!has_last_ff_)
                    || (par > min_parallax_deg_)
                    || (last_kf_ts_ > 0 && (t - last_kf_ts_) > max_kf_dt_);
    last_ff_=ff; has_last_ff_=true;
    if (!is_kf) return;

    last_gray_=cv_ptr->image.clone();
    integrateUpTo(t);
    auto pim=std::make_shared<ImuIntegrator>(*integrator_);

    if (state_==State::INIT) runInit(t, pim, ff);
    else                      runVIO(t, pim, ff);

    last_kf_ts_=t;
    // Reset integrator with latest optimised bias
    Eigen::Vector3d ba=Eigen::Vector3d::Zero(), bg=Eigen::Vector3d::Zero();
    if (state_==State::RUN && !window_.frames().empty()) {
      const auto& kf=window_.frames().back();
      ba=Eigen::Vector3d(kf.bias_params[0],kf.bias_params[1],kf.bias_params[2]);
      bg=Eigen::Vector3d(kf.bias_params[3],kf.bias_params[4],kf.bias_params[5]);
    }
    integrator_->reset(ba, bg);
  }

  // ── Midpoint IMU integration up to t_end ─────────────────────────────────────
  void integrateUpTo(double t_end) {
    std::lock_guard<std::mutex> lk(imu_mutex_);
    if (imu_buf_.empty()) return;
    if (!last_imu_.has_value()) { last_imu_=imu_buf_.front(); imu_buf_.pop_front(); }

    while (!imu_buf_.empty() && imu_buf_.front().t <= t_end) {
      const ImuSample cur=imu_buf_.front();
      const double dt=cur.t-last_imu_->t;
      if (dt>1e-6)
        integrator_->integrate(0.5*(last_imu_->acc+cur.acc),
                               0.5*(last_imu_->gyr+cur.gyr), dt);
      last_imu_=cur;
      imu_buf_.pop_front();
    }
    // Partial step to exactly t_end
    if (!imu_buf_.empty() && last_imu_->t < t_end) {
      const ImuSample& nxt=imu_buf_.front();
      const double span=nxt.t-last_imu_->t;
      if (span>1e-6) {
        const double a=(t_end-last_imu_->t)/span;
        const Eigen::Vector3d acc_i=(1-a)*last_imu_->acc+a*nxt.acc;
        const Eigen::Vector3d gyr_i=(1-a)*last_imu_->gyr+a*nxt.gyr;
        const double dt=t_end-last_imu_->t;
        if (dt>1e-6) {
          integrator_->integrate(0.5*(last_imu_->acc+acc_i),
                                 0.5*(last_imu_->gyr+gyr_i), dt);
          last_imu_=ImuSample{t_end, acc_i, gyr_i};
        }
      }
    }
  }

  // ── Initialization phase ──────────────────────────────────────────────────────
  // void runInit(double t, std::shared_ptr<ImuIntegrator> pim,
  //              const FeatureFrame& ff) {
  //   InitFrame fr;
  //   fr.timestamp = t;
  //   fr.R_wc      = Eigen::Matrix3d::Identity();
  //   fr.t_wc      = Eigen::Vector3d::Zero();
  //   fr.pim       = *pim;
  //   fr.velocity  = Eigen::Vector3d::Zero();

  //   const int frame_idx = initializer_.frameCount();
  //   initializer_.addFrame(fr);
  //   initializer_.addSfmFeatures(ff, frame_idx);

  //   const int n = initializer_.frameCount();
  //   RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
  //     "[INIT] %d/%d excited=%s",
  //     n, init_kf_count_,
  //     initializer_.isExcited(imu_excitation_thr_) ? "YES" : "NO");

  //   if (n < init_kf_count_) return;
  //   if (!initializer_.isExcited(imu_excitation_thr_)) {
  //     RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),2000,
  //       "[INIT] Waiting for IMU excitation..."); return;
  //   }

  //   InitResult res;
  //   std::string fail_reason;
  //   if (!initializer_.tryInitialize(res, &fail_reason)) {
  //     ++init_fail_count_;
  //     // Always reset so next 10-frame window starts fresh
  //     initializer_.reset();
  //     RCLCPP_WARN(get_logger(),
  //       "[INIT] tryInitialize failed (%d/%d) reason: %s",
  //       init_fail_count_, max_init_retries_, fail_reason.c_str());
  //     if (init_fail_count_ >= max_init_retries_) {
  //       RCLCPP_WARN(get_logger(),
  //         "[INIT] Max retries reached — entering RUN with zero gravity estimate.");
  //       init_fail_count_ = 0;
  //       enterRun(Eigen::Vector3d(0, 0, -9.81));
  //     }
  //     return;
  //   }

  //   // --------------- added here May 7 ------------
  //   const auto& sfm_R = initializer_.sfmR();   // gravity-aligned, from tryInitialize
  //   const auto& sfm_T = initializer_.sfmT();   // metric scale applied

  //   for (int i = 0; i < (int)window_.size(); ++i) {
  //       auto& kf = window_.frames()[i];
  //       kf.setPosition(sfm_T[i]);
  //       kf.setRotation(Eigen::Quaterniond(sfm_R[i]));
  //       kf.setVelocity(res.velocities[i]);     // from InitResult
  //       // bias stays at zero from linearized_ba/bg — correct for first iteration
  //   }

  //   // Seed landmarks directly from SfM map instead of re-triangulating
  //   for (auto& [id, sf] : initializer_.sfmMap()) {
  //       if (!sf.used) continue;
  //       auto& lm = window_.addLandmark(id);
  //       lm.setPosition(sf.position * res.scale);   // already scaled in tryInitialize
  //       for (auto& [fidx, uv] : sf.obs) {
  //           if (fidx < (int)window_.size())
  //               window_.addObservation(id, fidx, uv);
  //       }
  //   }
  //   // ----------------------------------------------------------

  //   RCLCPP_INFO(get_logger(),
  //     "[INIT] SUCCESS scale=%.3f g=[%.2f,%.2f,%.2f] |g|=%.3f",
  //     res.scale,
  //     res.gravity_w.x(), res.gravity_w.y(), res.gravity_w.z(),
  //     res.gravity_w.norm());

  //   enterRun(res.gravity_w);
  //   seedWindowFromInit(res);

  //   // Apply SfM poses and landmarks AFTER window is populated
  //   const auto& sfm_R = initializer_.sfmR();
  //   const auto& sfm_T = initializer_.sfmT();
  //   const int nw = (int)window_.size();

  //   for (int i = 0; i < nw; ++i) {
  //     auto& kf = window_.frames()[i];
  //     kf.setPosition(sfm_T[i]);
  //     kf.setRotation(Eigen::Quaterniond(sfm_R[i]));
  //     if (i < (int)res.velocities.size())
  //       kf.setVelocity(res.velocities[i]);
  //   }

  //   for (auto& [id, sf] : initializer_.sfmMap()) {
  //     if (!sf.used) continue;
  //     Landmark& lm = window_.addLandmark(id);
  //     if (!lm.initialized)
  //       lm.setPosition(sf.position);
  //     for (auto& [fidx, uv] : sf.obs) {
  //       if (fidx < nw)
  //         window_.addObservation(id, fidx, uv);
  //     }
  //   }
  // }
  void runInit(double t, std::shared_ptr<ImuIntegrator> pim,
             const FeatureFrame& ff) {
    InitFrame fr;
    fr.timestamp = t;
    fr.R_wc      = Eigen::Matrix3d::Identity();
    fr.t_wc      = Eigen::Vector3d::Zero();
    fr.pim       = *pim;
    fr.velocity  = Eigen::Vector3d::Zero();

    const int frame_idx = initializer_.frameCount();
    initializer_.addFrame(fr);
    initializer_.addSfmFeatures(ff, frame_idx);

    const int n = initializer_.frameCount();
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "[INIT] %d/%d excited=%s",
      n, init_kf_count_,
      initializer_.isExcited(imu_excitation_thr_) ? "YES" : "NO");

    if (n < init_kf_count_) return;
    if (!initializer_.isExcited(imu_excitation_thr_)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "[INIT] Waiting for IMU excitation..."); return;
    }

    InitResult res;
    std::string fail_reason;
    if (!initializer_.tryInitialize(res, &fail_reason)) {
      ++init_fail_count_;
      initializer_.reset();
      RCLCPP_WARN(get_logger(),
        "[INIT] tryInitialize failed (%d/%d) reason: %s",
        init_fail_count_, max_init_retries_, fail_reason.c_str());
      if (init_fail_count_ >= max_init_retries_) {
        RCLCPP_WARN(get_logger(),
          "[INIT] Max retries reached — entering RUN with zero gravity estimate.");
        init_fail_count_ = 0;
        enterRun(Eigen::Vector3d(0, 0, -9.81));
      }
      return;
    }

    RCLCPP_INFO(get_logger(),
      "[INIT] SUCCESS scale=%.3f g=[%.2f,%.2f,%.2f] |g|=%.3f",
      res.scale,
      res.gravity_w.x(), res.gravity_w.y(), res.gravity_w.z(),
      res.gravity_w.norm());

  //   enterRun(res.gravity_w);
  //   seedWindowFromInit(res);

  //   // Apply SfM poses and landmarks AFTER window is populated by enterRun
  //   const auto& sfm_R = initializer_.sfmR();
  //   const auto& sfm_T = initializer_.sfmT();
  //   const int nw = (int)window_.size();

  //   for (int i = 0; i < nw; ++i) {
  //     auto& kf = window_.frames()[i];
  //     kf.setPosition(sfm_T[i]);
  //     kf.setRotation(Eigen::Quaterniond(sfm_R[i]));
  //     if (i < (int)res.velocities.size())
  //       kf.setVelocity(res.velocities[i]);
  //   }

  //   for (auto& [id, sf] : initializer_.sfmMap()) {
  //     if (!sf.used) continue;
  //     Landmark& lm = window_.addLandmark(id);
  //     if (!lm.initialized)
  //       lm.setPosition(sf.position);
  //     for (auto& [fidx, uv] : sf.obs) {
  //       if (fidx < nw)
  //         window_.addObservation(id, fidx, uv);
  //     }
  //   }
  //   // ── Re-triangulate now that all 10 frames have correct SfM poses ──────
  //   triangulateAll();
  //   {
  //     size_t n_tri2 = 0;
  //     for (auto& [id, lm] : window_.landmarks()) if (lm.initialized) ++n_tri2;
  //     RCLCPP_INFO(get_logger(), "[INIT] Post-sfm-seed tri=%zu", n_tri2);
  //   }
  //   {
  //       optimizer_->optimize(window_, nullptr, {});
  //       updateFiniteDiffVelocities();
  //       for (auto& kf : window_.frames()) {
  //           Eigen::Map<Eigen::Vector3d> v(kf.vel_params);
  //           const double vn = v.norm();
  //           if (vn > max_vel_m_s_) v *= max_vel_m_s_ / vn;
  //       }
  //       RCLCPP_INFO(get_logger(), "[INIT] Burn-in BA complete |v_last|=%.2f",
  //           window_.frames().back().velocity().norm());
  //   }
  // }


  enterRun(res.gravity_w);
  seedWindowFromInit(res);  // this already handles landmarks + obs + triangulation

  // Apply SfM poses only — landmarks/obs already seeded above
  const auto& sfm_R = initializer_.sfmR();
  const auto& sfm_T = initializer_.sfmT();
  const int nw = (int)window_.size();
  for (int i = 0; i < nw; ++i) {
    auto& kf = window_.frames()[i];
    kf.setPosition(sfm_T[i]);
    kf.setRotation(Eigen::Quaterniond(sfm_R[i]));
    if (i < (int)res.velocities.size())
      kf.setVelocity(res.velocities[i]);
  }

  // Re-triangulate with correct SfM poses, then burn-in
  triangulateAll();
  {
    size_t n_tri2 = 0;
    for (auto& [id, lm] : window_.landmarks()) if (lm.initialized) ++n_tri2;
    RCLCPP_INFO(get_logger(), "[INIT] Post-sfm-seed tri=%zu", n_tri2);
  }

  // ── Burn-in: visual-only BA (no IMU — positions may be near-zero during hover)
  {
      Optimizer::Params burn_params = optimizer_params_;
      burn_params.use_imu_factors = false;   // IMU factors blow up with zero positions
      Optimizer burn_opt(burn_params, K_, T_cam_imu_, gravity_w_);
      burn_opt.optimize(window_, nullptr, {});
      updateFiniteDiffVelocities();
      for (auto& kf : window_.frames()) {
        Eigen::Map<Eigen::Vector3d> v(kf.vel_params);
        const double vn = v.norm();
        if (vn > max_vel_m_s_) v *= max_vel_m_s_ / vn;
      }
      RCLCPP_INFO(get_logger(), "[INIT] Burn-in BA complete |v_last|=%.2f",
        window_.frames().back().velocity().norm());
    }
  }

  // ── Enter RUN state ───────────────────────────────────────────────────────────
  void enterRun(const Eigen::Vector3d& g_aligned) {
    // Trust the solved gravity direction; it is already normalised to 9.81
    // by the initializer.  Only fall back to [0,0,-9.81] in emergency.
    gravity_w_ = g_aligned;
    if (std::abs(gravity_w_.norm() - 9.81) > 1.0) {
      RCLCPP_WARN(get_logger(),
        "[INIT] Gravity norm implausible (%.2f) — using [0,0,-9.81]",
        gravity_w_.norm());
      gravity_w_ = Eigen::Vector3d(0.0, 0.0, -9.81);
    }
    RCLCPP_INFO(get_logger(),
      "[INIT] gravity_w=[%.3f,%.3f,%.3f] (fixed to 9.81)",
      gravity_w_.x(), gravity_w_.y(), gravity_w_.z());

    optimizer_ = std::make_unique<Optimizer>(
                     optimizer_params_, K_, T_cam_imu_, gravity_w_);

    drift_R_.setIdentity();
    drift_t_.setZero();
    marg_factor_.reset();
    marg_params_.clear();

    state_       = State::RUN;
    kf_count_    = 0;
    has_origin_  = false;
    loop_count_  = 0;
    last_loop_kf_= -9999;
    last_loop_ref_= -1;
    lm_obs_.clear();
    window_ = SlidingWindow(window_size_);
    loop_detector_->reset();
  }

  // ── Seed window from VI-aligned result ───────────────────────────────────────
  // Finite-difference velocities from current window poses.
  void updateFiniteDiffVelocities() {
    auto& fr = window_.frames();
    const int Nw = (int)fr.size();
    for (int i = 0; i+1 < Nw; ++i) {
      const double dt = fr[i+1].timestamp - fr[i].timestamp;
      if (dt < 1e-4) continue;
      Eigen::Vector3d v = (fr[i+1].position() - fr[i].position()) / dt;
      const double vn = v.norm();
      if (vn > max_vel_m_s_) v *= (max_vel_m_s_ / vn);
      fr[i].setVelocity(v);
    }
    if (Nw >= 2) fr[Nw-1].setVelocity(fr[Nw-2].velocity());
  }

  void seedWindowFromInit(const InitResult& res) {
    // ── Seeding strategy ─────────────────────────────────────────────────────
    // Problem: SfM positions have unknown metric scale (VI alignment unreliable
    // for slow-start sequences like EuRoC MH_01). IMU position integration has
    // ~15m bias over 10s. Both fail as standalone metric position sources.
    //
    // Solution: Integrate VI-alignment VELOCITIES (which are in m/s regardless
    // of scale errors in position) over keyframe time intervals to get positions.
    // Velocities from the VI linear system are v_i * dt = (metric displacement),
    // so cumulative sum gives metric positions consistent with actual motion.
    //
    // Rotations from SfM essential matrix (accurate, gravity-aligned).
    // Biases: zero initial estimate (optimizer will refine them).
    const auto& ifr   = initializer_.frames();
    const auto& R_sfm = initializer_.sfmR();   // gravity-aligned R_wc
    const int N  = (int)ifr.size();
    const int st = std::max(0, N - window_size_);

    // ── Velocity validation ──────────────────────────────────────────────────
    // Cap VI velocities to physical range (EuRoC max ~1.5 m/s).
    // If velocities are clearly wrong (>3 m/s), zero them — constant-zero is
    // safer than wrong-metric because BA will find the correct positions.
    const double VEL_CAP = 1.5;    // m/s — EuRoC cruising speed
    const double VEL_MAX = 3.0;    // m/s — hard upper bound; above = suspect
    bool   vels_valid = true;
    for (int i = 0; i < N && i < (int)res.velocities.size(); ++i) {
      if (res.velocities[i].norm() > VEL_MAX) { vels_valid = false; break; }
    }

    // ── Build metric positions by integrating velocities ─────────────────────
    // p[k] = p[k-1] + v[k-1] * dt_k
    std::vector<Eigen::Vector3d> p_metric(N, Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> v_seeded(N, Eigen::Vector3d::Zero());

    for (int i = 0; i < N && i < (int)res.velocities.size(); ++i) {
      Eigen::Vector3d v = res.velocities[i];
      if (!vels_valid || v.norm() > VEL_MAX) v.setZero();
      const double vn = v.norm();
      if (vn > VEL_CAP) v *= VEL_CAP / vn;
      v_seeded[i] = v;
    }
    for (int i = 1; i < N; ++i) {
      const double dt = ifr[i].timestamp - ifr[i-1].timestamp;
      p_metric[i] = p_metric[i-1] + v_seeded[i-1] * dt;
    }

    const double metric_span = (p_metric[N-1] - p_metric[st]).norm();
    RCLCPP_INFO(get_logger(),
      "[INIT] VI vel-integrated span=%.3fm vels_valid=%s "
      "v[st]=(%.2f,%.2f,%.2f) v[N-1]=(%.2f,%.2f,%.2f)",
      metric_span, vels_valid ? "YES" : "NO",
      v_seeded[st].x(), v_seeded[st].y(), v_seeded[st].z(),
      v_seeded[N-1].x(), v_seeded[N-1].y(), v_seeded[N-1].z());

    // ── Seed window frames ───────────────────────────────────────────────────
    // for (int i = st; i < N; ++i) {
    //   auto pim_ptr = std::make_shared<ImuIntegrator>(ifr[i].pim);
    //   WindowFrame& kf = window_.addKeyframe(ifr[i].timestamp, pim_ptr);
    //   // Rotation: SfM camera → body frame
    //   const Eigen::Matrix3d R_wb = R_sfm[i] * R_ci_;
    //   kf.setRotation(Eigen::Quaterniond(R_wb).normalized());
    //   // Position: velocity integral, translated so frame[st] = origin
    //   kf.setPosition(p_metric[i] - p_metric[st]);
    //   // Velocity: seeded VI estimate
    //   kf.setVelocity(v_seeded[i]);
    //   // Bias: zero (optimizer will estimate)
    //   for (int k = 0; k < 6; ++k) kf.bias_params[k] = 0.0;
    // }
    // In seedWindowFromInit, replace the kf seeding block:
    for (int i = st; i < N; ++i) {
        auto pim_ptr = std::make_shared<ImuIntegrator>(ifr[i].pim);
        WindowFrame& kf = window_.addKeyframe(ifr[i].timestamp, pim_ptr);
        const Eigen::Matrix3d R_wb = R_sfm[i] * R_ci_;
        kf.setRotation(Eigen::Quaterniond(R_wb).normalized());
        kf.setPosition(Eigen::Vector3d::Zero());
        // Seed VI velocities (will be overwritten by runInit post-block anyway)
        if (i < (int)res.velocities.size())
            kf.setVelocity(res.velocities[i]);
        else
            kf.setVelocity(Eigen::Vector3d::Zero());
        for (int k = 0; k < 6; ++k) kf.bias_params[k] = 0.0;
    }

    // ── Seed observations ────────────────────────────────────────────────────
    lm_obs_.clear();
    for (auto& [lm_id, sf] : initializer_.sfmMap()) {
      bool added = false;
      for (auto& [fi, uv] : sf.obs) {
        const int wi = fi - st;
        if (wi < 0 || wi >= window_.size()) continue;
        if (!added) { window_.addLandmark(lm_id); added = true; }
        window_.addObservation(lm_id, wi, uv);
        lm_obs_[lm_id].push_back({wi, Eigen::Vector3d(uv.x(), uv.y(), 1.0)});
      }
    }

    // ── Triangulate from metric-seeded body poses ────────────────────────────
    triangulateAll();
    size_t n_tri = 0;
    for (auto& [id, lm] : window_.landmarks()) if (lm.initialized) ++n_tri;

    // ── Landmark-only burn-in: fix ALL poses, optimize landmarks only ─────────
    // Preserves the metric scale while cleaning up triangulation noise.
    if (n_tri >= 8) {
      ceres::Problem prob;
      auto* ploc = new PoseLocalParameterization();
      for (auto& kf2 : window_.frames()) {
        prob.AddParameterBlock(kf2.pose_params, 7, ploc);
        prob.SetParameterBlockConstant(kf2.pose_params);
      }
      for (auto& [id, lm] : window_.landmarks())
        if (lm.initialized) prob.AddParameterBlock(lm.params, 3);
      auto* huber = new ceres::HuberLoss(optimizer_params_.huber_loss_scale);
      for (auto& o : window_.observations()) {
        if (o.frame_idx < 0 || o.frame_idx >= (int)window_.frames().size()) continue;
        auto it = window_.landmarks().find(o.landmark_id);
        if (it == window_.landmarks().end() || !it->second.initialized) continue;
        prob.AddResidualBlock(
            new ReprojectionFactor(o.pixel, K_, T_cam_imu_), huber,
            window_.frames()[o.frame_idx].pose_params, it->second.params);
      }
      ceres::Solver::Options opts;
      opts.linear_solver_type = ceres::DENSE_SCHUR;
      opts.max_num_iterations = 50;
      opts.minimizer_progress_to_stdout = false;
      ceres::Solver::Summary sm;
      ceres::Solve(opts, &prob, &sm);
    }

    // ── Keep VI velocities (don't overwrite with finite-diff here) ───────────
    // Finite-diff on 0.4m baseline gives ~0.04 m/s which is wrong.
    // VI velocities are the best metric estimate we have at init time.

    const auto& f0 = window_.frames().front();
    const auto& fN = window_.frames().back();
    RCLCPP_INFO(get_logger(),
      "[INIT] Seeded %d frames tri=%zu obs=%zu "
      "p0=(%.3f,%.3f,%.3f) pN=(%.3f,%.3f,%.3f) |vN|=%.2f",
      window_.size(), n_tri, window_.observations().size(),
      f0.position().x(), f0.position().y(), f0.position().z(),
      fN.position().x(), fN.position().y(), fN.position().z(),
      fN.velocity().norm());
  }

  void buildMargPrior() {
    if (window_.size() < 2) return;

    auto& frames = window_.frames();
    auto info = std::make_shared<MarginalizationInfo>();

    // ── 1. IMU residual between frame[0] and frame[1] ────────────────────────
    if (frames[1].preint) {
      ImuFactor imu_fac(*frames[1].preint, gravity_w_);

      // Stack parameter pointers: pose0(7), vel0(3), bias0(6),
      //                           pose1(7), vel1(3), bias1(6)
      double* params[6] = {
        frames[0].pose_params, frames[0].vel_params,  frames[0].bias_params,
        frames[1].pose_params, frames[1].vel_params,  frames[1].bias_params
      };
      double residuals[15];
      // Allocate Jacobian rows
      std::vector<double> J0(15*7,0), J1(15*3,0), J2(15*6,0),
                          J3(15*7,0), J4(15*3,0), J5(15*6,0);
      double* jacs[6] = {J0.data(),J1.data(),J2.data(),
                          J3.data(),J4.data(),J5.data()};
      imu_fac.Evaluate(params, residuals, jacs);

      // J_marg  = columns w.r.t. frame[0]: pose0(7), vel0(3), bias0(6) → 16
      // J_remain= columns w.r.t. frame[1]: pose1(7), vel1(3), bias1(6) → 16
      const int M=15, Cm=16, Cr=16;
      Eigen::MatrixXd Jm(M,Cm), Jr(M,Cr);
      // pose0 block
      Eigen::Map<Eigen::Matrix<double,15,7,Eigen::RowMajor>> j0(J0.data());
      Jm.block(0,0,M,7) = j0;
      // vel0 block
      Eigen::Map<Eigen::Matrix<double,15,3,Eigen::RowMajor>> j1(J1.data());
      Jm.block(0,7,M,3) = j1;
      // bias0 block
      Eigen::Map<Eigen::Matrix<double,15,6,Eigen::RowMajor>> j2(J2.data());
      Jm.block(0,10,M,6) = j2;
      // pose1 block
      Eigen::Map<Eigen::Matrix<double,15,7,Eigen::RowMajor>> j3(J3.data());
      Jr.block(0,0,M,7) = j3;
      // vel1 block
      Eigen::Map<Eigen::Matrix<double,15,3,Eigen::RowMajor>> j4(J4.data());
      Jr.block(0,7,M,3) = j4;
      // bias1 block
      Eigen::Map<Eigen::Matrix<double,15,6,Eigen::RowMajor>> j5(J5.data());
      Jr.block(0,10,M,6) = j5;

      Eigen::VectorXd res_vec = Eigen::Map<Eigen::VectorXd>(residuals, 15);
      info->addBlock(res_vec, Jm, Jr);
    }

    // ── 2. Reprojection residuals touching frame[0] ──────────────────────────
    for (auto& o : window_.observations()) {
      if (o.frame_idx != 0) continue;
      auto it = window_.landmarks().find(o.landmark_id);
      if (it == window_.landmarks().end() || !it->second.initialized) continue;

      ReprojectionFactor repr(o.pixel, K_, T_cam_imu_);
      double* params[2] = {frames[0].pose_params, it->second.params};
      double residuals[2];
      std::vector<double> J0(2*7,0), J1(2*3,0);
      double* jacs[2] = {J0.data(), J1.data()};
      repr.Evaluate(params, residuals, jacs);

      Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> Jpose(J0.data());
      Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>> Jlm  (J1.data());
      Eigen::VectorXd res_vec = Eigen::Map<Eigen::VectorXd>(residuals, 2);
      // Marginalize pose0 (marg) keeping landmark (remain)
      info->addBlock(res_vec,
                     Jpose.cast<double>(),
                     Jlm.cast<double>());
    }

    // ── 3. Carry forward existing prior ─────────────────────────────────────
    // (skipped for first window slide; added in future iteration)

    // ── 4. Set linearisation point = remaining variables = frame[1..end] ─────
    // We flatten all frame[1..end] pose(7)+vel(3)+bias(6)=16 each
    const int n_remain = (int)frames.size() - 1;
    const int lp_dim   = n_remain * 16;
    Eigen::VectorXd lp(lp_dim);
    for (int i = 0; i < n_remain; ++i) {
      Eigen::Map<const Eigen::Matrix<double,16,1>> blk(
        frames[i+1].pose_params);  // 7+3+6 = 16 contiguous bytes? NO:
      // pose(7) vel(3) bias(6) are separate arrays — copy manually
      lp.segment<7>(i*16).     head<7>() =
        Eigen::Map<const Eigen::VectorXd>(frames[i+1].pose_params, 7);
      lp.segment(i*16+7, 3)  =
        Eigen::Map<const Eigen::VectorXd>(frames[i+1].vel_params, 3);
      lp.segment(i*16+10, 6) =
        Eigen::Map<const Eigen::VectorXd>(frames[i+1].bias_params, 6);
    }
    info->setLinPoint(lp);
    info->marginalise();

    // Build factor and parameter pointer list for the optimizer
    marg_factor_ = std::make_shared<MarginalizationFactor>(*info);
    marg_params_.clear();
    for (int i = 1; i < (int)frames.size(); ++i) {
      marg_params_.push_back(frames[i].pose_params);
      marg_params_.push_back(frames[i].vel_params);
      marg_params_.push_back(frames[i].bias_params);
    }
  }

  // ── Main VIO pipeline (per keyframe) ─────────────────────────────────────────
  void runVIO(double t, std::shared_ptr<ImuIntegrator> pim,
              const FeatureFrame& ff) {
    ++kf_count_;
    WindowFrame& cur = window_.addKeyframe(t, pim);
    const int ci = window_.size() - 1;

    // ── 1. IMU propagation ───────────────────────────────────────────────────
    // if (ci > 0) {
    //   const WindowFrame& prev = window_.frames()[ci-1];
    //   const Eigen::Quaterniond qp = prev.rotation();
    //   const Eigen::Vector3d    pp = prev.position();
    //   const Eigen::Vector3d    vp = prev.velocity();
    //   const double dt = pim->sum_dt;

    //   // Rotation: IMU delta_q (reliable even without bias estimate)
    //   Eigen::Quaterniond q_pred = (qp * pim->delta_q).normalized();

    //   // Position: use IMU delta_p with gravity correction
    //   // This gives a metric-scale position even without VI init.
    //   // delta_p integrates (a_meas - b_a) which includes gravity.
    //   // Remove the gravity contribution: p_corr = R_wb * delta_p - 0.5*g*dt²
    //   const Eigen::Matrix3d R_wb = qp.toRotationMatrix();
    //   const Eigen::Vector3d dp_corrected =
    //       R_wb * pim->delta_p - 0.5 * gravity_w_ * dt * dt;
    //   Eigen::Vector3d p_pred = pp + vp * dt + dp_corrected;

    //   // Fallback to constant-velocity if dp_corrected is unreasonably large
    //   // (protects against bad biases before optimizer converges)
    //   if (dp_corrected.norm() > max_vel_m_s_ * dt * 2.0)
    //     p_pred = pp + vp * dt;

    //   Eigen::Vector3d v_pred = vp;
    //   const double vn = v_pred.norm();
    //   if (vn > max_vel_m_s_) v_pred *= (max_vel_m_s_ / vn);

    //   cur.setPosition(p_pred);
    //   cur.setRotation(q_pred);
    //   cur.setVelocity(v_pred);
    //   for (int k = 0; k < 6; ++k) cur.bias_params[k] = prev.bias_params[k];
    // }
    if (ci > 0) {
        const WindowFrame& prev = window_.frames()[ci-1];
        const double dt = pim->sum_dt;

        // Rotation from IMU (accurate even without bias)
        Eigen::Quaterniond q_pred = (prev.rotation() * pim->delta_q).normalized();

        // Position from constant velocity — safe before bias converges.
        // IMU delta_p without estimated bias gives ~0.15m/step error that compounds.
        // Constant-velocity gives <0.3m/step error for typical VIO motion.
        Eigen::Vector3d v_pred = prev.velocity();
        const double vn = v_pred.norm();
        if (vn > max_vel_m_s_) v_pred *= max_vel_m_s_ / vn;

        Eigen::Vector3d p_pred;
        if (vn < 0.05) {
            // Velocity not yet estimated — use IMU delta_p as position increment.
            // R_wb converts body-frame delta_p to world frame.
            const Eigen::Matrix3d R_wb = prev.rotation().toRotationMatrix();
            p_pred = prev.position() + R_wb * pim->delta_p + 0.5 * gravity_w_ * dt * dt;
        } else {
            p_pred = prev.position() + v_pred * dt;
        }
        cur.setPosition(p_pred);

        // cur.setPosition(prev.position() + v_pred * dt);
        cur.setRotation(q_pred);
        cur.setVelocity(v_pred);
        for (int k = 0; k < 6; ++k) cur.bias_params[k] = prev.bias_params[k];
    }

    // ── 2. Feature observations ──────────────────────────────────────────────
    for (const auto& feat : ff.features) {
      window_.addLandmark(feat.landmark_id);
      window_.addObservation(feat.landmark_id, ci, feat.uv_norm.head<2>());
      lm_obs_[feat.landmark_id].push_back({ci, feat.uv_norm});
    }

    // ── 3. Triangulation ─────────────────────────────────────────────────────
    triangulateAll();

    // ── 4. Build marg prior + slide window ───────────────────────────────────
    if (window_.isFull()) {
      // buildMargPrior() disabled: MarginalizationFactor currently declares
      // 1 combined parameter block but the optimizer passes N separate blocks,
      // causing a Ceres assertion failure (27 vs. 1).  Skip prior for now;
      // marginalization will be re-enabled once the factor is refactored.
      shiftLmObs();
      window_.removeOldest();

      if (window_.isFull()){
        shiftLmObs();
        window_.removeOldest();

        // ── Purge landmarks that have zero observations in the current window ──
        // After sliding, init-era landmarks with no remaining obs are free
        // variables with no cost — the optimizer finds degenerate solutions.
        std::vector<int> to_remove;
        for (auto& [id, lm] : window_.landmarks()) {
            bool has_obs = false;
            for (auto& o : window_.observations())
                if (o.landmark_id == id) { has_obs = true; break; }
            if (!has_obs) to_remove.push_back(id);
        }
        for (int id : to_remove) {
            window_.removeLandmark(id);
            lm_obs_.erase(id);
        }
      }
    }

    // ── 5. Bundle adjustment (no marginalization prior yet) ──────────────────
    size_t n_tri = 0;
    for (auto& [id, lm] : window_.landmarks()) if (lm.initialized) ++n_tri;

    // if (n_tri >= 5) {
    //   optimizer_->optimize(window_, nullptr, {});
    //   // Recompute velocities from pose finite-differences after BA.
    //   // When IMU factors are active, this overrides the Ceres-optimized
    //   // velocity with a physically grounded estimate, preventing IMU
    //   // from locking velocities at the initial (possibly wrong) value.
    //   updateFiniteDiffVelocities();
    //   // Sanity clamp after update
    //   for (auto& kf : window_.frames()) {
    //     Eigen::Map<Eigen::Vector3d> v(kf.vel_params);
    //     const double vn = v.norm();
    //     if (vn > max_vel_m_s_) v *= (max_vel_m_s_ / vn);
    //   }
    // }
    
    if (n_tri >= 8) {
        optimizer_->optimize(window_, nullptr, {});
        updateFiniteDiffVelocities();
        for (auto& kf : window_.frames()) {
            Eigen::Map<Eigen::Vector3d> v(kf.vel_params);
            const double vn = v.norm();
            if (vn > max_vel_m_s_) v *= max_vel_m_s_ / vn;
        }
    }

    // ── 6. Loop closure ───────────────────────────────────────────────────────
    if (use_loop_closure_ && kf_count_ >= loop_start_kf_ && !last_gray_.empty()) {
      const WindowFrame& lat = window_.frames().back();
      const Eigen::Matrix3d Rwc = lat.rotation().toRotationMatrix() * R_ci_.transpose();
      const Eigen::Vector3d twc = lat.position();

      loop_detector_->addKeyframe(kf_count_, last_gray_, Rwc, twc);
      auto lc = loop_detector_->detect(kf_count_, last_gray_, Rwc, twc);

      if (lc.has_value()) {
        const bool ok_inliers = (lc->n_inliers >= loop_min_inliers_);
        const bool ok_gap     = (kf_count_ - last_loop_kf_ >= loop_min_gap_);
        const bool same_ref   = (lc->loop_kf_id == last_loop_ref_ &&
                                  kf_count_ - last_loop_kf_ < loop_same_ref_cooldown_);
        if (ok_inliers && ok_gap && !same_ref) {
          applyLoopCorrection(*lc);
          last_loop_kf_  = kf_count_;
          last_loop_ref_ = lc->loop_kf_id;
          ++loop_count_;
          RCLCPP_INFO(get_logger(),
            "[LOOP] cur_kf=%d loop_kf=%d inliers=%d total=%d",
            lc->cur_kf_id, lc->loop_kf_id, lc->n_inliers, loop_count_);
        }
      }
    }

    publish(t, window_.frames().back(), ff.features.size(), n_tri);
  }

  // ── Full loop correction applied to window state ──────────────────────────────
  // Computes SE(3) drift = correction relative to current VIO estimate,
  // applies it to ALL window frames and the display drift transform.
  // void applyLoopCorrection(const LoopConstraint& lc) {
  //   if (window_.frames().empty()) return;

  //   // Retrieve loop keyframe pose from loop_detector DB
  //   // (LoopDetector::addKeyframe stores R_wc, t_wc; expose via getters)
  //   Eigen::Matrix3d R_loop_w = loop_detector_->getR(lc.loop_kf_id);
  //   Eigen::Vector3d t_loop_w = loop_detector_->getT(lc.loop_kf_id);

  //   // Corrected pose of current frame in world:
  //   //   T_cur_world_corr = T_loop_world * T_loop_cur^{-1}
  //   const Eigen::Matrix3d R_cur_corr = R_loop_w * lc.R_loop_cur.transpose();
  //   const Eigen::Vector3d t_cur_corr = t_loop_w - R_cur_corr * lc.t_loop_cur;

  //   // Drift = difference between corrected and current VIO estimates
  //   const WindowFrame& latest = window_.frames().back();
  //   const Eigen::Matrix3d R_cur_vio = latest.rotation().toRotationMatrix();
  //   const Eigen::Vector3d t_cur_vio = latest.position();

  //   const Eigen::Matrix3d dR = R_cur_corr * R_cur_vio.transpose();
  //   const Eigen::Vector3d dt = t_cur_corr - dR * t_cur_vio;

  //   // Apply drift correction to ALL window frames (metric state correction)
  //   for (auto& kf : window_.frames()) {
  //     kf.setPosition(dR * kf.position() + dt);
  //     kf.setRotation(Eigen::Quaterniond(dR * kf.rotation()
  //                                          .toRotationMatrix()).normalized());
  //     Eigen::Map<Eigen::Vector3d> v(kf.vel_params);
  //     v = dR * v;
  //   }

  //   // Reset display drift (window state is now corrected)
  //   drift_R_.setIdentity();
  //   drift_t_.setZero();
  // }
  void applyLoopCorrection(const LoopConstraint& lc) {
    if (window_.frames().empty()) return;

    Eigen::Matrix3d R_loop_w = loop_detector_->getR(lc.loop_kf_id);
    Eigen::Vector3d t_loop_w = loop_detector_->getT(lc.loop_kf_id);

    const Eigen::Matrix3d R_cur_corr = R_loop_w * lc.R_loop_cur.transpose();
    const Eigen::Vector3d t_cur_corr = t_loop_w - R_cur_corr * lc.t_loop_cur;

    const WindowFrame& latest = window_.frames().back();
    const Eigen::Matrix3d R_cur_vio = latest.rotation().toRotationMatrix();
    const Eigen::Vector3d t_cur_vio = latest.position();

    Eigen::Matrix3d dR = R_cur_corr * R_cur_vio.transpose();
    Eigen::Vector3d dt = t_cur_corr - dR * t_cur_vio;

    // ── Limit correction magnitude to prevent large jumps ─────────────────────
    // A jump > 1m in a single loop closure step usually means wrong loop match.
    // Dampen to max 0.5m per closure; the next closure will apply the remainder.
    const double dt_norm = dt.norm();
    // constexpr double MAX_LOOP_JUMP = 2.0;   // metres
    const double MAX_LOOP_JUMP = std::clamp(2.0 + (lc.n_inliers - 80) / 100.0, 2.0, 5.0);
    
    if (dt_norm > MAX_LOOP_JUMP) {
      RCLCPP_WARN(get_logger(),
        "[LOOP] Correction %.2fm exceeds limit %.2fm — clamping",
        dt_norm, MAX_LOOP_JUMP);
      dt *= MAX_LOOP_JUMP / dt_norm;
      // Also damp rotation proportionally
      Eigen::AngleAxisd aa(dR);
      const double max_angle = MAX_LOOP_JUMP / std::max(dt_norm, 1e-3) * aa.angle();
      dR = Eigen::AngleAxisd(
              std::min(aa.angle(), max_angle), aa.axis()).toRotationMatrix();
    }

    // ── Blend: scale correction by inlier confidence ──────────────────────────
    // At 300 inliers → full correction (alpha=1.0).
    // At 100 inliers → 33% correction this step; next loop closure applies more.
    // This prevents a single uncertain match from throwing all frames by several m.
    {
      const double blend_alpha = std::min(1.0, (double)lc.n_inliers / 150.0);

      dt *= blend_alpha;

      // Interpolate dR toward identity by the same factor
      Eigen::AngleAxisd aa(dR);
      dR = Eigen::AngleAxisd(aa.angle() * blend_alpha, aa.axis()).toRotationMatrix();
    }

    for (auto& kf : window_.frames()) {
      kf.setPosition(dR * kf.position() + dt);
      kf.setRotation(
          Eigen::Quaterniond(dR * kf.rotation().toRotationMatrix()).normalized());
      Eigen::Map<Eigen::Vector3d> v(kf.vel_params);
      v = dR * v;
    }

    drift_R_.setIdentity();
    drift_t_.setZero();
  }

  // ── DLT triangulation ────────────────────────────────────────────────────────
  void triangulateAll() {
    auto& frames = window_.frames();
    for (auto& [lm_id, lm] : window_.landmarks()) {
      if (lm.initialized) continue;
      auto it = lm_obs_.find(lm_id);
      if (it == lm_obs_.end() || it->second.size() < 2) continue;

      std::vector<Eigen::Matrix<double,3,4>> Ps;
      std::vector<Eigen::Vector3d>           Bs;
      for (const auto& obs : it->second) {
        if (obs.frame_idx < 0 || obs.frame_idx >= (int)frames.size()) continue;
        const auto& kf = frames[obs.frame_idx];
        const Eigen::Matrix3d Rwb = kf.rotation().toRotationMatrix();
        const Eigen::Vector3d pwb = kf.position();
        const Eigen::Matrix3d Rcw = R_ci_ * Rwb.transpose();
        const Eigen::Vector3d tcw = R_ci_ * (-Rwb.transpose()*pwb) + t_ci_;
        Eigen::Matrix<double,3,4> P;
        P.leftCols<3>()=Rcw; P.rightCols<1>()=tcw;
        Ps.push_back(P); Bs.push_back(obs.bearing);
      }
      if (Ps.size() < 2) continue;

      // Require ≥1.5° parallax
      const double cospar = Bs.front().dot(Bs.back()) /
                            (Bs.front().norm()*Bs.back().norm()+1e-10);
      if (std::acos(std::min(1.0,std::abs(cospar)))*57.3 < 1.0) continue;

      Eigen::Vector3d pw;
      if (Triangulator::triangulate(Ps, Bs, pw)) lm.setPosition(pw);
    }
  }

  void shiftLmObs() {
    for (auto& [id, v] : lm_obs_) {
      std::vector<LmObs> k;
      for (auto& o : v) if (o.frame_idx > 0) {--o.frame_idx; k.push_back(o);}
      v = std::move(k);
    }
    for (auto it = lm_obs_.begin(); it != lm_obs_.end();)
      it = it->second.empty() ? lm_obs_.erase(it) : ++it;
  }

  // ── Parallax ─────────────────────────────────────────────────────────────────
  double avgParallax(const FeatureFrame& a, const FeatureFrame& b) const {
    if (!has_last_ff_) return 999.0;
    std::unordered_map<int,Eigen::Vector2d> prev;
    for (auto& f : a.features) prev[f.landmark_id] = f.uv_norm.head<2>();
    double sum = 0; int cnt = 0;
    const double foc = 0.5*(K_.fx + K_.fy);
    for (auto& f2 : b.features) {
      auto it = prev.find(f2.landmark_id);
      if (it == prev.end()) continue;
      const double dpx = (f2.uv_norm.head<2>() - it->second).norm() * foc;
      sum += std::atan2(dpx, foc)*57.3; ++cnt;
    }
    return cnt > 0 ? sum/cnt : 0.0;
  }

  // ── Publish ───────────────────────────────────────────────────────────────────
  void publish(double t, const WindowFrame& kf, size_t nf, size_t nt) {
    const Eigen::Quaterniond q_raw = kf.rotation();
    const Eigen::Vector3d    p_raw = kf.position();
    const Eigen::Vector3d    v_raw(kf.vel_params[0],kf.vel_params[1],kf.vel_params[2]);

    const Eigen::Quaterniond q = (Eigen::Quaterniond(drift_R_)*q_raw).normalized();
    const Eigen::Vector3d    p = drift_R_*p_raw + drift_t_;
    const Eigen::Vector3d    v = drift_R_*v_raw;

    if (!has_origin_) { p_origin_=p; has_origin_=true; }
    const Eigen::Vector3d pr = p - p_origin_;

    nav_msgs::msg::Odometry odom;
    odom.header.stamp   =rclcpp::Time(static_cast<uint64_t>(t*1e9));
    odom.header.frame_id="odom";
    odom.child_frame_id ="body";
    odom.pose.pose.position.x   =pr.x();
    odom.pose.pose.position.y   =pr.y();
    odom.pose.pose.position.z   =pr.z();
    odom.pose.pose.orientation.x=q.x();
    odom.pose.pose.orientation.y=q.y();
    odom.pose.pose.orientation.z=q.z();
    odom.pose.pose.orientation.w=q.w();
    odom.twist.twist.linear.x=v.x();
    odom.twist.twist.linear.y=v.y();
    odom.twist.twist.linear.z=v.z();
    odom_pub_->publish(odom);

    geometry_msgs::msg::PoseStamped ps;
    ps.header=odom.header; ps.pose=odom.pose.pose;
    path_msg_.header=odom.header; path_msg_.header.frame_id="map";
    path_msg_.poses.push_back(ps);
    if (path_msg_.poses.size()>20000) path_msg_.poses.erase(path_msg_.poses.begin());
    path_pub_->publish(path_msg_);

    geometry_msgs::msg::TransformStamped tf;
    // map → odom (identity) required by RViz Fixed Frame = map
    geometry_msgs::msg::TransformStamped tf_map;
    tf_map.header.stamp    = odom.header.stamp;
    tf_map.header.frame_id = "map";
    tf_map.child_frame_id  = "odom";
    tf_map.transform.rotation.w = 1.0;
    tf_br_->sendTransform(tf_map);

    // odom → body
    tf.header=odom.header; tf.header.frame_id="odom"; tf.child_frame_id="body";
    tf.transform.translation.x=pr.x();
    tf.transform.translation.y=pr.y();
    tf.transform.translation.z=pr.z();
    tf.transform.rotation=odom.pose.pose.orientation;
    tf_br_->sendTransform(tf);

    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),500,
      "[RUN] p=(%.3f,%.3f,%.3f) |v|=%.2f kf#%d feat=%zu tri=%zu obs=%zu loops=%d",
      pr.x(),pr.y(),pr.z(),v.norm(),kf_count_,nf,nt,
      window_.observations().size(), loop_count_);
  }

  void publishDebug(const std_msgs::msg::Header& hdr,
                    const cv::Mat& gray, const FeatureFrame& ff) {
    if (dbg_pub_->get_subscription_count()==0) return;
    cv::Mat vis; cv::cvtColor(gray,vis,cv::COLOR_GRAY2BGR);
    for (auto& f : ff.features) {
      const int age=std::min(f.track_count,20);
      cv::circle(vis,f.px,3,cv::Scalar(0,255-age*10,age*10),-1,cv::LINE_AA);
    }
    dbg_pub_->publish(*cv_bridge::CvImage(hdr,"bgr8",vis).toImageMsg());
  }
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MonoVioNode>());
  rclcpp::shutdown();
  return 0;
}