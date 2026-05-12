# 🚀 Mono_VINS_ROS2


> 🎯 **Monocular Visual-Inertial Navigation System (VINS) using Ceres Solver on ROS2 on EuRoC dataset (C++ version)**

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?style=for-the-badge&logo=ros)
![Status](https://img.shields.io/badge/Status-Experimental-success?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-EuRoC-orange?style=for-the-badge)

---

## 🧠 Overview

This repository demonstrates a **Monocular Visual-Inertial Navigation System (VINS)** using Ceres Solver in **ROS2**, tested on the **EuRoC MAV dataset**.

✨ It combines:
- 📷 Monocular camera input  
- 📊 IMU measurements  
- 🧭 Real-time state estimation  
- 📍 Camera trajectory reconstruction  

The system follows a **tightly-coupled optimization-based VINS pipeline**, inspired by modern visual-inertial estimation frameworks.

---

## 🎥 Demo

<p align="center">
  <img src="https://github.com/user-attachments/assets/your-demo-gif-here" width="700"/>
</p>

---

## ⚙️ Features

- 📷 Monocular image processing  
- 📈 IMU preintegration  
- 🎯 Feature tracking and management  
- 🧩 Sliding-window optimization  
- 📍 Real-time trajectory estimation  
- 🔄 Visual-Inertial sensor fusion  
- 🧪 Tested on EuRoC MAV dataset  
- 🧱 Modular ROS2 architecture  

---

## 🛠️ System Architecture

The pipeline consists of:

- **Feature Tracker**
  - FAST/KLT-based tracking

- **IMU Processing**
  - IMU integration and motion propagation
  - Bias handling and state prediction

- **Initialization**
  - Visual-Inertial alignment, Gravity and scale estimation

- **Optimization Backend**
  - Sliding window nonlinear optimization
 
- **Loop Closure**
  - Handling loop closure and Bundle Adjustment

- **Marginalization**
  - Converting older keyframes to priors and graph updation

---

## 📂 Dataset

This project is tested on the **EuRoC MAV Dataset**:

- MH_01_easy  


📥 Dataset Link:  
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

---

## 📚 References & Inspiration

This implementation is inspired by well-known Visual-Inertial Navigation frameworks and research:

### 🔹 VINS-Mono

- Qin, T., Li, P., & Shen, S.  
  *"VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator"*  
- 🔗 https://ieeexplore.ieee.org/document/8421746  
- 🔗 https://github.com/HKUST-Aerial-Robotics/VINS-Mono  

💡 One of the most influential optimization-based monocular VINS systems.

---

### 🔸 Ceres Solver

## 🎓 Suggested Reading

<details>
<summary>📘 Learn the theory behind this project</summary>

### Core Topics
- Visual-Inertial Odometry (VIO/VINS)
- Nonlinear optimization
- Sliding window estimation
- IMU preintegration
- Feature tracking and triangulation
- Sensor fusion

## 🧠 Notes

- This project follows a **Factor graph-based VIO approach**  
- Designed for **learning and experimentation** rather than production use  
- Tested on **EuRoC MH_01 easy dataset**  

---
