# power-consumption-ml-arm
This project evaluates the power consumption of machine learning inference on ARM Cortex-M microcontrollers. By implementing Support Vector Machines and Artificial Neural Networks with CMSIS libraries and custom kernels, it compares efficiency across Cortex-M0+ and M4, offering insights for energy-efficient TinyML in IoT devices.


# ⚡ Evaluating Power Consumption of ML Inference on ARM Cortex-M MCUs  

This project investigates the **power efficiency of machine learning inference** on ARM Cortex-M microcontrollers.  
It compares **CMSIS libraries (CMSIS-DSP, CMSIS-NN)** with **custom C kernels** for Support Vector Machines (SVMs) and Artificial Neural Networks (ANNs),  
providing insights for **energy-efficient TinyML in IoT and edge devices**.  

---

## 📖 Project Description  
Machine learning is increasingly deployed on microcontrollers for IoT and embedded applications.  
However, inference can drain battery life and reduce efficiency.  
This work evaluates and measures the power consumption of ML algorithms on two ARM Cortex-M development boards:  

- **FRDM-KL25Z (Cortex-M0+)** – ultra-low power, no FPU/SIMD.  
- **FRDM-K64F (Cortex-M4)** – higher performance, with FPU and SIMD support.  

Two ML algorithms were implemented:  
- **Support Vector Machines (SVMs)**  
- **Artificial Neural Networks (ANNs)**  

Each algorithm was deployed in two ways:  
1. Using **ARM CMSIS libraries** (CMSIS-DSP for SVM, CMSIS-NN for ANN).  
2. Using **custom C kernels** written from scratch.  

---

## 📊 Key Results  

| **MCU (Core)** | **Algorithm** | **CMSIS (mW)** | **Custom (mW)** |
|----------------|---------------|----------------|-----------------|
| KL25Z (M0+)    | SVM           | 8.02           | 3.59            |
|                | ANN           | 8.24           | 8.77            |
| K64F (M4)      | SVM           | 6.19           | 24.12           |
|                | ANN           | 5.76           | 16.12           |

**Insights:**  
- On **Cortex-M0+**, custom implementations were more efficient.  
- On **Cortex-M4**, CMSIS libraries saved significant power by leveraging FPU/SIMD.  
- Choosing the right implementation depends on the MCU architecture.  

---

  

