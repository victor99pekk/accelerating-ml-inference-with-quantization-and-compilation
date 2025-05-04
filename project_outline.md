# Research Project - Outline

Quantizing and Compiling a Model → Measuring Performance → Comparing → Documenting

__Timeline__: 5 weeks, May 4 → June 8

---

# 📜 Research Abstract

**Title**: Accelerating Deep Learning Model Inference through Quantization and Compilation: A Comparative Study

**Abstract**:  
This study explores the effectiveness of post-training quantization and compiler-based optimizations in accelerating deep learning model inference. We benchmark a ResNet-18 model under four different configurations: baseline FP32, INT8 quantized, compiled FP32, and compiled INT8. Metrics such as inference latency, memory usage, model size, and accuracy are collected and analyzed. The goal is to understand the trade-offs between quantization and compilation techniques, individually and combined, to optimize model deployment for efficient real-world applications.

---

# 📚 Fields to Research and Learn About
- Basics of **post-training quantization** (PTQ) in deep learning
- **PyTorch quantization API** (`torch.quantization`)
- Introduction to **Apache TVM** or **TorchScript** for ML compilation
- Understanding **model benchmarking** (latency, memory footprint)
- Basics of **accuracy evaluation** (Top-1, Top-5)
- Light understanding of **FP32 vs INT8 precision trade-offs**
- Compilation concepts: **Relay IR**, **graph optimization**, **operator fusion**

---

# 📋 Todo
1. Load and benchmark baseline FP32 model.
2. Apply post-training quantization to INT8, benchmark.
3. Compile FP32 model (TorchScript or TVM), benchmark.
4. Compile INT8 model (optional, depending on tooling support), benchmark.
5. Compare results (latency, memory, accuracy).
6. Visualize results (plots, tables).
7. Write research report / blog post.
8. Polish code, README, and publish to GitHub.

---

## 🧩 Model: ResNet-18 Architecture Overview

- Input: **224 × 224 × 3** RGB image
- ↓
- **Conv1**: 7×7 convolution, stride 2 → BatchNorm → ReLU
- ↓
- **MaxPooling**: 3×3 kernel, stride 2
- ↓
- **Residual Block × 2** (64 channels)
- ↓
- **Residual Block × 2** (128 channels)
- ↓
- **Residual Block × 2** (256 channels)
- ↓
- **Residual Block × 2** (512 channels)
- ↓
- **Global Average Pooling**
- ↓
- **Fully Connected (Dense) Layer** → 1000 classes (e.g., ImageNet)
- ↓
- **Softmax** (classification output)


Each **Residual Block** has:
- conv → batchnorm → ReLU → conv → batchnorm → addition (skip connection) → ReLU

---

# 🗓️ Weekly Breakdown

## Week 1 (May 4 – May 10)
- Research:
  - Study post-training quantization and basic TVM/TorchScript compilation.
  - Review ResNet-18 structure.
- Setup:
  - Install environments: PyTorch, ONNX, TVM (optional).
  - Load and benchmark FP32 model on sample inputs.

✅ Deliverable: FP32 model running and benchmarked (baseline).

---

## Week 2 (May 11 – May 17)
- Quantization:
  - Apply post-training quantization to ResNet-18.
  - Benchmark quantized INT8 model (latency, memory, accuracy).
- Start basic documentation of process.

✅ Deliverable: INT8 model benchmarked and basic comparison draft.

---

## Week 3 (May 18 – May 24)
- Compilation:
  - Compile FP32 model (TorchScript or TVM).
  - Compile INT8 model (if framework supports it cleanly).
- Benchmark compiled versions.

✅ Deliverable: Compiled model benchmark results.

---

## Week 4 (May 25 – May 31)
- Analysis:
  - Validate accuracy drops (if any).
  - Create plots:
    - Inference time bar graph
    - Memory usage comparison
    - Model size comparison
    - Accuracy drop comparison
- Start writing research report.

✅ Deliverable: Visuals + partial report draft.

---

## Week 5 (June 1 – June 8)
- Final polishing:
  - Complete and polish research report (4–6 pages).
  - Complete README.
  - Push code, report, and visuals to GitHub.
  - (Optional) Post a blog summary or LinkedIn post.

✅ Final Deliverable: Public GitHub repo + Final report + (Optional) blog post.

---

# 📊 Visualizing Results
- **Inference Latency Plot**:
  - X-axis: Model variant (FP32, INT8, compiled FP32, compiled INT8)
  - Y-axis: Average inference time (ms)
- **Memory Usage Plot**:
  - X-axis: Model variant
  - Y-axis: Peak RAM usage (MB)
- **Accuracy Comparison**:
  - Table format (Top-1 accuracy)

---

# ✅ Final Notes
- Benchmark with at least 100–200 inferences for stability.
- Use dummy inputs (batch size = 1) for latency unless otherwise testing throughput.
- Save models and versions for reproducibility.
- Keep your results honest: if quantization hurts accuracy, document it carefully.

---
