<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../imgs/ellipse.png">
  <img alt="Ellipse logo" src="../imgs/ellipse.png" width="50%">
</picture>

**Ellipse**: Something between [PyTorch](https://github.com/pytorch/pytorch), [micrograd](https://github.com/karpathy/micrograd), and [XLA](https://openxla.org/xla). Maintained by [nileAGI](https://www.nileagi.com/).

[Home Page](../README.md) | [Documentation](./documentation.md) | [Quick Start](./quick_start.md)

[Announcement Blog Post](https://www.nileagi.com/blog/ellipse-introduction)

</div>

---

# Progress and Roadmap

Ellipse is designed to be an accessible, low-level deep learning framework with a focus on simplicity and modularity. This page tracks what is implemented and what comes next.

**Legend:** ✅ shipped · ❌ not implemented yet.

A row flips to ✅ only when the code exists **and** a test in `test.c` exercises it. Statuses synced against `ellipse.c` / `ellipse.h` — November 2026.

## Tensor core

| Task | Status |
| --- | --- |
| `Tensor` struct (`data`, `shape`, `size`, `dtype`) | ✅ |
| `tensor()` — create from existing data (copies the input) | ✅ |
| `_rand`, `_randn` (Box–Muller), `_randint` | ✅ |
| `_zeros`, `_ones` | ✅ |
| `print` / `tensor_print` (1-D and 2-D) | ✅ |
| `free_t` — manual memory management (`malloc` / `memcpy` based) | ✅ |
| `ellipse.*` namespace and `size(...)` macro | ✅ |
| Seedable RNG — `torch.seed()` | ✅ |
| Dtypes: `_int`, `_float` | ✅ |
| Dtype: `double` | ❌ |
| 3-D+ (ND) tensors | ❌ |
| Broadcasting | ❌ |

## Math operations

All operations are element-wise except `matmul`. $G$ denotes the upstream gradient $\partial L / \partial C$; each backward formula is what you accumulate into the inputs' gradients.

| Op | Forward | Backward | Status |
| --- | --- | --- | --- |
| add | $C = A + B$ | $G_A = G,\; G_B = G$ | ❌ |
| sub | $C = A - B$ | $G_A = G,\; G_B = -G$ | ❌ |
| mul | $C = A \odot B$ | $G_A = G \odot B,\; G_B = G \odot A$ | ❌ |
| div | $C = A / B$ | $G_A = G / B,\; G_B = -G \odot A / B^{2}$ | ❌ |
| matmul | $C_{ij} = \sum_k A_{ik} B_{kj}$ | $G_A = G B^\top,\; G_B = A^\top G$ | ❌ |
| exp | $C = e^{X}$ | $G_X = G \odot e^{X}$ | ❌ |
| log (natural) | $C = \ln X$ | $G_X = G / X$ | ❌ |
| pow | $C = X^{n}$ | $G_X = G \odot n X^{n-1}$ | ❌ |

### Reductions

| Op | Forward | Backward | Status |
| --- | --- | --- | --- |
| sum | $c = \sum_i X_i$ | broadcast $G$ to every element | ❌ |
| mean | $c = \frac{1}{n} \sum_i X_i$ | broadcast $G / n$ | ❌ |

### Shape operations

| Op | Definition | Backward | Status |
| --- | --- | --- | --- |
| transpose | $(A^\top)_{ij} = A_{ji}$ | transpose $G$ back | ❌ |
| flatten | row-major $(m, n) \to (mn,)$ | reshape $G$ to the input shape | ❌ |
| reshape | new shape, same row-major element order, same element count | reshape $G$ to the input shape | ❌ |
| eye | $I_{ij} = 1$ if $i = j$, else $0$ | — (constant, no gradient) | ❌ |

Note: reshape is **not** transpose — it preserves the row-major order of the elements and only changes the shape. Transpose, flatten, and reshape all *do* have backward passes: undo the shape change on the gradient.

## Automatic differentiation

| Task | Status |
| --- | --- |
| `grad` — per-tensor gradient storage | ❌ |
| `requires_grad` flag | ❌ |
| Graph tracking: `prev`, `op`, `num_prev` | ❌ |
| `backward()` — reverse-mode over the recorded graph | ❌ |
| `zero_grad()` / gradient accumulation | ❌ |

## Activations

### Core

| Op | Forward | Backward | Status |
| --- | --- | --- | --- |
| ReLU | $\max(0, x)$ | $G$ if $x > 0$, else $0$ | ❌ |
| LeakyReLU | $x$ if $x \ge 0$, else $\alpha x$ | $G$ if $x \ge 0$, else $\alpha G$ | ❌ |
| sigmoid | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $G\,\sigma(x)(1 - \sigma(x))$ | ❌ |
| tanh | $\tanh(x)$ | $G\,(1 - \tanh^2(x))$ | ❌ |
| softmax | $s_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$ | $s \odot \left(G - \sum_j G_j s_j\right)$ | ❌ |

At $x = 0$: ReLU uses gradient $0$, LeakyReLU uses $1$ (PyTorch convention).

**Softmax stability:** subtracting $\max(x)$ before exponentiating prevents overflow and does not change the result. The full Jacobian is

$$\frac{\partial s_i}{\partial x_j} = \begin{cases} s_i (1 - s_i) & i = j \\ -s_i s_j & i \neq j \end{cases} \qquad J = \mathrm{diag}(s) - s s^\top$$

but you never need to build it: the backward column above is the fused vector–Jacobian product.

### Extended (wishlist)

Derivatives get documented here as each one is implemented.

| Op | Forward | Status |
| --- | --- | --- |
| ReLU6 | $\min(\max(0, x), 6)$ | ❌ |
| PReLU | $x$ if $x \ge 0$, else $\alpha x$ — $\alpha$ is learned | ❌ |
| RReLU | like LeakyReLU, $\alpha$ sampled from $[\ell, u]$ during training | ❌ |
| Threshold | $x$ if $x > \theta$, else $v$ (PyTorch semantics) | ❌ |
| Hardtanh | $\mathrm{clamp}(x, \min, \max)$, default $[-1, 1]$ | ❌ |
| Hardsigmoid | $\max(0, \min(1, x/6 + 0.5))$ | ❌ |
| Hardswish | $x \cdot \mathrm{ReLU6}(x + 3) / 6$ | ❌ |
| SiLU | $x\,\sigma(x)$ | ❌ |
| Mish | $x \tanh(\mathrm{softplus}(x))$ | ❌ |
| ELU | $x$ if $x \ge 0$, else $\alpha (e^x - 1)$ | ❌ |
| CELU | $\max(0, x) + \min(0, \alpha (e^{x/\alpha} - 1))$ | ❌ |
| SELU | $\lambda x$ if $x \ge 0$, else $\lambda \alpha (e^x - 1)$; $\lambda \approx 1.0507$, $\alpha \approx 1.6733$ | ❌ |
| GLU | $A \odot \sigma(B)$, input split into halves $A$, $B$ | ❌ |
| GELU | $x\,\Phi(x)$, $\Phi$ = standard normal CDF | ❌ |
| Softplus | $\ln(1 + e^x)$ | ❌ |
| Softsign | $x / (1 + \lvert x \rvert)$ | ❌ |
| Tanhshrink | $x - \tanh(x)$ | ❌ |
| LogSigmoid | $-\ln(1 + e^{-x})$ | ❌ |
| Softshrink | $x - \lambda$ if $x > \lambda$; $x + \lambda$ if $x < -\lambda$; else $0$ | ❌ |
| Hardshrink | $x$ if $\lvert x \rvert > \lambda$, else $0$ | ❌ |
| Softmin | $\mathrm{softmax}(-x)$ | ❌ |
| LogSoftmax | $x_i - \ln \sum_j e^{x_j}$ (max-shifted) | ❌ |
| Softmax2d | softmax over channels at each spatial position (needs ND tensors) | ❌ |

## Loss functions

| Loss | Type | When to use | Trade-off | Status |
| --- | --- | --- | --- | --- |
| MSE | regression | continuous targets | simple, differentiable; outlier-sensitive | ❌ |
| MAE | regression | noisy data | robust to outliers; non-smooth gradient | ❌ |
| Cross-Entropy | classification | binary / multi-class | pairs with softmax; class-imbalance sensitive | ❌ |
| Hinge | classification | margin classifiers (SVM) | no probabilities | ❌ |
| Huber | regression | data with outliers | robust and smooth; needs $\delta$ tuning | ❌ |
| KL divergence | probabilistic | distribution matching | asymmetric | ❌ |
| NLL | classification | with log-probability outputs | expects log-probs as input | ❌ |

**MSE** (torch convention, $\frac{1}{n}$):

$$\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \qquad \frac{\partial\,\mathrm{MSE}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

Some tutorials use $\frac{1}{2n}$ so the $2$ cancels — either is fine, just stay consistent.

**MAE:**

$$\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert \qquad \frac{\partial\,\mathrm{MAE}}{\partial \hat{y}_i} = \begin{cases} +\frac{1}{n} & \hat{y}_i > y_i \\ -\frac{1}{n} & \hat{y}_i < y_i \\ 0 & \hat{y}_i = y_i \end{cases}$$

At equality the derivative is undefined; use the subgradient $0$.

**Binary cross-entropy:** $\;-\frac{1}{n} \sum_i \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \right]$

**Categorical cross-entropy:** $\;-\sum_i \sum_c y_{ic} \ln \hat{y}_{ic}\;$ (one-hot $y$; with softmax this reduces to $-\ln \hat{y}_{i,\text{true}}$ — that is also NLL)

**Hinge:** $\;\sum_i \max(0, 1 - y_i \hat{y}_i)\;$ with labels $y_i \in \{+1, -1\}$

**Huber:**

$$L_\delta = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \lvert y - \hat{y} \rvert \le \delta \\ \delta \lvert y - \hat{y} \rvert - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

**KL divergence:** $\;\sum_i p_i \ln \frac{p_i}{q_i}\;$ — asymmetric: $\mathrm{KL}(p \parallel q) \neq \mathrm{KL}(q \parallel p)$

## Optimizers

| Task | Status |
| --- | --- |
| `step()` optimizer interface | ❌ |
| SGD | ❌ |
| SGD + momentum | ❌ |
| Adam | ❌ |
| AdamW | ❌ |
| RMSprop | ❌ |
| Adagrad | ❌ |
| Adadelta | ❌ |

## Layers

| Task | Status |
| --- | --- |
| Sequential | ❌ |
| Linear (dense) | ❌ |
| Dropout | ❌ |
| Conv2d / Conv3d | ❌ |
| MaxPool2d / MaxPool3d | ❌ |
| Multi-head attention | ❌ |

Attention, for reference: $\;\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$

## Training and evaluation

| Task | Status |
| --- | --- |
| Forward / backward pass driver | ❌ |
| Metrics tracking (loss, accuracy) | ❌ |
| Progress display (epochs, mini-batches) | ❌ |

## Suggested build order

Each step is testable on its own — write the `test.c` case before flipping the row to ✅.

1. `add` + `mul` (element-wise, both dtypes) — the simplest possible ops.
2. Autograd core: `grad`, `requires_grad`, graph tracking, `backward()` for those two ops.
3. `matmul`, `sum`, `mean` — enough for a linear model.
4. `sub`, `div`, `exp`, `log`, `pow` + shape ops (transpose, reshape, flatten, eye).
5. ReLU, sigmoid, tanh + MSE.
6. SGD + a training loop — learn XOR or a tiny regression end-to-end.
7. Linear + Sequential — a small MLP from the layers, not hand-wired ops.
8. softmax + cross-entropy — a first classifier.

## Later milestones

- **SIMD optimizations** for CPU tensor ops.
- **GPU support** — explore OpenCL / CUDA.
- **Serialization** — save/load weights; ONNX export for interop.
- **Python bindings** — a minimal API for experimentation and debugging.