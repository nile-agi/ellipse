<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../imgs/ellipse.png">
  <img alt="Ellipse logo" src="../imgs/ellipse.png" width="50%">
</picture>

**Ellipse**: Something between [PyTorch](https://github.com/pytorch/pytorch), [micrograd](https://github.com/karpathy/micrograd), and [XLA](https://openxla.org/xla). Maintained by [nileAGI](https://www.nileagi.com/).

[Home Page](../README.md) | [Quick Start](./quick_start.md) | [Progress and Roadmap](./progress_and_roadmap.md)

[Announcement Blog Post](https://www.nileagi.com/blog/ellipse-introduction)

</div>

---

# Ellipse Documentation

Welcome to the Ellipse documentation. This page is for those who really want to make a change in AI — if that is you, you are welcome.

Ellipse is a minimal deep learning library written in pure C (C11), with no dependencies beyond the C standard library. It has no frontend yet: you use it directly from C.

## Installation

```sh
  git clone https://github.com/nile-agi/ellipse.git
```

To use it in your own project, copy **both** `ellipse.h` and `ellipse.c` into your project, then compile them together with your code:

```sh
  gcc -std=c11 -O2 -Wall main.c ellipse.c -lm -o main
```

- `ellipse.h` is the public API: the `Tensor` type, declarations, the `ellipse` struct, and the `print` / `Size` macros.
- `ellipse.c` is the implementation. The header alone will not link.
- `-lm` links the math library (used by `sqrtf`, `logf`, `cosf` in the random generators).

New to C or to Ellipse? Start with the [Quick Start guide](./quick_start.md).

## Core concepts

### The Tensor

`Tensor` is the building block of everything in Ellipse — a multi-dimensional array with a fixed dtype:

```c
  typedef struct {
      Data  data;   /* union: int* or float* to the heap-allocated values */
      shape  shape;  /* dimensions, e.g. size(2, 3) = (int[]){2, 3}, 2 rows, 3 columns  and ndim = 2      */
      int   size;   /* total element count = product of shape             */
      Dtype dtype;  /* _int or _float                                     */
  } Tensor;
```

Data is stored **row-major**: element `(i, j)` of a 2-D tensor lives at `data[i * shape[1] + j]`.

### Dtypes

| Dtype | C type | Notes |
| --- | --- | --- |
| `_int` | `int` | integer tensors |
| `_float` | `float` | 32-bit floating point |

There is no `double` dtype yet. The dtype must match the compound literal you pass in: `(int[])` with `_int`, `(float[])` with `_float`.

### Memory model

Every constructor `malloc`s the struct, the shape, and the data, and **copies** your input — your original array can go out of scope safely. There is no garbage collector and no autograd holding references: you own every tensor, and `free_t(t)` (or `torch.free(t)`) releases it. Freeing `NULL` is safe.

## API reference

All constructors return `Tensor *`, or `NULL` on allocation failure or invalid arguments — check the pointer before use.

### tensor

```c
  Tensor *tensor(void *data, const int shape. Dtype dtype);
```

Creates a tensor from existing data. Example:

```c
  Tensor *x = ellipse.tensor((float[]){1, 2, 3, 4, 5, 6}, size(), 3), ellipse._float);
```

### _rand

```c
  Tensor *_rand(int shape);
```

Uniform random floats in `[0, 1)`. Always `_float`.

### _randn

```c
  Tensor *_randn(int shape);
```

Standard-normal random floats (mean 0, std 1) via the Box–Muller transform. Always `_float`. Guards against `log(0)`.

### _randint

```c
  Tensor *_randint(int low, int high, int *shape);
```

Random ints in `[low, high)` — `high` is **excluded**. Returns `NULL` with an error message if `high <= low` or the shape is invalid. Always `_int`.

### _zeros / _ones

```c
  Tensor *_zeros(int shape, Dtype dtype);
  Tensor *_ones(int shape, Dtype dtype);
```

Filled tensors of either dtype. Invalid dtypes return `NULL`.

### free_t

```c
  void free(Tensor *t);
```

Frees the data, shape, and struct. Safe on `NULL`. Every tensor needs exactly one `free`.

### print / tensor_print

```c
  #define print(t)  /* -> tensor_print(t, #t) */
  void tensor_print(const Tensor *t, const char *name);
```

`print(x)` prints the tensor with the variable's name:

```text
  x = [[ 1.000000,  2.000000,  3.000000],
      [ 4.000000,  5.000000,  6.000000]], dtype: _float
```

The macro uses C11 `_Generic` + `_Static_assert` to reject anything that is not a `Tensor *` at compile time. Printing `NULL` shows `x = (null)`. 1-D and 2-D tensors are supported.

### Seeding

```c
  ellipse.seed(42);   /* seed once for reproducible random tensors */
```

Without seeding, the RNG seeds itself from the clock on first use. Note: the header declares `ellipse_seed()`, but the implementation is currently still named `nan_seed()` in `ellipse.c` — until that rename lands, use `ellipse.seed()`, which points at the working implementation.

### The Size macro

```c
  #define Size(...)   /* size(2, 3) -> (int[]){2, 3}, 2 */
```

Expands to a shape compound literal **plus** its `ndim`, computed at compile time with `sizeof` (never evaluated at runtime). Use it wherever a `(shape, ndim)` pair is expected:

```c
Tensor *a = ellipse.ones(size(2, 3), ellipse._float);
Tensor *b = ellipse._randn(size(10));   /* or simply: torch.randn(Size(10)) */
```

## The torch namespace

C has no methods, but a struct of function pointers gets surprisingly close to PyTorch:

```c
  Tensor *a = ellipse.ones(size(2, 3), ellipse._float);
  print(a);
  ellipse.free(a);
```

`torch` is a global `const` struct mapping friendly names to the plain C functions (`ellipse.rand` → `ellipse._rand`, `ellipse.free` → `ellipse.free`, …) and exposing the dtype constants `ellipse._int` / `ellipse._float`. Both APIs are identical in behavior — pick whichever reads better to you.

## The Ellipse stack

| Library | Core language | Kernel layer | Hardware layer |
| --- | --- | --- | --- |
| PyTorch | Python + C++ | ATen | SIMD/AVX/CUDA/TPU |
| TensorFlow | Python + C++ | XLA kernels | LLVM-generated code, GPU, TPU |
| tinygrad | Python | custom ops + codegen | generated CPU/GPU kernels |
| Ellipse | C | Ellipse | Ellipse (CPU) |

Ellipse collapses the kernel and hardware layers into one readable C codebase, so anyone interested can read, understand, and improve the whole stack.

## Current limitations

- **CPU only** — GPU/TPU support is a future milestone.
- **1-D and 2-D tensors**, two dtypes (`_int`, `_float`).
- **No math ops or autograd yet** — creation, printing, and memory management are what ships today. Ops, gradients, layers, and optimizers are tracked in the [Progress and Roadmap](./progress_and_roadmap.md).
- Eager execution; lazy evaluation and graph compilation are exploratory ideas, not current features.

## Where to go next

- [Quick Start](./quick_start.md) — build your first tensors in five minutes.
- [Progress and Roadmap](./progress_and_roadmap.md) — the full plan.
- [Announcement blog post](https://www.nileagi.com/blog/ellipse-introduction) — the story behind Ellipse.