<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../imgs/ellipse.png">
  <img alt="Ellipse logo" src="../imgs/ellipse.png" width="50%">
</picture>

**Ellipse**: Something between [PyTorch](https://github.com/pytorch/pytorch), [micrograd](https://github.com/karpathy/micrograd), and [XLA](https://openxla.org/xla). Maintained by [nileAGI](https://www.nileagi.com/).

[Home Page](../README.md) | [Documentation](./documentation.md) | [Progress and Roadmap](./progress_and_roadmap.md)

[Announcement Blog Post](https://www.nileagi.com/blog/ellipse-introduction)

</div>

---

# Quick Start Guide

This guide assumes no prior knowledge of C, PyTorch, or any other deep learning framework. By the end you will have created, printed, and freed your first tensors.

> **Heads-up:** Ellipse is early-stage. Today it does tensor creation, printing, and memory management. Math operations and automatic differentiation are being built — see the [roadmap](./progress_and_roadmap.md).

## 1. Get the code

```sh
git clone https://github.com/nile-agi/ellipse.git
cd ellipse
```

Two files are the whole library:

- `ellipse.h` — the API: types, function declarations, and macros.
- `ellipse.c` — the implementation: the actual function bodies.

To use Ellipse in your own project, copy **both** files into your project folder. The header alone is not enough — it only declares the functions; the code lives in `ellipse.c`.

## 2. What you need

A C11 compiler (GCC or Clang) and nothing else — no external dependencies. On macOS, `gcc` is an alias for Clang, which works fine.

## 3. Your first tensor

Create `main.c` next to `ellipse.h` and `ellipse.c`:

```c
#include "ellipse.h"

int main(void) {
    Tensor *x = tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2, _float);
    print(x);
    free_t(x);
    return 0;
}
```

Build and run — note that `ellipse.c` is compiled **together with** your file:

```sh
gcc -std=c11 -O2 -Wall main.c ellipse.c -lm -o main
./main
```

Output:

```text
x = [[ 1.000000,  2.000000,  3.000000],
     [ 4.000000,  5.000000,  6.000000]], dtype: _float
```

If you see `undefined symbols` / `undefined reference` errors, you forgot `ellipse.c` in the compile command.

## 4. Breaking down the `tensor()` call

That one line looks dense, so let's take it apart:

| Piece | Meaning |
| --- | --- |
| `Tensor *x` | A variable pointing to a `Tensor` — Ellipse's multi-dimensional array type. |
| `tensor(...)` | Allocates a new tensor on the heap and **copies** your data into it. |
| `(float[]){1, 2, 3, 4, 5, 6}` | The data itself. `(float[])` is the array's type, `{...}` are the values. |
| `(int[]){2, 3}` | The shape: 2 rows, 3 columns. |
| `2` | `ndim` — how many dimensions the shape has. Currently `1` or `2`. |
| `_float` | The dtype. Two exist today: `_int` and `_float`. |

Two rules to remember:

1. The data array type must match the dtype: `(float[])` goes with `_float`, `(int[])` goes with `_int`.
2. The number of values must equal the product of the shape (`2 × 3 = 6` above).

> Older versions of this guide showed `FLOAT32`, `FLOAT64`, and a `true`/`false` gradient flag. Those are gone: today the dtypes are `_int` and `_float`, and gradient tracking is not implemented yet.

## 5. Printing and freeing

- `print(x)` is a macro that prints the tensor **together with the variable's name**, PyTorch-style. It only accepts a `Tensor *` — passing anything else fails at compile time with a clear message.
- `free_t(x)` releases everything the tensor owns (data, shape, struct). There is no garbage collector: **every tensor you create, you must free.**

## 6. More ways to create tensors

From existing integer data:

```c
Tensor *x = tensor((int[]){1, 2, 3, 4}, (int[]){2, 2}, 2, _int);
print(x);
free_t(x);
```

```text
x = [[  1,   2],
     [  3,   4]], dtype: _int
```

Filled with zeros or ones:

```c
Tensor *z = _zeros((int[]){2, 2}, 2, _int);
Tensor *o = _ones((int[]){4}, 1, _float);
print(z);
print(o);
free_t(z);
free_t(o);
```

```text
z = [[  0,   0],
     [  0,   0]], dtype: _int
o = [1.000000, 1.000000, 1.000000, 1.000000], dtype: _float
```

Random tensors (your values will differ):

```c
Tensor *u = _rand((int[]){2, 3}, 2);            /* uniform floats in [0, 1)   */
Tensor *n = _randn((int[]){2, 3}, 2);           /* standard normal (mean 0, std 1) */
Tensor *i = _randint(1, 10, (int[]){2, 3}, 2);  /* ints in [1, 10) — high excluded */
print(u);
print(n);
print(i);
free_t(u);
free_t(n);
free_t(i);
```

Want the same "random" numbers on every run? Seed once at the start of `main`:

```c
torch.seed(42);
```

## 7. The PyTorch-style API

Every function is also available through the `torch` struct, so call sites read like PyTorch:

| Plain C | PyTorch-style |
| --- | --- |
| `tensor(data, shape, ndim, dtype)` | `torch.tensor(data, shape, ndim, dtype)` |
| `_rand(shape, ndim)` | `torch.rand(shape, ndim)` |
| `_randn(shape, ndim)` | `torch.randn(shape, ndim)` |
| `_randint(low, high, shape, ndim)` | `torch.randint(low, high, shape, ndim)` |
| `_zeros(shape, ndim, dtype)` | `torch.zeros(shape, ndim, dtype)` |
| `_ones(shape, ndim, dtype)` | `torch.ones(shape, ndim, dtype)` |
| `free_t(t)` | `torch.free(t)` |

The `Size(...)` macro builds the shape **and** its dimension count for you: `Size(2, 3)` expands to `(int[]){2, 3}, 2`. And `torch._float` / `torch._int` mirror the dtype constants.

```c
#include "ellipse.h"

int main(void) {
    torch.seed(42);

    Tensor *a = torch.ones(Size(2, 3), torch._float);
    Tensor *b = torch.zeros(Size(2, 3), torch._float);
    Tensor *r = torch.randn(Size(2, 3));

    print(a);
    print(b);
    print(r);   /* random values — yours will differ */

    torch.free(a);
    torch.free(b);
    torch.free(r);
    return 0;
}
```

```text
a = [[ 1.000000,  1.000000,  1.000000],
     [ 1.000000,  1.000000,  1.000000]], dtype: _float
b = [[ 0.000000,  0.000000,  0.000000],
     [ 0.000000,  0.000000,  0.000000]], dtype: _float
```

## Where to go next

- [Documentation](./documentation.md) — concepts and the full API reference.
- [Progress and Roadmap](./progress_and_roadmap.md) — what works today and what is coming: math ops, autograd, layers, optimizers.