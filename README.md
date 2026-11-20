<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="imgs/ellipse.png">
  <img alt="Ellipse logo" src="imgs/ellipse.png" width="50%">
</picture>

# Ellipse

A PyTorch-like AI engine built from scratch in pure C.

Something between [tinygrad](https://tinygrad.org/), [PyTorch](https://github.com/pytorch/pytorch), [micrograd](https://github.com/karpathy/micrograd), and [XLA](https://openxla.org/xla).
Maintained by [nileAGI](https://www.nileagi.com/).

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: C11](https://img.shields.io/badge/language-C11-A8B9CC.svg?logo=c&logoColor=white)](ellipse.h)
[![GitHub stars](https://img.shields.io/github/stars/nile-agi/ellipse?cache_bust=1)](https://github.com/nile-agi/ellipse/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/nile-agi/ellipse)](https://github.com/nile-agi/ellipse/commits/main)
[![Open issues](https://img.shields.io/github/issues/nile-agi/ellipse)](https://github.com/nile-agi/ellipse/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing) -->

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: C11](https://img.shields.io/badge/language-C11-A8B9CC.svg?logo=c&logoColor=white)](ellipse.h)
[![GitHub stars](https://img.shields.io/github/stars/nile-agi/ellipse?cache_bust=1)](https://github.com/nile-agi/ellipse/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/nile-agi/ellipse)](https://github.com/nile-agi/ellipse/forks)
[![GitHub watchers](https://img.shields.io/github/watchers/nile-agi/ellipse)](https://github.com/nile-agi/ellipse/watchers)
[![Last commit](https://img.shields.io/github/last-commit/nile-agi/ellipse)](https://github.com/nile-agi/ellipse/commits/main)
[![Open issues](https://img.shields.io/github/issues/nile-agi/ellipse)](https://github.com/nile-agi/ellipse/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

[Documentation](Documentation/documentation.md) | [Quick Start](Documentation/quick_start.md) | [Progress and Roadmap](Documentation/progress_and_roadmap.md) | [Announcement Blog Post](https://www.nileagi.com/blog/ellipse-introduction)

</div>

---

Ellipse is a lightweight, minimalistic, high-performance numerical computing library for deep learning, written in pure C with no dependencies beyond the C standard library. It targets low-resource environments and serves as a foundational toolkit for machine learning enthusiasts, embedded developers, and researchers who want to experiment with deep learning concepts in an efficient, resource-conscious manner.

> **Status:** Ellipse is early-stage. Today it provides tensor creation (`tensor`, `rand`, `randn`, `randint`, `zeros`, `ones`), two dtypes (`_int`, `_float`), 1-D/2-D shapes, a PyTorch-style `torch.*` API, printing, and manual memory management. Math ops, autograd, layers, and optimizers are on the [roadmap](Documentation/progress_and_roadmap.md).

## Key Features

- **Lightweight design** — two files (`ellipse.h` + `ellipse.c`), zero external dependencies.
- **Pure C (C11)** — portable, readable, and built for low-level control.
- **Two APIs** — plain C (`tensor(...)`, `print(x)`, `free_t(x)`) or PyTorch-style (`torch.randn(...)`, `torch.free(x)`).
- **Seedable randomness** — uniform, standard-normal (Box–Muller), and integer tensors.
- **Modular architecture** — designed to grow into ops, autograd, layers, and optimizers.

## Who Is This For?

Ellipse is a good fit if you want to:

- Understand the inner workings of a deep learning library from the ground up.
- Run simple neural networks in resource-limited environments.
- Prototype and test custom ML operations in C.

## Quickstart

You need a C11 compiler (GCC or Clang) — nothing else.

```sh
git clone https://github.com/nile-agi/ellipse.git
cd ellipse
```

Create `main.c` next to `ellipse.h` and `ellipse.c`:

```c
#include "ellipse.h"

int main(void) {
    /* Plain C API: data, shape, ndim, dtype */
    Tensor *x = tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2, _float);
    print(x);          /* prints the tensor together with its variable name */
    free_t(x);

    /* PyTorch-style API */
    torch.seed(42);                            /* optional: reproducible randoms */
    Tensor *w = torch.randn((int[]){2, 3}, 2); /* 2x3, standard normal */
    print(w);
    torch.free(w);
    return 0;
}
```

Build and run (on macOS `gcc` is Clang — that works too):

```sh
gcc -std=c11 -O2 -Wall main.c ellipse.c -lm -o main
./main
```

Output (then `w` prints in the same format, with random values):

```text
x = [[ 1.000000,  2.000000,  3.000000],
     [ 4.000000,  5.000000,  6.000000]], dtype: _float
```

More examples: [Quick Start guide](Documentation/quick_start.md) · [Documentation](Documentation/documentation.md).

## Contributing

Ellipse is open for contributions! Whether you are fixing bugs, adding new features, or experimenting with optimizations, your input is welcome — open an issue or a pull request.

## License

Ellipse is released under the [MIT License](LICENSE) — free to use, modify, and distribute; just keep the copyright notice.