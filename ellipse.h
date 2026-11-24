// ellipse.h

/*
    Project name: ellipse
    Creater: nileAGI
    Start Date: Jan 01 2026

    This is the minimal library for Machine Learning, Pytorch, this project inspired by by micrograd by Andrej Karpathy, micrograd by Geo Hotz.
    Hopefull this project will go well and expand to inference engine creation capability, like gguf library by George Gragernov.
    The name nan, does not mean anything technically but I like it because it is pellindrom name, and for it minimallism nan is the perfect name.😍

    This project I do it without internet or powefull LLM's,to be host I'm just using delta 😃,... So most of the library code wrote from my head and my understanding,
    So if you find any error, it may be due to syntax or my lack of understanding.

    For nan is only CPU implemented.

    Will continue adding notes here,...
*/

#ifndef ELLIPSE_H
#define ELLIPSE_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

/*
    Union Data
    ----------
    Used to define pointer to integer and pointer to float data.

    This is called inside the Tensor structure as Data data;

    It give the Tensor Structure more room and capability to carry and manupulate different data of different data types
*/
typedef union {
    float   *_float;
    int     *_int;
} Data;
// end of union Data

/*
    Enum Dtype
    ----------
    Used to define integer and float data type as _int and _float respectively.
    
    This help to carry data type that latter passed in Tensor structure.

    This help for dynamic Tensor define (memory allocation), manipulation and operations
*/
typedef enum {
    _int    = 0,
    _float  = 1,
} Dtype;
// end Enum Dtype

/*
 * size
 * ----
 *  dims + ndims. Built by the size(...) macro from a compound liter
 */
typedef struct {
    const int   *dims;
    int         ndim;
} size;


/*
    Tensor Structure
    ----------------
    This is the skelleton or building block of Tensor's.

    It carry, data that also pointing to Data union with int and float pointers.

    shape that give us row and colums of the tensor, t->shape[0] - row and t->shape[1] - column.

    size this tell the length of the data, and it is used to allocate the memory for Data data, float or int.

    ndim this tell if your Tensor is 1D or 2D, also help to allocate memory for the shape, note that shape is the pointer, ndim is used as the length of the shape.

    dtype is the data type of the tensor, this help us to allocate proper memory location to tensor and help in defining defferent random's, also simplify print of tensor.
*/

typedef struct {
    Data    data;
    size    size;
    size_t  t_size;
    Dtype   dtype; 
} Tensor;
//end Tensor Structure


/* Shape-from-brace-list helper:
 *   size(2, 3)  →  int* dims, int ndim => (int[]){2, 3}, 2
 *   size(10)    →  int* dims, int ndim => (int[]){10},   1
 * The sizeof is never evaluated at runtime (C11 §6.5.3.4p2).
 */
#define size(...)                                                     \
    ((size){                                                          \
        .dims = (int[]){__VA_ARGS__},                                 \
        .ndim = (int)(sizeof((int[]){__VA_ARGS__}) / sizeof(int))     \
    })

/* Pick the FIRST argument so _Generic has something to dispatch on.
 * The sentinel makes 1-argument calls legal:  Arr(1)  →  FIRST(1, 0). */
#define ELLIPSE_FIRST_(a, ...) (a)
#define ELLIPSE_FIRST(...)     ELLIPSE_FIRST_(__VA_ARGS__, 0)


/*
 * Arr(...) → partially-initialized Tensor
 *   .data  : union holding the typed pointer  (int[]  or  float[])
 *   .dtype : Dtype inferred from the FIRST literal
 *   .size  : zero  (filled in by tensor())
 *   .t_size: zero  (filled in by tensor())
 *
 * Both branches of every _Generic are compiled (C11 requires it);
 * the one selected by the first literal's type is the one that runs.
 */


#define Arr(...)                                                                \
   ((Tensor){                                                                   \
       .data  = _Generic(ELLIPSE_FIRST(__VA_ARGS__),                            \
                         int:    (Data){ ._int   = (int[])  { __VA_ARGS__ } },  \
                         float:  (Data){ ._float = (float[]){ __VA_ARGS__ } },  \
                         double: (Data){ ._float = (float[]){ __VA_ARGS__ } },  \
                         default:(Data){ ._int   = (int[])  { __VA_ARGS__ } }), \
       .dtype = _Generic(ELLIPSE_FIRST(__VA_ARGS__),                            \
                         int: _int, float: _float, double: _float,              \
                         default: _int)                                         \
   })


/*
    PyTorch-style namespace
    -----------------------
    C has no 'torch.ones(...)' syntax on a type, but it can on a struct of function pointers. 
    Call sites read almost identically to PyTorch:
        Tensor *a = ellipse.ones(shape(2,3), _float);
        print(a, "a");
        ellipse.free(a);

    Not named 'nan' because <math.h> declare 'double nan(consta char *)'.
*/
typedef struct {
    /* dtype constants — read-only metadata, PyTorch-style */
    Dtype _int;
    Dtype _float;

    /* constructors */
    int *(*size) (int *);
    Tensor *(*tensor)(Tensor partial,size size, Dtype dtype);
    Tensor *(*rand)(size size);
    Tensor *(*randn)(size size);
    Tensor *(*randint)(int low, int high, size size);
    // Tensor *range(void *, void *, void *);
    Tensor *(*zeros)(size size, Dtype dtype);
    Tensor *(*ones)(size size, Dtype dtype);
    void (*free)(Tensor *);
    void (*seed)(unsigned int seed);
} ellipse_api;

extern const ellipse_api ellipse;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
    defining tensor_print function and print macro.

    Note:
        This explained in length on the tensor_print function, See at the bottom.
*/
void tensor_print(const Tensor *t, const char *name);   // const, named param

#define print(t)                                                            \
    do {                                                                    \
        _Static_assert(                                                     \
            _Generic((t),                                                   \
                     Tensor *:       1,                                     \
                     const Tensor *: 1,                                     \
                     int *: 1,                                              \
                     default:        0),                                    \
            "print expects a Tensor* / const Tensor*");                     \
        tensor_print((const Tensor *)(t), #t);                              \
    } while (0)

// end  defining tensor_print function and print macro.

#endif /* NAN_H */
