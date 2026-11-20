// ellipse.h

/*
    Project name: ellipse
    Creater: Isack odero
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
#include <math.h>

/*
    Union Data
    ----------
    Used to define pointer to integer and pointer to float data.

    This is called inside the Tensor structure as Data data;

    It give the Tensor Structure more room and capability to carry and manupulate different data of different data types
*/
typedef union 
{
    float *_float;
    int *_int;
}Data;
// end of union Data

/*
    Enum Dtype
    ----------
    Used to define integer and float data type as _int and _float respectively.
    
    This help to carry data type that latter passed in Tensor structure.

    This help for dynamic Tensor define (memory allocation), manipulation and operations
*/
typedef enum
{
    _int,
    _float,
} Dtype;
// end Enum Dtype

/*
    Tensor Structure
    ----------------
    This is the skelleton or building bloch of Tensor's.

    It carry, data that also pointing to Data union with int and float pointers.

    shape that give us row and colums of the tensor, t->shape[0] - row and t->shape[1] - column.

    size this tell the length of the data, and it is used to allocate the memory for Data data, float or int.

    ndim this tell if your Tensor is 1D or 2D, also help to allocate memory for the shape, note that shape is the pointer, ndim is used as the length of the shape.

    dtype is the data type of the tensor, this help us to allocate proper memory location to tensor and help in defining defferent random's, also simplify print of tensor.
*/
typedef struct {
    Data data;
    int *shape;
    int size;
    int ndim;
    Dtype dtype; 
} Tensor;
//end Tensor Structure

Tensor *tensor(void *data, const int *shape, Dtype dtype);
Tensor *_rand(int *shape, int ndim);
Tensor *_randn(int *shape, int ndim);
Tensor *_randint(int low, int high, int *shape, int ndim);
// Tensor *range(void *, void *, void *);
Tensor *_zeros(int *shape, int ndim, Dtype dtype);
Tensor *_ones(int *shape, int ndim, Dtype dtype);
void free_t(Tensor *);
void ellipse_seed(unsigned int seed);   /* seed the RNG once for reproducibility */

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
    int *(*shape) (int *);
    Tensor *(*tensor)(void *data,const int *shape, Dtype dtype);
    Tensor *(*rand)(int *shape , int ndim);
    Tensor *(*randn)(int *shape , int ndim);
    Tensor *(*randint)(int low, int high, int *shape, int ndim);
    // Tensor *range(void *, void *, void *);
    Tensor *(*zeros)(int *shape, int ndim, Dtype dtype);
    Tensor *(*ones)(int *shape, int ndim, Dtype dtype);
    void (*free)(Tensor *);
    void (*seed)(unsigned int seed);
} ellipse_api;

extern const ellipse_api ellipse;

/* Shape-from-brace-list helper:
 *   NAN_SHAPE(2, 3)  →  (int[]){2, 3}, 2
 *   NAN_SHAPE(10)    →  (int[]){10},   1
 * The sizeof is never evaluated at runtime (C11 §6.5.3.4p2).
 */
#define Size(...) \
    (int[]){__VA_ARGS__}, (int)(sizeof((int[]){__VA_ARGS__}) / sizeof(int))


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
                     default:        0),                                    \
            "print expects a Tensor* / const Tensor*");                     \
        tensor_print((const Tensor *)(t), #t);                              \
    } while (0)

// end  defining tensor_print function and print macro.

#endif /* NAN_H */