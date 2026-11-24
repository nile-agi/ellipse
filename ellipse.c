// ellipse.c

#define ELLIPSE_IMPLEMENTATION
// #ifndef ELLIPSE_IMPLEMENTED

#include "ellipse.h"

/* ---- seeding: call once, or call ellipse_seed() for reproducibility ---- */
static int ellipse_seeded = 0;

void ellipse_seed(unsigned int seed){
    srand(seed);
    ellipse_seeded = 1;
}

static void nan_ensure_seeded(void) {
    if (!ellipse_seeded) {
        srand((unsigned int)time(NULL));
        ellipse_seeded = 1;
    }
}

/*
    size of tensor
    -------------
    This define the total size of array which is of very important for initializing Tensor.

    Help to determine the total size of the data passed in Tensor.

    Note:
        This is for Internal Use only.
*/
static size_t  _size(size s) {
    size_t sizet = 1;
    for (int i = 0; i < s.ndim; i++) {
        sizet *= (size_t)s.dims[i];
    }
    return sizet;
}
// end of size

/*
    ndim of tensor
    --------------
    This define dimension of the tensor or array.
*/
// static int _ndim(const int *shape) {
//     return (int)sizeof(shape)/sizeof(shape[0]);
// }

/*
    ellipse.tensor(data, shape, dtype)
    ---------------------------------
    Initialize and Define Tensor *, that pass in array of data, size and data type(dtype).
    This will give the freedom for different operations and grad calculation.

    ndim - dimension type of your Tensor, if it is 2D you will pass - ndim = 2 and if it is 1D, you will pass - ndim = 1

    For now Tensor it suport only 2 type of dimensions , 1D and 2D

    Syntax:
        Tensor *x = ellipse.tensor((float[]){0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, size(2, 3), ellipse._float);

        print(x);

        ellipse.free(x);

        Output:
            x = [[ 0.100000,  0.200000,  0.300000],
                [ 0.400000,  0.500000,  0.600000]], dtype: _float

        Tensor *x = ellipse.tensor((int[]){1, 2, 3, 4}, size(4), ellipse._int);

         print(x);

        ellipse.free(x);

        Output:
            x = [     1,      2,      3,      4], dtype: _int
*/
Tensor *tensor(Tensor partial, size sz, Dtype dtype)
{
    Tensor *t = malloc(sizeof *t);
    if (!t) return NULL;

    t->size.ndim = sz.ndim;
    t->size.dims = malloc(sizeof(int) * (size_t)sz.ndim);
    if (!t->size.dims) { free(t); return NULL; }
    memcpy((void *)t->size.dims, sz.dims, sizeof(int) * (size_t)sz.ndim);

    t->t_size = _size(sz);          /* _size must return size_t — see below */
    t->dtype  = dtype;

    switch (dtype) {
    case _int: {
        t->data._int = malloc(sizeof(int) * t->t_size);
        if (!t->data._int) { free((void *)t->size.dims); free(t); return NULL; }

        if (partial.dtype == _int) {
            /* fast path: source already int[] */
            memcpy(t->data._int, partial.data._int,
                   sizeof(int) * t->t_size);
        } else {
            /* slow path: read the member indicated by partial.dtype */
            const float *src = partial.data._float;
            for (size_t i = 0; i < t->t_size; ++i)
                t->data._int[i] = (int)src[i];
        }
        break;
    }
    case _float: {
        t->data._float = malloc(sizeof(float) * t->t_size);
        if (!t->data._float) { free((void *)t->size.dims); free(t); return NULL; }

        if (partial.dtype == _float) {
            memcpy(t->data._float, partial.data._float,
                   sizeof(float) * t->t_size);
        } else {
            const int *src = partial.data._int;
            for (size_t i = 0; i < t->t_size; ++i)
                t->data._float[i] = (float)src[i];
        }
        break;
    }
    default:
        fprintf(stderr, "dtype must be _int or _float\n");
        free((void *)t->size.dims);
        free(t);
        return NULL;
    }

    return t;
}
// end of Tensor


/*
    ellipse._rand(size):
    ------------------
    Generates a tensor of random floats from a uniform distribution on [0, 1) / (mean 0, standard deviation 1).

    It uses the current generator.

        Formula:
            rand() / (RAND_MAX + 1)

    Syntax:
        ellipse._rand(size, ndim)

        Return:
            Tensor *

    Example:
        Tensor *a = ellipse._rand(size(2, 3));

        print(a);

        ellipse.free(a);

        Output:
            a = [[ 0.521612, -0.456259,  1.207947],
                 [-0.387763, -0.670170, -0.307190]], dtype: _float

        Tesnor *b = ellipse._rand((size(10));

        print(b);

        ellipse.free(b);

        Output:
            b = [-0.701645, -0.204627, 0.202153, -1.604854, -0.404581, -0.416625, 0.531158, 1.881386, -0.224445, -1.828255], dtype: _float
*/
Tensor *_rand(size size){
    Tensor *t = (Tensor *) malloc(sizeof(*t));
    if (!t) return NULL;

    size_t sizet = _size(size);
    t->size.ndim = size.ndim;
    t->t_size= sizet;
    t->dtype = _float;

    t->size.dims = (int *)malloc((sizeof(*t->size.dims)) * (size_t)t->size.ndim);
    if (!t->size.dims) { free(t); return NULL; }
    memcpy((void*)t->size.dims, size.dims, sizeof(int) * (size_t)t->size.ndim);

    t->data._float = (float *)malloc((sizeof(*t->data._float)) * t->t_size);
    if (!t->data._float) { free((void*)t->size.dims); free(t); return NULL; }

    nan_ensure_seeded();

    const float denom = (float)RAND_MAX + 1.0f;         /* strictly > RAND_MAX -> result < 1.0f */
    for (size_t i = 0; i < sizet; i++){
        t->data._float[i] = (float)rand() / denom;
    }

    return t;
}
//    end of randn


/*
    ellipse._randn(size)
    ------------------
    Generates a tensor of random floats from the standard normal distribution, mean 0, standard deviation 1, via the Box–Muller transform.

    It uses the current generator.

    Formula:
            sqrt(-2 * ln(u1)) * cos(2π * u2),  u1, u2 ~ U(0, 1)

                where:
                    u1 :    random numbers from a uniform distribution between 0 and 1 (inclusive).
                            = rand() / RAND_MAX

                    u2 :    random numbers from a uniform distribution between 0 and 1 (inclusive).
                            = rand() / RAND_MAX

    Syntax:
        ellipse._randn(size)

        Return:
            Tensor *

    Example:
        Tensor *a = ellipse._randn(size(2, 3));

        print(a);

        ellipse.free(a);

        Output:
            a = [[ 0.762251,  0.144962,  0.379993],
                 [ 0.547818,  0.175760,  0.999096]], dtype: _float

        Tesnor *b = ellipse._randn((size(10));

        print(b);

        ellipse.free(b);

        Output:
            b = [0.762892, 0.931061, 0.339630, 0.157129, 0.862674, 0.953750, 0.670879, 0.455913, 0.531983, 0.033875], dtype: _float
*/

Tensor *_randn(size size){
    Tensor *t = (Tensor *)malloc(sizeof(*t));
    if (!t) return NULL;

    size_t sizet = _size(size);
    t->size.ndim = size.ndim;
    t->t_size = sizet;
    t->dtype = _float;

    t->size.dims = (int *)malloc((sizeof(*t->size.dims)) * (size_t)t->size.ndim);
    if (!t->size.dims) { free(t); return NULL; }
    memcpy((void*)t->size.dims, size.dims, sizeof(int)*(size_t)t->size.ndim);

    t->data._float = (float *)malloc((sizeof(*t->data._float)) * t->t_size);
    if (!t->data._float) { free((void*)t->size.dims); free(t); return NULL; }

    nan_ensure_seeded();

    const float denom = (float)RAND_MAX + 1.0f;
    for (size_t i = 0; i < t->t_size; i++) {
        float u1, u2;

        /* Box–Muller needs u1 in (0, 1]; guard against logf(0) = -inf → NaN */

        do {
            u1 = (float)rand() / denom;
        } while (u1 <= 0.0f);

        u2 = (float)rand() / denom;

        t->data._float[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    }
    return t;
}
// end of rnadn


/*
    ellipse._randint(low, high, size)
    -------------------------------
        Generates a tensor of random integers between low (inclusive) and high (exclusive).
        It uses the current generator.

        Formular:
            low + (rand() % (high - low + 1))

    Syntax:
        ellipse._randint(low, high, size)

        Return:
            Tensor *

    Example:
        Tensor *a = ellipse._randint(1, 10, size(5, 2));

        print(a);

        ellipse.free(a);

        Output:
            a = [[4, 7],
                 [2, 2],
                 [6, 6],
                 [2, 3],
                 [1, 5]], dtype: _int

        Tesnor *b = ellipse._randint(20, 30, size(10));

        print(b);

        ellipse.free(b);

        Output:
            b = [    26,     20,     28,     24,     29,     20,     23,     23,     23,     24], dtype: _int
*/
Tensor *_randint(int low, int high, size size){
    /* --- 1. validate BEFORE allocating anything --- */
    if (!size.dims || size.ndim < 0) {
        fprintf(stderr, "randint: bad size/ndim\n");
        return NULL;
    }
    if (high <= low) {      // high is exclusive; empty or reversed range
        fprintf(stderr, "randint: empty range [%d, %d)\n", low, high);
        return NULL;
    }

    /* --- 2. only now allocate --- */
    Tensor *t = (Tensor *)malloc(sizeof(*t));
    if (!t) return NULL;

    size_t sizet = _size(size);
    t->size.ndim = size.ndim;
    t->t_size = sizet;
    t->dtype = _int;

    t->size.dims = (int *)malloc((sizeof(*t->size.dims)) * (size_t)t->size.ndim);
    if (!t->size.dims) { free(t); return NULL; }
    memcpy((void*)t->size.dims, size.dims, sizeof(int) * (size_t)t->size.ndim);

    t->data._int = (int *)malloc((sizeof(*t->data._int)) * t->t_size);
    if (!t->data._int) { free((void*)t->size.dims); free(t); return NULL; }

    nan_ensure_seeded();

    /* --- 3. fill --- */
    int range = high - low;         // >= 1 here, guaranteed by the guard above
    for (size_t i = 0; i < t->t_size; i++)
    {
        t->data._int[i] = low + rand() % range;
    }

    return t;
}
// end of randint

Tensor *_ones(size size, Dtype dtype){
    Tensor *t = (Tensor *)malloc(sizeof(*t));
    if (!t) return NULL;

    size_t sizet = _size(size);

    t->size.ndim = size.ndim;
    t->t_size = sizet;
    t->dtype = dtype;

    t->size.dims = (int *)malloc((sizeof(*t->size.dims)) * (size_t)t->size.ndim);
    if (!t->size.dims) { free(t); return NULL; }

    memcpy((void*)t->size.dims, size.dims, sizeof(int)*(size_t)t->size.ndim);

    switch (t->dtype)
    {
    case _int:
        t->data._int = (int *)malloc((sizeof(*t->data._int)) * t->t_size);
        if (!t->data._int) { free((void*)t->size.dims); free(t); return NULL; }
        for (size_t i = 0; i < t->t_size; i++)
        {
            t->data._int[i] = 1;
        }

        break;

    case _float:
        t->data._float = (float *)malloc((sizeof(*t->data._float)) * t->t_size);
        if (!t->data._float) { free((void*)t->size.dims); free(t); return NULL; }
        for (size_t i = 0; i < t->t_size; i++)
        {
            t->data._float[i] = 1.f;
        }

        break;

    default:
        fprintf(stderr, "_ones: invalid dtype\n");
        free((void*)t->size.dims);
        free(t);
        return NULL;
    }
    return t;
}

Tensor *_zeros(size size, Dtype dtype){
    Tensor *t = (Tensor *)malloc(sizeof(*t));
    if (!t) return NULL;

    size_t sizet = _size(size);

    t->size.ndim = size.ndim;
    t->t_size = sizet;
    t->dtype = dtype;

    t->size.dims = (int *)malloc((sizeof(*t->size.dims)) * (size_t)t->size.ndim);
    if (!t->size.dims) { free(t); return NULL; }

    memcpy((void*)t->size.dims, size.dims, sizeof(int)*(size_t)t->size.ndim);

    switch (t->dtype)
    {
    case _int:
        t->data._int = malloc((sizeof(*t->data._int)) * (size_t)t->t_size);
        if (!t->data._int) { free((void*)t->size.dims); free(t); return NULL; }
        for (size_t i = 0; i < t->t_size; i++) {
            t->data._int[i] = 0;
        }

        break;

    case _float:
        t->data._float = (float *)malloc((sizeof(*t->data._float)) * t->t_size);
        if (!t->data._float) { 
            free((void*)t->size.dims); free(t); 
            return NULL; 
        }
        for (size_t i = 0; i < t->t_size; i++) {
            t->data._float[i] = 0.f;
        }

        break;

    default:
        fprintf(stderr, "_ones: invalid dtype\n");
        free((void*)t->size.dims);
        free(t);
        return NULL;
    }
    return t;
}

// Tensor *range(void *start, void *end, void *steps){
//     Tensor *t = (Tensor *)malloc(sizeof(Tensor));

//    return t;
// }

/*
    free tensor
    -----------
    Free the assigned Tensor * from the memory address, that you don't use any more from the memory.

    Syntax:
        ellipse.free(Tensor *)

        Return:
                NONE

    Example:
        Tensor *w = ellipse.randd(size(2,5));

        print(w);

        // this free the w from the memory.
        ellipse.free(w);

        Output:
            w = [[-0.624843,  0.296725],
                 [ 0.748386,  1.131429],
                 [ 0.006623, -0.412209],
                 [-0.198537,  0.445824],
                 [ 1.724266,  1.211697],
                 [ 2.357287,  0.123609],
                 [ 0.450936, -0.115897],
                 [-1.349130,  0.659726],
                 [-1.197287,  1.331525],
                 [-0.267274,  0.136951]], dtype: _float

*/
void free_t(Tensor *m){

    if (!m) return;

    switch (m->dtype)
    {
    case _int:
        free(m->data._int);
        break;

    case _float:
        free(m->data._float);
        break;

    default:
        break;
    }

    free((void*)m->size.dims);
    free(m);
}
// end of free_t

/*
    tensor_print
    ------------
    This function help to define how I want the Tensor's to be printed for user, I choose this style inspired from Pytorch and Python.

    This function just print and not return anything (void function).

    What amaizing is that this function is not the one that actually printing our output although I mentiioned earlyer that is does.

    I define a macro function called print that take in Tensor *t and special preprocessor directive (#t) and implement tensor_print function in it's body.

    Note: At torch.h header I defined
        void tensor_print(Tensor *t, const char *);
        #define print(t) tensor_print(t, #t)

    Now:
        void tensor_print(Tensor *t, const char *);
            - This declare the tensor_print function
            - it takes Tensor structure (Tensor *t) and const cahr *,.. const keyword indicate function will not modify the string pointer.

        #define print(t) tensor_print(t, #t)
            - This define print macro (#define), This tell compiler to link it during compilation stage.
            - print(t) is the macro name, whenever print(t) is used the compiler replace it with tensor_print(t, #t).
            - In turn tensor_print(t, #t) substituted with tensor_print(Tensor *t) function.
            - t is Tensor pointer to structure Tensor *t from tensor_print function.
            - #t is a special directive preprocessor, this is the macro substitution that expands to the name of the variable or expression that was originally passes to the macro print(t).
            - In other words, If you call print(my_tensor), then #t will be replaced with my_tensor
*/

void tensor_print(const Tensor *t, const char * name){
    if (!t) { printf("%s = (null)\n", name); return; }

    printf("%s = ", name);

    switch (t->size.ndim)
    {
    case 1:
        switch (t->dtype)
        {
            case _int:
                printf("[");
                for (size_t i = 0; i < t->t_size; i++)
                {
                    if (i == t->t_size - 1)
                    {
                        printf("%2d", t->data._int[i]);
                        break;
                    }
                    if (i == 0)
                    {
                        printf("%d, ", t->data._int[i]);
                    } else
                    {
                        printf("%2d, ", t->data._int[i]);
                    }

                }
                printf("], dtype: %s\n", "_int");
                break;

            case _float:
                printf("[");
                for (size_t i = 0; i < t->t_size; i++)
                {
                    if (i==t->t_size - 1)
                    {
                        printf("%8f", t->data._float[i]);
                        break;
                    }
                    if (i == 0)
                    {
                        printf("%f, ", t->data._float[i]);
                    } else
                    {
                        printf("%8f, ", t->data._float[i]);
                    }

                }
                printf("], dtype: %s\n", "_float");
                break;

            default:
                break;
        }

        break;

    case 2:
        switch (t->dtype)
            {
            case _int:
                printf("[");
                for (int i = 0; i < t->size.dims[0]; i++)
                {
                    if (i > 0 && i < t->size.dims[0])
                    {
                        printf("     [");
                    } else
                    {
                        printf("[");
                    }

                    for (int j = 0; j < t->size.dims[1]; j++)
                    {
                        // if (j == 0)
                        // {
                        //     printf("%d", t->data._int[i*t->size[1] + j]);
                        //     continue;
                        // }

                        // if (j > 0)
                        // {
                        //     printf(", %2d", t->data._int[i*t->shape[1] + j]);
                        // }
                        if (j < t->size.dims[1]-1)
                        {
                            printf("%3d, ", t->data._int[i*t->size.dims[1] + j]);
                        } else
                        {
                            printf("%3d", t->data._int[i*t->size.dims[1] + j]);
                        }

                    }

                    if (i<t->size.dims[0]-1)
                    {
                    printf("],\n");
                    } else
                    {
                        printf("]");
                    }
                }
                // switch to print dtype at the end of the tensor x = [[...]], dtype = _int/_float

                printf("], dtype: %s\n", "_int");
                break;

                case _float:
                printf("[");
                for (int i = 0; i < t->size.dims[0]; i++)
                {
                    if (i > 0 && i < t->size.dims[0])
                    {
                        printf("     [");
                    } else
                    {
                        printf("[");
                    }

                    for (int j = 0; j < t->size.dims[1]; j++)
                    {
                        if (j < t->size.dims[1]-1)
                        {
                            printf("%9f, ", t->data._float[i*t->size.dims[1] + j]);
                        } else
                        {
                            printf("%9f", t->data._float[i*t->size.dims[1] + j]);
                        }

                    }

                    if (i<t->size.dims[0]-1)
                    {
                    printf("],\n");
                    } else
                    {
                        printf("]");
                    }

                }

                printf("], dtype: %s\n", "_float");
                break;

            default:
                break;

            }
        break;

    default:
        break;
    }
}

const ellipse_api ellipse = {
    ._float  = _float,      /* the enum value */
    ._int    = _int,
    .tensor  = tensor,
    .zeros   = _zeros,
    .ones    = _ones,
    .rand    = _rand,
    .randn   = _randn,
    .randint = _randint,
    .free    = free_t,
    .seed    = ellipse_seed,
};

// end of the tensor_print

// #endif
// #endif
