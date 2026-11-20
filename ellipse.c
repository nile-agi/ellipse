// ellipse.c

#define ELLIPSE_IMPLEMENTATION
// #ifndef ELLIPSE_IMPLEMENTED

#include "ellipse.h"

/* ---- seeding: call once, or call ellipse_seed() for reproducibility ---- */
static int ellipse_seeded = 0;

void nan_seed(unsigned int seed) {
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
static int  _size(const int *shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}
// end of size

/*
    ndim of tensor
    --------------
    This define dimension of the tensor or array.
*/
static int _ndim(const int *shape) {
    return (int)sizeof(shape)/sizeof(shape[0]);
}



/*
    tensor(data, shape, ndim, dtype)
    --------------------------------
    Initialize and Define Tensor *, that pass in array of data, shape, number of dimension (ndim) and data type(dtype).
    This will give the freedom for different operations and grad calculation.

    ndim - dimension type of your Tensor, if it is 2D you will pass - ndim = 2 and if it is 1D, you will pass - ndim = 1

    For now Tensor it suport only 2 type of dimensions , 1D and 2D

    Syntax:
        Tensor *x = tensor((float[]){0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, (int[]){2, 3}, 2, _float);

         print(x);

        free_t(x);

        Output:
            x = [[ 0.100000,  0.200000,  0.300000],
                [ 0.400000,  0.500000,  0.600000]], dtype: _float

        Tensor *x = tensor((int[]){1, 2, 3, 4}, (int[]){4}, 1, _int);

         print(x);

        free_t(x);

        Output:
            x = [     1,      2,      3,      4], dtype: _int
*/
Tensor *tensor(void *data, const int *shape, Dtype dtype){
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    if (!t) return NULL;

    int ndim = _ndim(shape);
    int sizet = _size(shape, ndim);

    t->ndim = ndim;
    t->size = sizet;
    t->dtype = dtype;

    t->shape = (int *)malloc(sizeof(int)*t->ndim);
    if (!t->shape) {free(t); return NULL; }

    memcpy(t->shape, shape, t->ndim*sizeof(int));

    switch (dtype)
    {
    case 0:
        t->data._int = (int *) malloc(sizeof(int)*t->size);
        if (!t->data._int) { free( t->shape); free(t); return NULL; }
        memcpy(t->data._int, data, t->size*sizeof(int));
        break;

    case 1:
        t->data._float = (float *) malloc(sizeof(float)*t->size);
        if (!t->data._float) { free( t->shape); free(t); return NULL; }
        memcpy(t->data._float, data, t->size*sizeof(float));
        break;
    
    default:
        fprintf(stderr, "Please make sure you pass the correct dtype : \"_int\" - for an integer dtype and \"_float\" for float dtype");
        free(t->shape);
        free(t);
        return NULL;
        break;
    }
    
    return t;
}
// end of Tensor


/*
    _rand(shape, ndim): 
    ------------------
    Generates a tensor of random floats from a uniform distribution on [0, 1) / (mean 0, standard deviation 1).

    It uses the current generator.

        Formula: 
            rand() / (RAND_MAX + 1)  

    Syntax:
        _rand(shape, ndim)

        Return:
            Tensor *
    
    Example:
        Tensor *a = _rand((int[]){2, 3}, 2);

         print(a);

        free_t(a);

        Output:
            a = [[ 0.521612, -0.456259,  1.207947],
                 [-0.387763, -0.670170, -0.307190]], dtype: _float

        Tesnor *b = _rand((int[]){10}, 1);

         print(b);

        free_t(b);

        Output:
            b = [-0.701645, -0.204627, 0.202153, -1.604854, -0.404581, -0.416625, 0.531158, 1.881386, -0.224445, -1.828255], dtype: _float
*/
Tensor *_rand(int *shape, int ndim){
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    if (!t) return NULL;

    int sizet = _size(shape, ndim);
    t->ndim = ndim;
    t->size = sizet;
    t->dtype = _float;

    t->shape = (int *)malloc(sizeof(int)*t->ndim);
    if (!t->shape) { free(t); return NULL; }
    memcpy(t->shape, shape, sizeof(int)*t->ndim);

    t->data._float = (float *)malloc(sizeof(float)*t->size);
    if (!t->data._float) { free(t->shape); free(t); return NULL; }

    nan_ensure_seeded();

    const float denom = (float)RAND_MAX + 1.0f;         /* strictly > RAND_MAX -> result < 1.0f */
    for (int i = 0; i < sizet; i++){
        t->data._float[i] = (float)rand() / denom;

    }
    
    return t;
}
//    end of randn


/*
    _randn(shape, ndim)
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
        _randn(shape, ndim)

        Return:
            Tensor *
    
    Example:
        Tensor *a = _randn((int[]){2, 3}, 2);

         print(a);

        free_t(a);

        Output:
            a = [[ 0.762251,  0.144962,  0.379993],
                 [ 0.547818,  0.175760,  0.999096]], dtype: _float

        Tesnor *b = _randn((int[]){10}, 1);

         print(b);

        free_t(b);

        Output:
            b = [0.762892, 0.931061, 0.339630, 0.157129, 0.862674, 0.953750, 0.670879, 0.455913, 0.531983, 0.033875], dtype: _float
*/

Tensor *_randn(int *shape, int ndim){
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    int sizet = _size(shape, ndim);
    t->ndim = ndim;
    t->size = sizet;
    t->dtype = _float;

    t->shape = (int *)malloc(sizeof(int)*t->ndim);
    if (!t->shape) { free(t); return NULL; }
    memcpy(t->shape, shape, sizeof(int)*t->ndim);

    t->data._float = (float *)malloc(sizeof(float)*t->size);
    if (!t->data._float) { free(t->shape); free(t); return NULL; }

    nan_ensure_seeded();

    const float denom = (float)RAND_MAX + 1.0f;
    for (int i = 0; i < t->size; i++) {
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
    _randint(low, high, shape, ndim)
    -------------------------------
        Generates a tensor of random integers between low (inclusive) and high (exclusive). 
        It uses the current generator.

        Formular:
            low + (rand() % (high - low + 1))

    Syntax:
        _randint(low, high, shape, ndim)

        Return:
            Tensor *
    
    Example:
        Tensor *a = _randint(1, 10 (int[]){5, 2}, 2);

        print(a);

        free_t(a);

        Output:
            a = [[4, 7],
                 [2, 2],
                 [6, 6],
                 [2, 3],
                 [1, 5]], dtype: _int

        Tesnor *b = _randint(20, 30, (int[]){10}, 1);

         print(b);

        free_t(b);

        Output:
            b = [    26,     20,     28,     24,     29,     20,     23,     23,     23,     24], dtype: _int
*/
Tensor *_randint(int low, int high, int *shape, int ndim){
    /* --- 1. validate BEFORE allocating anything --- */
    if (!shape || ndim < 0) {
        fprintf(stderr, "randint: bad shape/ndim\n");
        return NULL;
    }
    if (high <= low) {      // high is exclusive; empty or reversed range
        fprintf(stderr, "randint: empty range [%d, %d)\n", low, high);
        return NULL;
    }

    /* --- 2. only now allocate --- */
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    int sizet = _size(shape, ndim);
    t->ndim = ndim;
    t->size = sizet;
    t->dtype = _int;

    t->shape = (int *)malloc(sizeof(int)*t->ndim);
    if (!t->shape) { free(t); return NULL; }
    memcpy(t->shape, shape, sizeof(int)*t->ndim);

    t->data._int = (int *)malloc(sizeof(int)*t->size);
    if (!t->data._int) { free(t->shape); free(t); return NULL; }

    nan_ensure_seeded();

    /* --- 3. fill --- */
    int range = high - low;         // >= 1 here, guaranteed by the guard above
    for (int i = 0; i < t->size; i++)
    {
        t->data._int[i] = low + rand() % range;
    }

    return t;
}
// end of randint

Tensor *_ones(int* shape, int ndim, Dtype dtype){
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    int sizet = _size(shape, ndim);

    t->ndim = ndim;
    t->size = sizet;
    t->dtype = dtype;

    t->shape = (int *)malloc(sizeof(int)*t->ndim);
    if (!t->shape) { free(t); return NULL; }

    memcpy(t->shape, shape, sizeof(int)*t->ndim);

    switch (t->dtype)
    {
    case _int:
        t->data._int = (int *)malloc(sizeof(int)*t->size);
        if (!t->data._int) { free(t->shape); free(t); return NULL; }
        for (int i = 0; i < t->size; i++)
        {
            t->data._int[i] = 1;
        }
        
        break;

    case _float:
        t->data._float = (float *)malloc(sizeof(float)*t->size);
        if (!t->data._float) { free(t->shape); free(t); return NULL; }
        for (int i = 0; i < t->size; i++)
        {
            t->data._float[i] = 1.f;
        }
        
        break;
    
    default:
        fprintf(stderr, "_ones: invalid dtype\n");
        free(t->shape);
        free(t);
        return NULL;
    }
    return t;
}

Tensor *_zeros(int* shape, int ndim, Dtype dtype){
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    int sizet = _size(shape, ndim);

    t->ndim = ndim;
    t->size = sizet;
    t->dtype = dtype;

    t->shape = (int *)malloc(sizeof(int)*t->ndim);
    if (!t->shape) { free(t); return NULL; }

    memcpy(t->shape, shape, sizeof(int)*t->ndim);

    switch (t->dtype)
    {
    case _int:
        t->data._int = malloc(sizeof(int) * t->size);
        if (!t->data._int) { free(t->shape); free(t); return NULL; }
        for (int i = 0; i < t->size; i++)
        {
            t->data._int[i] = 0;
        }
        
        break;

    case _float:
        t->data._float = (float *)malloc(sizeof(float)*t->size);
        if (!t->data._float) { free(t->shape); free(t); return NULL; }
        for (int i = 0; i < t->size; i++)
        {
            t->data._float[i] = 0.f;
        }
        
        break;
    
    default:
        fprintf(stderr, "_ones: invalid dtype\n");
        free(t->shape);
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
        free_t(Tensor *)

        Return:
                NONE

    Example:
        Tensor *w = randd((int[]){10,2}, 2);

        print(w);

        // this free the w from the memory.
        free_t(w);

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
    
    free(m->shape);
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

    switch (t->ndim)
    {
    case 1:
        switch (t->dtype)
        {
            case _int:
                printf("[");    
                for (int i = 0; i < t->size; i++)
                {
                    if (i == t->size - 1)
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
                for (int i = 0; i < t->size; i++)
                {
                    if (i==t->size-1)
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
                for (int i = 0; i < t->shape[0]; i++)
                {
                    if (i > 0 && i < t->shape[0])
                    {
                        printf("     [");
                    } else
                    {
                        printf("[");
                    }

                    for (int j = 0; j < t->shape[1]; j++)
                    {
                        // if (j == 0)
                        // {
                        //     printf("%d", t->data._int[i*t->shape[1] + j]);
                        //     continue;
                        // }
                        
                        // if (j > 0)
                        // {
                        //     printf(", %2d", t->data._int[i*t->shape[1] + j]);
                        // } 
                        if (j < t->shape[1]-1)
                        {
                            printf("%3d, ", t->data._int[i*t->shape[1] + j]);
                        } else
                        {
                            printf("%3d", t->data._int[i*t->shape[1] + j]);
                        }  
                        
                    }
        
                    if (i<t->shape[0]-1)
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
                for (int i = 0; i < t->shape[0]; i++)
                {
                    if (i > 0 && i < t->shape[0])
                    {
                        printf("     [");
                    } else
                    {
                        printf("[");
                    }

                    for (int j = 0; j < t->shape[1]; j++)
                    {
                        if (j < t->shape[1]-1)
                        {
                            printf("%9f, ", t->data._float[i*t->shape[1] + j]);
                        } else
                        {
                            printf("%9f", t->data._float[i*t->shape[1] + j]);
                        }  
                        
                    }
        
                    if (i<t->shape[0]-1)
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
    .seed    = nan_seed,
};

// end of the tensor_print

// #endif
// #endif