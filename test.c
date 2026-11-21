#include "ellipse.h"
#include <stdio.h>

int main(void){
    /* Plain C API: data, shape, ndim, dtype  |  PyTorch-style API */
    
    Tensor *x = ellipse.tensor((float[]){1, 2, 3, 4, 5, 6}, size(2, 3), ellipse._float);
    print(x);                                     /* prints the tensor together with its variable name */
    ellipse.free(x);

    
    ellipse.seed(42);                             /* optional: reproducible randoms */
    Tensor *w = ellipse.randn(size(3,4));         /* 3x4, standard normal */
    print(w);
    ellipse.free(w);
    
    return 0;
}