#include "ellipse.h"
#include <stdio.h>

int main(void){

    Tensor *x = ellipse.tensor((int[]){1,2,3,4}, (int[]){2,2}, ellipse._int);
    print(x);

    ellipse.free(x);
    printf("Hello, ellipse!\n");

    return 0;
}
