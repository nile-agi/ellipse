#include "ellipse.h"

int main(void){
    
    Tensor *w = ellipse.tensor(Arr(1,2,3,4), size(1, 4), ellipse._float);

    print(w);

    ellipse.free(w);
    
    return 0;
}

/*
    Expected Output
    ----------------

*/

