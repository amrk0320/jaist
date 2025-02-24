#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#define N 1024

void main(){
    int i,j,k,NumT,MyID[N];
    double t,x[N][N],y[N][N],z[N][N];
    
    // 直列
    t = omp_get_wtime();
    #pragma omp parallel for
    for( i = 0; i < N; i++ ){
        for( j = 0; j < N; j++ ){
            for( k = 0; k < N; k++ ){
                z[i][j] += x[i][k] * y[k][j];
            }
        }
    }

    i = lrand48() % N;
    j = lrand48() % N;
    t = omp_get_wtime()-t;
    printf( "x[%d][%d]=%f\n", i, j, x[i][j]);  
    printf( "t=%f sec\n", t );
}
