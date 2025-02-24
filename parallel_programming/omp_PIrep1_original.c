#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define ITER (512*1024*1024)

main(argc, argv )
int     argc;
char    *argv[];
{
    int    numthreads, total_threads,MyID;
    int    i, j, sum=0;
    double x, y, t;

    #pragma omp parallel
    total_threads = omp_get_num_threads();

    t = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for( i = 0; i < ITER; i++ ){
        x = drand48();
        y = drand48();
        if( x*x+y*y < 1 )       sum++;
    }

    t=omp_get_wtime()-t;
    printf( "total_threads=%d,t=%f sec\n", total_threads,t );
    printf( "PI=%f\n", 4.0*sum/ITER );
}