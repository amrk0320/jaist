#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define ITER (1000000000)

main(argc, argv )
int     argc;
char    *argv[];
{
    // プログラム開始時刻
    int thread_num;
    double exec_start_time = omp_get_wtime();
    // #pragma omp parallel
    // {
    //     thread_num = omp_get_thread_num();
    //     printf( "thread_num=%d\n", thread_num );    
    // }

    int i,sum=0;
    #pragma omp parallel for reduction(+:sum)
    for( i = 0; i < ITER; i++ ){
        sum+=i;
    }

    printf( "sum=%d\n", sum );
    // プログラム終了時刻
    double exec_time = omp_get_wtime() - exec_start_time;
    printf( "exec_time=%fsec\n", exec_time );
}