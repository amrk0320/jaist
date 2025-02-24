#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define SIZE 1024


void init_ary(double m[SIZE][SIZE]){
    long i, j;

    // 外側のループを並列化した
    #pragma omp parallel for
    for( i = 0; i < SIZE; i++ ){
        for( j = 0; j < SIZE; j++ ){
            m[i][j] = i*SIZE+j+1;
        }
    }

    m[0][0] = -1;
}

// 行列の乗算
void mul_ary( m, x, y )
double m[SIZE][SIZE], x[SIZE][SIZE], y[SIZE][SIZE];
{
    int i, j, k,total_threads, thread_num;
    double t;
    struct timespec tp;

#ifdef _OPENMP
    // 並列化版
    t=omp_get_wtime();
#else
    clock_gettime(CLOCK_REALTIME, &tp);
    t = tp.tv_sec + 1.0e-9*tp.tv_nsec;
#endif
    #pragma omp parallel
    {
        #pragma omp single
        total_threads = omp_get_num_threads();
    }

    #pragma omp parallel for
    for( i = 0; i < SIZE; i++ ){
        // i,j毎に計算は独立なので２重ループを並列にした
        #pragma omp parallel for
        for( j = 0; j < SIZE; j++ ){
            // temp変数を導入してメモリアクセス数を減らした
            double temp = 0.0;
            for( k = 0; k < SIZE; k++ ){
                temp += x[i][j]*y[k][j];
            }

            m[i][j] = temp;
        }
    }
#ifdef _OPENMP
    // 並列化版
    t=omp_get_wtime()-t;
#else
    clock_gettime(CLOCK_REALTIME, &tp);
    t = tp.tv_sec + 1.0e-9*tp.tv_nsec - t;
#endif
    // 計算結果をランダムに出力する
    srand48((long int)time(NULL));
    i = lrand48() % SIZE;
    j = lrand48() % SIZE;
    printf( "x[%d][%d]=%f\n", i, j, m[i][j]);  
    printf( "t=%f sec\n",t );
}

void main(){
    double (*a)[SIZE], (*x)[SIZE], (*y)[SIZE];
    double proctime;
    struct timespec tp;

    // メモリ割り当て
    a = malloc( (long)sizeof(double) * SIZE * SIZE );
    x = malloc( (long)sizeof(double) * SIZE * SIZE );
    y = malloc( (long)sizeof(double) * SIZE * SIZE );

    if( (long)a+(long)x+(long)y == 0 )
	printf( "malloc error\n" );

    double t=omp_get_wtime();
    init_ary( x );
    init_ary( y );
    t=omp_get_wtime()-t;

#ifdef _OPENMP
    proctime = omp_get_wtime();
#else
    clock_gettime(CLOCK_REALTIME, &tp);
    proctime = tp.tv_sec + 1.0e-9*tp.tv_nsec;
#endif

    mul_ary( a, x, y );

#ifdef _OPENMP
    proctime = omp_get_wtime() - proctime;
#else
    clock_gettime(CLOCK_REALTIME, &tp);
    proctime = tp.tv_sec + 1.0e-9*tp.tv_nsec - proctime;
#endif
    printf( "det(km) = %e\t%f Second\n", a[0][0], proctime );
}