#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define ITER (1024*1024*512)

// 線形合同法による擬似乱数生成器
double my_random(unsigned int *seed) {
    const unsigned int a = 1664525;
    const unsigned int c = 1013904223;
    *seed = (a * (*seed) + c);

    // 0.0から1.0の範囲に正規化
    return (double)(*seed) / UINT_MAX;
}

unsigned int make_thread_seed(int thread_num) {
    return (unsigned int)time(NULL) + (unsigned int)thread_num;
}

int main(int argc, char *argv[]) {
    int total_threads;
    long int i, sum = 0;

    // プログラム開始時刻
    double exec_start_time = omp_get_wtime();

    #pragma omp parallel
    {
        // スレッド内でシードを生成
        unsigned int seed = make_thread_seed(omp_get_thread_num());

        // 並列ループを指定する
        // sumはreductionで１つの共有変数を利用しない
        #pragma omp for reduction(+:sum)
        for (i = 0; i < ITER; i++) {
            // スレッドのシード値を取得
            double x = my_random(&seed);
            double y = my_random(&seed);
            if (x * x + y * y < 1) sum++;
        }
    }

    double calc_pi = (4.0 * sum) / ITER;
    double pi = M_PI;
    double pi_abs_error = fabs(calc_pi - pi) / pi;

    printf("PI=%f\n", calc_pi);
    printf("PI validity=%f\n", pi_abs_error);

    // プログラム終了時刻
    double exec_time = omp_get_wtime() - exec_start_time;
    printf("total_threads=%d, exec_time=%f sec\n", omp_get_max_threads(), exec_time);

    return 0;
}