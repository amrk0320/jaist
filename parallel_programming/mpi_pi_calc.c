#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#define ITER (1024L*1024*1024)

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

int calc_in_round_sum(int chunk_size, int myID){
    int sum=0;
    int i=0;
    double x, y;

    // スレッド独立なシードを生成
    unsigned int seed = make_thread_seed(myID);

    for( i = 0; i < chunk_size; i++ ){
            // x = drand48();
            // y = drand48();
            x = my_random(&seed);
            y = my_random(&seed);
            if( x*x+y*y < 1 )       sum++;
    }

    return sum;
}

int main( int argc, char *argv[] ){
    int  MyID = 0;
    int  numprocs;
    int  chunk_size;
    int    i, j, sum=0;
    int    tag_id=0;
    double  tstart;
    MPI_Status  mstat;

    // おまじない
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &numprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &MyID );

    // プロセスあたりのループ計算数
    chunk_size = ITER / numprocs;
    
    // if( MyID == 0 ){
    //     // プロセスID0で実行時間計測する
    //     tstart = MPI_Wtime();
    //     sum += calc_in_round_sum(chunk_size, MyID);
    // }  

    // 直列
    if( MyID == 0 ){
        // プロセスID0で実行時間計測する
        tstart = MPI_Wtime();
        sum += calc_in_round_sum(chunk_size, MyID);
        double calc_pi = 4.0*sum/ITER;
        printf( "PI=%f\n", calc_pi );
        double pi = M_PI;
        double pi_abs_error = fabs(calc_pi - pi) / pi;
        printf("PI validity=%f\n", pi_abs_error);
        printf( "Sequencial: time = %lf sec\n",
                (MPI_Wtime() - tstart) );
    }

    // if( MyID == 0 ){
    //     // mpi reduceを使えたら使う
    //     int  id, sum0;
    //     for( id = 1; id < numprocs; id++ ){
    //         // 0番以外が計算結果を受信する
    //         MPI_Recv(&sum0, 1, MPI_INT, MPI_ANY_SOURCE, tag_id, MPI_COMM_WORLD, &mstat);
    //         sum += sum0;
    //     }

    //     double calc_pi = 4.0*sum/ITER;
    //     printf( "PI=%f\n", calc_pi );
    //     double pi = M_PI;
    //     double pi_abs_error = fabs(calc_pi - pi) / pi;
    //     printf("PI validity=%f\n", pi_abs_error);
    // }else{
    //     // sumは各プロセス毎の異なるポインタ値
    //     sum += calc_in_round_sum(chunk_size, MyID);
    //     MPI_Send(&sum, 1, MPI_INT, 0, tag_id, MPI_COMM_WORLD);
    // }

    // if( MyID == 0 ){
    //     printf( "Parallel: time = %lf sec\n",
    //             (MPI_Wtime() - tstart) );
    // }

    // おまじない
    MPI_Finalize();
}