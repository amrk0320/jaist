#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define SIZE 2048

void init_ary(double m[SIZE][SIZE]){
    int i, j;

    for( i = 0; i < SIZE; i++ ){
        for( j = 0; j < SIZE; j++ ){
            m[i][j] = i*SIZE+j+1;
        }
    }
    m[0][0] = -1;
}


// 転置行列に変換
void transposeMatrix(double x[SIZE][SIZE], double result[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[j][i] = x[i][j];
        }
    }
}

int main( int argc, char *argv[] ){
    
    int    MyID = 0, NumP;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &NumP );
    MPI_Comm_rank( MPI_COMM_WORLD, &MyID );

    int    i=5,j=6,k;
    double (*a)[SIZE], (*x)[SIZE], (*y)[SIZE], (*x_t)[SIZE],(*a_t)[SIZE];
    double proctime, sum=0.0;
    // 分散行列の列長
    a = malloc( (long)sizeof(double) * SIZE * SIZE );
    a_t = malloc( (long)sizeof(double) * SIZE * SIZE );
    x = malloc( (long)sizeof(double) * SIZE * SIZE );
    x_t = malloc( (long)sizeof(double) * SIZE * SIZE );
    y = malloc( (long)sizeof(double) * SIZE * SIZE );
    long int current_time = (long int)time(NULL);    

    if( MyID == 0 ){
        if( (long)a+(long)x+(long)y == 0 )
            printf( "malloc error\n" );

        init_ary( x );
        init_ary( y );

        printf( "i:%d, j:%d,MyID:%d, x:%f \n", SIZE-1,SIZE-1, MyID, x[SIZE-1][SIZE-1] );

        proctime = MPI_Wtime();
    }

    // broascast
    // ランク0のプロセスからデータをブロードキャスト
    int y_tag_id = 0;
    MPI_Bcast(y, SIZE*SIZE, MPI_DOUBLE, y_tag_id, MPI_COMM_WORLD);
    // debug
    printf( "Bcast i:%d, j:%d,MyID:%d, y:%f \n", i,j, MyID, y[i][j] );

    if( MyID == 0 ){
        // xの行毎に分散したいので転置行列にして列毎に分散する(元の値は行毎に分散する)
        transposeMatrix(x, x_t);
    }

    // 部分行列のメモリ割り当て
    int col_size = SIZE/NumP;
    double** local_x_t = malloc(SIZE * sizeof(double*));
    for (int i = 0; i < SIZE; i++) {
        local_x_t[i] = malloc(col_size * sizeof(double));
    }

    if (a == NULL || x == NULL || y == NULL || local_x_t == NULL) {
        printf("Failed to allocate memory MyID:%d\n", MyID);
        MPI_Finalize();
        return 1;
    }

    // 行列Xを分散、2次元配列を分散
    for( i = 0; i < SIZE; i++ ){ 
        // 行毎に分散する
        if (i== 0){
            printf( "scatter start \n");
        }

        MPI_Scatter(x_t[i], col_size, MPI_DOUBLE, local_x_t[i], col_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (i== (SIZE-1)){
            printf( "scatter end\n");
        }
    }

    // // // 現在の時間を取得してシード値として使用
    // // // // シード値を設定
    // srand48(current_time);
    // i = lrand48() % SIZE;
    // j = lrand48() % col_size;

    // // debug
    // printf( "col_size:%d\n", col_size);
    // printf( "i:%d, j:%d,MyID:%d\n", i,j, MyID);
    // printf( "local_x:%f\n", local_x_t[i][j]);

    // local_xのメモリ割り当て
    double** local_x = malloc(col_size * sizeof(double*));
    for (int i = 0; i < col_size; i++) {
        local_x[i] = malloc(SIZE * sizeof(double));
    }

    if (local_x == NULL) {
        printf("Failed to local_x allocate memory MyID:%d\n", MyID);
        MPI_Finalize();
        return 1;
    }

    // local_xの転置行列を元に戻す
    // local_x_t -> local_x
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < col_size; j++) {
            local_x[j][i] = local_x_t[i][j];
        }
    }

    printf( "create local_x MyID:%d\n", MyID);

    // local_aのメモリ割り当て
    double** local_a = malloc(col_size * sizeof(double*));
    for (int i = 0; i < col_size; i++) {
        local_a[i] = malloc(SIZE * sizeof(double));
    }

    if (local_a == NULL) {
        printf("Failed to local_a allocate memory MyID:%d\n", MyID);
        MPI_Finalize();
        return 1;
    }

    // 行列計算する、計算量削減
    for( i = 0; i < col_size; i++ ){ 
        for( j = 0; j < SIZE; j++ ){
            local_a[i][j] = 0.0;
            for( k = 0; k < SIZE; k++ ){
                // xは列と行を逆に計算する
                local_a[i][j] += local_x[i][k] * y[k][j];
            }
        }
    }

    srand48(current_time);
    i = lrand48() % col_size;
    j = lrand48() % SIZE;

    printf( "caclulate local_a i:%d, j:%d,MyID:%d, a:%f\n", i,j, MyID, local_a[i][j]);

    // local_aのメモリ割り当て
    double** local_a_t = malloc(SIZE * sizeof(double*));
    for (int i = 0; i < SIZE; i++) {
        local_a_t[i] = malloc(col_size * sizeof(double));
    }

    if (local_a_t == NULL) {
        printf("Failed to local_a_t allocate memory MyID:%d\n", MyID);
        MPI_Finalize();
        return 1;
    }

    // local_aの転置行列にする
    // local_a -> local_a_t
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < col_size; j++) {
            local_a_t[i][j] = local_a[j][i];
        }
    }

    // 行列Xを分散、2次元配列を分散
    for( i = 0; i < SIZE; i++ ){ 
        // 行毎を列毎に集約する
        MPI_Gather(local_a_t[i], col_size, MPI_DOUBLE, a_t[i], col_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    printf( "gather end,MyID:%d \n", MyID);

    if( MyID == 0 ){
        // aは転置行列なので元の行列に戻す
        transposeMatrix(a_t, a);

        proctime = MPI_Wtime() - proctime;

        for( i = 0; i < SIZE; i++ )
            for( j = 0; j < SIZE; j++ )
                sum += a[i][j];

        printf( "sum = %f\n", sum );;
        printf( "x[%d][%d](=%f) * y[%d][%d](=%f) = a[%d][%d](=%f)\n",  
                i,j,x[i][j], i,j,y[i][j], i,j,a[i][j] );
        printf( "Time = %f Sec.\n", proctime );
    }

    MPI_Finalize();
    return(0);
}

