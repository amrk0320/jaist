#include <mpi.h>
#include <stdio.h>
#include <string.h>

void main(int argc, char *argv[]){
    int i,NumProcess, MyProcessNo=0,dest=1;
    int a[10],N=3,result;
    MPI_Status mstat;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumProcess);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyProcessNo);

    if (MyProcessNo==0){
        for (i = 0; i < N; i++){
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 84,MPI_COMM_WORLD, &mstat);
            printf("SourceNode=%d,result=%d\n", mstat.MPI_SOURCE, result);
        }
    } else {
        a[0]= MyProcessNo*10;
        MPI_Send(a, 1, MPI_INT, 0, 84,MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
