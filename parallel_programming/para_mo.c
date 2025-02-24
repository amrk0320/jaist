#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[])
{
	int niter = 1000000000;						//number of iterations per FOR loop
	double x,y;							//x,y value for the random coordinate
	int i;								//loop counter
        int count=0;							//Count holds all the number of how many good coordinates
	double z;							//Used to check if x^2+y^2<=1
	double pi;							//holds approx value of pi
	
    // プログラム開始時刻
    double exec_start_time = omp_get_wtime();
    double numthreads = omp_get_num_threads();

	#pragma omp parallel firstprivate(x, y, z, i) reduction(+:count)
	{
		srand48((int)time(NULL) ^ omp_get_thread_num());	//Give random() a seed value
		for (i=0; i<niter; ++i)					//main loop
		{
			x = (double)drand48();				//gets a random x coordinate
			y = (double)drand48();				//gets a random y coordinate
			z = ((x*x)+(y*y));				//Checks to see if number is inside unit circle
			if (z<=1)
			{
				++count;				//if it is, consider it a valid random point	
			}		
		}
	} 
	pi = ((double)count/(double)(niter*numthreads))*4.0;
	printf("Pi: %f\n", pi);
	// プログラム終了時刻
    double exec_time = omp_get_wtime() - exec_start_time;
    printf( "exec_time=%fsec\n", exec_time );
}