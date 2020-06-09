/*
 ============================================================================
 Name        : example.c
 Author      : Guillermo Trabes
 Version     :
 Copyright   : @2020, Carleton University
 Description : Hello OpenMP World in C
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include "mytime.h"
#include <sched.h>

using namespace std;

/**
 * Inner product
 */

int main (int argc, char *argv[]) {

  int numThreads, tid;

  double *x,*y, time, *partial_result, result=0;
  int P, n, i;
  CLOCK_TYPE begin, end;

  P=atoi(argv[1]);
  n=atoll(argv[2]);

  x = (double*)malloc(n*sizeof(double));
  y = (double*)malloc(n*sizeof(double));

  partial_result = (double*)malloc(P*sizeof(double));

  /* initialization */

  for(int i=0; i<n; i++){
	  x[i]=1.0;
	  y[i]=1.0;
  }

  for(int i=0; i<P; i++){
	  partial_result[i]=0.0;
  }

  /* sequential version */

  GETTIME(begin);

  for(int i=0; i<n; i++){
	  result+=x[i]*y[i];
  }

  GETTIME(end);
  DIFTIME(end,begin,time);

  cout << "Sequential -> Result:" << result << " time: "<< time << endl ;

  result=0;

  /* parallel version */

  /* inner product */
/*  #pragma omp parallel shared (x,y,partial_result) private(i){
  	  int tid   = omp_get_thread_num();
  	  int procs = omp_get_num_threads();
  	  unsigned long len;
  	  cpu_set_t mascara;
  	  CPU_ZERO (&mascara);

  	  //Esto es para 1 Thread por core
  	  len = sizeof(cpu_set_t);
  	  CPU_SET (tid, &mascara);
//		if (sched_setaffinity(0, len, &mascara) < 0)
//			printf("\n\nError :: sched_setaffinity\n\n");
*/



  int sum=0, psum=0;


//	#pragma omp parallel for reduction (+:sum) //shared (x,y,partial_result) private(i)
//  	  tid   = omp_get_thread_num();
//		#pragma omp for schedule(static)
//  	  	  	  for(i=0; i<n; i++){
//  		  for(int j=0;j<100000;j++){
//  			  partial_result[0]+=x[i]*y[i];
//  		  }
			//				#pragma omp critical
//				cout << "ID:" << omp_get_thread_num()  << endl ;
//  	  	  	  }

  GETTIME(begin);

#pragma omp parallel private(i,tid,psum) shared(x,y) num_threads(P)
{
psum = 0.0;
tid = omp_get_thread_num();

#pragma omp for schedule(static) reduction(+:result)
  for (i=0; i<n; i++) {
		  result += (x[i] * y[i]);
//      psum = result;
  }
//printf("Thread %d partial sum = %f\n",tid, psum);
//cout << "Thread: " << tid << " partial sum = "<< psum << endl ;

}
	GETTIME(end);
	DIFTIME(end,begin,time);

	cout << "Parallel for -> Result:" << result << " time: "<< time << endl ;

	result = 0;

	GETTIME(begin);

	#pragma omp parallel for schedule(static) reduction(+:result) proc_bind(spread) num_threads(P)
		for (i=0; i<n; i++) {
				result += (x[i] * y[i]);
		}

	GETTIME(end);
	DIFTIME(end,begin,time);

	cout << "Parallel for Spread -> Result:" << result << " time: "<< time << endl ;

	result = 0;

		GETTIME(begin);

		#pragma omp parallel for schedule(static) reduction(+:result) proc_bind(close) num_threads(P)
			for (i=0; i<n; i++) {
					result += (x[i] * y[i]);
			}

		GETTIME(end);
		DIFTIME(end,begin,time);

		cout << "Parallel for Close -> Result:" << result << " time: "<< time << endl ;


	return 0;
}
