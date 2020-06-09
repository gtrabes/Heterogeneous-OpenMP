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
#include <vector>
#include <math.h>       /* sqrt */
#include <utility>
#include <tuple>

using namespace std;

/**
 * Inner product
 */

double square_root(double number) {
    double squareRoot;
    squareRoot =  sqrt(number);
    return squareRoot;
}



template<typename ITERATOR, typename FUNC>
std::vector<double> parallel_for_each2(int thread_number, ITERATOR first, ITERATOR last, FUNC& f) {
//    		int size = obj.size();
	const size_t n = std::distance(first, last);
	std::vector<double> y2;

	/* set number of threads */
	omp_set_num_threads(thread_number);


	cout << "ACA: PARALEL" << endl ;
/*
	#pragma omp parallel for private(f) shared(obj)
		for (int i = 0; i < size; i++) {
			f(obj[i]);
		}
*/


//    		#pragma omp parallel for firstprivate(f) shared(first,last)
//    		for (ITERATOR it = first; it != last; it++) {
//    			f(*it);
//    		}

	#pragma omp parallel for firstprivate(f) shared(first)
		for(size_t i = 0; i < n; i++) {
			auto& elem = *(first + i);
			// do whatever you want with elem
			f(elem);
//            cout << result << "ACA: PARALEL LOOP" << endl ;
//			y2.push_back(5.2);
			//cout << std::experimental::apply(f, elem) << '\n';
//			cout << f(elem) << '\n';
		}

	y2.push_back(5.2);
	return y2;

}




template<class T, class Function>
void parallel_for_each(int thread_number, std::vector<T> & obj, Function f){
	/* set number of threads */
	omp_set_num_threads(thread_number);
	int size = obj.size();
	double result;
	#pragma omp parallel for firstprivate(f) shared(obj)
		for (int i = 0; i < size; i++){
			result = f(obj[i]);
		}
	cout << "Aca parallel after loop" << result << '\n';
}






int main (int argc, char *argv[]) {

  int numThreads, tid;

  double time, *partial_result, result=0;
  int P, n, i;
  CLOCK_TYPE begin, end;

  P=atoi(argv[1]);
  n=atoll(argv[2]);

  std::vector<double> x, y;
//  x = (double*)malloc(n*sizeof(double));

  /* initialization */

  for(int i=0; i<n; i++){
	  x.push_back(3.15);
  }

  cout << "ACA:" << endl ;


  /* sequential version */

  GETTIME(begin);

  for(int i=0; i<n; i++){
	  y.push_back(square_root(x.at(i)));
//	  y.push_back(4.234);
  }

  GETTIME(end);
  DIFTIME(end,begin,time);

  cout << "Sequential -> Result:" << y.at(0) << " time: "<< time << endl;

  result=0;

  /* parallel version */

  GETTIME(begin);

  parallel_for_each(P, x, square_root);

  GETTIME(end);
  DIFTIME(end,begin,time);

  cout << "Parallel for -> Result:" << y.at(0) << " time: "<< time << endl ;

  return 0;

}
