/**
 * Copyright (c) 2020, Guillermo G. Trabes
 * Carleton University, Universidad Nacional de San Luis
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CPU_PARALLEL_OPENMP_HPP
#define CPU_PARALLEL_OPENMP_HPP

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <sched.h>
#include <vector>
#include <utility>
#include <tuple>

namespace openmp {

	template<typename T>
	void cpu_parallel_add_objects_to_vector_openmp(std::vector<T>& objects, double value, long long unsigned int n, unsigned int thread_number){

		//#pragma omp declare reduction (merge : std::vector<T> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

		//std::vector<T>& objects = nullptr;

		//#pragma omp parallel for num_threads(thread_number) schedule (static) reduction(merge: objects)
		for(int i=0; i<n; i++) {
			objects.push_back(T(value));
		}

	//	return objects();
	}

	template<class T, class Function>
	void cpu_parallel_for_each_openmp(std::vector<T>& obj, Function& f, unsigned int thread_number){
		/* set number of threads */
		//omp_set_num_threads(thread_number);
		size_t size = obj.size();

		#pragma omp parallel for num_threads(thread_number) firstprivate(f) shared(obj)
		for(size_t i = 0; i < size; i++){
			f(obj[i]);
		}

	}

	template<typename ITERATOR, typename FUNC>
	void parallel_for_each_iterator_openmp(ITERATOR first, ITERATOR last, FUNC& f, unsigned int thread_number){
		/* set number of threads */
		//omp_set_num_threads(thread_number);
		size_t n = std::distance(first, last);

		#pragma omp parallel for num_threads(thread_number) firstprivate(f, first)
		for(int i = 0; i < n; i++){
			f(*(i+first));
		}

//		#pragma omp parallel for firstprivate(f) shared(first,last)
//    	for (ITERATOR it = first; it != last; it++) {
//    		f(*it);
//    	}

//		#pragma omp parallel for firstprivate(f) shared(first)
//		for(size_t i = 0; i < n; i++){
//			auto& elem = *(first + i);
			// do whatever you want with elem
//			f(elem);
//		}
	}


	template<class T, class Function>
	void gpu_parallel_for_each_openmp(std::vector<T>& obj, Function& f, unsigned int thread_number){
                /* set number of threads */
                //omp_set_num_threads(thread_number);
                size_t size = obj.size();

		//#pragma omp declare target
  		//auto GPUf = f;
		//auto objGPU = obj;
		//#pragma omp end declare target

                #pragma omp target teams distribute parallel for map(tofrom:obj)
                for(size_t i = 0; i < size; i++){
                        f(obj[i]);
                }

        }


	template<class T, class Function>
	void multi_gpu_parallel_for_each_openmp(std::vector<T>& obj, Function& f, unsigned int thread_number){
		/* set number of threads */
		//omp_set_num_threads(thread_number);
		size_t size = obj.size();

		//#pragma omp declare target
	  	//auto GPUf = f;
		//auto objGPU = obj;
		//#pragma omp end declare target
		//omp_get_num_devices();
		int num_devices = omp_get_num_devices();

		#pragma omp parallel num_threads(num_devices) proc_bind(close)
    	{
    		/* get thread id */
    		size_t tid = omp_get_thread_num();

    		/* if it's not last thread compute n/thread_number elements */
    		if(tid != num_devices-1) {
				#pragma omp target teams distribute parallel for map(tofrom:obj) device(tid)
    			for(size_t i = (size/num_devices) * tid; i < (size/num_devices)*(tid+1); i++) {
    				f(obj[i]);
    			}
    		/* if it's last thread compute till the end of the vector */
			} else {
				#pragma omp target teams distribute parallel for map(tofrom:obj) device(tid)
				for(size_t i = (size/num_devices) * tid; i < size ; i++) {
					f(obj[i]);
				}
			}
    	}


	}



	template<typename ITERATOR, typename FUNC>
        void gpu_parallel_for_each_iterator_openmp(ITERATOR first, ITERATOR last, FUNC& f, unsigned int thread_number){
                /* set number of threads */
                //omp_set_num_threads(thread_number);
                size_t n = std::distance(first, last);

		#pragma omp declare target
                auto gpuf = f;
		auto firstgpu = *first;
                #pragma omp end declare target

                #pragma omp target
                for(int i = 0; i < n; i++){
                        gpuf(firstgpu);
			firstgpu=*(first+i);
                }

//              #pragma omp parallel for firstprivate(f) shared(first,last)
//      for (ITERATOR it = first; it != last; it++) {
//              f(*it);
//      }

//              #pragma omp parallel for firstprivate(f) shared(first)
//              for(size_t i = 0; i < n; i++){
//                      auto& elem = *(first + i);
                        // do whatever you want with elem
//                      f(elem);
//              }
        }



	template<typename ITERATOR, typename FUNC>
	void cpu_parallel_min_element_openmp(ITERATOR first, ITERATOR last, unsigned int thread_number){
		/* set number of threads */
		//omp_set_num_threads(thread_number);
		size_t n = std::distance(first, last);

//		#pragma omp parallel for num_threads(thread_number) firstprivate(f, first)
//		for(int i = 0; i < n; i++){
//			f(*(i+first));
//		}

//		#pragma omp parallel for firstprivate(f) shared(first,last)
//    	for (ITERATOR it = first; it != last; it++) {
//    		f(*it);
//    	}

//		#pragma omp parallel for firstprivate(f) shared(first)
//		for(size_t i = 0; i < n; i++){
//			auto& elem = *(first + i);
			// do whatever you want with elem
//			f(elem);
//		}
	}


}

#endif //CPU_PARALLEL_OPENMP_HPP
