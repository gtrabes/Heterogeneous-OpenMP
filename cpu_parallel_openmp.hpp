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

#ifndef PARALLEL_FOR_EACH_HPP
#define PARALLEL_FOR_EACH_HPP

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <sched.h>
#include <vector>
#include <utility>
#include <tuple>

namespace parallel {

	template<class T, class Function>
	void parallel_for_each_object(std::vector<T>& obj, Function& f, unsigned int thread_number){
		/* set number of threads */
		//omp_set_num_threads(thread_number);
		size_t size = obj.size();

		#pragma omp parallel for num_threads(thread_number) firstprivate(f) shared(obj)
		for(size_t i = 0; i < size; i++){
			f(obj[i]);
		}

	}

	template<typename ITERATOR, typename FUNC>
	void parallel_for_each_iterator(ITERATOR first, ITERATOR last, FUNC& f, unsigned int thread_number){
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

}

#endif //PARALLEL_FOR_EACH_HPP
