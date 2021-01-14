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

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <sched.h>
#include <chrono>
#include <memory>
#include <algorithm>


#include <cuda_runtime_api.h>
#include <cuda.h>

#include "gpu_parallel_cuda.hpp"
#include "cpu_parallel_openmp.hpp"
#include "sample_class.cpp"

using namespace std;
using hclock=std::chrono::high_resolution_clock;

int main (int argc, char *argv[]) {

	double time, value;
	unsigned int P;
	unsigned long long int n;
	//std::vector<std::shared_ptr<sample>> objects;
	std::vector<sample> objects;
	auto begin_time=hclock::now(), end_time = hclock::now();

	/* get programs parameters */
	P=atoi(argv[1]);
	n=atoll(argv[2]);

	/* initialize value to initialize sample class objects */
	value=3.14;

	int num_cuda_gpus = 0, omp_num_devices = 0;

	omp_num_devices = omp_get_num_devices();

	std::cout << "GPUs detected OpenMP: "<< omp_num_devices << std::endl;

	cudaGetDeviceCount(&num_cuda_gpus);

	std::cout << "GPUs detected Cuda: "<< num_cuda_gpus << std::endl;

	cuda::printCudaVersion();
	//printCudaVersion();

	begin_time = hclock::now();

	/* sequential initialization */
	for(unsigned long long int i=0; i<n; i++){
		objects.push_back(sample(value));
	}

	end_time = hclock::now();

	/*calculate and print time */
	std::cout << "Sequential initialization time: "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end_time - begin_time).count() << std::endl;

	objects.clear();

	begin_time = hclock::now();

	/* parallel initialization */
	openmp::cpu_parallel_add_objects_to_vector_openmp(objects, value, n, P);

	end_time = hclock::now();

	/*calculate and print time */
	std::cout << "CPU Parallel initialization time: "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end_time - begin_time).count() << std::endl;

	//#pragma omp declare target
	#pragma omp declare reduction (merge : std::vector<sample> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
	//#pragma omp end declare target

	begin_time = hclock::now();

	#pragma omp parallel for reduction(merge: objects)
	for(int i=0; i<n; i++) {
		objects.push_back(sample(value));
	}

	end_time = hclock::now();

	/*calculate and print time */
	//std::cout << "GPU Parallel initialization time: "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end_time - begin_time).count() << std::endl;

	cout << "Initial values:" << endl;

	for(unsigned long long int i=0; i<1; i++){
		cout << "Position:(" << i << "): "<< objects.at(i).get_number() << endl ;
	}

	auto square = [](sample& c)->void { c.square_root(); };

	/* sequential version */
	begin_time = hclock::now();
	std::for_each(objects.begin(), objects.end(), square);
	end_time =  hclock::now();

	cout << "Values after sequential:" << endl;

	for(unsigned long long int i=0; i<1; i++){
		cout << "Position:(" << i << "): "<< objects.at(i).get_number() << endl ;
	}

	/*calculate and print time */
	std::cout << "Sequential time: "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end_time - begin_time).count() << std::endl;

	/* initialization */
	for(unsigned long long int i=0; i<n; i++){
		objects.at(i).set_number(value);
	}

	/* parallel version */
	begin_time = hclock::now();
	openmp::parallel_for_each_iterator_openmp(objects.begin(), objects.end(), square, P);
	end_time =  hclock::now();

	cout << "Values after CPU parallel:" << endl;

	for(unsigned long long int i=0; i<1; i++){
		cout << "Position:(" << i<< "): "<< objects.at(i).get_number() << endl ;
	}

	/*calculate and print time */
	std::cout << "CPU parallel time: "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end_time - begin_time).count() << std::endl;

	/* initialization */
        for(unsigned long long int i=0; i<n; i++){
                objects.at(i).set_number(value);
        }


	/* openMP GPU version */
        begin_time = hclock::now();
        openmp::gpu_parallel_for_each_openmp(objects, square, P);
        end_time =  hclock::now();

        cout << "Values after GPU parallel:" << endl;

        for(unsigned long long int i=0; i<1; i++){
                cout << "Position:(" << i<< "): "<< objects.at(i).get_number() << endl ;
        }

        /*calculate and print time */
        std::cout << "OpenMP GPU parallel time: "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end_time - begin_time).count() << std::endl;

        /* initialization */
         //for(unsigned long long int i=0; i<n; i++){
        //	 objects.at(i).set_number(value);
         //}

     	 /* cuda GPU version */
		 //auto size = objects.size();

		 //std::vector<sample> gpu_objects;

		 // Allocate memory for each vector on GPU
		 //cudaMalloc(&gpu_objects, v.size*);

         //begin_time = hclock::now();
         //cuda::gpu_parallel_for_each_cuda(objects, square, P);
         //end_time =  hclock::now();

         //cout << "Values after GPU parallel:" << endl;

         //for(unsigned long long int i=0; i<1; i++){
        //	 cout << "Position:(" << i<< "): "<< objects.at(i).get_number() << endl ;
         //}

         /*calculate and print time */
         //std::cout << "CUDA GPU parallel time: "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end_time - begin_time).count() << std::endl;

    	/* initialization */
            for(unsigned long long int i=0; i<n; i++){
                    objects.at(i).set_number(value);
            }


    	/* openMP Multi-GPU version */
            begin_time = hclock::now();
            openmp::multi_gpu_parallel_for_each_openmp(objects, square, P);
            end_time =  hclock::now();

            cout << "Values after Multi GPU parallel:" << endl;

            for(unsigned long long int i=0; i<1; i++){
                    cout << "Position:(" << i<< "): "<< objects.at(i).get_number() << endl ;
            }

            /*calculate and print time */
            std::cout << "OpenMP Multi-GPU parallel time: "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end_time - begin_time).count() << std::endl;


	return 0;
}
