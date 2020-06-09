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

#include <parallel_for_each.hpp>
#include <sample_class.cpp>

using namespace std;
using hclock=std::chrono::high_resolution_clock;

int main (int argc, char *argv[]) {

	double time;
	unsigned int P;
	unsigned long long int n;
	//std::vector<std::shared_ptr<sample>> objects;
	std::vector<sample> objects;
	auto begin_time=hclock::now(), end_time = hclock::now();

	/* get programs parameters */
	P=atoi(argv[1]);
	n=atoll(argv[2]);

	/* initialization */
	for(unsigned long long int i=0; i<n; i++){
		objects.push_back(sample(3.14));
	}

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
		objects.at(i).set_number(3.14);
	}

	/* parallel version */
	begin_time = hclock::now();
	parallel::parallel_for_each_iterator(objects.begin(), objects.end(), square, P);
	end_time =  hclock::now();

	cout << "Values after parallel:" << endl;

	for(unsigned long long int i=0; i<1; i++){
		cout << "Position:(" << i << "): "<< objects.at(i).get_number() << endl ;
	}

	/*calculate and print time */
	std::cout << "Parallel time: "<< std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end_time - begin_time).count() << std::endl;

	return 0;
}
