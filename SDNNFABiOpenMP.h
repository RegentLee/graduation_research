//
// Created by 李明曄 on 2019/12/25.
//

#ifndef GRADUATION_RESEARCH_SDNNFABIOPENMP_H
#define GRADUATION_RESEARCH_SDNNFABIOPENMP_H

#include <vector>
#include <random>
#include <algorithm>
#include <ctime>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

struct NEURON_OUTPUT{
    int potential;
    int index;
};

class SDNNFABiOpenMP {
public:
    SDNNFABiOpenMP(int input_size, std::vector<std::vector<int> > pattern, int element = 20,
                   std::vector<std::vector<std::vector<int> > > w = std::vector<std::vector<std::vector<int> > >());
    float Train(std::vector<int> input, int target);
    std::vector<float> Predict(std::vector<int> input);
    std::vector<std::vector<std::vector<int> > > GetWeight();

private:
    int output_neuron_size;

    std::vector<std::vector<int> > original_pattern;
    std::vector<std::vector<std::vector<char> > > bi_pattern;
    std::vector<std::vector<std::vector<int> > > weight;

    void MakeBiPattern(std::vector<std::vector<int> > og_pattern,
                       std::vector<std::vector<std::vector<char> > >& bi_pattern);
    std::vector<int> SD(std::vector<int> input);
    float NNTrain(std::vector<int> nn_input, std::vector<int> target);
    std::vector<float> NNPredict(std::vector<int> nn_input);
    static bool cmp(struct NEURON_OUTPUT& x, struct NEURON_OUTPUT& y);
};


#endif //GRADUATION_RESEARCH_SDNNFABIOPENMP_H
