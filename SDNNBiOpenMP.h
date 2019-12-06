//
// Created by 李明曄 on 2019/12/03.
//

#ifndef GRADUATION_RESEARCH_SDNNBIOPENMP_H
#define GRADUATION_RESEARCH_SDNNBIOPENMP_H

#include <vector>
#include <random>
#include <algorithm>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

class SDNNBiOpenMP {
public:
    SDNNBiOpenMP(int input_size, std::vector<std::vector<int> > pattern, std::vector<std::vector<int> > w = std::vector<std::vector<int> >());
    float Train(std::vector<int> input, int target);
    std::vector<int> Predict(std::vector<int> input);
    std::vector<std::vector<int> > GetWeight();

private:
    std::vector<std::vector<int> > original_pattern;
    std::vector<std::vector<std::vector<char> > > bi_pattern;
    std::vector<std::vector<int> > weight;

    void MakeBiPattern(std::vector<std::vector<int> > og_pattern,
                           std::vector<std::vector<std::vector<char> > >& bi_pattern);
    std::vector<int> SD(std::vector<int> input);
    float NNTrain(std::vector<int> nn_input, std::vector<int> target);
    std::vector<int> NNPredict(std::vector<int> nn_input);
};


#endif //GRADUATION_RESEARCH_SDNNBIOPENMP_H
