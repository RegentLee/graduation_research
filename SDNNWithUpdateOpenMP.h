//
// Created by 李明曄 on 2020/01/06.
//

#ifndef GRADUATION_RESEARCH_SDNNWITHUPDATEOPENMP_H
#define GRADUATION_RESEARCH_SDNNWITHUPDATEOPENMP_H


#include <vector>
#include <random>
#include <algorithm>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

struct NEURON_OUTPUT{
    int potential;
    int index;
};

class SDNNWithUpdateOpenMP {
public:
    SDNNWithUpdateOpenMP(int input_size, std::vector<std::vector<int> > pattern, std::vector<std::vector<int> > w = std::vector<std::vector<int> >());
    float Train(std::vector<int> input, int target);
    float TrainWithUpdate(std::vector<int> input, int target);
    void Update();
    std::vector<int> Predict(std::vector<int> input);
    std::vector<std::vector<int> > GetWeight();
    std::vector<std::vector<int> > GetPattern();

private:
    std::vector<std::vector<int> > original_pattern;
    std::vector<std::vector<int> > changed_pattern;
    std::vector<std::vector<int> > random_pattern;
    std::vector<std::vector<int> > weight;
    std::vector<std::vector<int> > zeros;

    void MakeRandomPattern(std::vector<std::vector<int> > pattern,
                           std::vector<std::vector<int> >& random_pattern);
    std::vector<int> SD(std::vector<int> input);
    float NNTrain(std::vector<int> nn_input, int target);
    float NNTrainWithUpdate(std::vector<int> nn_input, int target);
    std::vector<int> NNPredict(std::vector<int> nn_input);
    static bool cmp(struct NEURON_OUTPUT& x, struct NEURON_OUTPUT& y);
};


#endif //GRADUATION_RESEARCH_SDNNWITHUPDATEOPENMP_H
