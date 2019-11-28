//
// Created by 李明曄 on 2019/11/27.
//

#ifndef GRADUATION_RESEARCH_SDNNOPENMP_H
#define GRADUATION_RESEARCH_SDNNOPENMP_H

#include <vector>
#include <random>

class SDNNOpenMP {
public:
    SDNNOpenMP(int input_size, std::vector<std::vector<int> > pattern);
    float Train(std::vector<int> input, int target);
    std::vector<int> Predict(std::vector<int> input);
    std::vector<std::vector<int> > GetPattern();

private:
    std::vector<std::vector<int> > original_pattern;
    std::vector<std::vector<std::vector<int> > > random_pattern;
    std::vector<std::vector<float> > weight;

    void MakeRandomPattern(std::vector<std::vector<int> > pattern,
                           std::vector<std::vector<std::vector<int> > >& random_pattern);
    std::vector<float> SD(std::vector<int> input);
    float NNTrain(std::vector<float> nn_input, std::vector<float> target);
    std::vector<int> NNPredict(std::vector<float> nn_input);
};


#endif //GRADUATION_RESEARCH_SDNNOPENMP_H
