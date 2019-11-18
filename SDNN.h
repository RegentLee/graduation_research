//
// Created by 李明曄 on 2019/11/16.
//

#ifndef GRADUATION_RESEARCH_SDNN_H
#define GRADUATION_RESEARCH_SDNN_H

#include <vector>
#include <random>

class SDNN {
public:
    SDNN(int input_size, std::vector<std::vector<int> > pattern);
    std::vector<std::vector<int> > Forward(std::vector<std::vector<int> > input);
    void Backward(std::vector<std::vector<int> > output, std::vector<int> target);
    std::vector<std::vector<int> > GetPattern();

private:
    std::vector<std::vector<int> > original_pattern;
    std::vector<std::vector<std::vector<int> > > random_pattern;
    std::vector<std::vector<int> > nn_ip;
    std::vector<std::vector<float> > weight;

    void MakeRandomPattern(std::vector<std::vector<int> > pattern,
                           std::vector<std::vector<std::vector<int> > >& random_pattern);
    std::vector<std::vector<int> > SD(std::vector<std::vector<int> > input);
    std::vector<std::vector<int> > NNForward(std::vector<std::vector<int> > nn_input);
    void NNBackward(std::vector<std::vector<int> > output, std::vector<std::vector<int> > target);
};

#endif //GRADUATION_RESEARCH_SDNN_H