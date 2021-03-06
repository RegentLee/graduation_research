//
// Created by 李明曄 on 2019/11/17.
//

#ifndef GRADUATION_RESEARCH_TRAINER_H
#define GRADUATION_RESEARCH_TRAINER_H

#include <random>
#include <iostream>
#include <ctime>

#include "SDNN.h"
#include "SDNNOpenMP.h"
#include "SDNNBiOpenMP.h"
//#include "SDNNFABiOpenMP.h"
#include "SDNNWithUpdateOpenMP.h"

class Trainer {
public:
    void fit(SDNN& model, std::vector<std::vector<int> > sample, std::string fp, int max_epoch = 10, int batch_size = 32);
    void fit(SDNNOpenMP& model, std::vector<std::vector<int> > sample, std::string fp, int max_epoch = 10, int batch_size = 1);
    void fit(SDNNBiOpenMP& model, std::vector<std::vector<int> > sample, std::string fp, int max_epoch = 10, int batch_size = 1);
    //void fit(SDNNFABiOpenMP& model, std::vector<std::vector<int> > sample, std::string fp, int max_epoch = 10, int batch_size = 1);
    void fit(SDNNWithUpdateOpenMP& model, std::vector<std::vector<int> > sample, std::string fp, int max_epoch = 10, int batch_size = 1);
};



#endif //GRADUATION_RESEARCH_TRAINER_H
