//
// Created by 李明曄 on 2019/11/17.
//

#ifndef GRADUATION_RESEARCH_TESTER_H
#define GRADUATION_RESEARCH_TESTER_H

#include <random>
#include <iostream>
#include <ctime>
#include <string>

#include "SDNN.h"
#include "SDNNOpenMP.h"
#include "SDNNBiOpenMP.h"

class Tester {
public:
    void predict(SDNN& model, std::vector<std::vector<int> > sample, std::string fp, int batch_size = 32);
    void predict(SDNNOpenMP& model, std::vector<std::vector<int> > sample, std::string fp, int batch_size = 1);
    void predict(SDNNBiOpenMP& model, std::vector<std::vector<int> > sample, std::string fp, int batch_size = 1);
};


#endif //GRADUATION_RESEARCH_TESTER_H
