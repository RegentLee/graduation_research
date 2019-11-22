//
// Created by 李明曄 on 2019/11/17.
//

#ifndef GRADUATION_RESEARCH_TESTER_H
#define GRADUATION_RESEARCH_TESTER_H

#include <random>
#include <iostream>
#include <ctime>

#include "SDNN.h"

class Tester {
public:
    void predict(SDNN& model, std::vector<std::vector<int> > sample, int batch_size = 32);
};


#endif //GRADUATION_RESEARCH_TESTER_H
