//
// Created by 李明曄 on 2019/11/17.
//

#ifndef GRADUATION_RESEARCH_TRAINER_H
#define GRADUATION_RESEARCH_TRAINER_H

#include <random>
#include <iostream>
#include <time.h>

#include "SDNN.h"

class Trainer {
public:
    void fit(SDNN& model, std::vector<std::vector<int> > sample, int max_epoch = 10, int batch_size = 32);
};


#endif //GRADUATION_RESEARCH_TRAINER_H
