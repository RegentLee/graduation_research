//
// Created by 李明曄 on 2019/12/07.
//

#ifndef GRADUATION_RESEARCH_FUNC_H
#define GRADUATION_RESEARCH_FUNC_H

#include <vector>
#include <random>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>

std::vector<std::vector<int> > CosPattern(int len);
std::vector<std::vector<int> > CosPatternKai(int len);
std::vector<std::vector<int> > CosPatternKai2(int len);
int dot(std::vector<int> a, std::vector<int> b);
std::vector<int> equal(std::vector<int> a, std::vector<int> b, bool c);
std::vector<int> equal(std::vector<int> a, std::vector<int> b, std::vector<int> c,bool d);

struct param_list{
    std::string train_sample_file;
    std::string test_sample_file;
    std::string pattern_file;
    std::string train_result_file;
    std::string test_result_file;
    std::string save_weight_file;
    std::string read_weight_file;
};

struct param_list ReadParam();

#endif //GRADUATION_RESEARCH_FUNC_H
