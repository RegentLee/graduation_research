//
// Created by 李明曄 on 2019/11/16.
//

#include <random>
#include "SDNN.h"

using namespace std;

SDNN::SDNN(int input_size, std::vector<std::vector<int> > pattern) {
    original_pattern.resize(pattern.size());
    for(int i = 0; i < pattern.size(); i++){
        original_pattern[i].resize(pattern[0].size());
        for(int j = 0; j < pattern[0].size(); j++){
            original_pattern[i][j] = pattern[i][j];
        }
    }

    random_pattern.resize(input_size);
    for(int i = 0; i < input_size; i++){
        random_pattern[i].resize(pattern.size());
        for(int j = 0; j < pattern.size(); j++) random_pattern[i][j].resize(pattern[0].size());
    }
    SDNN::MakeRandomPattern(pattern, random_pattern);

    weight.resize(input_size*(input_size - 1)*pattern[0].size());
    for(int i = 0; i < weight.size(); i++) weight[i].resize(pattern[0].size());
}

void SDNN::MakeRandomPattern(vector<vector<int> > pattern,
                             vector<vector<vector<int> > > &random_pattern) {

    vector<int> range(pattern[0].size());
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));

    for(int i = 0; i < pattern[0].size(); i++) range[i] = i;
    for(int i = 0; i < random_pattern.size(); i++){
        shuffle(range.begin(), range.end(), mt);
        for(int j = 0; j < pattern.size(); j++){
            for(int k = 0; k < pattern[0].size(); k++){
                random_pattern[i][j][k] = pattern[j][range[k]];
            }
        }
    }
}

vector<vector<int> > SDNN::Forward(std::vector<std::vector<int> > input) {
    vector<vector<int> > nn_input(input.size());
    vector<vector<int> > nn_output(input.size(), vector<int>(original_pattern[0].size()));
    nn_input = SDNN::SD(input);
    nn_ip.resize(nn_input.size());
    for(int i = 0; i < nn_input.size(); i++){
        nn_ip[i].resize(nn_input[0].size());
        for(int j = 0; j < nn_input[0].size(); j++){
            nn_ip[i][j] = nn_input[i][j];
        }
    }
    nn_output = SDNN::NNForward(nn_input);

    return nn_output;
}

vector<vector<int> > SDNN::SD(std::vector<std::vector<int> > input) {
    vector<vector<int> > nn_input(input.size());
    for(int i = 0; i < input.size(); i++){
        //nn_output[i].resize(original_pattern[0].size());
        //for(int j = 0; j < original_pattern[0].size(); j++) nn_output[i][j] = 0;
        for(int s = 0; s < input[0].size(); s++){
            for(int c = s + 1; c < input[0].size(); c++){
                vector<int> s_temp(original_pattern[0].size());
                vector<int> c_temp(original_pattern[0].size());
                for(int j = 0; j < original_pattern[0].size(); j++){
                    s_temp[j] = ((1 + random_pattern[c][input[i][c]][j])/2)*original_pattern[input[i][s]][j];
                    c_temp[j] = ((1 + random_pattern[s][input[i][s]][j])/2)*original_pattern[input[i][c]][j];
                }
                for(int j = 0; j < original_pattern[0].size(); j++) nn_input[i].push_back(s_temp[j]);
                for(int j = 0; j < original_pattern[0].size(); j++) nn_input[i].push_back(c_temp[j]);
            }
        }
    }

    return nn_input;
}

vector<vector<int> > SDNN::NNForward(vector<vector<int> > nn_input) {
    //printf("%s\n", "before forward");
    vector<vector<int> > nn_output(nn_input.size(), vector<int>(original_pattern[0].size(), 0));
    for(int i = 0; i < nn_input.size(); i++){
        for(int j = 0; j < nn_input[0].size(); j++){
            for(int k = 0; k < nn_output[0].size(); k++){
                nn_output[i][k] += nn_input[i][j]*weight[j][k];
            }
        }
    }
    for(int i = 0; i < nn_output.size(); i++){
        for(int k = 0; k < nn_output[0].size(); k++){
            if(nn_output[i][k] < 0) nn_output[i][k] = -1;
            else nn_output[i][k] = 1;
        }
    }

    return nn_output;
}

void SDNN::Backward(vector<vector<int> > output, vector<int> target) {
    vector<vector<int> > pattern_target(output.size(), vector<int>(output[0].size()));
    for(int i = 0; i < output.size(); i++){
        for(int j = 0; j < output[0].size(); j++){
            pattern_target[i][j] = original_pattern[target[i]][j];
        }
    }
    NNBackward(output, pattern_target);
}

void SDNN::NNBackward(vector<vector<int> > output, vector<vector<int> > target) {
    vector<vector<float> > loss(output.size(), vector<float>(output[0].size()));
    vector<vector<float> > grad(weight.size(), vector<float>(weight[0].size(), 0));
    for(int i = 0; i < output.size(); i++){
        for(int j = 0; j < output[0].size(); j++){
            //printf("i:%d, j:%d, output:%d, target:%d, ", i, j, output[i][j], target[i][j]);
            loss[i][j] = (float)(output[i][j] - target[i][j])/2.0;
            loss[i][j] /= output.size();
            //printf("loss:%f\n", loss[i][j]);
        }
    }
    //printf("%s\n", "before grad");
    //printf("loss.size():%d, loss[0].size():%d\n", loss.size(), loss[0].size());
    //printf("weight.size():%d, weight[0].size():%d\n", weight.size(), weight[0].size());
    //printf("nn_ip.size():%d, nn_ip[0].size():%d\n", nn_ip.size(), nn_ip[0].size());
    //printf("grad.size():%d, grad[0].size():%d\n", grad.size(), grad[0].size());

    for(int i = 0; i < nn_ip[0].size(); i++){
        for(int j = 0; j < loss[0].size(); j++){
            for(int k = 0; k < loss.size(); k++){
                //printf("nn_ip:%f, loss:%f\n", (float)nn_ip[k][i], loss[k][j]);
                grad[i][j] += (float)nn_ip[k][i]*loss[k][j];
                //printf("i:%d, j:%d, k:%d, grad[i][j]:%f\n", i, j, k, grad[i][j]);
            }
        }
    }
    //printf("%s\n", "before weight");
    //printf("weight.size():%d, weight[0].size():%d\n", weight.size(), weight[0].size());
    //printf("grad.size():%d, grad[0].size():%d\n", grad.size(), grad[0].size());
    for(int i = 0; i < weight.size(); i++){
        for(int j = 0; j < weight[0].size(); j++) weight[i][j] -= grad[i][j];
    }
    //printf("%s\n", "after weight");
}


