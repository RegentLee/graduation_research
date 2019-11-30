//
// Created by 李明曄 on 2019/11/27.
//

#include "SDNNOpenMP.h"

using namespace std;

SDNNOpenMP::SDNNOpenMP(int input_size, vector<vector<int> > pattern, vector<vector<int> > w) {
    int pattern_size = pattern.size();
    int pattern0_size = pattern[0].size();

    //omp_set_num_threads(24);

    original_pattern.resize(pattern_size);
    for(int i = 0; i < pattern_size; i++){
        original_pattern[i].resize(pattern0_size);
        original_pattern[i].assign(pattern[i].begin(), pattern[i].end());
    }

    random_pattern.resize(input_size);
    for(int i = 0; i < input_size; i++){
        random_pattern[i].resize(pattern_size);
        for(int j = 0; j < pattern_size; j++) random_pattern[i][j].resize(pattern0_size);
    }
    SDNNOpenMP::MakeRandomPattern(pattern, random_pattern);

    if(w.size() == 0) {
        int weight_size = input_size * (input_size - 1) * pattern0_size;
        weight.resize(pattern0_size);
        for (int i = 0; i < pattern0_size; i++) weight[i].resize(weight_size);
    } else {
        weight = w;
    }
}

void SDNNOpenMP::MakeRandomPattern(vector<vector<int> > pattern,
                             vector<vector<vector<int> > > &random_pattern) {

    int pattern_size = pattern.size();
    int pattern0_size = pattern[0].size();
    int random_pattern_size = random_pattern.size();

    vector<int> range(pattern0_size);
    //mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    mt19937_64 mt(0);

    for(int i = 0; i < pattern0_size; i++) range[i] = i;
    for(int i = 0; i < random_pattern_size; i++){
        shuffle(range.begin(), range.end(), mt);
        for(int j = 0; j < pattern_size; j++){
            //int *prp = &random_pattern[i][j][0];
            for(int k = 0; k < pattern0_size; k++){
                random_pattern[i][j][k] = pattern[j][range[k]];
                //*prp++ = pattern[j][range[k]];
            }
        }
    }
}

vector<int> SDNNOpenMP::SD(vector<int> input) {
    int input_size = input.size();
    int original_pattern0_size = original_pattern[0].size();

    vector<int> nn_input(1, 0);
    for(int s = 0; s < input_size; s++){
        for(int c = s + 1; c < input_size; c++){
            int *prpc = &random_pattern[c][input[c]][0];
            int *prps = &random_pattern[s][input[s]][0];
            int *pops = &original_pattern[input[s]][0];
            int *popc = &original_pattern[input[c]][0];
            for(int j = 0; j < original_pattern0_size; j++) {
                int st = ((1 + *prpc++) / 2) * *pops++;
                if (st == 0) {
                    nn_input.back()++;
                } else {
                    nn_input.push_back(st);
                    nn_input.push_back(0);
                }
            }
            for(int j = 0; j < original_pattern0_size; j++) {
                int ct = ((1 + *prps++) / 2) * *popc++;
                if(ct == 0){
                    nn_input.back()++;
                } else {
                    nn_input.push_back(ct);
                    nn_input.push_back(0);
                }
            }
        }
    }

    return nn_input;
}

float SDNNOpenMP::Train(vector<int> input, int target) {
    int output_size = input.size();
    int output0_size = original_pattern[0].size();

    vector<int> nn_input;
    vector<vector<int> > nn_output(output_size, vector<int>(output0_size));

    vector<int> pattern_target(output0_size);
    pattern_target.assign(original_pattern[target].begin(), original_pattern[target].end());

    nn_input = SDNNOpenMP::SD(input);
    NNTrain(nn_input, pattern_target);

    return 0;
}

float SDNNOpenMP::NNTrain(vector<int> nn_input, vector<int> target) {
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
    // int weight_size = weight[0].size(); //nn_input0_size

#pragma omp parallel for
    for(int o = 0; o < original_pattern0_size; o++){
        //int *pt = &target[0];
        int potential = 0;
        int *pni = &nn_input[0];
        int *pw = &weight[o][0];
        for(int j = 0; j < nn_input_size; j++){
            pw += *pni++;
            potential += *pni++**pw++;
        }
        int nn_output = potential < 0 ? -1 : 1;
        int loss = (nn_output - target[o])/2;
        //float loss = (nn_output - *(pt + o))/2.0;
        //float loss = (nn_output - *pt++)/2.0;
        if(loss == 0) continue;
        pni = &nn_input[0];
        pw = &weight[o][0];
        for(int j = 0; j < nn_input_size; j++){
            pw += *pni++;
            *pw++ -= *pni++*loss;
        }
    }

    return 0;
}

vector<int> SDNNOpenMP::Predict(vector<int> input) {
    vector<int> nn_input;
    vector<vector<int> > nn_output(input.size(), vector<int>(original_pattern[0].size()));
    //int output_size = input.size();
    //int output0_size = original_pattern[0].size();

    nn_input = SDNNOpenMP::SD(input);
    return NNPredict(nn_input);
}

vector<int> SDNNOpenMP::NNPredict(vector<int> nn_input) {
    int original_pattern_size = original_pattern.size();
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
    //int weight_size = weight[0].size(); //nn_input0_size

    vector<int> nn_output(original_pattern0_size);
    vector<int> result(original_pattern_size + 1, 0);

#pragma omp parallel for
    for(int o = 0; o < original_pattern0_size; o++){
        int potential = 0;
        int *pni = &nn_input[0];
        int *pw = &weight[o][0];
        for(int j = 0; j < nn_input_size; j++){
            pw += *pni++;
            potential += *pni++**pw++;
        }
        nn_output[o] = potential < 0 ? -1 : 1;
    }

    //int *pr = &result[1];
#pragma omp parallel for
    for(int i = 0; i < original_pattern_size; i++){
        //int *pr = &result[1];
        int *po = &nn_output[0];
        int *pp = &original_pattern[i][0];
        for(int j = 0; j < original_pattern0_size; j++){
            result[i + 1] += *po++ * *pp++;
            //*(pr + i) += *po++ * *pp++;
        }
        //pr++;
    }

    return result;
}

vector<vector<int> > SDNNOpenMP::GetWeight() { return weight; }
