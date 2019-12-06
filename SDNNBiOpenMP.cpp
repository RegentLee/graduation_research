//
// Created by 李明曄 on 2019/12/03.
//

#include "SDNNBiOpenMP.h"

using namespace std;

SDNNBiOpenMP::SDNNBiOpenMP(int input_size, vector<vector<int> > pattern, vector<vector<int> > w) {
    int pattern_size = pattern.size();
    int pattern0_size = pattern[0].size();

#ifdef _OPENMP
    omp_set_num_threads(24);
#endif

    original_pattern.insert(original_pattern.end(), pattern.begin(), pattern.end());

    bi_pattern.resize(input_size + 1);
    for(int i = 0; i < input_size + 1; i++){
        bi_pattern[i].resize(pattern_size);
        for(int j = 0; j < pattern_size; j++) bi_pattern[i][j].resize(pattern0_size / 4, 0);
    }
    SDNNBiOpenMP::MakeBiPattern(pattern, bi_pattern);

    if(w.size() == 0) {
        int weight_size = input_size * (input_size - 1) * pattern0_size;
        weight.resize(pattern0_size);
        for (int i = 0; i < pattern0_size; i++) weight[i].resize(weight_size);
    } else {
        weight = w;
    }
}

void SDNNBiOpenMP::MakeBiPattern(vector<vector<int> > og_pattern,
                                   vector<vector<vector<char> > > &bi_pattern) {

    int og_pattern_size = og_pattern.size();
    int og_pattern0_size = og_pattern[0].size();
    int og_pattern0_size_4 = og_pattern[0].size()/4;
    int bi_pattern_size_1 = bi_pattern.size() - 1;

    vector<int> range(og_pattern0_size);
    //mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    mt19937_64 mt(0);

    for(int j = 0; j < og_pattern_size; j++){
        for(int k = 0; k < og_pattern0_size_4; k++){
            for(int l = 0; l < 4; l++){
                //printf("%d\n", og_pattern[j][k*4 + l]);
                switch(og_pattern[j][k*4 + l]){
                    case 1:{
                        bi_pattern[bi_pattern_size_1][j][k] |= 1 << (2*l);
                        bi_pattern[bi_pattern_size_1][j][k] |= 1 << (2*l + 1);
                        break;
                    }
                    case -1:{
                        bi_pattern[bi_pattern_size_1][j][k] |= 1 << (2*l);
                        //bi_pattern[bi_pattern_size_1][j][k] |= 0 << (2*l + 1);
                        //printf("%d\n", bi_pattern[bi_pattern_size_1][j][k]);
                        break;
                    }
                    /*case 0:{
                        //bi_pattern[bi_pattern_size_1][j][k] |= 0 << (2*l);
                        //bi_pattern[bi_pattern_size_1][j][k] |= 0 << (2*l + 1);
                        //break;
                    }*/
                }
            }
            //printf("%d\n", bi_pattern[bi_pattern_size_1][j][k]);
        }
    }
    for(int i = 0; i < og_pattern0_size; i++) range[i] = i;
    for(int i = 0; i < bi_pattern_size_1; i++){
        shuffle(range.begin(), range.end(), mt);
        for(int j = 0; j < og_pattern_size; j++){
            //int *prp = &bi_pattern[i][j][0];
            for(int k = 0; k < og_pattern0_size_4; k++){
                for(int l = 0; l < 4; l++){
                    switch(og_pattern[j][range[k*4 + l]]){
                        case 1:{
                            bi_pattern[i][j][k] |= 1 << (2*l);
                            bi_pattern[i][j][k] |= 1 << (2*l + 1);
                            break;
                        }
                        case -1:{
                            bi_pattern[i][j][k] |= 1 << (2*l);
                            //bi_pattern[i][j][k] |= 0 << (2*l + 1);
                            break;
                        }
                            /*case 0:{
                                //bi_pattern[i][j][k] |= 0 << (2*l);
                                //bi_pattern[i][j][k] |= 0 << (2*l + 1);
                                //break;
                            }*/
                    }
                }
                //*prp++ = og_pattern[j][range[k]];
            }
        }
    }
}

vector<int> SDNNBiOpenMP::SD(vector<int> input) {
    int input_size = input.size();
    int original_pattern0_size_4 = original_pattern[0].size()/4;
    int bi_og_pattern_position = bi_pattern.size() - 1;

    vector<int> nn_input(1, 0);
    for(int s = 0; s < input_size; s++){
        for(int c = s + 1; c < input_size; c++){
            char *prpc = &bi_pattern[c][input[c]][0];
            char *prps = &bi_pattern[s][input[s]][0];
            char *pops = &bi_pattern[bi_og_pattern_position][input[s]][0];
            char *popc = &bi_pattern[bi_og_pattern_position][input[c]][0];
            for(int j = 0; j < original_pattern0_size_4; j++) {
                char popst = *pops++;
                char prpct = *prpc++;
                for(int k = 0; k < 4; k++){
                    if((popst & 0x1) && (prpct & 0x1)){
                        popst >>= 1;
                        prpct >>= 1;
                        if(prpct & 0x1){
                            nn_input.push_back(2*(popst & 0x1) - 1);
                            nn_input.push_back(0);
                        } else {
                            nn_input.back()++;
                        }
                        popst >>= 1;
                        prpct >>= 1;
                    } else {
                        popst >>= 2;
                        prpct >>= 2;
                        nn_input.back()++;
                    }
                }
            }
            for(int j = 0; j < original_pattern0_size_4; j++) {
                char prpst = *prps++;
                char popct = *popc++;
                for(int k = 0; k < 4; k++){
                    if((prpst & 0x1) && (popct & 0x1)){
                        prpst >>= 1;
                        popct >>= 1;
                        if(prpst & 0x1){
                            nn_input.push_back(2*(popct & 0x1) - 1);
                            nn_input.push_back(0);
                        } else {
                            nn_input.back()++;
                        }
                        prpst >>= 1;
                        popct >>= 1;
                    } else {
                        prpst >>= 2;
                        popct >>= 2;
                        nn_input.back()++;
                    }
                }
            }
        }
    }

    return nn_input;
}

/*vector<int> SDNNOpenMP::SD(vector<int> input) {
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
}*/

float SDNNBiOpenMP::Train(vector<int> input, int target) {
    int output_size = input.size();
    int output0_size = original_pattern[0].size();

    vector<int> nn_input;
    vector<vector<int> > nn_output(output_size, vector<int>(output0_size));

    vector<int> pattern_target(output0_size);
    pattern_target.assign(original_pattern[target].begin(), original_pattern[target].end());

    nn_input = SDNNBiOpenMP::SD(input);
    NNTrain(nn_input, pattern_target);

    return 0;
}

float SDNNBiOpenMP::NNTrain(vector<int> nn_input, vector<int> target) {
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
    // int weight_size = weight[0].size(); //nn_input0_size
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

vector<int> SDNNBiOpenMP::Predict(vector<int> input) {
    vector<int> nn_input;
    vector<vector<int> > nn_output(input.size(), vector<int>(original_pattern[0].size()));
    //int output_size = input.size();
    //int output0_size = original_pattern[0].size();

    nn_input = SDNNBiOpenMP::SD(input);
    return NNPredict(nn_input);
}

vector<int> SDNNBiOpenMP::NNPredict(vector<int> nn_input) {
    int original_pattern_size = original_pattern.size();
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
    //int weight_size = weight[0].size(); //nn_input0_size

    vector<int> nn_output(original_pattern0_size);
    vector<int> result(original_pattern_size + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

vector<vector<int> > SDNNBiOpenMP::GetWeight() { return weight; }
