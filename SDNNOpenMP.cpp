//
// Created by 李明曄 on 2019/11/27.
//

#include "SDNNOpenMP.h"

using namespace std;

SDNNOpenMP::SDNNOpenMP(int input_size, std::vector<std::vector<int> > pattern) {
    int pattern_size = pattern.size();
    int pattern0_size = pattern[0].size();

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

    int weight_size = input_size*(input_size - 1)*pattern0_size;
    weight.resize(pattern0_size);
    for(int i = 0; i < pattern0_size; i++) weight[i].resize(weight_size);
}

void SDNNOpenMP::MakeRandomPattern(vector<vector<int> > pattern,
                             vector<vector<vector<int> > > &random_pattern) {

    int pattern_size = pattern.size();
    int pattern0_size = pattern[0].size();
    int random_pattern_size = random_pattern.size();

    vector<int> range(pattern0_size);
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    //mt19937_64 mt(0);

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

vector<float> SDNNOpenMP::SD(vector<int> input) {
    int input_size = input.size();
    int original_pattern0_size = original_pattern[0].size();

    vector<float> nn_input(1, 0);
    for(int s = 0; s < input_size; s++){
        for(int c = s + 1; c < input_size; c++){
            vector<float> s_temp(1, 0);//(original_pattern0_size, 0);
            vector<float> c_temp(1, 0);//(original_pattern0_size, 0);
            //float *pst = &s_temp[0];
            //float *pct = &c_temp[0];
            int *prpc = &random_pattern[c][input[c]][0];
            int *prps = &random_pattern[s][input[s]][0];
            int *pops = &original_pattern[input[s]][0];
            int *popc = &original_pattern[input[c]][0];
            for(int j = 0; j < original_pattern0_size; j++) {
                int st = ((1 + *prpc++) / 2) * *pops++;
                if (st == 0) {
                    //(*pst)++;
                    //s_temp.back()++;
                    nn_input.back()++;
                } else {
                    //*pst++;
                    //*pst++ = (float)st;
                    //s_temp.push_back((float) st);
                    //if (j != original_pattern0_size - 1) s_temp.push_back(0.0);
                    nn_input.push_back((float) st);
                    //if (j != original_pattern0_size - 1) nn_input.push_back(0.0);
                    nn_input.push_back(0.0);
                }
            }
            for(int j = 0; j < original_pattern0_size; j++) {
                int ct = ((1 + *prps++) / 2) * *popc++;
                if(ct == 0){
                    //(*pct)++;
                    nn_input.back()++;
                } else {
                    //*pct++;
                    //*pct++ = (float)ct;
                    nn_input.push_back((float)ct);
                    //if(j != original_pattern0_size - 1) nn_input.push_back(0.0);
                    nn_input.push_back(0.0);
                }
                //*pst++ = (float)(((1 + *prpc++)/2)**pops++);
                //*pct++ = (float)(((1 + *prps++)/2)**popc++);
            }
            /*if(nn_input.size()%2 != 0){

                printf("%f, ", nn_input.back());
                for(int i = 0; i < nn_input.back(); i++){
                    printf("%d, ", ((1 + random_pattern[s][input[s]][original_pattern0_size - 1 - i]) / 2) * original_pattern[input[c]][original_pattern0_size - 1 - i]);
                }
                printf("\n");

                nn_input.pop_back();
            }*/
            //for(int i = 0; i < nn_input.size(); i++) printf("%f\n", nn_input[i]);
            //nn_input.insert(nn_input.end(), s_temp.begin(), s_temp.end());
            //nn_input.insert(nn_input.end(), c_temp.begin(), c_temp.end());
            //for(int j = 0; j < original_pattern[0].size(); j++) nn_input[i].push_back(s_temp[j]);
            //for(int j = 0; j < original_pattern[0].size(); j++) nn_input[i].push_back(c_temp[j]);
        }
    }

    return nn_input;
}

vector<vector<int> > SDNNOpenMP::GetPattern() { return original_pattern; }

float SDNNOpenMP::Train(vector<int> input, int target) {
    vector<float> nn_input;
    vector<vector<int> > nn_output(input.size(), vector<int>(original_pattern[0].size()));
    int output_size = input.size();
    int output0_size = original_pattern[0].size();

    vector<float> pattern_target(output0_size);
    pattern_target.assign(original_pattern[target].begin(), original_pattern[target].end());

    nn_input = SDNNOpenMP::SD(input);
    NNTrain(nn_input, pattern_target);

    return 0;
}

float SDNNOpenMP::NNTrain(vector<float> nn_input, vector<float> target) {
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
    // int weight_size = weight[0].size(); //nn_input0_size

    //printf("%d\n", nn_input_size);

    //float *pt = &target[0];
    for(int o = 0; o < original_pattern0_size; o++){
        //float *pt = &target[0];
        float potential = 0;
        float *pni = &nn_input[0];
        float *pw = &weight[o][0];
        for(int j = 0; j < nn_input_size; j++){
            //printf("%p, %f, ", pw, *pni);
            pw += (int)*pni++;
            //printf("%p\n", pw);
            potential += *pni++**pw++;
        }
        float nn_output = potential < 0 ? -1.0 : 1.0;
        float loss = (nn_output - target[o])/2.0;
        //float loss = (nn_output - *(pt + o))/2.0;
        //float loss = (nn_output - *pt++)/2.0;
        if(loss < 0.5 && loss > -0.5) continue;
        pni = &nn_input[0];
        pw = &weight[o][0];
        for(int j = 0; j < nn_input_size; j++){
            pw += (int)*pni++;
            *pw++ -= *pni++*loss;
        }
    }

    return 0;
}

vector<int> SDNNOpenMP::Predict(vector<int> input) {
    vector<float> nn_input;
    vector<vector<int> > nn_output(input.size(), vector<int>(original_pattern[0].size()));
    int output_size = input.size();
    int output0_size = original_pattern[0].size();

    nn_input = SDNNOpenMP::SD(input);
    return NNPredict(nn_input);
}

vector<int> SDNNOpenMP::NNPredict(vector<float> nn_input) {
    int original_pattern_size = original_pattern.size();
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
    int weight_size = weight[0].size(); //nn_input0_size

    vector<int> nn_output(original_pattern0_size);
    vector<int> result(original_pattern_size + 1, 0);

    for(int o = 0; o < original_pattern0_size; o++){
        float potential = 0;
        float *pni = &nn_input[0];
        float *pw = &weight[o][0];
        for(int j = 0; j < nn_input_size; j++){
            //printf("%p, %f, ", pw, *pni);
            pw += (int)*pni++;
            //printf("%p\n", pw);
            potential += *pni++**pw++;
        }
        nn_output[o] = potential < 0 ? -1 : 1;
    }


    //int *pr = &result[1];
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
