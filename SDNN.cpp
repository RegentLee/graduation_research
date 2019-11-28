//
// Created by 李明曄 on 2019/11/16.
//


#include "SDNN.h"

using namespace std;

SDNN::SDNN(int input_size, std::vector<std::vector<int> > pattern) {
    int pattern_size = pattern.size();
    int pattern0_size = pattern[0].size();

    original_pattern.resize(pattern_size);
    //original_pattern.assign(pattern.begin(), pattern.end());
    for(int i = 0; i < pattern_size; i++){
        original_pattern[i].resize(pattern0_size);
        original_pattern[i].assign(pattern[i].begin(), pattern[i].end());
        /*for(int j = 0; j < pattern0_size; j++){
            original_pattern[i][j] = pattern[i][j];
        }*/
    }

    random_pattern.resize(input_size);
    for(int i = 0; i < input_size; i++){
        random_pattern[i].resize(pattern_size);
        for(int j = 0; j < pattern_size; j++) random_pattern[i][j].resize(pattern0_size);
    }
    SDNN::MakeRandomPattern(pattern, random_pattern);

    weight.resize(input_size*(input_size - 1)*pattern0_size);
    for(int i = 0; i < weight.size(); i++) weight[i].resize(pattern0_size);
}

void SDNN::MakeRandomPattern(vector<vector<int> > pattern,
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

vector<vector<int> > SDNN::Forward(std::vector<std::vector<int> > input) {
    vector<vector<int> > nn_input(input.size());
    vector<vector<int> > nn_output(input.size(), vector<int>(original_pattern[0].size()));
    nn_input = SDNN::SD(input);
    nn_ip.resize(nn_input.size());
    nn_ip.assign(nn_input.begin(), nn_input.end());
/*
    for(int i = 0; i < nn_input.size(); i++){
        nn_ip[i].resize(nn_input[0].size());
        for(int j = 0; j < nn_input[0].size(); j++){
            nn_ip[i][j] = nn_input[i][j];
        }
    }
*/
    nn_output = SDNN::NNForward(nn_input);

    return nn_output;
}

vector<vector<int> > SDNN::SD(std::vector<std::vector<int> > input) {
    int input_size = input.size();
    int input0_size = input[0].size();
    int original_pattern0_size = original_pattern[0].size();

    vector<vector<int> > nn_input(input_size);
    for(int i = 0; i < input_size; i++){
        //nn_output[i].resize(original_pattern[0].size());
        //for(int j = 0; j < original_pattern[0].size(); j++) nn_output[i][j] = 0;
        for(int s = 0; s < input0_size; s++){
            for(int c = s + 1; c < input0_size; c++){
                vector<int> s_temp(original_pattern0_size);
                vector<int> c_temp(original_pattern0_size);
                int *pst = &s_temp[0];
                int *pct = &c_temp[0];
                int *prpc = &random_pattern[c][input[i][c]][0];
                int *prps = &random_pattern[s][input[i][s]][0];
                int *pops = &original_pattern[input[i][s]][0];
                int *popc = &original_pattern[input[i][c]][0];
                for(int j = 0; j < original_pattern0_size; j++){
                    //s_temp[j] = ((1 + random_pattern[c][input[i][c]][j])/2)*original_pattern[input[i][s]][j];
                    //c_temp[j] = ((1 + random_pattern[s][input[i][s]][j])/2)*original_pattern[input[i][c]][j];
                    *pst++ = ((1 + *prpc++)/2)**pops++;
                    *pct++ = ((1 + *prps++)/2)**popc++;
                }
                nn_input[i].insert(nn_input[i].end(), s_temp.begin(), s_temp.end());
                nn_input[i].insert(nn_input[i].end(), c_temp.begin(), c_temp.end());
                //for(int j = 0; j < original_pattern[0].size(); j++) nn_input[i].push_back(s_temp[j]);
                //for(int j = 0; j < original_pattern[0].size(); j++) nn_input[i].push_back(c_temp[j]);
            }
        }
    }

    return nn_input;
}

vector<vector<int> > SDNN::NNForward(vector<vector<int> > nn_input) {
    int nn_input_size = nn_input.size();
    int nn_input0_size = nn_input[0].size();
    int original_pattern0_size = original_pattern[0].size();

    //printf("%s\n", "before forward");
    vector<vector<float> > potential(nn_input_size, vector<float>(original_pattern0_size, 0));
    vector<vector<int> > nn_output(nn_input_size, vector<int>(original_pattern0_size, 0));
    for(int i = 0; i < nn_input_size; i++){
        for(int j = 0; j < nn_input0_size; j++){
            if(nn_input[i][j] == 0) continue;
            float nn_input_ij = (float)nn_input[i][j];
            float *pz = &potential[i][0];
            float *py = &weight[j][0];
            for(int k = 0; k < original_pattern0_size; k++){
                //potential[i][k] += nn_input[i][j]*weight[j][k];
                //*pz++ += (float)nn_input[i][j]**py++;
                *pz++ += nn_input_ij**py++;
            }
        }
    }
    for(int i = 0; i < nn_input_size; i++){
        float *pz = &potential[i][0];
        int *po = &nn_output[i][0];
        for(int k = 0; k < original_pattern0_size; k++){
            //if(potential[i][k] < 0) nn_output[i][k] = -1;
            //else nn_output[i][k] = 1;
            *po++ = *pz++ < 0 ? -1 : 1;
            //nn_output[i][k] = potential[i][k] < 0 ? -1 : 1;
        }
    }

    return nn_output;
}

float SDNN::Backward(vector<vector<int> > output, vector<int> target) {
    int output_size = output.size();
    int output0_size = output[0].size();

    vector<vector<int> > pattern_target(output_size, vector<int>(output0_size));
    for(int i = 0; i < output_size; i++){
        for(int j = 0; j < output0_size; j++){
            pattern_target[i][j] = original_pattern[target[i]][j];
        }
    }
    return NNBackward(output, pattern_target);
}

float SDNN::NNBackward(vector<vector<int> > output, vector<vector<int> > target) {
    int output_size = output.size();
    int output0_size = output[0].size();
    int weight_size = weight.size();
    //int weight0_size = weight[0].size();

    vector<vector<float> > loss(output_size, vector<float>(output0_size));
    //vector<vector<float> > grad(weight.size(), vector<float>(weight[0].size(), 0));
    float o2 = (float)output_size*2;
    int err = 0;

    for(int i = 0; i < output_size; i++){
        float *pl = &loss[i][0];
        int *po = &output[i][0];
        int *pt = &target[i][0];
        for(int j = 0; j < output0_size; j++){
            //printf("i:%d, j:%d, output:%d, target:%d, ", i, j, output[i][j], target[i][j]);
            //loss[i][j] = (float)(output[i][j] - target[i][j])/2.0;
            //loss[i][j] /= output_size;
            err += abs(*po - *pt);
            //err += *po > *pt ? 2 : *pt - *po;
            *pl++ += (float)(*po++ - *pt++)/o2;
            //printf("loss:%f\n", loss[i][j]);
        }
    }
    //printf("%s\n", "before grad");
    //printf("loss.size():%d, loss[0].size():%d\n", loss.size(), loss[0].size());
    //printf("weight.size():%d, weight[0].size():%d\n", weight.size(), weight[0].size());
    //printf("nn_ip.size():%d, nn_ip[0].size():%d\n", nn_ip.size(), nn_ip[0].size());
    //printf("grad.size():%d, grad[0].size():%d\n", grad.size(), grad[0].size());

    //for(int i = 0; i < nn_ip[0].size(); i++){
    for(int i = 0; i < weight_size; i++){
        for(int k = 0; k < output_size; k++){
            if(nn_ip[k][i] == 0) continue;
            float nn_ip_ki = (float)nn_ip[k][i];
            float *pz = &weight[i][0];
            float *py = &loss[k][0];
            for(int j = 0; j < output0_size; j++){
                //printf("nn_ip:%f, loss:%f, ", (float)nn_ip[k][i], loss[k][j]);
                //grad[i][j] += (float)nn_ip[k][i]*loss[k][j];
                //weight[i][j] -= (float)nn_ip[k][i]*loss[k][j];
                *pz++ -= nn_ip_ki**py++;
                //printf("i:%d, j:%d, k:%d, grad[i][j]:%f\n", i, j, k, grad[i][j]);
            }
        }
    }
    //printf("%s\n", "before weight");
    //printf("weight.size():%d, weight[0].size():%d\n", weight.size(), weight[0].size());
    //printf("grad.size():%d, grad[0].size():%d\n", grad.size(), grad[0].size());
    /*
    for(int i = 0; i < weight.size(); i++){
        for(int j = 0; j < weight[0].size(); j++){
            weight[i][j] -= grad[i][j];
            //printf("%f\n", weight[i][j]);
        }
    }
     */
    //printf("%s\n", "after weight");

    return (float)err/(float)(2*output_size*output0_size);
}

vector<vector<int> > SDNN::GetPattern() { return original_pattern; }
