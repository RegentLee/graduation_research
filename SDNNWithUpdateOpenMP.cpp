//
// Created by 李明曄 on 2020/01/06.
//

#include "SDNNWithUpdateOpenMP.h"

using namespace std;

SDNNWithUpdateOpenMP::SDNNWithUpdateOpenMP(int input_size, vector<vector<int> > pattern, vector<vector<int> > w) {
    int pattern_size = pattern.size();
    int pattern0_size = pattern[0].size();

#ifdef _OPENMP
    omp_set_num_threads(24);
#endif

    /*original_pattern.resize(pattern_size);
    for(int i = 0; i < pattern_size; i++){
        original_pattern[i].resize(pattern0_size);
        original_pattern[i].assign(pattern[i].begin(), pattern[i].end());
    }*/
    original_pattern.insert(original_pattern.end(), pattern.begin(), pattern.end());
    changed_pattern.resize(pattern_size);
    zeros.resize(pattern_size);
    for(int i = 0; i < pattern_size; i++){
        changed_pattern[i].resize(pattern0_size);
        zeros[i].resize(pattern0_size);
    }

    random_pattern.resize(input_size);
    for(int i = 0; i < input_size; i++){
        random_pattern[i].resize(pattern_size);
    }
    SDNNWithUpdateOpenMP::MakeRandomPattern(pattern, random_pattern);

    if(w.size() == 0) {
        int weight_size = input_size * (input_size - 1) * pattern0_size;
        weight.resize(pattern0_size);
        for (int i = 0; i < pattern0_size; i++) weight[i].resize(weight_size);
    } else {
        weight = w;
    }
}

void SDNNWithUpdateOpenMP::MakeRandomPattern(vector<vector<int> > pattern,
                                   vector<vector<int> > &random_pattern) {

    int pattern_size = pattern.size();
    int pattern0_size = pattern[0].size();
    int random_pattern_size = random_pattern.size();

    vector<int> range(pattern0_size);
    //mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    mt19937_64 mt(0);

    for(int i = 0; i < pattern0_size; i++) range[i] = i;
    for(int i = 0; i < random_pattern_size; i++){
        shuffle(range.begin(), range.end(), mt);
        random_pattern[i].assign(range.begin(), range.end());
    }
}

vector<int> SDNNWithUpdateOpenMP::SD(vector<int> input) {
    int input_size = input.size();
    int original_pattern0_size = original_pattern[0].size();

    vector<int> nn_input(1, 0);
    for(int s = 0; s < input_size; s++){
        for(int c = s + 1; c < input_size; c++){
            int *prpc = &random_pattern[c][0];
            int *prps = &random_pattern[s][0];
            int *pops = &original_pattern[input[s]][0];
            int *popc = &original_pattern[input[c]][0];
            for(int j = 0; j < original_pattern0_size; j++) {
                int st = ((1 + original_pattern[input[c]][*prpc++]) / 2) * *pops++;
                if (st == 0) {
                    nn_input.back()++;
                } else {
                    nn_input.push_back(st);
                    nn_input.push_back(0);
                }
            }
            for(int j = 0; j < original_pattern0_size; j++) {
                int ct = ((1 + original_pattern[input[s]][*prps++]) / 2) * *popc++;
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

float SDNNWithUpdateOpenMP::Train(vector<int> input, int target) {
    int output_size = input.size();
    int output0_size = original_pattern[0].size();

    vector<int> nn_input;
    vector<vector<int> > nn_output(output_size, vector<int>(output0_size));

    //vector<int> pattern_target(output0_size);
    //pattern_target.assign(original_pattern[target].begin(), original_pattern[target].end());

    nn_input = SDNNWithUpdateOpenMP::SD(input);
    NNTrain(nn_input, target);

    return 0;
}

float SDNNWithUpdateOpenMP::NNTrain(vector<int> nn_input, int target) {
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
        int loss = (nn_output - original_pattern[target][o])/2;
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

float SDNNWithUpdateOpenMP::TrainWithUpdate(std::vector<int> input, int target) {
    int output_size = input.size();
    int output0_size = original_pattern[0].size();

    vector<int> nn_input;
    vector<vector<int> > nn_output(output_size, vector<int>(output0_size));

    //vector<int> pattern_target(output0_size);
    //pattern_target.assign(original_pattern[target].begin(), original_pattern[target].end());

    nn_input = SDNNWithUpdateOpenMP::SD(input);
    NNTrainWithUpdate(nn_input, target);

    return 0;
}

float SDNNWithUpdateOpenMP::NNTrainWithUpdate(vector<int> nn_input, int target) {
    /*
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
    vector<int> potential_holder(original_pattern0_size);
    vector<int> nn_output(original_pattern0_size);
    vector<NEURON_OUTPUT> result0;
    vector<NEURON_OUTPUT> result1;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int o = 0; o < original_pattern0_size; o++) {
        //int *pt = &target[0];
        int potential = 0;
        int *pni = &nn_input[0];
        int *pw = &weight[o][0];
        for (int j = 0; j < nn_input_size; j++) {
            pw += *pni++;
            potential += *pni++ * *pw++;
        }
        potential_holder[o] = potential;
    }
    vector<int> copy(potential_holder.begin(), potential_holder.end());
    sort(copy.begin(), copy.end());
    int middle = (copy[original_pattern0_size/2 - 1] + copy[original_pattern0_size/2])/2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int o = 0; o < original_pattern0_size; o++) {
        nn_output[o] = (potential_holder[o] > middle) ? 1 : -1;
        int loss = nn_output[o] - original_pattern[target][o];
        if(loss > 0){
            NEURON_OUTPUT temp;
            temp.potential = potential_holder[o] - middle;
            temp.index = o;
            result0.push_back(temp);
        } else if(loss < 0){
            NEURON_OUTPUT temp;
            temp.potential = middle - potential_holder[o];
            temp.index = o;
            result1.push_back(temp);
        }
    }
    sort(result0.begin(), result0.end(), cmp);
    int result0_len_2 = result0.size()/2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int e = 0; e < result0_len_2; e++){
        int *pni = &nn_input[0];
        int *pw = &weight[result0[e].index][0];
        for(int j = 0; j < nn_input_size; j++){
            pw += *pni++;
            *pw++ -= *pni++;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int e = result0_len_2; e < result0.size(); e++){
        original_pattern[target][result0[e].index] = 1;
    }
    sort(result1.begin(), result1.end(), cmp);
    int result1_len_2 = result1.size()/2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int e = 0; e < result1_len_2; e++){
        int *pni = &nn_input[0];
        int *pw = &weight[result1[e].index][0];
        for(int j = 0; j < nn_input_size; j++){
            pw += *pni++;
            *pw++ += *pni++;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int e = result1_len_2; e < result1.size(); e++){
        original_pattern[target][result1[e].index] = -1;
    }
    int sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(int i = 0; i < original_pattern0_size; i++) sum += original_pattern[target][i];
    sum /= 2;
    if(sum > 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int e = result0_len_2; e < result0_len_2 + sum; e++){
            original_pattern[target][result0[e].index] = -1;
            int *pni = &nn_input[0];
            int *pw = &weight[result0[e].index][0];
            for(int j = 0; j < nn_input_size; j++){
                pw += *pni++;
                *pw++ -= *pni++;
            }
        }
    } else if(sum < 0){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int e = result1_len_2; e < result1_len_2 - sum; e++){
            original_pattern[target][result1[e].index] = 1;
            int *pni = &nn_input[0];
            int *pw = &weight[result1[e].index][0];
            for(int j = 0; j < nn_input_size; j++){
                pw += *pni++;
                *pw++ += *pni++;
            }
        }
    }
*/
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
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
        changed_pattern[target][o] += nn_output;
        int loss = (nn_output - original_pattern[target][o])/2;
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

void SDNNWithUpdateOpenMP::Update(){
    int original_pattern_size = original_pattern.size();
    int original_pattern0_size = original_pattern[0].size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i = 1; i < original_pattern_size - 1; i++){
        vector<int> copy(changed_pattern[i].begin(), changed_pattern[i].end());
        sort(copy.begin(), copy.end());
        int middle = (copy[original_pattern0_size/2 - 1] + copy[original_pattern0_size/2])/2;
        for(int j = 0; j < original_pattern0_size; j++){
            original_pattern[i][j] = (changed_pattern[i][j] > middle) ? 1 : -1;
        }
    }

    changed_pattern.assign(zeros.begin(), zeros.end());
}

vector<int> SDNNWithUpdateOpenMP::Predict(vector<int> input) {
    vector<int> nn_input;
    vector<vector<int> > nn_output(input.size(), vector<int>(original_pattern[0].size()));
    //int output_size = input.size();
    //int output0_size = original_pattern[0].size();

    nn_input = SDNNWithUpdateOpenMP::SD(input);
    return NNPredict(nn_input);
}

vector<int> SDNNWithUpdateOpenMP::NNPredict(vector<int> nn_input) {
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

vector<vector<int> > SDNNWithUpdateOpenMP::GetWeight() { return weight; }

bool SDNNWithUpdateOpenMP::cmp(struct NEURON_OUTPUT& x, struct NEURON_OUTPUT& y) {
    return x.potential < y.potential;
}

std::vector<std::vector<int> > SDNNWithUpdateOpenMP::GetPattern() { return original_pattern; }
