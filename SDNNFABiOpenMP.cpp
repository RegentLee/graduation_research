//
// Created by 李明曄 on 2019/12/25.
//

#include "SDNNFABiOpenMP.h"
#include "csv.h"

using namespace std;

SDNNFABiOpenMP::SDNNFABiOpenMP(int input_size, vector<vector<int> > pattern, int element, vector<vector<vector<int> > > w) {
    int pattern_size = pattern.size();
    int pattern0_size = pattern[0].size();
    int original_pattern0_size;
#ifdef _OPENMP
    omp_set_num_threads(24);
#endif

    //original_pattern.insert(original_pattern.end(), pattern.begin(), pattern.end());
    csv::ReadCsv(original_pattern, "vector.csv");
    original_pattern0_size = original_pattern[0].size();

    output_neuron_size = original_pattern0_size*element;

    bi_pattern.resize(input_size + 1);
    for(int i = 0; i < input_size + 1; i++){
        bi_pattern[i].resize(pattern_size);
        for(int j = 0; j < pattern_size; j++) bi_pattern[i][j].resize(pattern0_size / 4, 0);
    }
    SDNNFABiOpenMP::MakeBiPattern(pattern, bi_pattern);

    if(w.size() == 0) {
        int weight_size = input_size * (input_size - 1) * pattern0_size;
        weight.resize(original_pattern0_size);
        for (int i = 0; i < original_pattern0_size; i++){
            weight[i].resize(element);
            for(int j = 0; j < element; j++){
                weight[i][j].resize(weight_size);
            }
        }
    } else {
        weight = w;
    }
}

void SDNNFABiOpenMP::MakeBiPattern(vector<vector<int> > og_pattern,
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
    //cout << bi_pattern[bi_pattern_size_1][0][0] << endl;
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

vector<int> SDNNFABiOpenMP::SD(vector<int> input) {
    int input_size = input.size();
    int original_pattern0_size_4 = bi_pattern[0][0].size();
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

/*vector<int> SDNNFAOpenMP::SD(vector<int> input) {
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

float SDNNFABiOpenMP::Train(vector<int> input, int target) {
    int output_size = input.size();
    int output0_size = original_pattern[0].size();

    vector<int> nn_input;

    vector<int> pattern_target(output0_size);
    pattern_target.assign(original_pattern[target].begin(), original_pattern[target].end());

    nn_input = SDNNFABiOpenMP::SD(input);
    NNTrain(nn_input, pattern_target);

    return 0;
}

float SDNNFABiOpenMP::NNTrain(vector<int> nn_input, vector<int> target) {
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
    int element = weight[0].size();
    vector<vector<NEURON_OUTPUT> > result0(original_pattern0_size, vector<NEURON_OUTPUT>(element));
    vector<vector<NEURON_OUTPUT> > result1(original_pattern0_size, vector<NEURON_OUTPUT>(element));

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int o = 0; o < original_pattern0_size; o++){
        int neuron_output_sum = -10;
        for(int e = 0; e < element; e++){
            int potential = 0;
            int *pni = &nn_input[0];
            int *pw = &weight[o][e][0];
            for(int j = 0; j < nn_input_size; j++){
                pw += *pni++;
                potential += *pni++**pw++;
            }

            neuron_output_sum += (potential >= 0);
            result0[o][e].potential = (potential >= 0) ? potential : 0x8fffff;
            result0[o][e].index = e;
            result1[o][e].potential = (potential < 0) ? -potential : 0x8fffff;
            result1[o][e].index = e;
        }
        if(neuron_output_sum < -8) neuron_output_sum = -8;
        else if(neuron_output_sum > 8) neuron_output_sum = 8;
        if(neuron_output_sum == target[o]) continue;

        int error = neuron_output_sum - target[o];

        if(error > 0) {
            partial_sort(result0[o].begin(), result0[o].begin() + error, result0[o].end(), cmp);
            for(int e = 0; e < error; e++){
                int *pni = &nn_input[0];
                int *pw = &weight[o][result0[o][e].index][0];
                for(int j = 0; j < nn_input_size; j++){
                    pw += *pni++;
                    *pw++ -= *pni++;
                }
            }
        }
        else if(error < 0) {
            error = -error;
            partial_sort(result1[o].begin(), result1[o].begin() + error, result1[o].end(), cmp);
            for(int e = 0; e < error; e++){
                int *pni = &nn_input[0];
                int *pw = &weight[o][result1[o][e].index][0];
                for(int j = 0; j < nn_input_size; j++){
                    pw += *pni++;
                    *pw++ += *pni++;
                }
            }
        }
    }

    return 0;
}

vector<float> SDNNFABiOpenMP::Predict(vector<int> input) {
    vector<int> nn_input;
    vector<vector<int> > nn_output(input.size(), vector<int>(original_pattern[0].size()));
    //int output_size = input.size();
    //int output0_size = original_pattern[0].size();

    nn_input = SDNNFABiOpenMP::SD(input);
    return NNPredict(nn_input);
}

vector<float> SDNNFABiOpenMP::NNPredict(vector<int> nn_input) {
    int original_pattern_size = original_pattern.size();
    int original_pattern0_size = original_pattern[0].size();
    int nn_input_size = nn_input.size()/2;
    int element = weight[0].size();
    //int weight_size = weight[0].size(); //nn_input0_size

    vector<int> nn_output(original_pattern0_size);
    vector<float> result(original_pattern_size + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int o = 0; o < original_pattern0_size; o++) {
        int neuron_output_sum = -10;
        for (int e = 0; e < element; e++) {
            int potential = 0;
            int *pni = &nn_input[0];
            int *pw = &weight[o][e][0];
            for (int j = 0; j < nn_input_size; j++) {
                pw += *pni++;
                potential += *pni++ * *pw++;
            }
            neuron_output_sum += (potential >= 0);
        }
        if (neuron_output_sum < -8) neuron_output_sum = -8;
        else if (neuron_output_sum > 8) neuron_output_sum = 8;

        nn_output[o] = neuron_output_sum;
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < original_pattern_size; i++) {
        int *po = &nn_output[0];
        int *pp = &original_pattern[i][0];
        int sum = 0;
        for (int j = 0; j < original_pattern0_size; j++) {
            result[i + 1] += abs(*po++ - *pp++);
        }
    }
    /*
    for(int i = 0; i < original_pattern_size; i++){
        int *po = &nn_output[0];
        int *pp = &original_pattern[i][0];
        int sum = 0;
        int nn_output_norm = 0;
        int original_pattern_norm = 0;
        for(int j = 0; j < original_pattern0_size; j++){
            nn_output_norm += *po * *po;
            original_pattern_norm += *pp * *pp;
            sum += *po++ * *pp++;
        }
        result[i + 1] = (float)sum/(sqrtf((float)nn_output_norm) * sqrtf((float)original_pattern_norm));
    }
    */
    return result;
}

vector<vector<vector<int> > > SDNNFABiOpenMP::GetWeight() { return weight; }

bool SDNNFABiOpenMP::cmp(struct NEURON_OUTPUT& x, struct NEURON_OUTPUT& y) {
    return x.potential < y.potential;
}
