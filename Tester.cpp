//
// Created by 李明曄 on 2019/11/17.
//

#include "Tester.h"
#include "csv.h"

using namespace std;

void Tester::predict(SDNN &model, std::vector<std::vector<int> > sample, int batch_size) {
    int max_iters = sample.size()/batch_size;
    vector<vector<int> > pattern = model.GetPattern();
    vector<vector<int> > answer(sample.size(), vector<int>(pattern.size() + 1, 0));

    //train
    for(int iters = 0; iters < max_iters; iters++){
        vector<vector<int> > input(batch_size, vector<int>(sample[0].size() - 1));
        vector<int> target(batch_size);
        vector<vector<int> > output;

        for(int i = 0; i < batch_size; i++){
            input[i].assign(sample[iters*batch_size + i].begin(), sample[iters*batch_size + i].end() - 1);
            target[i] = sample[iters*batch_size + i].back();
            //printf("input:%d, %d, epoch:%d, target:%d\n", input[i][0], input[i][1], epoch, target[i]);
        }

        output = model.Forward(input);

        for(int i = 0; i < output.size(); i++){
            answer[iters*batch_size + i][0] = target[i];
            for(int j = 0; j < pattern.size(); j++){
                for(int k = 0; k < pattern[0].size(); k++){
                    answer[iters*batch_size + i][j + 1] += output[i][k]*pattern[j][k];
                }
            }
        }

        //usleep(1000000);
        cout << "\r" << "iters: " << iters + 1 << "/" << max_iters << flush;// << string(20, ' ') ;
    }

    if(batch_size*max_iters != sample.size()){
        vector<vector<int> > input(sample.size() - max_iters*batch_size, vector<int>(sample[0].size() - 1));
        vector<int> target(sample.size() - max_iters*batch_size);
        vector<vector<int> > output;

        for(int i = 0; i < input.size(); i++){
            input[i].assign(sample[max_iters*batch_size + i].begin(), sample[max_iters*batch_size + i].end() - 1);
            target[i] = sample[max_iters*batch_size + i].back();
            //printf("except   input:%d, %d, epoch:%d, target:%d\n", input[i][0], input[i][1], epoch, target[i]);
        }

        output = model.Forward(input);

        for(int i = 0; i < output.size(); i++){
            answer[max_iters*batch_size + i][0] = target[i];
            for(int j = 0; j < pattern.size(); j++){
                for(int k = 0; k < pattern[0].size(); k++){
                    answer[max_iters*batch_size + i][j + 1] += output[i][k]*pattern[j][k];
                }
            }
        }
    }

    cout << endl;

    csv::ToCsv(answer, "result.csv");
}
