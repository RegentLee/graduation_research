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

    clock_t start, iters_start;
    start = clock();

    //test
    for(int iters = 0; iters < max_iters; iters++){
        iters_start = clock();

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
        cout << "\r" << "sample: " << (iters + 1)*batch_size << "/" << sample.size()
             << " time: " << (double)(clock() - iters_start)/CLOCKS_PER_SEC << "/"
             << (double)(clock() - start)/CLOCKS_PER_SEC << string(20, ' ') << flush;
    }

    if(batch_size*max_iters != sample.size()){
        iters_start = clock();

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

        cout << "\r" << "sample: " << sample.size() << "/" << sample.size()
             << " time: " << (double)(clock() - iters_start)/CLOCKS_PER_SEC << "/"
             << (double)(clock() - start)/CLOCKS_PER_SEC << string(20, ' ') << flush;
    }

    cout << endl;

    csv::ToCsv(answer, "result.csv");
}

void Tester::predict(SDNNOpenMP &model, std::vector<std::vector<int> > sample, int batch_size) {
    int max_iters = sample.size();
    vector<vector<int> > answer(sample.size(), vector<int>());

    clock_t start, iters_start;
    start = clock();

    //test
    for(int iters = 0; iters < max_iters; iters++){
        iters_start = clock();

        vector<int> one_answer;
        vector<int> input(sample[0].size() - 1);
        int target;

        input.assign(sample[iters].begin(), sample[iters].end() - 1);
        target = sample[iters].back();

        one_answer = model.Predict(input);

        one_answer[0] = target;
        answer[iters].insert(answer[iters].end(), one_answer.begin(), one_answer.end());
        //answer[iters].assign(one_answer.begin(), one_answer.end());

        cout << "\r" << "sample: " << iters + 1 << "/" << sample.size()
             << " time: " << (double)(clock() - iters_start)/CLOCKS_PER_SEC << "/"
             << (double)(clock() - start)/CLOCKS_PER_SEC << string(20, ' ') << flush;
    }

    cout << endl;

    csv::ToCsv(answer, "result.csv");
}
