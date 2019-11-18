//
// Created by 李明曄 on 2019/11/17.
//

#include "Trainer.h"

using namespace std;

void Trainer::fit(SDNN &model, vector<vector<int> > sample, int max_epoch, int batch_size) {

    int max_iters = sample.size()/batch_size;
    vector<vector<int> > data(sample.size(), vector<int>(sample[0].size()));

    clock_t start, epoch_start, iters_start;
    start = clock();

    vector<int> range(sample.size());
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    for(int i = 0; i < sample.size(); i++) range[i] = i;

    for(int epoch = 0; epoch < max_epoch; epoch++){
        epoch_start = clock();
        //サンプルの順番を変える
        shuffle(range.begin(), range.end(), mt);
        for(int i = 0; i < sample.size(); i++){
            data[i].assign(sample[range[i]].begin(), sample[range[i]].end());
        }

        //train
        for(int iters = 0; iters < max_iters; iters++){
            iters_start = clock();
            vector<vector<int> > input(batch_size, vector<int>(sample[0].size() - 1));
            vector<int> target(batch_size);
            vector<vector<int> > output;

            for(int i = 0; i < batch_size; i++){
                input[i].assign(data[iters*batch_size + i].begin(), data[iters*batch_size + i].end() - 1);
                target[i] = data[iters*batch_size + i].back();
                //printf("input:%d, %d, epoch:%d, target:%d\n", input[i][0], input[i][1], epoch, target[i]);
            }

            /*cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch
                 << " iters: " << iters + 1 << "/" << max_iters //<< flush;// << string(20, ' ') ;
                 << " time: " << (double)(clock() - start)/CLOCKS_PER_SEC << flush;*/

            output = model.Forward(input);

            /*cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch
                 << " iters: " << iters + 1 << "/" << max_iters //<< flush;// << string(20, ' ') ;
                 << " time: " << (double)(clock() - start)/CLOCKS_PER_SEC << flush;*/

            model.Backward(output, target);

            //usleep(1000000);
            cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch
                 << " iters: " << iters + 1 << "/" << max_iters //<< flush;// << string(20, ' ') ;
                 << " time: " << (double)(clock() - iters_start)/CLOCKS_PER_SEC << "/" << (double)(clock() - start)/CLOCKS_PER_SEC << flush;
        }

        if(batch_size*max_iters != sample.size()){
            cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch
                 << " iters: extra/" << max_iters //<< flush;// << string(20, ' ') ;
                 << " time: " << (double)(clock() - start)/CLOCKS_PER_SEC << flush;

            vector<vector<int> > input(sample.size() - max_iters*batch_size, vector<int>(sample[0].size() - 1));
            vector<int> target(sample.size() - max_iters*batch_size);
            vector<vector<int> > output;

            for(int i = 0; i < input.size(); i++){
                input[i].assign(data[max_iters*batch_size + i].begin(), data[max_iters*batch_size + i].end() - 1);
                target[i] = data[max_iters*batch_size + i].back();
                //printf("except   input:%d, %d, epoch:%d, target:%d\n", input[i][0], input[i][1], epoch, target[i]);
            }

            output = model.Forward(input);
            model.Backward(output, target);
        }


        //usleep(1000000);
        //cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch << flush;// << string(20, ' ') ;
    }
    cout << endl;
}
