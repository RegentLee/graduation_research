//
// Created by 李明曄 on 2019/11/17.
//

#include "Trainer.h"
#include "csv.h"

using namespace std;

void Trainer::fit(SDNN &model, vector<vector<int> > sample, string fp, int max_epoch, int batch_size) {

    int max_iters = sample.size()/batch_size;
    vector<vector<int> > data(sample.size(), vector<int>(sample[0].size()));

    clock_t start, iters_start;
    start = clock();

    vector<int> range(sample.size());
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    for(int i = 0; i < sample.size(); i++) range[i] = i;

    for(int epoch = 0; epoch < max_epoch; epoch++){
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

            float err = model.Backward(output, target);

            //usleep(1000000);
            cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch
                 << " iters: " << iters + 1 << "/" << max_iters //<< flush;// << string(20, ' ') ;
                 << " time: " << (double)(clock() - iters_start)/CLOCKS_PER_SEC << "/" << (double)(clock() - start)/CLOCKS_PER_SEC
                 << " loss: " << err << flush;
        }

        if(batch_size*max_iters != sample.size()){
            iters_start = clock();

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

            cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch
                 << " iters: extra/" << max_iters
                 << " time: " << (double)(clock() - iters_start)/CLOCKS_PER_SEC << "/" << (double)(clock() - start)/CLOCKS_PER_SEC << flush;
        }


        //usleep(1000000);
        //cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch << flush;// << string(20, ' ') ;
    }
    cout << endl;
}

void Trainer::fit(SDNNOpenMP &model, vector<vector<int> > sample, string fp, int max_epoch, int batch_size) {

    int max_iters = sample.size();
    vector<vector<int> > data(sample.size(), vector<int>(sample[0].size()));

    clock_t start, iters_start;
    //start = clock();

    vector<int> range(sample.size());
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    //mt19937_64 mt(0);
    for(int i = 0; i < sample.size(); i++) range[i] = i;

    for(int epoch = 0; epoch < max_epoch; epoch++){
        //サンプルの順番を変える
        shuffle(range.begin(), range.end(), mt);
        for(int i = 0; i < sample.size(); i++){
            data[i].assign(sample[range[i]].begin(), sample[range[i]].end());
        }
        start = clock();

        //train
        for(int iters = 0; iters < max_iters; iters++){
            iters_start = clock();
            vector<int> input(sample[0].size() - 1);
            int target;

            input.assign(data[iters].begin(), data[iters].end() - 1);
            target = data[iters].back();

            model.Train(input, target);

            float err = 0;//model.Backward(output, target);

            if((iters + 1) % 100 == 0){
                cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch
                     << " iters: " << iters + 1 << "/" << max_iters //<< flush;// << string(20, ' ') ;
                     << " time: " << (double)(clock() - iters_start)/CLOCKS_PER_SEC << "/" << (double)(clock() - start)/CLOCKS_PER_SEC
                     << " loss: " << err << string(10, ' ') << flush;
            }

        }
    }
    cout << endl;

    vector<vector<int> > weight = model.GetWeight();
    csv::ToCsv(weight, fp);
}

void Trainer::fit(SDNNBiOpenMP &model, vector<vector<int> > sample, string fp, int max_epoch, int batch_size) {

    int max_iters = sample.size();
    vector<vector<int> > data(sample.size(), vector<int>(sample[0].size()));

    clock_t start, iters_start;
    //start = clock();

    vector<int> range(sample.size());
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    //mt19937_64 mt(0);
    for(int i = 0; i < sample.size(); i++) range[i] = i;

    for(int epoch = 0; epoch < max_epoch; epoch++){
        //サンプルの順番を変える
        shuffle(range.begin(), range.end(), mt);
        for(int i = 0; i < sample.size(); i++){
            data[i].assign(sample[range[i]].begin(), sample[range[i]].end());
        }
        start = clock();

        //train
        for(int iters = 0; iters < max_iters; iters++){
            iters_start = clock();
            vector<int> input(sample[0].size() - 1);
            int target;

            input.assign(data[iters].begin(), data[iters].end() - 1);
            target = data[iters].back();

            model.Train(input, target);

            float err = 0;//model.Backward(output, target);

            if((iters + 1) % 100 == 0){
                cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch
                     << " iters: " << iters + 1 << "/" << max_iters //<< flush;// << string(20, ' ') ;
                     << " time: " << (double)(clock() - iters_start)/CLOCKS_PER_SEC << "/" << (double)(clock() - start)/CLOCKS_PER_SEC
                     << " loss: " << err << string(10, ' ') << flush;
            }
        }
    }
    cout << endl;

    vector<vector<int> > weight = model.GetWeight();
    csv::ToCsv(weight, fp);
}

void Trainer::fit(SDNNFABiOpenMP &model, vector<vector<int> > sample, string fp, int max_epoch, int batch_size) {

    int max_iters = sample.size();
    vector<vector<int> > data(sample.size(), vector<int>(sample[0].size()));

    clock_t start, iters_start;
    //start = clock();

    vector<int> range(sample.size());
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    //mt19937_64 mt(0);
    for(int i = 0; i < sample.size(); i++) range[i] = i;

    for(int epoch = 0; epoch < max_epoch; epoch++){
        //サンプルの順番を変える
        shuffle(range.begin(), range.end(), mt);
        for(int i = 0; i < sample.size(); i++){
            data[i].assign(sample[range[i]].begin(), sample[range[i]].end());
        }
        start = clock();

        //train
        for(int iters = 0; iters < max_iters; iters++){
            iters_start = clock();
            vector<int> input(sample[0].size() - 1);
            int target;

            input.assign(data[iters].begin(), data[iters].end() - 1);
            target = data[iters].back();

            model.Train(input, target);

            float err = 0;//model.Backward(output, target);

            cout << "\r" << "epoch: " << epoch + 1 << "/" << max_epoch
                 << " iters: " << iters + 1 << "/" << max_iters //<< flush;// << string(20, ' ') ;
                 << " time: " << (double)(clock() - iters_start)/CLOCKS_PER_SEC << "/" << (double)(clock() - start)/CLOCKS_PER_SEC
                 << " loss: " << err << string(10, ' ') << flush;
        }
    }
    cout << endl;

    //vector<vector<int> > weight = model.GetWeight();
    //csv::ToCsv(weight, fp);
}