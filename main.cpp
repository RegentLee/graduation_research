#include <iostream>
#include <string>
#include "csv.h"
#include "SDNN.h"
#include "SDNNOpenMP.h"
#include "SDNNBiOpenMP.h"
//#include "SDNNFABiOpenMP.h"
#include "SDNNWithUpdateOpenMP.h"
#include "Trainer.h"
#include "Tester.h"
#include "func.h"

using namespace std;

int main() {
    //CosPatternKai2(1000);
/*
    vector<vector<int> > pattern(2, vector<int>(16));
    vector<vector<int> > learning_sample(4, vector<int>(3));
    vector<vector<int> > output;
    //vector<int> target(5);

    pattern[0][0] = 1;
    pattern[0][1] = 1;
    pattern[0][2] = -1;
    pattern[0][3] = -1;
    pattern[0][4] = 1;
    pattern[0][5] = 1;
    pattern[0][6] = -1;
    pattern[0][7] = -1;
    pattern[1][0] = 1;
    pattern[1][1] = -1;
    pattern[1][2] = 1;
    pattern[1][3] = -1;
    pattern[1][4] = 1;
    pattern[1][5] = -1;
    pattern[1][6] = 1;
    pattern[1][7] = -1;
    for(int i = 8; i < 16; i++){
        pattern[0][i] = pattern[1][i - 8];
        pattern[1][i] = pattern[0][i - 8];
    }

    learning_sample[0][0] = 0;
    learning_sample[0][1] = 1;
    learning_sample[0][2] = 1;

    learning_sample[1][0] = 1;
    learning_sample[1][1] = 0;
    learning_sample[1][2] = 1;

    learning_sample[2][0] = 0;
    learning_sample[2][1] = 0;
    learning_sample[2][2] = 0;

    learning_sample[3][0] = 1;
    learning_sample[3][1] = 1;
    learning_sample[3][2] = 1;

    //vector<vector<int> > weight;
    //csv::ReadCsv(weight, "weight.csv");

    SDNNWithUpdateOpenMP model(2, pattern);
    Trainer trainer;
    trainer.fit(model, learning_sample, "weight.csv", 5);
    //output = model.Forward(learning_sample);
    //model.Backward(output, target);

    Tester tester;
    tester.predict(model, learning_sample, "learning_result.csv");
*/

    vector<vector<int> > pattern;
    vector<vector<int> > learning_sample;

    param_list param;
    param = ReadParam();

    csv::ReadCsv(pattern, param.pattern_file);
    csv::ReadCsv(learning_sample, param.train_sample_file);

    vector<vector<int> > weight;
    if(param.read_weight_file != ""){
        csv::ReadCsv(weight, param.read_weight_file);
    }

    cout << "read end" << endl;

    SDNNOpenMP model(learning_sample[0].size() - 1, pattern, param.thread);
    pattern.clear();

    Trainer trainer;
    trainer.fit(model, learning_sample, param.save_weight_file, 1, 1);

    Tester tester;
    tester.predict(model, learning_sample, param.train_result_file);

    learning_sample.clear();

    vector<vector<int> > test_sample;
	csv::ReadCsv(test_sample, param.test_sample_file);

	Tester tester2;
	tester2.predict(model, test_sample, param.test_result_file);

	test_sample.clear();

/*
    vector<vector<int> > pattern;
    vector<vector<int> > learning_sample;
    vector<vector<int> > test_sample;

    param_list param;
    param = ReadParam();

    csv::ReadCsv(pattern, param.pattern_file);
    csv::ReadCsv(learning_sample, param.train_sample_file);
    csv::ReadCsv(test_sample, param.test_sample_file);

    vector<vector<int> > weight;
    if(param.read_weight_file != ""){
        csv::ReadCsv(weight, param.read_weight_file);
    }

    cout << "read end" << endl;

    vector<string> train_result;
    train_result.push_back("wikitext-2_num_train_result_3.csv");
    train_result.push_back("wikitext-2_num_train_result_6.csv");
    train_result.push_back("wikitext-2_num_train_result_9.csv");
    train_result.push_back("wikitext-2_num_train_result_12.csv");

    vector<string> test_result;
    test_result.push_back("wikitext-2_num_test_result_3.csv");
    test_result.push_back("wikitext-2_num_test_result_6.csv");
    test_result.push_back("wikitext-2_num_test_result_9.csv");
    test_result.push_back("wikitext-2_num_test_result_12.csv");

    vector<string> w;
    w.push_back("wikitext-2_num_weight_3.csv");
    w.push_back("wikitext-2_num_weight_6.csv");
    w.push_back("wikitext-2_num_weight_9.csv");
    w.push_back("wikitext-2_num_weight_12.csv");

    SDNNBiOpenMP model(learning_sample[0].size() - 1, pattern, thread, weight);
    pattern.clear();

    Trainer trainer;
    Tester tester1;
    Tester tester2;

    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));
    shuffle(learning_sample.begin(), learning_sample.end(), mt);

    for(int i = 0; i < 4; i++){
        vector<vector<int> > ls;
        if(i != 3){
            ls.insert(ls.begin(), learning_sample.begin() + i*300000, learning_sample.begin() + (i + 1)*300000);
        } else {
            ls.insert(ls.begin(), learning_sample.begin() + i*300000, learning_sample.end());
        }

        trainer.fit(model, ls, w[i], 1, 1);
        vector<vector<int> > tls;
        if(i != 3){
            tls.insert(tls.begin(), learning_sample.begin(), learning_sample.begin() + (i + 1)*300000);
        } else {
            tls.insert(tls.begin(), learning_sample.begin(), learning_sample.end());
        }
        tester1.predict(model, tls, train_result[i]);
        tester2.predict(model, test_sample, test_result[i]);
    }

    learning_sample.clear();
    test_sample.clear();
*/
    cout << "\npress to end";
    cin.get();

    return 0;
}
