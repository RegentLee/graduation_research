//
// Created by 李明曄 on 2019/12/07.
//

#include "func.h"
#include "csv.h"

using namespace std;

vector<vector<int> > CosPattern(int len){
    vector<vector<int> > cos_sort;
    vector<vector<int> > cos_arg;

    csv::ReadCsv(cos_sort, "cos_sort.csv");
    csv::ReadCsv(cos_arg, "cos_arg.csv");

    int num = cos_sort.size();
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));

    vector<vector<int> > pattern(num + 1, vector<int>(len, 0));
    vector<vector<int> > pattern_bp(num + 1, vector<int>(len, 0));
    vector<vector<int> > pattern_bp2(num + 1, vector<int>(len, 0));

    while(1) {
        bool flag = true;
        int sum_bp = 0x7FFFFFFF;
        int sum_bp2 = 0x7FFFFFFF;

        //random pattern
        for(int i = 0; i < num; i++){
            for(int j = 0; j < len/2; j++) pattern[i][j] = 1;
            for(int j = len/2; j < len; j++) pattern[i][j] = -1;
            while(1){
                bool flag = true;
                shuffle(pattern[i].begin(), pattern[i].end(), mt);
                for(int k = 0; k < i; k++){
                    if(pattern[i] == pattern[k]){
                        flag = false;
                        break;
                    }
                }
                if(flag) break;
            }
        }

        //make pattern
        int idx = 0;
        while (1) {
            cout << "idx: "<< ++idx << endl;
            for (int i = 0; i < num; i++) {
                //cout << cos_sort[i][num - 2] << endl;
                for (int j = num - 2; cos_sort[i][j] > 900; j--) {
                    int c = (dot(pattern[i], pattern[cos_arg[i][j]]) - cos_sort[i][j]) / 40;
                    //cout << "c: " << c << endl;
                    vector<int> temp;
                    if (c > 0) {
                        vector<int> d = equal(pattern[i], pattern[cos_arg[i][j]], true);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        for (int k = 0; k < c; k++) {
                            while (1) {
                                //cout << "k: "<< k << endl;
                                int a = dist(mt);
                                int b = dist(mt);
                                if (pattern[i][d[a]] * pattern[i][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    pattern[i][d[a]] *= -1;
                                    pattern[i][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    } else if (c < 0) {
                        vector<int> d = equal(pattern[i], pattern[cos_arg[i][j]], false);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        for (int k = 0; k < -c; k++) {
                            while (1) {
                                int a = dist(mt);
                                int b = dist(mt);
                                //cout << a << " " << b << " " << pattern[cos_arg[i][j]][d[a]] << " " << pattern[cos_arg[i][j]][d[b]] << endl;
                                if (pattern[i][d[a]] * pattern[i][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    //cout << "if"<< k << endl;
                                    pattern[i][d[a]] *= -1;
                                    pattern[i][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    }
                }
                //cout << "negative" << endl;
                //cout << cos_sort[i][0] << endl;
                for (int j = 0; cos_sort[i][j] < 100 || j < 10; j++) {
                    int c = (dot(pattern[i], pattern[cos_arg[i][j]]) - cos_sort[i][j]) / 40;
                    vector<int> temp;
                    if (c > 0) {
                        vector<int> d = equal(pattern[i], pattern[cos_arg[i][j]], true);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        for (int k = 0; k < c; k++) {
                            while (1) {
                                int a = dist(mt);
                                int b = dist(mt);
                                if (pattern[i][d[a]] * pattern[i][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    pattern[i][d[a]] *= -1;
                                    pattern[i][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    } else if (c < 0) {
                        vector<int> d = equal(pattern[i], pattern[cos_arg[i][j]], false);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        for (int k = 0; k < -c; k++) {
                            while (1) {
                                int a = dist(mt);
                                int b = dist(mt);
                                if (pattern[i][d[a]] * pattern[i][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    pattern[i][d[a]] *= -1;
                                    pattern[i][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            int sum = 0;
            for (int i = 0; i < num; i++) {
                //for (int j = num - 2; cos_sort[i][j] > 800; j--) {
                for (int j = 0; j < num - 1; j++) {
                    sum += abs(dot(pattern[i], pattern[cos_arg[i][j]]) - cos_sort[i][j]);
                }
            }
            cout << "sum: "<< sum << endl;
            if (sum > sum_bp && sum_bp > sum_bp2) break;
            sum_bp2 = sum_bp;
            sum_bp = sum;
            pattern_bp2.assign(pattern_bp.begin(), pattern_bp.end());
            pattern_bp.assign(pattern.begin(), pattern.end());
        }
        cout << "check" << endl;
        for (int i = 0; i < num; i++) {
            for (int j = 0; j < num; j++) {
                if (i == j) continue;
                if (pattern_bp2[i] == pattern_bp2[j]) {
                    flag = false;
                    break;
                }
            }
            if (!flag) break;
        }
        if(flag) break;
    }

    csv::ToCsv(pattern_bp2, "cos_pattern.csv");

    return pattern_bp2;
}

vector<vector<int> > CosPatternKai(int len){
    vector<vector<int> > cos_sort;
    vector<vector<int> > cos_arg;

    csv::ReadCsv(cos_sort, "cos_sort.csv");
    csv::ReadCsv(cos_arg, "cos_arg.csv");

    int num = cos_sort.size();
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));

    vector<vector<int> > pattern(num + 1, vector<int>(len, 0));
    vector<vector<int> > pattern_bp(num + 1, vector<int>(len, 0));
    vector<vector<int> > pattern_bp2(num + 1, vector<int>(len, 0));

    while(1) {
        bool flag = true;
        int sum_bp = 0x7FFFFFFF;
        int sum_bp2 = 0x7FFFFFFF;

        //random pattern
        for(int i = 0; i < num; i++){
            for(int j = 0; j < len/2; j++) pattern[i][j] = 1;
            for(int j = len/2; j < len; j++) pattern[i][j] = -1;
            while(1){
                bool flag_rp = true;
                shuffle(pattern[i].begin(), pattern[i].end(), mt);
                for(int k = 0; k < i; k++){
                    if(pattern[i] == pattern[k]){
                        flag_rp = false;
                        break;
                    }
                }
                if(flag_rp) break;
            }
        }

        //make pattern
        int idx = 0;
        while (1) {
            cout << "idx: "<< ++idx << endl;
            for (int i = 0; i < num; i++) {
                //cout << cos_sort[i][num - 2] << endl;
                for (int j = num - 2; cos_sort[i][j] > 900; j--) {
                    int c = ((dot(pattern[i], pattern[cos_arg[i][j]]) - cos_sort[i][j]) + (dot(pattern[i], pattern[cos_arg[i][j - 1]]) - cos_sort[i][j - 1]))/ 80;
                    //cout << "c: " << c << endl;
                    vector<int> temp;
                    if (c > 0) {
                        vector<int> d = equal(pattern[cos_arg[i][j - 1]], pattern[cos_arg[i][j]], pattern[i], true);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        //cout << c << ", " << d.size() << endl;
                        for (int k = 0; k < c; k++) {
                            //while (1) {
                            for(int m = 0; m < 100; m++) {
                                //cout << "k: "<< k << endl;
                                int a = dist(mt);
                                int b = dist(mt);
                                if (pattern[i][d[a]] * pattern[i][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    pattern[i][d[a]] *= -1;
                                    pattern[i][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    } else if (c < 0) {
                        vector<int> d = equal(pattern[cos_arg[i][j - 1]], pattern[cos_arg[i][j]], pattern[i], false);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        //cout << c << ", " << d.size() << endl;
                        for (int k = 0; k < -c; k++) {
                            //while (1) {
                            for(int m = 0; m < 100; m++) {
                                int a = dist(mt);
                                int b = dist(mt);
                                //cout << a << " " << b << " " << pattern[cos_arg[i][j]][d[a]] << " " << pattern[cos_arg[i][j]][d[b]] << endl;
                                if (pattern[i][d[a]] * pattern[i][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    //cout << "if"<< k << endl;
                                    pattern[i][d[a]] *= -1;
                                    pattern[i][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    }
                }
                //cout << "negative" << endl;
                //cout << cos_sort[i][0] << endl;
                /*
                for (int j = 0; cos_sort[i][j] < 100 || j < 10; j++) {
                    int c = (dot(pattern[i], pattern[cos_arg[i][j]]) - cos_sort[i][j]) / 40;
                    vector<int> temp;
                    if (c > 0) {
                        vector<int> d = equal(pattern[i], pattern[cos_arg[i][j]], true);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        for (int k = 0; k < c; k++) {
                            while (1) {
                                int a = dist(mt);
                                int b = dist(mt);
                                if (pattern[i][d[a]] * pattern[i][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    pattern[i][d[a]] *= -1;
                                    pattern[i][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    } else if (c < 0) {
                        vector<int> d = equal(pattern[i], pattern[cos_arg[i][j]], false);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        for (int k = 0; k < -c; k++) {
                            while (1) {
                                int a = dist(mt);
                                int b = dist(mt);
                                if (pattern[i][d[a]] * pattern[i][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    pattern[i][d[a]] *= -1;
                                    pattern[i][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    }
                }*/
            }
            int sum = 0;
            for (int i = 0; i < num; i++) {
                //for (int j = num - 2; cos_sort[i][j] > 800; j--) {
                for (int j = 0; j < num - 1; j++) {
                    sum += abs(dot(pattern[i], pattern[cos_arg[i][j]]) - cos_sort[i][j]);
                }
            }
            cout << "sum: "<< sum << endl;
            if (sum > sum_bp && sum_bp > sum_bp2) break;
            sum_bp2 = sum_bp;
            sum_bp = sum;
            pattern_bp2.assign(pattern_bp.begin(), pattern_bp.end());
            pattern_bp.assign(pattern.begin(), pattern.end());
        }
        cout << "check" << endl;
        for (int i = 0; i < num; i++) {
            for (int j = 0; j < num; j++) {
                if (i == j) continue;
                if (pattern_bp2[i] == pattern_bp2[j]) {
                    flag = false;
                    break;
                }
            }
            if (!flag) break;
        }
        if(flag) break;
    }

    csv::ToCsv(pattern_bp2, "cos_pattern.csv");

    return pattern_bp2;
}

vector<vector<int> > CosPatternKai2(int len){
    vector<vector<int> > cos_sort;
    vector<vector<int> > cos_arg;

    csv::ReadCsv(cos_sort, "cos_sort.csv");
    csv::ReadCsv(cos_arg, "cos_arg.csv");

    int num = cos_sort.size();
    mt19937_64 mt(static_cast<unsigned int>(time(nullptr)));

    vector<vector<int> > pattern(num + 1, vector<int>(len, 0));
    vector<vector<int> > pattern_bp(num + 1, vector<int>(len, 0));
    vector<vector<int> > pattern_bp2(num + 1, vector<int>(len, 0));

    while(1) {
        bool flag = true;
        int sum_bp = 0x7FFFFFFF;
        int sum_bp2 = 0x7FFFFFFF;

        //random pattern
        for(int i = 0; i < num; i++){
            for(int j = 0; j < len/2; j++) pattern[i][j] = 1;
            for(int j = len/2; j < len; j++) pattern[i][j] = -1;
            while(1){
                bool flag = true;
                shuffle(pattern[i].begin(), pattern[i].end(), mt);
                for(int k = 0; k < i; k++){
                    if(pattern[i] == pattern[k]){
                        flag = false;
                        break;
                    }
                }
                if(flag) break;
            }
        }

        //make pattern
        int idx = 0;
        while (1) {
            cout << "idx: "<< ++idx << endl;
            for (int i = 0; i < num; i++) {
                //cout << cos_sort[i][num - 2] << endl;
                for (int j = num - 2; j >= 0; j--) {
                    int c = (dot(pattern[i], pattern[cos_arg[i][j]]) - cos_sort[i][j]) / 4;
                    //cout << "c: " << c << endl;
                    vector<int> temp;
                    if (c > 0) {
                        vector<int> d = equal(pattern[i], pattern[cos_arg[i][j]], true);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        for (int k = 0; k < c; k++) {
                            while (1) {
                                //cout << "k: "<< k << endl;
                                int a = dist(mt);
                                int b = dist(mt);
                                if (pattern[cos_arg[i][j]][d[a]] * pattern[cos_arg[i][j]][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    pattern[cos_arg[i][j]][d[a]] *= -1;
                                    pattern[cos_arg[i][j]][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    } else if (c < 0) {
                        vector<int> d = equal(pattern[i], pattern[cos_arg[i][j]], false);
                        uniform_int_distribution<> dist(0, d.size() - 1);
                        for (int k = 0; k < -c; k++) {
                            while (1) {
                                int a = dist(mt);
                                int b = dist(mt);
                                //cout << a << " " << b << " " << pattern[cos_arg[i][j]][d[a]] << " " << pattern[cos_arg[i][j]][d[b]] << endl;
                                if (pattern[cos_arg[i][j]][d[a]] * pattern[cos_arg[i][j]][d[b]] == -1 &&
                                    find(temp.begin(), temp.end(), a) == temp.end() &&
                                    find(temp.begin(), temp.end(), b) == temp.end()) {
                                    //cout << "if"<< k << endl;
                                    pattern[cos_arg[i][j]][d[a]] *= -1;
                                    pattern[cos_arg[i][j]][d[b]] *= -1;
                                    temp.push_back(a);
                                    temp.push_back(b);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            int sum = 0;
            for (int i = 0; i < num; i++) {
                //for (int j = num - 2; cos_sort[i][j] > 800; j--) {
                for (int j = 0; j < num - 1; j++) {
                    sum += abs(dot(pattern[i], pattern[cos_arg[i][j]]) - cos_sort[i][j]);
                }
            }
            cout << "sum: "<< sum << endl;
            if (sum > sum_bp && sum_bp > sum_bp2) break;
            sum_bp2 = sum_bp;
            sum_bp = sum;
            pattern_bp2.assign(pattern_bp.begin(), pattern_bp.end());
            pattern_bp.assign(pattern.begin(), pattern.end());
        }
        cout << "check" << endl;
        for (int i = 0; i < num; i++) {
            for (int j = 0; j < num; j++) {
                if (i == j) continue;
                if (pattern_bp2[i] == pattern_bp2[j]) {
                    flag = false;
                    break;
                }
            }
            if (!flag) break;
        }
        if(flag) break;
    }

    csv::ToCsv(pattern_bp2, "cos_pattern.csv");

    return pattern_bp2;
}


int dot(vector<int> a, vector<int> b){
    int sum = 0;
    int a_size = a.size();
    int *pa = &a[0];
    int *pb = &b[0];
    for(int i = 0; i < a_size; i++){
        sum += *pa++ * *pb++;
    }
    return sum;
}

vector<int> equal(vector<int> a, vector<int> b, bool c){
    int a_size = a.size();
    vector<int> where;
    if(c){
        for(int i = 0; i < a_size; i++){
            if(a[i] == b[i]) where.push_back(i);
        }
    } else {
        for(int i = 0; i < a_size; i++){
            if(a[i] != b[i]) where.push_back(i);
        }
    }
    return where;
}

vector<int> equal(vector<int> a, vector<int> b, vector<int> c, bool d){
    int a_size = a.size();
    vector<int> where;
    if(d){
        for(int i = 0; i < a_size; i++){
            if(a[i] == b[i] && a[i] == c[i]) where.push_back(i);
        }
    } else {
        for(int i = 0; i < a_size; i++){
            if(a[i] == b[i] && a[i] != c[i]) where.push_back(i);
        }
    }
    return where;
}

struct param_list ReadParam(){
    struct param_list param;
    vector<vector<string> > p;

    ifstream ifs("param.txt");

    for(int i = 0; i < 7; i++){
        string temp;
        ifs >> temp;
        if(temp == "<train_sample>"){
            ifs >> param.train_sample_file;
        } else if(temp == "<test_sample>") {
            ifs >> param.test_sample_file;
        } else if(temp == "<pattern>") {
            ifs >> param.pattern_file;
        } else if(temp == "<thread>") {
            ifs >> temp;
            param.thread = stoi(temp);
        } else if(temp == "<train_result>") {
            ifs >> param.train_result_file;
        } else if(temp == "<test_result>") {
            ifs >> param.test_result_file;
        } else if(temp == "<save_weight>") {
            ifs >> param.save_weight_file;
        } else if(temp == "<read_weight>") {
            ifs >> param.read_weight_file;
        }
    }

    ifs.close();

    return param;
}