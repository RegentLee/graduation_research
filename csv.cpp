//
// Created by 李明曄 on 2019/11/16.
//

#include "csv.h"

using namespace std;

// int 値の csv ファイルを読み込む
void csv::ReadCsv(vector<vector<int> >& map, string fp) {
    ifstream ifs(fp);

    string one_line;
    while(getline(ifs, one_line)){
        istringstream line_buf(one_line);
        string one_buf;
        vector<int> line;
        while(getline(line_buf, one_buf, ',')){
            int num = stoi(one_buf);
            line.push_back(num);
        }
        map.push_back(line);
    }

    ifs.close();
}

// int 値の csv ファイルを書き込む
void csv::ToCsv(vector<vector<int> >& map, string fp) {
    ofstream ofs;
    ofs.open(fp, ios::out | ios::trunc);

    for(int i = 0; i<map.size(); i++){
        int j = 0;
        for(; j<map[i].size() - 1; j++){
            ofs << map[i][j] << ",";
        }
        ofs << map[i][j] << endl;
    }

    ofs.close();
}

// float 値の csv ファイルを読み込む
void csv::ReadCsv(vector<vector<float> >& map, string fp) {
    ifstream ifs(fp);

    string one_line;
    while(getline(ifs, one_line)){
        istringstream line_buf(one_line);
        string one_buf;
        vector<float> line;
        while(getline(line_buf, one_buf, ',')){
            float num = stof(one_buf);
            line.push_back(num);
        }
        map.push_back(line);
    }

    ifs.close();
}

// float 値の csv ファイルを書き込む
void csv::ToCsv(vector<vector<float> >& map, string fp) {
    ofstream ofs;
    ofs.open(fp, ios::out | ios::trunc);

    for(int i = 0; i<map.size(); i++){
        int j = 0;
        for(; j<map[i].size() - 1; j++){
            ofs << map[i][j] << ",";
        }
        ofs << map[i][j] << endl;
    }

    ofs.close();
}