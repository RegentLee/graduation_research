//
// Created by 李明曄 on 2019/11/16.
//

#ifndef GRADUATION_RESEARCH_CSV_H
#define GRADUATION_RESEARCH_CSV_H

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

namespace csv {
    // 读取 int 值的csv文件
    // int 値の csv ファイルを読み込
    void ReadCsv(std::vector<std::vector<int> >& map, std::string fp);
    // 写入 int 值的csv文件
    // int 値の csv ファイルを書き込み
    void ToCsv(std::vector<std::vector<int> >& map, std::string fp);

    // 读取 float 值的csv文件
    // float 値の csv ファイルを読み込み
    void ReadCsv(std::vector<std::vector<float> >& map, std::string fp);
    // 写入 float 值的csv文件
    // float 値の csv ファイルを書き込み
    void ToCsv(std::vector<std::vector<float> >& map, std::string fp);
}

#endif //GRADUATION_RESEARCH_CSV_H