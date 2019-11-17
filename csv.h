//
// Created by 李明曄 on 2019/11/16.
//

#ifndef GRADUATION_RESEARCH_CSV_H
#define GRADUATION_RESEARCH_CSV_H

#include <vector>
#include <string>

namespace csv {
    // int 値の csv ファイルを読み込みや書き込み
    void ReadCsv(std::vector<std::vector<int> >& map, std::string fp);
    void ToCsv(std::vector<std::vector<int> >& map, std::string fp);

    // float 値の csv ファイルを読み込みや書き込み
    void ReadCsv(std::vector<std::vector<float> >& map, std::string fp);
    void ToCsv(std::vector<std::vector<float> >& map, std::string fp);
}

#endif //GRADUATION_RESEARCH_CSV_H