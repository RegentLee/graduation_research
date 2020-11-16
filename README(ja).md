# input file(train_sample.csv, test_sample.csv)
前は入力、最後の一列は目標値

# output file(train_result.csv, test_result.csv)
一列目は目標値、後ろはSDNNの出力と各単語patternとの内積

# param.txt
\<train_sample\> 学習サンプル(train_sample.csv)\
\<test_sample\> テストサンプル(test_sample.csv)\
\<pattern\> pattern(pattern.csv)\
\<thread\> スレッド数(20)\
\<train_result\> 学習結果(train_result.csv)\
\<test_result\> テスト結果(test_result.csv)\
\<save_weight\> 重みを保存(weight.csv)\
\<read_weight\> 重みを読み取る(weight.csv)

(かっこの中は例)
- 重みを読み取らない場合、\<read_weight\> は何も書かなくていい\
- 重みを保存しない場合、\<save_weight\> は何も書かなくていい\
- OpenMPを使う場合、\<thread\> で使用するスレッド数を書く\
他の５つは正しく書かなければいけない、\
- プログラムはファイルが存在するかどうかをチェックしない\
- もしプログラムが read end を出力した後にエラーが出た場合、\
ファイル名が正しくない可能性が高い

# SDNNOpenMP 和 SDNNBiOpenMP
単語のパターン長は4の倍数の場合 SDNNBiOpenMP を使うことができる\
他の場合は SDNNOpenMP を使ってください
