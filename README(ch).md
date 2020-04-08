#input file(train_sample.csv, test_sample.csv)
前面是输入，最后一列目标值

#output file(train_result.csv, test_result.csv)
第一列目标值，后面是 SDNN 的输出和各个单词的pattern的内积

#param.txt
\<train_sample\> 学习集(train_sample.csv)\
\<test_sample\> 测试集(test_sample.csv)\
\<pattern\> pattern(pattern.csv)\
\<thread\> 线程数(20)\
\<train_result\> 学习结果(train_result.csv)\
\<test_result\> 测试结果(test_result.csv)\
\<save_weight\> 保存权重(weight.csv)\
\<read_weight\> 读取权重(weight.csv)

(括号内为例)
- 如不需读取权重，\<read_weight\> 空着即可\
- 如不需保存权重，\<save_weight\> 空着即可\
- 如开启OpenMP, 请在 \<thread\> 填写使用的线程数,\
- 其余5项必须正确填写，程序不会检查文件是否存在\
- 如果程序输出 read end 后立刻出错，多半是文件名没正确填写

#SDNNOpenMP 和 SDNNBiOpenMP
单词的pattern长为4的倍数时可以使用 SDNNBiOpenMP\
其他pattern长请用SDNNOpenMP

