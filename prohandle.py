#coding=utf-8
'''
Created on Apr 16, 2017

@author: binlin

本模块主是对中文文本文件进行预处理，使用结巴分词进行处理

input file: preinputfile.txt
output file: preoutputfile.txt 

'''
import sys  # 引入系统模块，可以输入输出
import os 
import jieba  # 引入结巴分词模块
import jieba.analyse #引入结巴topk分析工具
# import __future__  import print_function, unicode_literals

#自定义常量                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
PRE_INPUT_FILE="preinputfile.txt"
PRE_OUTPUT_FILE="preoutputfile.txt"
PRE_TOPK_FILE="pretopkfile.txt"



#测试模块
def testopenfile():
    "打开文件"
    f = open(PRE_INPUT_FILE,'r')
    for eachLine in f:
        print eachLine
    f.close()
def testjieba():
    "使用结巴进行分词"
    f = open(PRE_INPUT_FILE,'r')
    for eachLine in f:
        seg_list = jieba.cut(eachLine,cut_all=False)
        print ",".join(seg_list)  
    f.close()
def testoutputfile():
    "使用结巴进行分词，并将分词结果写入目标文件中"
    fi = open(PRE_INPUT_FILE,'r')
    fo = open(PRE_OUTPUT_FILE,'w')
    for eachLine in fi:
        seg_list = jieba.lcut(eachLine.strip(),cut_all=False)
        #fo.writelines(str(seg_list))
        for w in seg_list:
            s = w.encode("utf-8").strip()
            if s is not None:
                print s, type(s)
                fo.write(s+",")
        fo.write("\n")
        #print ",".join(seg_list)
        #=======================================================================
        # s = ""
        # s = [s+ words.word for words in seg_list]
        # fo.write(s)
        #=======================================================================
    # fo.write("just for test")
    #print os.getcwd()
    fi.close()
    fo.close()

def testgettopk(topK=10,srcfile=PRE_INPUT_FILE,dstfile=PRE_TOPK_FILE):
    "使用结巴进行中文分析，将文件中出现频率最多的词语写入目标文件中"
    fi=open(srcfile,'r')
    fo=open(dstfile,'w')
    content =  fi.read()
    tags = jieba.analyse.extract_tags(content,topK=topK)
    print(",".join(tags))
    for w in tags:
        s = w.encode("utf-8").strip()
        fo.write(s+"\n")
    fi.close()
    fo.close()
      
    
if __name__ == '__main__':
    pass
    #testoutputfile()
    #testgettopk(10, PRE_INPUT_FILE, PRE_TOPK_FILE)