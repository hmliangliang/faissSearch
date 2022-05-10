#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 18:51
# @Author  : Liangliang
# @File    : execution.py
# @Software: PyCharm

import argparse
import faissSearch
import datetime
import s3fs
import time
import multiprocessing
import math
import os


def multiprocessingWrite(file_number,data,output_path):
    print("开始写第{}个文件 {}".format(file_number,datetime.datetime.now()))
    n = len(data)  # 列表的长度
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    with fs.open(output_path + 'pred_{}.csv'.format(int(file_number)), mode="a") as resultfile:
        if n > 1:#说明此时的data是[[],[],...]的二级list形式
            for i in range(n):
                line = "{},{},{},{}\n".format(int(data[i][0]),data[i][1],data[i][2],data[i][3])
                resultfile.write(line)
        else:#说明此时的data是[x,x,...]的list形式
            line = ",".join(map(str, data)) + "\n"
            resultfile.write(line)
    print("第{}个文件已经写入完成,写入数据的行数{} {}".format(file_number,n,datetime.datetime.now()))

class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )
class S3Filewrite:
    def __init__(self, args):
        super(S3Filewrite, self).__init__()
        self.output_path = args.data_output

    def write(self, data, args):
        #注意在此业务中data是一个二维list
        n_data = len(data) #数据的数量
        n = math.ceil(n_data/args.file_max_num) #列表的长度
        s3fs.S3FileSystem = S3FileSystemPatched
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        start = time.time()
        for i in range(0,n,args.file_max_num):
            pool.apply_async(multiprocessingWrite, args=(i, data[i*args.file_max_num:min((i+1)*args.file_max_num,n_data)],self.output_path,))
        pool.close()
        pool.join()
        cost = time.time() - start
        print("write is finish. write {} lines with {:.2f}s".format(n_data, cost))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train_incremental')
    parser.add_argument("--nlist", help="聚类的数目", type=int, default=100)
    parser.add_argument("--k_vectors", help="KNN的最近邻邻居数目", type=int, default=100)
    parser.add_argument("--nprobe", help="在多少个聚类中进行搜索,其值越大搜索越慢", type=int, default=10)
    parser.add_argument("--dim", help="数据的维数", type=int, default=64)
    parser.add_argument("--file_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=300000)

    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument('--tdw_user', type=str, default="",help='tdw_user.')
    parser.add_argument('--tdw_pwd', type=str, default="", help='tdw_pwd.')
    parser.add_argument('--tdw_password', type=str, default="",help='tdw_password.')
    parser.add_argument('--task_name', type=str, default="",
                        help='task_name.')
    parser.add_argument("--model_output", help="模型的输出位置", type=str,default='s3://JK/models/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    print("开始执行算法!{}".format(datetime.datetime.now()))
    start = datetime.datetime.now()
    data = faissSearch.run(args)
    '''
    #采用pyspark来计算召回
    import faissSearch_pyspark
    data = faissSearch_pyspark.run(args)
    '''
    writer = S3Filewrite(args)
    writer.write(data, args)
    end = datetime.datetime.now()
    print("算法的开始时间为:",start)
    print("算法的结束时间为:", end)