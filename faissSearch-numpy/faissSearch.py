#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 17:10
# @Author  : Liangliang
# @File    : faissSearch.py
# @Software: PyCharm
import os
os.system("conda install nomkl numpy scipy scikit-learn numexpr")
os.system("conda remove mkl mkl-service")
os.system("conda install faiss-cpu -c pytorch")
import faiss
import pandas as pd
import time
import s3fs
import datetime
import numpy as np
import math
import multiprocessing


#https://zhuanlan.zhihu.com/p/148413517

result = np.zeros((1,4)).astype('str')
ranklist = [rank for rank in range(100)]

def multiprocessingWrite(file_number,count,data,output_path):
    print("开始写第{}批第{}个文件 {}".format(count,file_number,datetime.datetime.now()))
    n = len(data)  # 列表的长度
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    with fs.open(output_path + 'pred_{}_{}.csv'.format(count,int(file_number)), mode="a") as resultfile:
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

    def write(self, data, args,count):
        #注意在此业务中data是一个二维list
        n_data = len(data) #数据的数量
        n = math.ceil(n_data/args.file_max_num) #列表的长度
        s3fs.S3FileSystem = S3FileSystemPatched
        pool = multiprocessing.Pool(processes=1)
        start = time.time()
        for i in range(n):
            pool.apply_async(multiprocessingWrite, args=(i,count, data[i*args.file_max_num:min((i+1)*args.file_max_num,n_data)],self.output_path,))
        pool.close()
        pool.join()
        cost = time.time() - start
        print("第{}批文件写入完成,共写入{}行,耗时:{:.2f}s".format(count,n_data, cost))

def searchClubs(num,inputs,data_index,args,index):
    #inputs的第一列为节点id(roleid) string类型,剩余列为embedding vectors且inputs为pandas.dataframe类型
    #data_index为pandas.dataframe类型,第一列代表玩家的roleid,剩余列为embedding vectors
    src_data = inputs.iloc[1::].values.astype("float32") #向量
    src_id = str(int(inputs.iloc[0])) #查询向量对象的roleid
    index.nprobe = args.nprobe #在多少个类簇中进行搜索
    D, I = index.search(np.ascontiguousarray(src_data.reshape(1,-1)),args.k_vectors) #D为k最近邻向量与查询向量之间的距离,I为k最近邻向量的id编号
    n = num*args.k_vectors
    # 该结果有四列,第一列为玩家的roleid,第二列为俱乐部的clubid,第三列为ranking值,第四列为距离
    result[n:n + args.k_vectors, 0] = src_id
    result[n:n + args.k_vectors, 1] = data_index.iloc[I[0,:],0].astype("str")
    result[n:n + args.k_vectors, 2] = ranklist
    result[n:n + args.k_vectors, 3] = D


def searchVectors(input_data,data,args,count):
    N = input_data.shape[0]
    #训练faiss的索引结构,采用倒排索引IndexIVFFlat方法
    print("开始训练索引模型! {}".format(datetime.datetime.now()))
    quantizer = faiss.IndexFlatL2(args.dim)
    print("已构建完quantizer {}".format(datetime.datetime.now()))
    index = faiss.IndexIVFFlat(quantizer, args.dim, args.nlist)
    print("已构建完index结构 {}".format(datetime.datetime.now()))
    #训练索引结构
    index.train(np.ascontiguousarray(data.iloc[::,1::].values.astype("float32")))
    print("训练完index结构 {}".format(datetime.datetime.now()))
    #建立索引结构
    index.add(np.ascontiguousarray(data.iloc[::,1::].values.astype("float32")))
    print("已构建完俱乐部的index结构 {}".format(datetime.datetime.now()))
    global  result
    global ranklist
    result = np.zeros((N*args.k_vectors, 4)).astype('str')
    ranklist = [str(rank) for rank in range(args.k_vectors)]
    for i in range(N):
        t1 = time.time()
        searchClubs(i,input_data.iloc[i,::],data,args,index)
        t2 = time.time()
        print("第{}批数据一共{}个任务,已完成第{}个任务的执行,检索耗时:{} 时间为:{}".format(count,N,i,t2-t1,datetime.datetime.now()))


def run(args):
    #读取tdw分布式数据库中的表
    '''读取俱乐部的数据'''
    path = args.data_input.split(',')[1]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    print("开始读取俱乐部数据! {}".format(datetime.datetime.now()))
    data = pd.DataFrame()
    for file in input_files:
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        data = pd.concat([data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)  # 读取边结构数据

    '''读取玩家的数据'''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    print("开始读取玩家数据! {}".format(datetime.datetime.now()))
    for file in input_files:
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        input_data = pd.read_csv("s3://" + file, sep=',', header=None)  # 读取边结构数据
        # 进行向量召回
        searchVectors(input_data, data, args,count)
        writer = S3Filewrite(args)
        writer.write(result.tolist(), args,count)
        print("第{}个文件已经写完! {}".format(count,datetime.datetime.now()))
