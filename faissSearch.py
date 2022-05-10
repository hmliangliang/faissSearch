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
import multiprocessing
import datetime
import numpy as np

#https://zhuanlan.zhihu.com/p/148413517

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
        n = len(data) #列表的长度
        s3fs.S3FileSystem = S3FileSystemPatched
        fs = s3fs.S3FileSystem()
        start = time.time()
        for i in range(n):
            with fs.open(self.output_path + 'pred_{}.csv'.format(int(i/args.file_max_num)), mode="a") as resultfile:
                # data = [line.decode('utf8').strip() for line in data.tolist()]
                #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                line = "{},{},{},{}\n".format(int(data[i][0]),data[i][1],data[i][2],data[i][3])
                resultfile.write(line)
        cost = time.time() - start
        print("write is finish. write {} lines with {:.2f}s".format(len(data), cost))


def searchClubs(inputs,data_index,args,index,result):
    #inputs的第一列为节点id(roleid) string类型,剩余列为embedding vectors且inputs为pandas.dataframe类型
    #data_index为pandas.dataframe类型,只有一列,第一列代表clubid string类型
    src_data =inputs.iloc[1::].values.astype(np.float32)#向量
    src_id = inputs.iloc[0] #查询向量对象的roleid
    index.nprobe = args.nprobe #在多少个类簇中进行搜索
    D, I = index.search(np.ascontiguousarray(src_data.reshape(1,-1)),args.k_vectors) #D为k最近邻向量与查询向量之间的距离,I为k最近邻向量的id编号
    for i in range(args.k_vectors):
        #该结果有四列,第一列为玩家的roleid,第二列为俱乐部的clubid,第三列为ranking值,第四列为距离
        result.append([src_id,data_index.iloc[int(I[0,i])],i,D[0,i]])
    print("roleid:{}的俱乐部检索完成 {}".format(int(src_id), datetime.datetime.now()))


def searchVectors(input_data,data,args):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    result = multiprocessing.Manager().list([])
    N = input_data.shape[0]
    #训练faiss的索引结构,采用倒排索引IndexIVFFlat方法
    quantizer = faiss.IndexFlatL2(args.dim)
    index = faiss.IndexIVFFlat(quantizer, args.dim, args.nlist)
    #训练索引结构
    index.train(np.ascontiguousarray(data.iloc[::,1::].values.astype(np.float32)))
    #建立索引结构
    index.add(np.ascontiguousarray(data.iloc[::,1::].values.astype(np.float32)))
    data = data.iloc[:,0]
    for i in range(N):
        print("开始分发第{}个节点搜索任务! {}".format(i,datetime.datetime.now()))
        pool.apply_async(func=searchClubs,args=(input_data.iloc[i,::],data,args,index,result,))
    pool.close()
    pool.join()
    return result

def run(args):
    #读取tdw分布式数据库中的表

    '''读取玩家的数据'''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    print("开始读取玩家数据! {}".format(datetime.datetime.now()))
    input_data = pd.DataFrame()
    for file in input_files:
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        input_data = pd.concat([input_data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)  # 读取边结构数据
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
    #进行向量召回
    result = searchVectors(input_data,data,args)
    return result
