#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 17:10
# @Author  : Liangliang
# @File    : faissSearch_pyspark.py
# @Software: PyCharm
import faiss
from pyspark.sql import SparkSession
from pytoolkit import TDWSQLProvider
from pytoolkit import TDWUtil
from pytoolkit import TableDesc
from pyspark.sql.types import StructField, StringType, FloatType, StructType,IntegerType
#import pandas
import numpy as np
import multiprocessing

def searchClubs(inputs,data_index,args,index,result):
    #inputs的第一列为节点id(roleid) string类型,剩余列为embedding vectors且inputs为pandas.dataframe类型
    #data_index为pandas.dataframe类型,第一列代表玩家的roleid,剩余列为embedding vectors
    src_data = inputs.iloc[1::].values.astype(float) #向量
    src_id = inputs.iloc[0] #查询向量对象的roleid
    index.nprobe = args.nprobe #在多少个类簇中进行搜索
    D, I = index.search(np.ascontiguousarray(src_data.reshape(1,-1)),args.k) #D为k最近邻向量与查询向量之间的距离,I为k最近邻向量的id编号
    for i in range(args.k):
        #该结果有四列,第一列为玩家的roleid,第二列为俱乐部的clubid,第三列为ranking值,第四列为距离
        result.append([src_id,data_index.iloc[int(I[0,i])],i,D[0,i]])


def searchVectors(input_data,data,args):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    result = multiprocessing.Manager().list([])
    N = input_data.shape[0]
    #训练faiss的索引结构,采用倒排索引IndexIVFFlat方法
    quantizer = faiss.IndexFlatL2(args.dim)
    index = faiss.IndexIVFFlat(quantizer, args.dim, args.nlist)
    #训练索引结构
    index.train(np.ascontiguousarray(data.iloc[::,1::].values.astype(float)))
    #建立索引结构
    index.add(np.ascontiguousarray(data.iloc[::,1::].values.astype(float)))
    for i in range(N):
        pool.apply_async(func=searchClubs,args=(input_data.iloc[i,::],data,args,index,result,))
    pool.close()
    pool.join()
    return result

def run(args):
    #读取分布式数据库中的表
    print('Begin Load Dataset.')
    spark = SparkSession.builder.appName("read_table").getOrCreate()
    player_path = args.data_input.split(",")[0]
    club_path = args.data_input.split(",")[1]

    player_db_name = player_path.split("::")[0]
    player_td_name = player_path.split("::")[1]

    club_db_name = club_path.split("::")[0]
    club_td_name = club_path.split("::")[1]

    player_tdw = TDWSQLProvider(spark, db=player_db_name, group='tl')
    club_tdw = TDWSQLProvider(spark, db=club_db_name, group='tl')

    #得到数据
    player_df = player_tdw.table(player_td_name).toPandas()
    club_df = club_tdw.table(club_td_name).toPandas()

    #执行召回过程
    result = searchVectors(player_df, club_df, args)
    print("召回算法执行完毕!")
    #result = pandas.DataFrame(result)#list->pandas.DataFrame

    #将召回结果result转化为spark.Dataframe
    #values = result.values.tolist()
    fields = [
        StructField("roleid", StringType(), True),
        StructField("clubid", StringType(), True),
        StructField("ranking", IntegerType(), True),
        StructField("distance", FloatType(), True)
    ]
    t_schema = StructType(fields)
    result = spark.createDataFrame(result, t_schema)

    #将result写回tdw表中
    db_name = args.data_output.split("::")[0]
    tb_name = args.data_output.split("::")[1]
    tdwUtil = TDWUtil(user=args.tdw_user, passwd=args.tdw_pwd, dbName=db_name)
    if not tdwUtil.tableExist(tb_name):
        table_desc = TableDesc().setTblName(tb_name). \
            setCols([['roleid', 'string', 'roleid'],
                     ['clubid', 'string', 'clubid'],
                     ['ranking', 'int', 'ranking'],
                     ['distance', 'double', "distance"]]). \
            setComment("This is result!")

        tdwUtil.createTable(table_desc)
        print('Table created')

    print("output db %s table %s" % (db_name, tb_name))
    tdw = TDWSQLProvider(spark, db=db_name, group='tl')
    tdw.saveToTable(result, tb_name)
    print("数据输出完毕!")

