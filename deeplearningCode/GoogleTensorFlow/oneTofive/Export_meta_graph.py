#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 16:46
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : Export_meta_graph.py
# @Software: PyCharm
# 使用这个函数能够以json格式导出MetaGraphDef Protocol Buffer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

v1 = tf.Variable(tf.constant(1., shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2., shape=[1]), name='v2')
result = v1 + v2

saver = tf.train.Saver()
# 通过函数导出tensorflow计算原图保存为json格式
saver.export_meta_graph("/Users/cjv/Documents/deeplearningCode/GoogleTensorFlow/Models/model3.ckpt.meta.json",
                        as_text=True)
# meta_info_def 属性：记录tf计算图中的元数据以及tf程序中运算方法的信息
# 元数据类型的定义原型如下：
# message MetaInfoDef{
#     string meta_graph_version = 1;    版本号
#     OpList stripped_op_list = 2;      在.json文件中，此处不为空
#     google.protobuf.Any any_info = 3;
#     repeated string tags = 4;
# }
# stripped_op_list记录了tf计算图所有的运算方法信息

# OpList是一个OpDef类型的列表
# op {
#       name: "Add"
#       input_arg {
#         name: "x"
#         type_attr: "T"
#       }
#       input_arg {
#         name: "y"
#         type_attr: "T"
#       }
#       output_arg {
#         name: "z"
#         type_attr: "T"
#       }
#       attr {
#         name: "T"
#         type: "type"
#         allowed_values {
#           list {
#             type: DT_HALF
#             type: DT_FLOAT
#             type: DT_DOUBLE
#             type: DT_UINT8
#             type: DT_INT8
#             type: DT_INT16
#             type: DT_INT32
#             type: DT_INT64
#             type: DT_COMPLEX64
#             type: DT_COMPLEX128
#             type: DT_STRING
#           }
#         }
#       }
#     }
# 这个操作是名称为add的运算。2个输入，1个输出

# graph_def记录的是tf计算图中的节点信息。每个节点对应一个运算。
# graph_def {
#   node {
#     name: "Const"
#     op: "Const"
#     attr {
#       key: "_output_shapes"
#       value {
#         list {
#           shape {
#             dim {
#               size: 1
#             }
#           }
#         }
#       }
#     }
#     attr {
#       key: "dtype"
#       value {
#         type: DT_FLOAT
#       }
#     }
#     attr {
#       key: "value"
#       value {
#         tensor {
#           dtype: DT_FLOAT
#           tensor_shape {
#             dim {
#               size: 1
#             }
#           }
#           float_val: 1.0
#         }
#       }
#     }
#   }

# saver_def {
#   filename_tensor_name: "save/Const:0" 张量名称
#   save_tensor_name: "save/control_dependency:0"   持久化tf模型运算对应的节点名称
#   restore_op_name: "save/restore_all"  加载模型的运算
#   max_to_keep: 5   保存模型超过5次那么第一次的就会被清理
#   keep_checkpoint_every_n_hours: 10000.0
#   version: V2
# }
# save_def记录持久化模型需要的一些参数。文件名、保存操作、加载操作、保存频率、清理历史记录
