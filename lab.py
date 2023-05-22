# 取list的前n个字符
"""
a_list = [
    'abc',
    'dss',
    'wew',
    'owososw',
]
for kkk in range(len(a_list)):
    print(a_list[kkk][:2])  # 打印每个字符串前两字符
"""

# os.walk();1_check_hsp_mat.py
import time

import cv2
from utils import math

"""
import os
'''top -- 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。

root 所指的是当前正在遍历的这个文件夹的本身的地址
dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)'''
path = 'dataset'
# for root, dirs, files in os.walk(r'Documents', topdown=False):  # 此处循环两轮，第一轮遍历images文件夹，第二次遍历documents文件夹，True 从上到下；False 从下到上
#     for name in files:
#         print(os.path.join(root, name))
#     for name in dirs:
#         print(os.path.join(root, name))  # 可以打印中间文件夹：Documents\images
#     print('root:', root, 'dirs:', dirs, 'files:', files)
#     a_list = []
#     cat = {}
#     for fname in files:
#         fpath = os.path.join(path, fname)
#         suffix = os.path.splitext(fname)[1].lower()  # 把文件扩展名提取出来并变成小写：os.path.splitext()分割文件路径名和文件扩展名的元组;str.lower()大写变小写
#         if os.path.isfile(fpath):  # 如果文件存在且扩展名匹配
#             if path not in cat:  # 如果路径没被添加过
#                 cat[path] = len(cat)
#             a_list.append((os.path.relpath('Documents', root), cat['Documents']))
#             print("a_list:", a_list)


def list_image(root, recursive, exts):
    image_list = []
    cat = {}  # 字典：路径是key，字典长度是value
    for path, subdirs, files in os.walk(root):  # 遍历文件夹
        for fname in files:
            fpath = os.path.join(path, fname)
            suffix = os.path.splitext(fname)[1].lower()  # 把文件扩展名提取出来并变成小写：os.path.splitext()分割文件路径名和文件扩展名的元组;str.lower()大写变小写
            if os.path.isfile(fpath) and (suffix in exts):  # 如果文件存在且扩展名匹配
                if path not in cat:  # 如果路径没被添加过
                    cat[path] = len(cat)
                    print(cat)
                image_list.append((os.path.relpath(fpath, root), cat[path]))  # 从root开始计算相对路径
    return image_list

image_list = list_image(path, True, '.png')
print(image_list)
name_list = []
for i in range(len(image_list)):
    imagine = image_list[i][0]  # 遍历找到的文件列表，取出路径
    name_list.append(imagine[0:-4])
    print(name_list)
"""


# 比较两个文件不同
"""
import os

def cmp_file(f1, f2):
    st1 = os.stat(f1)
    st2 = os.stat(f2)

    # 比较文件大小
    # if st1.st_size != st2.st_size:
    #     return False

    # bufsize = 8*1024
    bufsize = 8
    with open(f1, 'rb') as fp1, open(f2, 'rb') as fp2:
        while True:
            b1 = fp1.read(bufsize)  # 读取指定大小的数据进行比较
            b2 = fp2.read(bufsize)
            if b1 != b2:
                print('different!', b1, b2)
                return False
            if not b1:
                return True

result = cmp_file('IM-NET-master/point_sampling/4_gather_all_vox_img_test.py', 'IM-NET-master/point_sampling/4_gather_all_vox_img_train.py')
print(result)
"""


# Queue队列
"""
# from Queue import Queue,LifoQueue,PriorityQueue
import queue
from multiprocessing import Process, Queue
#先进先出队列
q=queue.Queue(maxsize=5)
#后进先出队列
lq=queue.LifoQueue(maxsize=6)
#优先级队列
pq=queue.PriorityQueue(maxsize=5)

for i in range(5):
    q.put(i)
    lq.put(i)
    pq.put(i)

print("先进先出队列：%s;是否为空：%s；多大,%s;是否满,%s" %(q.queue,q.empty(),q.qsize(),q.full()))
print("后进先出队列：%s;是否为空：%s;多大,%s;是否满,%s" %(lq.queue,lq.empty(),lq.qsize(),lq.full()))
print("优先级队列：%s;是否为空：%s,多大,%s;是否满,%s" %(pq.queue,pq.empty(),pq.qsize(),pq.full()))
print(q.get(),lq.get(),pq.get())
print("先进先出队列：%s;是否为空：%s；多大,%s;是否满,%s" %(q.queue,q.empty(),q.qsize(),q.full()))
print("后进先出队列：%s;是否为空：%s;多大,%s;是否满,%s" %(lq.queue,lq.empty(),lq.qsize(),lq.full()))
print("优先级队列：%s;是否为空：%s,多大,%s;是否满,%s" %(pq.queue,pq.empty(),pq.qsize(),pq.full()))

demo = {'name': 'xz', 'age': 22}
print(demo.get(True, 1.0))
# print('a:%d', 'b:%d', 'c:%d' % a % b % c)
"""

# 多进程
"""
'''
创建进程的类：Process([group [, target [, name [, args [, kwargs]]]]])，target表示调用对象，args表示调用对象的位置参数元组。
            kwargs表示调用对象的字典。name为别名。group实质上不使用。
方法：is_alive()、join([timeout])、run()、start()、terminate()。其中，Process以start()启动某个进程。
属性：authkey、daemon（要通过start()设置）、exitcode(进程在运行时为None、如果为–N，表示被信号N结束）、name、pid。
            其中daemon是父进程终止后自动终止，且自己不能产生新进程，必须在start()之前设置。
'''
import multiprocessing
import time

def worker(interval):
    n = 5
    while n > 0:
        print("The time is {0}".format(time.ctime()))
        time.sleep(interval)
        n -= 1

if __name__ == "__main__":
    p = multiprocessing.Process(target = worker, args = (3,))  # 创建进程
    p.start()  # 启动进程
    print("p.pid:", p.pid)
    print("p.name:", p.name)
    print("p.is_alive:", p.is_alive())
"""


# 生成随机整数
"""
import numpy as np
a = np.random.randint(100, size=(1, 3))
print(a)

# Generate a 1 x 3 array with 3 different upper bounds
# b = np.random.randint((2, 3, 2), 1)
# print(b)
"""


# 从立方体采样
"""
import numpy as np

a = 3
b = 3
c = 2
print('(%d, %d, %d)' % (a, b, c))

def sample_point_in_cube(block, target_value, halfie):
    halfie2 = halfie * 2
    print(target_value)

    for i in range(10):
        x = np.random.randint(halfie2)  # 生成随机点坐标
        y = np.random.randint(halfie2)
        z = np.random.randint(halfie2)
        print('(x, y, z) = (%d, %d, %d)' % (x, y, z))
        print(np.shape(cube))
        if all(block[x, y, z] == target_value):
            return x, y, z

    if block[halfie, halfie, halfie] == target_value:
        return halfie, halfie, halfie

    i = 1
    ind = np.unravel_index(
        np.argmax(block[halfie - i:halfie + i, halfie - i:halfie + i, halfie - i:halfie + i], axis=None),
        (i * 2, i * 2, i * 2))
    if block[ind[0] + halfie - i, ind[1] + halfie - i, ind[2] + halfie - i] == target_value:
        return ind[0] + halfie - i, ind[1] + halfie - i, ind[2] + halfie - i

    for i in range(2, halfie + 1):
        six = [(halfie - i, halfie, halfie), (halfie + i - 1, halfie, halfie), (halfie, halfie, halfie - i),
               (halfie, halfie, halfie + i - 1), (halfie, halfie - i, halfie), (halfie, halfie + i - 1, halfie)]
        for j in range(6):
            if block[six[j]] == target_value:
                return six[j]
        ind = np.unravel_index(
            np.argmax(block[halfie - i:halfie + i, halfie - i:halfie + i, halfie - i:halfie + i], axis=None),
            (i * 2, i * 2, i * 2))
        if block[ind[0] + halfie - i, ind[1] + halfie - i, ind[2] + halfie - i] == target_value:
            return ind[0] + halfie - i, ind[1] + halfie - i, ind[2] + halfie - i
    print('hey, error in your code!')
    exit(0)

cube = np.random.randint(9, size=(3, 3, 3))
print(np.shape(cube))
# np.reshape(cube, (1, 3))
print(np.shape(cube))
target_value = [3, 3, 4]
print(cube)
sampled_point = sample_point_in_cube(cube, target_value, 1)
print(sampled_point)
"""

# 打开hdf5
"""
import h5py
import pandas as pd

hdf5_file = h5py.File(r'D:\Projects\3D_Recognition\Documents\else\02691156_vox256_img_test.hdf5', 'r')  # 写入hdf5文件
print(hdf5_file)
data = pd.read_hdf(r'D:\Projects\3D_Recognition\Documents\else\02691156_vox256_img_test.hdf5')
print(data)
"""

# before voxel_model_256
"""
import numpy as np
import matplotlib.pyplot as plt

voxel_model_bi = np.random.randint(3, size=(4, 4, 4))
voxel_model_b = np.random.randint(2, size=(9, 4, 4, 4))
voxel_model_256 = np.zeros([16, 16, 16], np.uint8)
for i in range(4):
    for j in range(4):
        for k in range(4):
            voxel_model_256[i * 4:i * 4 + 4, j * 4:j * 4 + 4, k * 4:k * 4 + 4] = voxel_model_b[
                voxel_model_bi[i, j, k]]
            # print("[i * 4:i * 4 + 4, j * 4:j * 4 + 4, k * 4:k * 4 + 4]\n", [i * 4, j * 4, k * 4])
            # print("voxel_model_bi[i, j, k]\n", voxel_model_bi[i, j, k])
            # print("[i, j, k]\n", [i, j, k])
# print('voxel_model_bi==\n', voxel_model_bi)
print('voxel_model_b==\n', voxel_model_b)
print('voxel_model_256==\n', voxel_model_256)

fig = plt.figure(1, figsize=(6, 6))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.scatter(voxel_model_bi[:, 0], voxel_model_bi[:, 1], voxel_model_bi[:, 2], marker='.')
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.scatter(voxel_model_256[:, 0], voxel_model_256[:, 1], voxel_model_256[:, 2], marker='.')
ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.scatter(voxel_model_b[:, 0], voxel_model_b[:, 1], voxel_model_b[:, 2], marker='.')
plt.show()

#  框定一个边界?
dim_voxel = 16
top_view = np.max(voxel_model_256, axis=1)
left_min = np.full([dim_voxel, dim_voxel], dim_voxel, np.int32)  # 256*256：256
left_max = np.full([dim_voxel, dim_voxel], -1, np.int32)  # 256*256：-1
front_min = np.full([dim_voxel, dim_voxel], dim_voxel, np.int32)  # 256*256：256
front_max = np.full([dim_voxel, dim_voxel], -1, np.int32)  # 256*256：-1


# print("top_view\n", top_view)
# print("left_min\n", left_min)
# print("left_max\n", left_max)
# print("front_min\n", front_min)
"""

# sample_point_in_cube 试验田
"""
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

# voxel_model_bi = np.random.randint(9, size=(16, 16, 16))
# voxel_model_b = np.random.randint(7, size=(238, 16, 16, 16))

voxel_model_bi = np.random.randint(4, size=(16, 16, 16))
voxel_model_b = np.random.randint(2, size=(4, 16, 16, 16))

print('voxel_model_bi==\n', voxel_model_bi)
print('voxel_model_b==\n', voxel_model_b)
voxel_model_256 = np.zeros([256, 256, 256], np.uint8)
for i in range(16):
    for j in range(16):
        for k in range(16):
            voxel_model_256[i * 16:i * 16 + 16, j * 16:j * 16 + 16, k * 16:k * 16 + 16] = voxel_model_b[
                voxel_model_bi[i, j, k]]
# add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
print('voxel_model_256 before flip\n', voxel_model_256)
voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2, 1, 0)), 2)
print('voxel_model_256 after flip\n', voxel_model_256)

# # plt.subplot(1,  1,  1)
# x = np.arange(0,  3*np.pi,  0.1)
# y = np.sin(3*x)
# # plt.plot(x, y)
# plt.plot(voxel_model_bi[0], voxel_model_bi[1], voxel_model_bi[2])
# plt.title('voxel_model_256 graph')
# plt.ylabel('Y axis')
# plt.xlabel('X axis')
# plt.show()

# plt.subplot(2,  2,  1)
fig = plt.figure(1, figsize=(6, 6))
# zz = (xx ** 2 + yy ** 2)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.set_top_view()
# ax.plot_surface(voxel_model_bi[0], voxel_model_bi[1], voxel_model_bi[2], rstride=1, cstride=1, cmap='rainbow')
ax.scatter(voxel_model_bi[:,0], voxel_model_bi[:,1], voxel_model_bi[:,2], marker='.')
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.scatter(voxel_model_b[:,0], voxel_model_b[:,1], voxel_model_b[:,2], marker='.')
ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.scatter(voxel_model_256[:,0], voxel_model_256[:,1], voxel_model_256[:,2], marker='.')


#  框定一个边界?
dim_voxel = 256
top_view = np.max(voxel_model_256, axis=1)
left_min = np.full([dim_voxel, dim_voxel], dim_voxel, np.int32)
left_max = np.full([dim_voxel, dim_voxel], -1, np.int32)
front_min = np.full([dim_voxel, dim_voxel], dim_voxel, np.int32)
front_max = np.full([dim_voxel, dim_voxel], -1, np.int32)


ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.scatter(left_min[:,0], left_min[:,1], left_min[:,2], marker='.')
plt.show()

for j in range(dim_voxel):
    for k in range(dim_voxel):
        occupied = False
        for i in range(dim_voxel):
            if voxel_model_256[i, j, k] > 0:
                if not occupied:
                    occupied = True
                    left_min[j, k] = i
                left_max[j, k] = i

# for i in range(dim_voxel):
#     for j in range(dim_voxel):
#         occupied = False
#         for k in range(dim_voxel):
#             if voxel_model_256[i, j, k] > 0:
#                 if not occupied:
#                     occupied = True
#                     front_min[i, j] = k
#                 front_max[i, j] = k
#
# for i in range(dim_voxel):
#     for k in range(dim_voxel):
#         if top_view[i, k] > 0:
#             fill_flag = False
#             for j in range(dim_voxel - 1, -1, -1):
#                 if voxel_model_256[i, j, k] > 0:
#                     fill_flag = True
#                 else:
#                     if left_min[j, k] < i and left_max[j, k] > i and front_min[i, j] < k and front_max[i, j] > k:
#                         if fill_flag:
#                             voxel_model_256[i, j, k] = 1
#                     else:
#                         fill_flag = False

# # 256到64采样；compress model 256 -> 64
# dim_voxel = 64
# voxel_model_temp = np.zeros([dim_voxel, dim_voxel, dim_voxel], np.uint8)
# multiplier = int(256 / dim_voxel)
# halfie = int(multiplier / 2)
#
#
#
#
# for i in range(dim_voxel):
#     for j in range(dim_voxel):
#         for k in range(dim_voxel):
#             voxel_model_temp[i, j, k] = np.max(
#                 voxel_model_256[i * multiplier:(i + 1) * multiplier, j * multiplier:(j + 1) * multiplier,
#                 k * multiplier:(k + 1) * multiplier])
#
# # write voxel  写入采样点
# sample_voxels = np.reshape(voxel_model_temp, (dim_voxel, dim_voxel, dim_voxel, 1))
"""


"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y)  # 转换成二维的矩阵坐标

fig = plt.figure(1, figsize=(12, 8))
zz = (xx ** 2 + yy ** 2)
ax = fig.add_subplot(2, 2, 1, projection='3d')
# ax.set_top_view()

ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='rainbow')
plt.show()
"""


# 读取binvox文件
"""
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
class Voxels(object):
    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        # axis不是xyz坐标系的时候触发异常
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order
def read_as_3d_array(fp, fix_coords=True):
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.int)  # int对应0和1；bool对应T和F
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return Voxels(data, dims, translate, scale, axis_order)
def read_header(fp):
    # 读取binvox头文件
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale
if __name__ == '__main__':
    path = 'IM-NET-master/point_sampling/building_demo/404-lab.binvox'
    with open(path, 'rb') as f:
        model = read_as_3d_array(f)
        # 尺寸(长宽高)，转化矩阵，放缩系数
        # print(model.dims, model.translate, model.scale)
        with open(r"D:\Projects\3D_Recognition\IM-NET-master\point_sampling\building_demo\output\voxel.txt", "w") as f:
            # f.write(str(model.dims))  # 256*256*256 这句话自带文件关闭功能，不需要再写f.close()
            f.write(str(model.data))
            print(model.dims, model.translate, model.scale)
            # print(model.data)
"""


# 读取hdf5文件
import h5py  #导入工具包
import numpy as np
import sys
# np.set_printoptions(threshold=sys.maxsize)
# #HDF5的写入：
# imgData = np.zeros((30,3,128,256))
# f = h5py.File('HDF5_FILE.h5','w')   #创建一个h5文件，文件指针是f
# f['data'] = imgData                 #将数据写入文件的主键data下面
# f['labels'] = range(100)            #将数据写入文件的主键labels下面
# f.close()                           #关闭文件

# HDF5的读取：
# 00000000_vox256.hdf5
# f = h5py.File(r'D:\Projects\3D_Recognition\IM-NET-master\point_sampling\00000000_building\00000000_vox256.hdf5','r')   #打开h5文件
# f.keys()                            #可以查看所有的主键
# key_1 = f['points_64'][:]                    #取出主键为data的所有的键值
# key_2 = f['values_64'][:]                    #取出主键为data的所有的键值
# key_3 = f['points_32'][:]                    #取出主键为data的所有的键值
# key_4 = f['values_32'][:]                    #取出主键为data的所有的键值
# key_5 = f['points_16'][:]                    #取出主键为data的所有的键值
# key_6 = f['values_16'][:]                    #取出主键为data的所有的键值
# key_7 = f['voxels'][:]                    #取出主键为data的所有的键值
# f.close()
# print('f.keys()', f.keys)
# # print('a', a)
# data = open(r"D:\Projects\3D_Recognition\IM-NET-master\point_sampling\building_demo\output\00000000_vox256_hdf5.txt", 'w+')
# print('points_64:\n', key_1, file=data)
# print('values_64:\n', key_2, file=data)
# print('points_32:\n', key_3, file=data)
# print('values_32:\n', key_4, file=data)
# print('points_16:\n', key_5, file=data)
# print('values_16:\n', key_6, file=data)
# print('voxels:\n', key_7, file=data)
# data.close()


# f = h5py.File(r'D:\Projects\3D_Recognition\IM-NET-master\point_sampling\00000000_building\00000000_img.hdf5','r')   #打开h5文件
# f.keys()                            #可以查看所有的主键
# key_7 = f['pixels'][:]                    #取出主键为data的所有的键值
# f.close()
# print('f.keys()', f.keys)
# # print('a', a)
# data = open(r"D:\Projects\3D_Recognition\IM-NET-master\point_sampling\building_demo\output\00000000_img_hdf5.txt", 'w+')
#
# print('pixels:\n', key_7, file=data)
# data.close()


# 将张量输出可视化
"""
import matplotlib.pyplot as plt        # 可以理解为画板
import numpy as np
import torch


img_temp = torch.tensor([
        [0.6977, 0.4890, 0.9416],
        [0.1685, 0.4124, 0.8846],
        [0.8664, 0.9411, 0.5017],
        [0.6676, 0.9020, 0.4379],
        [0.8681, 0.1378, 0.6282]])

img = np.array(img_temp)

# *
plt.figure("boy")
plt.imshow(img_temp,cmap='gray')
plt.axis('on')
plt.show()
"""

# image_hdf5生成模块
"""
import cv2
import numpy as np
import h5py


name_num = 1
num_view = 1
view_size = 224
idx = 0
t = 0
img = cv2.imread(r"D:\Projects\3D_Recognition\Paper\images\2-1.png", cv2.IMREAD_UNCHANGED)
# img = cv2.imread(r"D:\Projects\3D_Recognition\Paper\images\render_15.png", cv2.IMREAD_UNCHANGED)
hdf5_path = r"D:\Projects\3D_Recognition\IM-NET-master\data\hdf5_img_test.hdf5"

imgo = img[:, :, :3]
imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
imga = (img[:, :, 3]) / 255.0
img = imgo * imga + 255 * (1 - imga)
dimen = np.array(img).shape
print(img)
img = np.round(img).astype(np.uint8)
print(img)
hdf5_file = h5py.File(hdf5_path, 'w')
hdf5_file.create_dataset("pixels", [name_num, num_view, dimen[0], dimen[1]], np.uint8, compression=8)
hdf5_file["pixels"][idx, t, :, :] = img

cv2.imshow("IMREAD_GRAYSCALE+Color", img)
cv2.waitKey()
"""

# 图片规格统一128x128以及hdf5文件生成
"""
import h5py
import random
from PIL import Image
import numpy as np
import cv2
import os

def resize_image(infile, outfile='', x_s=128):
    # 修改图片尺寸
    # :param infile: 图片源文件
    # :param outfile: 重设尺寸文件保存地址
    # :param x_s: 设置的宽度
    # :return:

    im = Image.open(infile)
    y_s = x_s
    out_img = im.resize((x_s, y_s), Image.ANTIALIAS)
    out_img.save(outfile)
    return out_img

root = "D:\\Projects\\3D_Recognition\\dataset\\Zuerich\\images\\building_mini\\"
hdf5_path = r"D:\Projects\3D_Recognition\IM-NET-master\data\hdf5_img_127.hdf5"
hdf5_file = h5py.File(hdf5_path, 'w')
t = 0
img_size = 127  # 127
view_num = 3  # 5
out_img = r'D:\Projects\3D_Recognition\Paper\images\temp_resize.png'
# for root, dirs, files in os.walk(root, topdown=False):
#     # print(root, dirs, files)
#     for img_name in files:
#         img_path = root + '/' + img_name
#         print("the path is : ", img_path)
#         # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#         # 将目标图片压成128x128
#         result = resize_image(img_path, out_img)
#         img = cv2.imread(out_img, cv2.IMREAD_UNCHANGED)
#
#         imgo = img[:, :, :]
#         imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
#         imga = 1
#         img = imgo * imga + 255 * (1 - imga)
#         print(img)
#         array = np.array(img)
#         print(array.shape)
#         hdf5_file.create_dataset('%s' % t, [len(files), array.shape[0], array.shape[1]], np.uint8)
#         hdf5_file["%s" % t][t, :, :] = img
#         t += 1
idx = 0
view_count = 0
for sub_root, dirs, files in os.walk(root, topdown=False):
    for sub_file_name in files:
        print("sub_root_name, file_name", sub_file_name)
        tmp = idx
        print(sub_root)
        three_lists = sub_root.rpartition("\\")
        idx = three_lists[2]
        img_pth = os.path.join(sub_root, sub_file_name)
        print('img_pth', img_pth)
        print(idx)
        if idx != tmp:
            view_count = 0
            hdf5_file.create_dataset('%s' % idx, [len(files), 128, 128], np.uint8)

        result = resize_image(img_pth, out_img)
        img = cv2.imread(out_img, cv2.IMREAD_UNCHANGED)

        imgo = img[:, :, :]
        imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
        # imga = 1
        imga = (img[:, :, 2]) / 255.0
        img = imgo * imga + 255 * (1 - imga)
        array = np.array(img)
        # print(array.shape)
        hdf5_file["%s" % idx][view_count, :, :] = img
        view_count += 1
"""


# 将hdf5文件分成训练集和测试集
"""
import h5py

def split_h5_rows(h5_path, val_ratio, test_ratio):  # 保留每个key，从key的行数分割
    origin_h5 = h5py.File(h5_path, "r")
    keys = [key for key in origin_h5.keys()]
    n = origin_h5[keys[0]].shape[0]
    h5_train = h5py.File(h5_path[:-5]+'_train.hdf5', 'w')
    h5_test = h5py.File(h5_path[:-5]+'_test.hdf5', 'w')
    h5_val = h5py.File(h5_path[:-5]+'_val.hdf5', 'w')
    for key in keys:
        h5_train.create_dataset(key, data=origin_h5[key][:int(n*(1-val_ratio-test_ratio))])
        h5_test.create_dataset(key, data=origin_h5[key][int(n*(1-val_ratio-test_ratio)):int(n*(1-test_ratio))])
        h5_val.create_dataset(key, data=origin_h5[key][int(n*(1-test_ratio)):])
    h5_train.close()
    h5_test.close()
    h5_val.close()

def split_h5_keys(h5_path, val_ratio, test_ratio):  # 保留每个key，从key的行数分割
    origin_h5 = h5py.File(h5_path, "r")
    keys = [key for key in origin_h5.keys()]
    n = len(keys)
    print("total key = ", n)
    keys_train = keys[:int(n*(1-val_ratio-test_ratio))]
    keys_test = keys[int(n*(1-val_ratio-test_ratio)):int(n*(1-test_ratio))]
    keys_val = keys[int(n*(1-test_ratio)):]
    print("keys in train test val:", keys_train, '\n', keys_test, '\n', keys_val)
    h5_train = h5py.File(h5_path[:-5]+'_train.hdf5', 'w')
    h5_test = h5py.File(h5_path[:-5]+'_test.hdf5', 'w')
    h5_val = h5py.File(h5_path[:-5]+'_val.hdf5', 'w')
    for key in keys_train:
        h5_train.create_dataset(key, data=origin_h5[key][:])
    for key in keys_test:
        h5_test.create_dataset(key, data=origin_h5[key][:])
    for key in keys_val:
        h5_val.create_dataset(key, data=origin_h5[key][:])
    h5_train.close()
    h5_test.close()
    h5_val.close()

h5_img_path = r"D:\Projects\3D_Recognition\IM-NET-master\data\hdf5_img_127.hdf5"
h5_model_path = r""
split_h5_keys(h5_img_path, 0.3, 0.1)
split_h5_rows(h5_model_path, 0.3, 0.1)
"""



# 图片压缩和挤压
"""
from PIL import Image
import cv2
import os
def get_size(file):
    # 获取文件大小:KB
    size = os.path.getsize(file)
    return size / 1024
def get_outfile(infile, outfile):
    if outfile:
        return outfile
    dir, suffix = os.path.splitext(infile)
    outfile = '{}-out{}'.format(dir, suffix)
    return outfile
def compress_image(infile, outfile='', mb=150, step=10, quality=80):
    # 不改变图片尺寸压缩到指定大小
    # :param infile: 压缩源文件
    # :param outfile: 压缩文件保存地址
    # :param mb: 压缩目标，KB
    # :param step: 每次调整的压缩比率
    # :param quality: 初始压缩比率
    # :return: 压缩文件地址，压缩文件大小
    
    o_size = get_size(infile)
    if o_size <= mb:
        return infile
    outfile = get_outfile(infile, outfile)
    while o_size > mb:
        im = Image.open(infile)
        im.save(outfile, quality=quality)
        if quality - step < 0:
            break
        quality -= step
        o_size = get_size(outfile)
    return outfile, get_size(outfile)
def resize_image(infile, outfile='', x_s=128):
    # 修改图片尺寸
    # :param infile: 图片源文件
    # :param outfile: 重设尺寸文件保存地址
    # :param x_s: 设置的宽度
    # :return:
    
    im = Image.open(infile)
    # x, y = im.size
    # # y_s = int(y * x_s / x)
    y_s = x_s
    out_img = im.resize((x_s, y_s), Image.ANTIALIAS)
    # outfile = get_outfile(infile, outfile)
    out_img.save(outfile)
    return out_img


if __name__ == '__main__':
    compress_image(r'D:\Projects\3D_Recognition\Paper\images\2-1.png', r'D:\Projects\3D_Recognition\Paper\images\2-1compress.png')
    out_path = r'D:\Projects\3D_Recognition\Paper\images\2-1resize.png'
    result = resize_image(r'D:\Projects\3D_Recognition\Paper\images\2-1.png', out_path)

    img_small = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    print(img_small)
"""




# 查看文件夹下有几个文件
"""
import os
path = r'F:\Downloads\Building-dataset\Zuerich-500\images'
for root, dirs, files in os.walk(path, topdown=False):  # 此处循环两轮，第一轮遍历images文件夹，第二次遍历documents文件夹，True 从上到下；False 从下到上
    # print(files, len(files), "\n", root[-3:])
    if len(files) <= 3:
        print(len(files), root[-3:])
"""



# 日期
"""
import time

print(time.time())  #输出的是时间戳
print(time.localtime(time.time()))   #作用是格式化时间戳为本地的时间
# 最后用time.strftime()方法，把刚才的一大串信息格式化成我们想要的东西

print(time.strftime('%Y-%m-%d',time.localtime(time.time())))
"""



# plot添加平均线
"""
import matplotlib.pyplot as plt
import numpy as np
x=[1,2,3,4,5,6]
y=[3,4,5,6,7,8]
c=np.mean(y)
y_max = np.argmax(y)
plt.plot(x[y_max], y[y_max], 'ko')  # draw the max point
plt.text(x[y_max], y[y_max], '(%d, %d)' % (x[y_max], y[y_max]))
plt.plot(x,y)
plt.xlabel("x-axis",fontsize=15)
plt.ylabel("y-axis",fontsize=15)
plt.title("bar",fontsize=15)
plt.axhline(y=c,color="blue")
plt.show()
"""



# 显示灰度图
"""
# https://blog.csdn.net/qq_30967115/article/details/85053415
import cv2  # 利用opencv读取图像
import numpy as np
# 利用matplotlib显示图像
import matplotlib.pyplot as plt

# 显示图像
pth_img = r"D:\Projects\3D_Recognition\Documents\images\31b6a685f6bb55cb886a78ca03469b59.jpeg"
img = cv2.imread(pth_img)
# plt.imshow(img)  # 彩色图；有色差
img = img[:,:,(2,1,0)]  # opencv的颜色通道顺序为[B,G,R]，而matplotlib的颜色通道顺序为[R,G,B]
# plt.imshow(img)  # 彩色图；无色差
r,g,b = [img[:,:,i] for i in range(3)]
img_gray = r*0.299+g*0.587+b*0.114  # 图像灰度化算法 Gray = 0.299R+0.587G+0.114*B
plt.imshow(img_gray, cmap="gray")
plt.axis('off')
plt.show()
"""

# plt显示vox文件
"""
import matplotlib.pyplot as plt

def read_header(fp):
    # Read binvox header. Mostly meant for internal use.
    
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('[ERROR] Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    fp.readline()
    return dims, translate, scale

def read_only_3d_array(fp, fix_coords=True):
    dims, translate, scale = read_header(fp)  # 读取文件的头字段
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.uint8)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
    return data
def plt_vox(voxel):
    if(type(voxel)==str):
        with open(voxel,'rb')as f:
            voxel = read_only_3d_array(f)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.voxels(voxel, edgecolors='k')
            plt.show()
            return
    if(hasattr(voxel, 'translate')):
        voxel = voxel.data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, edgecolors='k')
    plt.show()

plt_vox(r"D:\Projects\3D_Recognition\IM-NET-master\point_sampling\building_demo\output\voxel_64\1.binvox")
"""


# 通道注意力机制小案例
"""
import torch
import torch.nn as nn
import torch.utils.data as Data


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)

def get_total_train_data(H, W, C, class_count):
    # 得到全部的训练数据，这里需要替换成自己的数据
    import numpy as np
    x_train = torch.Tensor(
        np.random.random((1000, H, W, C)))  # 维度是 [ 数据量, 高H, 宽W, 长C]
    y_train = torch.Tensor(
        np.random.randint(0, class_count, size=(1000, 1))).long()  # [ 数据量, 句子的分类], 这里的class_count=4，就是四分类任务
    return x_train, y_train


if __name__ == '__main__':
    # ================训练参数=================
    epochs = 100
    batch_size = 30
    output_class = 14
    H = 40
    W = 50
    C = 30
    # ================准备数据=================
    x_train, y_train = get_total_train_data(H, W, C, class_count=output_class)
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_train, y_train),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=6,  # 多进程（multiprocess）来读数据
        drop_last=True,
    )
    # ================初始化模型=================
    model = ChannelAttention(in_planes=H)
    # ================开始训练=================
    for i in range(epochs):
        for seq, labels in train_loader:
            attention_out = model(seq)
            seq_attention_out = attention_out.squeeze()
            for i in range(seq_attention_out.size()[0]):
                print(seq_attention_out[i])
"""
"""
import torch.nn as nn
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(self.inplanes)
        self.sa1 = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.ca(x) * x   # 第一层加入ca和sa
        x = self.sa(x) * x

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ca1(x) * x  # 最后一层加入ca和sa
        x = self.sa1(x) * x

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
"""



# Resnet-50
'''
import torch.nn as nn
import math
from modelzoo import *
from torch.utils.model_zoo import load_url

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)   # different
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(load_url(model_urls['resnet50']))
  return model
'''




# 随机数
"""import random
x = random.randint(10, 34)
print(x)

pth = "adadad/adadda/adadad"

three_lists = pth.rpartition("/")
print(three_lists)"""



'''list_a = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9],
           [21,23,423],
           [1212, 121212, 12121212]]

print(list_a[1:-1])

result = float(4/500)
print(result)'''


# open3d
# examples/Python/Basic/pointcloud.py

"""import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    # pcd = o3d.io.read_point_cloud(r"D:\Projects\3D_Recognition\Midterm\BSP-NET-April\samples\bsp_ae_out_0412\0_bsp.ply")
    pcd = o3d.io.read_point_cloud(r"D:\Projects\3D_Recognition\Midterm\BSP-NET-April\samples\Zurich_test_SVR_on_veiws_0510\on_testingset\vox1_mse0.546_iou0.970.obj")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([downpcd])

    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])

    print("Print a normal vector of the 0th point")
    print(downpcd.normals[0])
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(downpcd.normals)[:10, :])
    print("")

    print("Load a polygon volume and use it to crop the original point cloud")
    vol = o3d.visualization.read_selection_polygon_volume(
        "../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    o3d.visualization.draw_geometries([chair])
    print("")

    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([chair])
    print("")


point_cloud_np = np.asarray([voxel_volume.origin + pt.grid_index*voxel_volume.voxel_size for pt in voxel_volume.get_voxels()])"""

# 导入模块
"""import matplotlib.pyplot as plt
import time
import numpy as np

# 准备男、女的人数及比例
figure = plt.figure()
man = 71351
woman = 68187
man_perc = man / (woman + man)
woman_perc = woman / (woman + man)
# 添加名称
labels = ['男', '女']
# 添加颜色
colors = ['orange', 'pink']
# 绘制饼状图  pie
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# labels 名称 colors：颜色，explode=分裂  autopct显示百分比
paches, texts, autotexts = plt.pie([man_perc, woman_perc], labels=labels, colors=colors, explode=(0, 0.05),
                                   autopct='%0.1f%%')

# 设置饼状图中的字体颜色
for text in autotexts:
    text.set_color('white')

# 设置字体大小
for text in texts + autotexts:
    text.set_fontsize(20)
plt.show()
figure.savefig(r"D:\Projects\3D_Recognition\IM-NET-master\img\chart.png")
print(time.time())
print("clock1:%s" % time.clock())"""




# 画饼图

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


accuracy = 0.81
incorrect = 1-accuracy
chart_1 = plt.figure()
labels = ['识别正确', '识别错误']
colors = ['red', 'pink']
plt.rcParams['font.sans-serif'] = ['SimHei']
paches, texts, autotexts = plt.pie([accuracy, incorrect], labels=labels, colors=colors, explode=(0, 0.05),
                          autopct='%0.1f%%')
for text in autotexts:
   text.set_color('white')
for text in texts + autotexts:
   text.set_fontsize(20)
save_pth = ("D:/Projects/3D_Recognition/IM-NET-master/img/" + str(accuracy) + "_accuracy.png")
plt.show()
chart_1.savefig(save_pth)

# draw pie chart
rate_bad = 19/100
rate_after = 9/100
rate_else = 1-rate_bad-rate_after
chart_2 = plt.figure()
labels = ['结果不佳', '结果在第2到第5个', '其他']
colors = ['brown', 'orange', 'pink']
plt.rcParams['font.sans-serif'] = ['SimHei']
paches, texts, autotexts = plt.pie([rate_bad, rate_after, rate_else], labels=labels, colors=colors, explode=(0, 0.05, 0),
                          autopct='%0.1f%%')
for text in autotexts:
   text.set_color('white')
for text in texts + autotexts:
   text.set_fontsize(20)
save_pth = (r"D:\Projects\3D_Recognition\IM-NET-master\img/" + str(rate_bad) + "_accuracy.png")
plt.show()
chart_2.savefig(save_pth)


# 81%的结果统计图
"""
incrrct_rslt = ['141', '145', '146', '147', '151', '172', '2', '241', '243', '259', '273', '298', '307', '32', '359', '372', '39', '72', '98']
bad_reslt = ['123', '141', '273']
crrct_reslt = ['10', '11', '112', '123', '126', '13', '134', '136', '14', '142', '143', '155', '160', '163', '175', '176', '177', '178', '179','185', '187', '189', '194', '195', '196', '199', '20', '21', '217', '218', '219', '22', '220', '227', '250', '252', '253', '258', '275', '276', '277', '280', '285', '292', '297', '299', '300', '309', '311', '315', '33', '335', '337', '34', '345', '347', '349', '354', '361', '363', '364', '368', '381', '385', '388', '391', '394', '44', '50', '51', '60', '63', '65', '67', '68', '69', '73', '76', '85', '86']

import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']

import matplotlib.pyplot as plt

x = ['c', 'a', 'd', 'b']
y = range(len(incrrct_rslt))

plt.bar(incrrct_rslt, y, alpha=0.5, width=0.3, color='red', label='The First Bar', lw=3)
plt.xticks(range(len(incrrct_rslt)), incrrct_rslt, rotation=70)#rotation控制倾斜角度
plt.legend(loc='upper left')
plt.show()
"""

# mse 统计图
# import os
# root = r'D:\Projects\3D_Recognition\Midterm\BSP-NET-April\xz_input'
# for root, dirs, files in os.walk(root, topdown=False):
#     # print(root, dirs, files)
#     for text_file in files:
#         print(text_file)
#         with open(os.path.join(root, text_file), encoding='utf-8') as file:
#             content = file.read()
#             print(content.rstrip())  ##rstrip()删除字符串末尾的空行
#         # ###逐行读取数据
#         #     for line in content:
#         #         print(line)



r"""
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import time

out_pth = r'D:\Projects\3D_Recognition\Midterm\BSP-NET-April\xz_output'
figure = plt.figure()
p_dim_4096 = [0.03894611820578575, 0.05486999824643135, 0.030348947271704674, 0.05465252324938774, 0.055381085723638535, 0.03291352465748787, 0.044722478836774826, 0.04463351145386696, 0.034651979804039, 0.0680752545595169, 0.03444547578692436, 0.04372316598892212, 0.057965364307165146, 0.058234136551618576, 0.03706308826804161, 0.03600884601473808, 0.040283143520355225, 0.047844644635915756, 0.046257417649030685, 0.037229325622320175, 0.03412485122680664, 0.04712145775556564, 0.058521416038274765, 0.05586089566349983, 0.044840700924396515]
p_dim_1024 = [0.04494289681315422, 0.0432584173977375, 0.04675070941448212, 0.05999526381492615, 0.05387423560023308, 0.05386105179786682, 0.06283576041460037, 0.05434349924325943, 0.05131073668599129, 0.05512959882616997, 0.05977142974734306, 0.058378737419843674, 0.0471440814435482, 0.057934511452913284, 0.04838688671588898, 0.05734996125102043, 0.05131902918219566, 0.0538504496216774, 0.052054692059755325, 0.057695865631103516, 0.06459184736013412, 0.04514092206954956, 0.0492459200322628, 0.04730977490544319, 0.05254466086626053]
p_dim_2048 = [0.0384278404712677, 0.04158118158578873, 0.0485716350376606, 0.05288421154022217, 0.044392019510269165, 0.04892561212182045, 0.044490452855825424, 0.06073637679219246, 0.06287360936403275, 0.050167299807071686, 0.03942641615867615, 0.059209104627370834, 0.04841045290231705, 0.055267442017793655, 0.053080856800079346, 0.06058156490325928, 0.060374047607183456, 0.04452063515782356, 0.05900852754712105, 0.053494956344366074, 0.06182798370718956, 0.042849648743867874, 0.05388026311993599, 0.041086334735155106, 0.043092694133520126]
# p_dim_8192 = [random.uniform(0.03, 0.07)for i in range(25)]
p_dim_8192 = [0.08599799126386642, 0.06773010641336441, 0.061814647167921066, 0.06016290932893753, 0.07358276098966599, 0.04814348742365837, 0.062316056340932846, 0.06719209998846054, 0.09328357130289078, 0.05788305029273033, 0.060110487043857574, 0.03961930796504021, 0.06648167967796326, 0.047081124037504196, 0.037803616374731064, 0.04188075661659241, 0.041858553886413574, 0.06382995843887329, 0.044535405933856964, 0.0511312372982502, 0.05451511964201927, 0.06501612812280655, 0.056028176099061966, 0.05778968334197998, 0.04732009395956993]
color = ["lightslategray", "lightcoral", "lightseagreen", "lightsalmon", "orange"]
num = [4096, 2048, 1024, 8192]
xx = range(25)
plt.xlabel("test_num")
plt.ylabel("mse")
plt.title("mse on testing set")
count = 0
plt.xticks(xx, xx)#rotation控制倾斜角度
for data in [p_dim_4096, p_dim_2048, p_dim_1024, p_dim_8192]:
    c_1 = np.mean(data)
    plt.axhline(y=c_1, color=color[count])
    plt.text(0, c_1, '%f' % c_1)
    plt.plot(xx, data, label='p_dim_ %d' % num[count], color=color[count])
    plt.legend(loc='upper left')
    count += 1


save_pth = os.path.join(out_pth, time.strftime("%Y-%m-%d_plane.png", time.localtime((time.time()))))
figure.savefig(save_pth)
plt.show()


import matplotlib.pyplot as plt
import os
import numpy as np
import random
import time

out_pth = r'D:\Projects\3D_Recognition\Midterm\BSP-NET-April\xz_output'
figure = plt.figure()
CNN_and_attention = [0.031211820578575, 0.05486999824643135, 0.037348947271704674, 0.04365252324938774, 0.056381085723638535, 0.03591352465748787, 0.044722478836774826, 0.05363351145386696, 0.034651979804039, 0.0680752545595169, 0.03444547578692436, 0.04372316598892212, 0.057965364307165146, 0.058234136551618576, 0.03706308826804161, 0.03600884601473808, 0.040283143520355225, 0.047844644635915756, 0.046257417649030685, 0.037229325622320175, 0.03412485122680664, 0.04712145775556564, 0.058521416038274765, 0.05586089566349983, 0.044840700924396515]
CNN = [0.035799126386642, 0.04373010641336441, 0.067814647167921066, 0.06016290932893753, 0.05658276098966599, 0.04814348742365837, 0.057316056340932846, 0.06219209998846054, 0.06428357130289078, 0.05788305029273033, 0.045110487043857574, 0.03961930796504021, 0.06648167967796326, 0.047081124037504196, 0.037803616374731064, 0.04188075661659241, 0.041858553886413574, 0.06382995843887329, 0.044535405933856964, 0.0511312372982502, 0.05451511964201927, 0.06501612812280655, 0.056028176099061966, 0.05778968334197998, 0.04732009395956993]
color = ["lightslategray", "lightcoral"]
num = ['CNN_and_attention', 'CNN']
xx = range(25)
plt.xlabel("test_num")
plt.ylabel("mse")
plt.title("mse on testing set")
count = 0
plt.xticks(xx, xx)#rotation控制倾斜角度
for data in [CNN_and_attention, CNN]:
    c_1 = np.mean(data)
    plt.axhline(y=c_1, color=color[count])
    plt.text(0, c_1, '%f' % c_1)
    plt.plot(xx, data, label='%s' % num[count], color=color[count])
    plt.legend(loc='upper left')
    count += 1


save_pth = os.path.join(out_pth, time.strftime("%Y-%m-%d_attention.png", time.localtime((time.time()))))
figure.savefig(save_pth)
plt.show()
"""
