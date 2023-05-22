# 18, Dec
## completed：
1. 完成airplane的数据预处理
   - 最后生成的文件夹中有8个文件：
      - 02691156_img.hdf5
      - statistics.txt
      - 02691156_vox256.hdf5
      - 02691156_vox256.txt
      - 02691156_vox256_img_test.hdf5
      - 02691156_vox256_img_test.txt
      - 02691156_vox256_img_train.hdf5
      - 02691156_vox256_img_train.txt
## notes：
1. point_sampling
   1. 1_check_hsp_mat.py
      0. 执行时间几十分钟
      1. 检查是否都是能加载的.mat体素文件
   2. 2_gather_256vox_16_32_64.py
      0. 执行一个类别用了一晚上
   3. 2_test_hdf5.py
      0. 执行速度很快
   4. 3_gather_img.py
      0. 执行一个类别用了近7h
2. os模块：
   0. 对目录和文件进行处理
   1. os.walk():遍历文件夹
      - for root, dirs, files in os.walk(r'Documents', topdown=False):
      - root 所指的是当前正在遍历的这个文件夹的本身的地址
      - dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
      - files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
   2. os.path():
      - os.path.split(path) 把路径分割成 dirname 和 basename，返回一个元组
      - os.path.splitext(path) 分割路径，返回路径名和文件扩展名的元组
      - os.path.isfile(path)	判断路径是否为文件
      - os.path.relpath(path[, start])	从start开始计算相对路径
## errors：

# 19, Dec
## completed:
1. 了解IM-NET采样原理：
    - 根据IM-NET-data-preparation.pdf
    - 与二维像素采样比较，三维体素采样对形状边界采样更密集：目的是减少总采样点数
## notes:
1. 多进程：
    - Process 创建进程：p = multiprocessing.Process
    - start 启动进程：p.start()
2. obj文件：
    ~~~ python
      # 这个就相当于C++代码里面的//，如果一行开始时#，那么就可以理解为这一行完全是注释，解析的时候可以无视
      g 这个应该是geometry的缩写，代表一个网格，后面的是网格的名字。
      v v是Vertex的缩写，很简单，代表一个顶点的局部坐标系中的坐标，可以有三个到四个分量。我之分析了三个分量，因为对于正常的三角形的网格来说，第四个分量是1，可以作为默认情况忽略。如果不是1，那可能这个顶点是自由曲面的参数顶点，这个我们这里就不分析了，因为大部分的程序都是用三角形的。
      vn 这个是Vertex Normal，就是代表法线，这些向量都是单位的，我们可以默认为生成这个obj文件的软件帮我们做了单位化。
      vt  这个是Vertex Texture Coordinate，就是纹理坐标了，一般是两个，当然也可能是一个或者三个，这里我之分析两个的情况。
      mtllib <matFileName> 这个代表后面的名字是一个材质描述文件的名字，可以根据后面的名字去找相应的文件然后解析材质。
      usemtl <matName> 这里是说应用名字为matName的材质，后面所有描述的面都是用这个材质，直到下一个usemtl。
      f 这里就是face了，真正描述面的关键字。后面会跟一些索引。一般索引的数量是三个，也可能是四个（OpenGL里面可以直接渲染四边形，Dx的话只能分成两个三角形来渲染了）。每个索引数据中可能会有顶点索引，法线索引，纹理坐标索引，以/分隔。
## errors:

# 20, Dec
## completed:
1. BSP-NET第一次训练：
- 每阶段500轮
- 共四阶段
- phase 0 -> phase 0 -> phase 0 -> phase 1
## notes:
1. 动态查看内存：
    - nvidia-smi
    
## errors：
1. ubuntu内存不够问题，一般是使用策略问题，实际上够用
    - cat /proc/meminfo | grep Commit 查看是否真的够用：https://blog.csdn.net/wolfcode_cn/article/details/82665717
    - cat /proc/sys/vm/overcommit_memory 查看是否为0:https://blog.csdn.net/wolfcode_cn/article/details/82665717
    - su root：进入root
    - 如果认证失败先初始化：https://blog.csdn.net/qq_41076734/article/details/79518361
    - echo 1 > /proc/sys/vm/overcommit_memory 改成模式1：https://blog.csdn.net/WILDCHAP_/article/details/107558421
    - Ctrl + D 退出root

# 21, Dec
## completed:
1. 训练模型：
   - iteration: 8000000
   - phase: 0
   - epoch: 2472
   - loss_sp: 0.000069
   - loss_total: 0.007163
   - time: 71742.3196(~20h)
   - output:
      - 16*2+16 .ply; 
      - BSP_AE.model-16: .data-00000-of-00001; .index; .meta
      - BSP_AE.model-2459: .data-00000-of-00001; .index; .meta
      - 02691156_vox256_img_train_z.hdf5
      - checkpoint
2. 转化obj为mat
   - 可以使用matlab查看obj和mat；
   - 了解obj文件的结构：v；vn；vt；f
   - 了解mat文件的结构：
        - 'b'：(238, 16, 16, 16) 都是0或1; 
        - 'bi'：(16, 16, 16) 有1和其他大于1的整数;
        - 不明白信息含义及怎样转化成256*256*256矩阵
   - 经检验，voxel_model_256是体素矩阵；换方案为转obj为体素
## notes:
1. 采样步骤：
   1. 先将体素文件存放在两个矩阵中，并新建256*256*256的空矩阵
   2. 将体素的值扩充到256*256*256矩阵中：每个体素扩充17倍
   3. 规定一个大小为256*256*256的立方体边界：包含左边界；前边界；和256矩阵决定的上边界
2. 不省略地打印np数组：
    - import numpy
    - import sys
    - numpy.set_printoptions(threshold=sys.maxsize)
## errors:

# 22, Dec
## completed:
1. 完成第二次airplane模型训练
2. 使用binvox将obj转成binvox格式
    - python读取binvox文件
    - 将binvox文件存入voxel_model_256
## notes:

# 23, Dec
## completed：
1. 拷贝了苏黎世和多特蒙德building数据集
    - 其中苏黎世的models中缺少237~239的obj文件，所以一共538-3=535个obj
    - 苏黎世的images中237~239是空文件夹，那没事了。综上，苏黎世一共535个building
    - 多特蒙德-100building的images中75号有三个，其他都是100个
## notes：
## errors：

# 24, Dec
## completed:
1. 批处理obj转binvox文件
    - 选择转成256*256*256，这样可以直接使用IMNET的预处理程序
    - 但是在排序时出问题，10在2前面等等
2. 
## notes:
1. windows批处理：
    1. for循环：
        - 参数：D L R F
        - D: 仅对目录而不是文件执行for命令
        - R: 参数之后可带盘符及路径
        - L: 后面接(起始值，每次增值，结束时的比较值)等差数列
        - F: 打开（集）里的文件
2. binvox：
    1. viewvox操作：
    ~~~cmd
       鼠标：
        左=自由旋转视角
        中=平移
        右=缩放
    
       按键：
        r=重置视图
        箭头键=沿x（左、右）或y（上、下）移动1个体素步长
        =，-=沿z方向移动1个体素步长
        q=退出
        a=切换交替颜色
        p=在正交投影和透视投影之间切换
        x、 y，z=设置相机向下看x、y或z轴
        X、 Y，Z=设置相机向上看X、Y或Z轴
        1=切换显示x、y和z坐标
        s=显示单个切片
        n=显示两个/上/下切片相邻点
        t=切换邻居透明度
        j=向下移动切片
        k=向上移动切片
        g=在切片级别切换显示网格
    ~~~
   
    2. binvox命令参数：
    ~~~cmd 
        用法：
        binvox[-d<体素维度>][-t<体素文件类型>][-c][-v]<文件路径>
        -license：显示软件许可证
        -d：指定体素网格大小（默认256，最大512）
        -t：指定体素文件类型（默认binvox，也支持：hips、mira、vtk、raw、schematic、msh、nrrd）
        -c：z-buffer based carving method only(仅基于z缓冲区的雕刻方法(?))
        -dc：扩展雕刻，在相交前停止雕刻1个体素
        -v：z-buffer based parity voting method only(仅限基于z缓冲区的奇偶校验投票方法(?))（默认同时使用-c和-v）
        -e：精确体素化（设置与凸多边形相交的任何体素）（不使用图形卡graphics card）
        
        其他参数：
        -bb<minx><miny><minz><maxx><maxy><maxz>：强制使用不同的输入模型边界框
        -ri：移除内部体素
        -cb：将模型放置于单位立方体内部中心位置
        -rotx：在体素化之前，将对象绕x轴逆时针旋转90度
        -rotz：在体素化之前，将对象围绕z轴顺时针旋转90度
        -rotx和-rotz可以多次使用
        -nf<value>：使用标准化因子<value>（默认值1.0）
        -aw: 在线框图中渲染模型（有助于处理薄零件）
        -fit：仅在体素边界框中写入体素
        -bi<id>：转换为原理图时，使用块id<id>
        -mb：使用-e从.obj转换时。obj到原理图，从材料规格“usemtl blockid_uId>”解析块ID（仅允许ID范围1-255）
        -down：将每个维度中的体素向下采样2倍（可多次使用）
        -dmin<nr>：when downsampling, destination voxel is on if >= <nr> source voxels are (default 4)(下采样时，如果大于等于<nr>源体素为（默认值4），则目标体素处于启用状态(?))
        
        支持的三维模型文件格式：
        VRML v2.0: 几乎完全支持
        UG、OBJ、OFF、DXF、XGL、POV、BREP、PLY、JOT：仅支持多边形
        
        例子：
        binvox -c -d 200 -t mira plane.wrl
    ~~~
       
## errors:
1. '.' 不是内部或外部命令，也不是可运行的程序：
    - 把./xxx.bat换成.\xxx.bat
    
# 30, Dec
## completed:
1. 尝试用3_gather_img.py生成00000000_img.hdf5
    - 但是执行速度过快，明显有问题
    - 尝试读取hdf5文件的内容，比较building和airplane，对比哪个hdf5是有问题的
    - 在lab里打印00000000_vox256_hdf5.txt，三个采样的格式无明显错误
    - 但是00000000_img.hdf5中却是pixels: []

## notes:
## errors:


# 2, Feb
## completed:
## notes:
1. 使用binvox讲obj文件转化成体素的时候最小体素个数为64
    - 如果使用32和16会强行终止
    - 可能与具体的obj文件有关？
## errors:

# 6, Feb
## completed:
## notes:
1. 透明图对应的通道读取：
    1. 一个使用16位存储的图片，可能5位表示红色，5位表示绿色，5位表示蓝色，1位是阿尔法。
## errors&tips:
1. 查明00000000_img_hdf5.txt为空的原因
    1. 文件路径与代码中不符合，所以无法进入遍历
        - 更改路径后解决
    2. 由于building的多视角图片没有按照标准的命名方式（01.png，02.png……）所以路径中找不到图片
        - 尝试读取图片的文件名称并存放在列表里: 使用os.walk()解决
    3. 显示img[]列表的规格与索引不符
        - 尝试查看airplane的img[]列表规格
        - 使用ubuntu执行airplane的采样第三步
        - airplane使用cv2.imread()是（137，137，4）
        - building使用cv2.imread()是（298，298，3）:按理说彩色图片应该是RGB三个通道，不知道为什么airplane读了四个
            - cv2.IMREAD_UNCHANGED # 读取结果为图片本身维度，如透明图为4维数组，彩色图为3维，黑白图为2维
        - building的多视角图片既没有固定的视角个数也没有固定的分辨率规格，所以不能在第三步设置num_view和view_size
            - 而且为什么两个苏黎世数据集的视角图片个数不一样呢？在小u盘里的caps和00000000
2. 理解airplane对应文件的内容含义
3. u盘出现插入读不到的现象（右下角有图标但无盘符）
    - 多次插拔解决，第一时间转移资料


# 10, Feb
## completed：
1. 苏黎世数据集有以下缺点：
    - 图片分辨率不统一
    - building包含背景没有扣成透明图
    - 视角图片个数不统一
    - 有空文件夹
    - 质量有限（遮挡，不位于图片中心等）
    - 同一个文件夹下不同建筑物
2. 多特蒙德-100数据集：缺少视角图片，可以用模型手动截视角图片？
## notes：
1. 通过设置hdf5_file.create_dataset的view_size为较大值解决列表shape不兼容问题，并修改了hdf5_file["pixels"][] = img


## 2, Mar
## completed:
1. 使用苏黎世mini_dataset训练模型，下午三点开始，还在训练中
## tips：
1. 如果出现GPU内存不足的情况需要在终端使用:
    - fuser -v /dev/nvidia0
    - 查看是否有进程占用内存
    - 如果有python程序则关闭pycharm再打开即可
2. 在modelAE.py程序中：
    - 有shape_batch_size参数，原本是16但因为我的训练样本很少，当shape_num/shape_batch_size的时候分母为零会报错
    - 设置shape_batch_size为4的时候batch_num为4，解决了batch_num循环问题
    
## 3, Mar
## completed:
1. 20220303模型分析：
    1. 训练参数：
        - 用phase-0训练16，32，64规格各80000轮（因为有16个训练样本，所以整体是5000轮）
        - 用phase-1训练64规格80000轮
        - 训练单视图输入，提取相应代码
        - 耗时：两个多小时
        - 可视化结果保存间隔：10 epochs
        - checkpoint文件保存间隔：20 epochs
    2. 评估结果（64）：
        - 测试集平均CD（倒角距离）：0.0054999765125
        - 测试集各自CD：0.005090516； 0.0056006345； 0.0013534443； 0.009955311
## tips:
1. 评估过程：
    1. 不确定倒角距离是否需要乘以1000作为最后结果；
    2. .plt文件中没有读取到法向量的值，所以不能计算法向量的误差，不清楚是不是.plt文件选错了


# 4, Mar
## completed:
1. 20220304模型分析：
    1. 训练参数：
        - 用phase-0-1-2-3-4训练64规格80000轮（因为有16个训练样本，所以整体是5000轮）
        - 用phase-0-0-0-0-4训练32规格80000轮
        - 训练单视图输入，提取相应代码
        - 耗时：15：50 - 19：40  四个小时
        - 可视化结果保存间隔：50 epochs
        - checkpoint文件保存间隔：20 epochs
    2. 评估结果（64）：
        - 测试集平均CD（倒角距离）：
        - 测试集各自CD：
2. 20220305模型分析：
    1. 训练参数：
        - 用phase-0-3训练64规格80000轮（因为有16个训练样本，所以整体是5000轮）
        - 耗时：15：30 - 17: 02
        - 可视化结果保存间隔：50 epochs
        - checkpoint文件保存间隔：20 epochs
    2. 评估结果（64）：
        1. after phase-0：
            - 测试集平均CD（倒角距离）：0.012203640537336469
            - 测试集各自CD：0.005821651；0.004275405；0.012462886；0.02625462
        2. after phase-3：
            - 测试集平均CD（倒角距离）：0.008548823650926352
            - 测试集各自CD：0.0062189763；0.005398132；0.0011876961；0.02139049
            - 可见经过phase-3之后效果有变好
## tips:
1. 几个问题：
    1. 为什么经过phase0和phase3训练之后结果不是好
       - 可以提高epoch和样本数试试
    2. 为什么plt文件中没有evaluate步骤需要的法向量信息
    3. loss_sp和loss_total的关系
       - loss_sp是倒角距离；loss_total应该是倒角距离与L（overlap）的和
       - 如果对应phase不包含L（overlap）两者应该相同：phase-1
    4. loss与倒角距离的关系
    
# 5, Mar
## completed:
1. 20220306模型分析：
    1. 训练参数：
        - 用phase-0-4训练64规格800000轮（因为有16个训练样本，所以整体是50000轮）
        - 训练集：mini_dataset
        - 耗时：22:00 - 11:40  13小时
        - 可视化结果保存间隔：100 epochs
        - checkpoint文件保存间隔：50 epochs
    2. 评估结果（64）：
        1. after phase-0：
            - 测试集平均CD（倒角距离）：
            - 测试集各自CD：
        2. after phase-4：
            - 测试集平均CD（倒角距离）：0.008086681016720831
            - 测试集各自CD：0.009065325; 0.008930169; 0.0016763392; 0.01267489
## tips:
1. 制作middle_dataset
2. 

# 7, Mar
## completed:
## notes:
1. 题目理解：
    - 实现二维图片到三维模型的识别；
    - 两个部分：先实现二维图片的模型重建；再将重建结果在模型库中进行检索和相似度匹配；将相似度最高的模型输出
    - 测试使用的二维图片是陌生图片，但房子是训练集中的房子 
2. 评价指标：   
    - 倒角距离的大小与采样点的个数和位置有关，需要适当的设置
    - 也可以使用IOU作为模型相似度的评估指标
3. 模型检索：
    - 直接比较BSP-NET学习过程中保存的隐层向量与测试图片生成的三维模型的隐层向量的相似程度
    - 也可以另加一个网络实现模型与模型的匹配
4. BSP-NET：
    - 可以通过调整model—AE文件中的超参数改进效果；
    - 训练包括两个部分模型：AE-NET和SVR-NET；前者实现的是三维模型与模型的匹配（？）；后者实现的是单幅图的识别网络
    - 预处理过程可以不用IM-NET的预处理程序，而自己编写，其中使用pillow对图片进行像素转化等处理
    
# 9, Apr
## completed:
1. 制作Zurich_Hundred数据集
2. 整理打包预处理程序

# 10, Apr
## completed:
1. model-0410:
    parameter:
        - dataset: Zurich-Hundred(60+30+10)
        - Epoch: AE-60000 CVR-6000
        - Phase: 0-1 (64*64*64)
        - shape_batch_size: AE-20; CVR-20
        - image_size: 128*128
        - preprocess: create_hdf5
        - svr-out: h3 layer
    result:
        - AE:
            - N/A
        - SVR:
            - loss: 0.00008093
            - mse_mean = 0.17455631670103092
            - time: 1409.4887
   
# 11, Apr
## completed:
0. 问题:
    1. if BSP-NET don't use z_vector as input, why it is an input variable
    


# 20, Apr
## completed:






    
# Notes：
1. Python
    1. 调用其他文件夹中的函数
        - from name-of-file.name-of-py import name-of-function
    2. numpy
        1. np.random.randint: 生成随机整数
        2. array.astype(): 转换数据类型
2. BSP-NET
    1. model_SVR:
        1. BSP_SVR类
            1. 初始化：
                - 输入体素的规格（64）
                - 设置batch_size
                - 设置图片的参数：view_num; crop_size;view_size
                - 设置data路径：checkpoint路径；hdf5文件路径；数据集路径
                - 获取点坐标：使用numpy获取相应规格的坐标数组
                - 使用设备：cuda/cpu
                - 建立模型：
                    - 实例化bsp_network类：输入相应初始化参数（图片体素参数），前向传播函数调用img_encoder, decoder, generator
                    - 优化算法：Adam
                    - 设置输出路径：保存模型
                    - 损失函数：torch.mean
            2. train方法：
                - 加载AE模型
                - 设置训练参数：shape_num从字典长度获取; batch_index_list; epoch
                - 开始训练：
                    1. shuffle(batch_index_list)
                    2. 将之前的numpy数组转化为张量：torch.from_numpy()
                    3. 后向传播
                    4. 优化
                - 测试：间隔若干轮
                - 保存模型：间隔若干轮
            3. test_SVR_on_view(FLAGS, 数据集类型：train/test, 输出层：h3/h2):
                - 加载隐藏向量 z：_z.hdf5
                - 加载图像数据集
                - mapping
                - 加载之前的checkpoint
                - 加载模型数据集用于计算IOU
                - 开始测试：
                    1. h3
                    2. h2
                - 打印测试结果
    2. BSP-NET再认识
        1. AE-NET: 实现了模型的特征提取、再重建的过程
            1. 输入model
            2. model编码成zs向量
            3. zs向量解码成model
            4. 输出model
            5. 生辰_z.hdf5文件：存放每个模型的zs向量（num，256）
        2. SVR-NET: 实现了单视图重建成三维模型的过程
            1. 输入单视角图像
            2. image编码为z向量
            3. 将z向量与上一步中zs向量比较，计算损失函数
            4. 将相似度最高的z向量进行解码（重建）生成model
            5. 输出model
            6. 输出层选择：h2-论文中分割后的mesh；h3-没有分割效果的体素模型
        3. 我的毕设：
            1. 使用AE-NET训练，并保留zs向量与解码模块
            2. 训练SVR-NET提取图片（old building new angle）的向量z
            3. 比较向量z和向量zs
            4. 将相似度最高的zs作为AE-NET解码模块的输入（或者直接保存训练阶段的所有模型，并生成对应编号的字典索引）
            5. 输出模型即为匹配结果模型
            6. 将匹配的准确率作为评价指标：建立库中三维模型与图片序号的对应字典，如果key对应value符合则检索正确
            7. 做快点把其他种类的目标也做了
3. Ubuntu:
    1. 清理内存：
        - nvidia-smi
        - kill -9 进程号（pid）
    2. 打开终端：Ctrl + Alt + T
    3. 查看磁盘空间：
        - df      
        - df -h   显示mb和G为单位
        - df -ha  所有文件系统的磁盘空间
4. PyTorch
    1. Xavier初始化
        - https://blog.csdn.net/shuzfan/article/details/51338178
        - 为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等。基于这个目标，现在我们就去推导一下：每一层的权重应该满足哪种条件。
    2. matmul
# tips:
1. 绝对路径
    - 如果使用相对路径也要用复制的方法，不要自己写，因为如果根目录定位错误会有莫名其妙的问题
    - 通过再文件夹上点右键，可以 mark dir as 将其设置为蓝色的根目录
    - 需要注意新建文件的时候要带上文件名称
    - 如果将某个文件夹设置为根源文件夹，可以从工程下的任意.py文件调用其中的函数
    