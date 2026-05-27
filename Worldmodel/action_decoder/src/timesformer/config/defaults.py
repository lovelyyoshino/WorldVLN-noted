# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""TimeSformer 默认配置。"""
from fvcore.common.config import CfgNode
# -----------------------------------------------------------------------------
# 配置定义
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# BatchNorm 选项
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# 是否使用 Precise BN 统计。
_C.BN.USE_PRECISE_STATS = False

# 用于计算精确 BN 的 batch 数量。
_C.BN.NUM_BATCHES_PRECISE = 200

# 应用于 BN 参数的权重衰减。
_C.BN.WEIGHT_DECAY = 0.0

# Norm 类型，可选 `batchnorm`、`sub_batchnorm`、`sync_batchnorm`。
_C.BN.NORM_TYPE = "batchnorm"

# SubBatchNorm 参数：把 batch 维度切成 NUM_SPLITS 份，并分别独立运行 BN。
_C.BN.NUM_SPLITS = 1

# NaiveSyncBatchNorm3d 参数：同步 `NUM_SYNC_DEVICES` 个设备上的统计量。
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# 训练选项
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# 为 True 时训练模型，否则跳过训练。
_C.TRAIN.ENABLE = True

# 训练数据集名称。
_C.TRAIN.DATASET = "kinetics"

##
_C.TRAIN.FINETUNE = False

# 全局 mini-batch 总大小。
_C.TRAIN.BATCH_SIZE = 64

# 每隔多少个 epoch 在测试数据上评估模型。
_C.TRAIN.EVAL_PERIOD = 10

# 每隔多少个 epoch 保存一次模型检查点。
_C.TRAIN.CHECKPOINT_PERIOD = 10

# 从输出目录中的最新检查点恢复训练。
_C.TRAIN.AUTO_RESUME = True

# 用于加载初始权重的检查点路径。
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# 检查点类型，可选 `caffe2` 或 `pytorch`。
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# 为 True 时，加载检查点时执行 inflation。
_C.TRAIN.CHECKPOINT_INFLATE = False

# 为 True 时，加载检查点后重置 epoch 计数。
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# 如果设置，则按给定模式清理所有层名。
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # 形状/映射说明：("backbone.",)

# ---------------------------------------------------------------------------- #
# 测试选项
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# 为 True 时测试模型，否则跳过测试。
_C.TEST.ENABLE = True

# 测试数据集名称。
_C.TEST.DATASET = "kinetics"

# 测试时的全局 mini-batch 总大小。
_C.TEST.BATCH_SIZE = 8

# 用于加载测试权重的检查点路径。
_C.TEST.CHECKPOINT_FILE_PATH = ""

# 从每个视频中均匀采样多少个片段，用于聚合预测结果。
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# 从每帧的空间维度采样多少个裁剪区域，用于聚合预测结果。
_C.TEST.NUM_SPATIAL_CROPS = 3

# 检查点类型，可选 `caffe2` 或 `pytorch`。
_C.TEST.CHECKPOINT_TYPE = "pytorch"
# 保存预测结果文件的路径。
_C.TEST.SAVE_RESULTS_PATH = ""
# -----------------------------------------------------------------------------
# ResNet 选项
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# ResNet block 中使用的变换函数。
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# 分组数量；ResNet 为 1，ResNeXt 通常大于 1。
_C.RESNET.NUM_GROUPS = 1

# 每个分组的通道宽度（64 -> ResNet；4 -> ResNeXt）。
_C.RESNET.WIDTH_PER_GROUP = 64

# 是否以原地方式执行 ReLU。
_C.RESNET.INPLACE_RELU = True

# 是否把步幅放在 1x1 卷积上。
_C.RESNET.STRIDE_1X1 = False

# 为 True 时，将每个 block 最后一个 BN 的 gamma 初始化为 0。
_C.RESNET.ZERO_INIT_FINAL_BN = False

# 网络权重层数量。
_C.RESNET.DEPTH = 50

# 如果当前 block 数超过 NUM_BLOCK_TEMP_KERNEL，剩余 block 使用 temporal kernel=1。
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# 不同 ResNet stage 的步幅大小。
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# 不同 ResNet stage 的膨胀系数大小。
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# ---------------------------------------------------------------------------- #
# X3D 选项
# X3D 网络细节见 https://arxiv.org/abs/2004.04730。
# ---------------------------------------------------------------------------- #
_C.X3D = CfgNode()

# 宽度扩展因子。
_C.X3D.WIDTH_FACTOR = 1.0

# 深度扩展因子。
_C.X3D.DEPTH_FACTOR = 1.0

# 3x3x3 卷积的 bottleneck 扩展因子。
_C.X3D.BOTTLENECK_FACTOR = 1.0  #

# 分类前最后一个线性层的维度。
_C.X3D.DIM_C5 = 2048

# 第一个 3x3 卷积层的维度。
_C.X3D.DIM_C1 = 12

# 是否缩放 Res2 的宽度，默认 false。
_C.X3D.SCALE_RES2 = False

# 是否在分类器前使用 BatchNorm (BN) 层，默认 false。
_C.X3D.BN_LIN5 = False

# 是否在 residual blocks 中间的 (3x3x3) 卷积使用逐通道（=depthwise）卷积。
_C.X3D.CHANNELWISE_3x3x3 = True

# -----------------------------------------------------------------------------
# Nonlocal 选项
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# 要加入 Nonlocal 层的 stage 和 block 索引。
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# 每个 stage 中 Nonlocal 的分组数量。
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Nonlocal 层使用的实例化方式。
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Nonlocal 中使用的池化层大小。
_C.NONLOCAL.POOL = [
    # 中文说明：Res2
    [[1, 2, 2], [1, 2, 2]],
    # 中文说明：Res3
    [[1, 2, 2], [1, 2, 2]],
    # 中文说明：Res4
    [[1, 2, 2], [1, 2, 2]],
    # 中文说明：Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# 模型选项
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# 模型架构。
_C.MODEL.ARCH = "slowfast"

# 模型名称。
_C.MODEL.MODEL_NAME = "SlowFast"

# 模型要预测的类别数量。
_C.MODEL.NUM_CLASSES = 400

# 损失函数。
_C.MODEL.LOSS_FUNC = "cross_entropy"

# 只有单一路径的模型架构。
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slow", "x3d"]

# 有多条路径的模型架构。
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# 主干网络最终投影前的 dropout 比率。
_C.MODEL.DROPOUT_RATE = 0.5

# Res-block 的随机丢弃比例，会从 res2 到 res5 线性增加。
_C.MODEL.DROPCONNECT_RATE = 0.0

# 初始化 fc 层时使用的标准差。
_C.MODEL.FC_INIT_STD = 0.01

# 输出头的激活层。
_C.MODEL.HEAD_ACT = "softmax"


# -----------------------------------------------------------------------------
# SlowFast 选项
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# 对应 Slow/Fast 路径之间通道缩减比例 $\beta$ 的倒数。
_C.SLOWFAST.BETA_INV = 8

# 对应 Slow/Fast 路径之间帧率缩减比例 $\alpha$。
_C.SLOWFAST.ALPHA = 8

# Slow/Fast 路径之间的通道维度比例。
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# 用于把 Fast pathway 信息融合到 Slow pathway 的卷积核尺寸。
_C.SLOWFAST.FUSION_KERNEL_SZ = 5

####### TimeSformer 选项
_C.TIMESFORMER = CfgNode()
_C.TIMESFORMER.ATTENTION_TYPE = 'divided_space_time'
_C.TIMESFORMER.PRETRAINED_MODEL = ''

## MixUp 参数
_C.MIXUP = CfgNode()
_C.MIXUP.ENABLED = False
_C.MIXUP.ALPHA = 0.8
_C.MIXUP.CUTMIX_ALPHA = 1.0
_C.MIXUP.CUTMIX_MINMAX = None
_C.MIXUP.PROB = 1.0
_C.MIXUP.SWITCH_PROB = 0.5
_C.MIXUP.MODE = 'batch'

_C.EMA = CfgNode()
_C.EMA.ENABLED = False

# -----------------------------------------------------------------------------
# 数据选项
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# 数据目录路径。
_C.DATA.PATH_TO_DATA_DIR = ""

# 路径和标签之间使用的分隔符。
_C.DATA.PATH_LABEL_SEPARATOR = " "

# 视频路径前缀（如果有）。
_C.DATA.PATH_PREFIX = ""

# 输入片段的空间裁剪尺寸。
_C.DATA.CROP_SIZE = 224

# 输入片段的帧数。
_C.DATA.NUM_FRAMES = 8

# 输入片段的视频采样间隔。
_C.DATA.SAMPLING_RATE = 8

# 视频原始像素在 R/G/B 通道上的均值。
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# 输入帧通道维度列表。

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# 视频原始像素在 R/G/B 通道上的标准差。
_C.DATA.STD = [0.225, 0.225, 0.225]

# 训练时空间增强的抖动尺度范围。
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# 训练时的空间裁剪尺寸。
_C.DATA.TRAIN_CROP_SIZE = 224

# 测试时的空间裁剪尺寸。
_C.DATA.TEST_CROP_SIZE = 256

# 输入视频可能有不同 fps，采样帧之前先转换到目标视频 fps。
_C.DATA.TARGET_FPS = 30

# 解码后端，可选 `pyav` 或 `torchvision`。
_C.DATA.DECODING_BACKEND = "pyav"

# 如果为 True，在 [1 / max_scale, 1 / min_scale] 中均匀采样后取倒数得到 scale；
# 如果为 False，直接在 [min_scale, max_scale] 中均匀采样。
_C.DATA.INV_UNIFORM_SAMPLE = False

# 为 True 时，训练期间对视频帧执行随机水平翻转。
_C.DATA.RANDOM_FLIP = True

# 为 True 时，使用 mAP 作为指标。
_C.DATA.MULTI_LABEL = False

# 集成方法，可选 "sum" 和 "max"。
_C.DATA.ENSEMBLE_METHOD = "sum"

# 为 True 时，反转默认输入通道顺序（RBG <-> BGR）。
_C.DATA.REVERSE_INPUT_CHANNEL = False

############
_C.DATA.TEMPORAL_EXTENT = 8
_C.DATA.DEIT_TRANSFORMS = False
_C.DATA.COLOR_JITTER = 0.
_C.DATA.AUTO_AUGMENT = ''
_C.DATA.RE_PROB = 0.0

# ---------------------------------------------------------------------------- #
# 优化器选项
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# 基础学习率。
_C.SOLVER.BASE_LR = 0.1

# 学习率策略（选项和示例见 utils/lr_policy.py）。
_C.SOLVER.LR_POLICY = "cosine"

# 'cosine' 策略的最终学习率。
_C.SOLVER.COSINE_END_LR = 0.0

# 指数衰减因子。
_C.SOLVER.GAMMA = 0.1

# 'exp' 和 'cos' 策略的步长（单位：epoch）。
_C.SOLVER.STEP_SIZE = 1

# 'steps_' 策略的步点（单位：epoch）。
_C.SOLVER.STEPS = []

# 'steps_' 策略对应的学习率列表。
_C.SOLVER.LRS = []

# 最大 epoch 数。
_C.SOLVER.MAX_EPOCH = 300

# 中文说明：Momentum。
_C.SOLVER.MOMENTUM = 0.9

# 中文说明：Momentum dampening。
_C.SOLVER.DAMPENING = 0.0

# 是否使用 Nesterov momentum。
_C.SOLVER.NESTEROV = True

# L2 正则化。
_C.SOLVER.WEIGHT_DECAY = 1e-4

# 预热从 SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR 开始。
_C.SOLVER.WARMUP_FACTOR = 0.1

# 用这些 epoch 逐步预热到 SOLVER.BASE_LR。
_C.SOLVER.WARMUP_EPOCHS = 0.0

# 预热的起始学习率。
_C.SOLVER.WARMUP_START_LR = 0.01

# 优化方法。
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# 基础学习率是否随 NUM_SHARDS 线性缩放。
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# ---------------------------------------------------------------------------- #
# 其他选项
# ---------------------------------------------------------------------------- #

# 使用的 GPU 数量（训练和测试都会用到）。
_C.NUM_GPUS = 1

# 本任务使用的机器数量。
_C.NUM_SHARDS = 1

# 当前机器的索引。
_C.SHARD_ID = 0

# 输出根目录。
_C.OUTPUT_DIR = "./tmp"

# 注意：GPU 算子库中某些算子实现仍可能带来非确定性。
_C.RNG_SEED = 1

# 日志记录间隔（单位：iter）。
_C.LOG_PERIOD = 10

# 为 True 时记录模型信息。
_C.LOG_MODEL_INFO = False

# 分布式后端。
_C.DIST_BACKEND = "nccl"

# 全局 batch 大小。
_C.GLOBAL_BATCH_SIZE = 64

# ---------------------------------------------------------------------------- #
# 基准测试选项
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()

# 数据加载基准测试的 epoch 数。
_C.BENCHMARK.NUM_EPOCHS = 5

# 数据加载基准测试的日志间隔（单位：iter）。
_C.BENCHMARK.LOG_PERIOD = 100

# 为 True 时，基准测试期间每个 epoch 都打乱数据加载器。
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# 训练/测试通用数据加载器选项
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# 每个训练进程使用的数据加载器 worker 数量。
_C.DATA_LOADER.NUM_WORKERS = 8

# 是否把数据加载到固定页主机内存。
_C.DATA_LOADER.PIN_MEMORY = True

# 是否启用多线程解码。
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# ---------------------------------------------------------------------------- #
# 检测选项
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# 是否启用视频检测。
_C.DETECTION.ENABLE = False

# RoI 的对齐版本，更多细节见 slowfast/models/head_helper.py。
_C.DETECTION.ALIGNED = True

# 空间缩放因子。
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI 变换的输出分辨率。
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


# -----------------------------------------------------------------------------
# AVA 数据集选项
# -----------------------------------------------------------------------------
_C.AVA = CfgNode()

# 帧目录路径。
_C.AVA.FRAME_DIR = ""

# 帧列表文件所在目录。
_C.AVA.FRAME_LIST_DIR = (
    ""
)

# 标注文件所在目录。
_C.AVA.ANNOTATION_DIR = (
    ""
)

# 训练样本列表文件名。
_C.AVA.TRAIN_LISTS = ["train.csv"]

# 测试样本列表文件名。
_C.AVA.TEST_LISTS = ["val.csv"]

# 训练用框列表文件名。这里假设包含预测框的文件名会带有
# "predicted_boxes" 后缀。
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]
_C.AVA.TRAIN_PREDICT_BOX_LISTS = []

# 测试用框列表文件名。
_C.AVA.TEST_PREDICT_BOX_LISTS = ["ava_val_predicted_boxes.csv"]

# 使用预测框时的分数阈值。
_C.AVA.DETECTION_SCORE_THRESH = 0.9

# 输入帧是否使用 BGR 格式。
_C.AVA.BGR = False

# 训练增强参数。
# 是否使用颜色增强方法。
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

# 使用颜色增强时，是否只使用 PCA 抖动；否则会与颜色抖动组合使用。
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

# PCA 抖动的特征值。注意 PCA 基于 RGB。
_C.AVA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# PCA 抖动的特征向量。
_C.AVA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# 测试期间是否执行水平翻转。
_C.AVA.TEST_FORCE_FLIP = False

# 是否在验证划分上使用完整测试集。
_C.AVA.FULL_TEST_ON_VAL = False

# AVA 标签映射文件名。
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# AVA 排除列表文件名。
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

# AVA 真值文件名。
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"

# 图像处理后端，包括 `pytorch` 和 `cv2`。
_C.AVA.IMG_PROC_BACKEND = "cv2"

# ---------------------------------------------------------------------------- #
# Multigrid 训练选项
# Multigrid 训练细节见 https://arxiv.org/abs/1912.00998。
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid 训练允许用更少迭代训练更多 epoch。
# 该超参数指定训练 epoch 数相比基线增加多少倍。
# 论文默认设置为基线的 1.5 倍。
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# 是否启用短周期。
_C.MULTIGRID.SHORT_CYCLE = False
# 短周期相对默认裁剪尺寸的额外空间尺寸比例。
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
# 相对默认形状的 (Temporal, Spatial) 维度比例。
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

# 标准 BN 会在单个 GPU 的所有样本上计算统计量；Multigrid 训练中，我们固定用于
# 计算 BN 统计量的片段数量。细节见 https://arxiv.org/abs/1912.00998。
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid 训练的 epoch 与实际训练时间/计算量不成比例，因此 _C.TRAIN.EVAL_PERIOD
# 可能导致评估过于频繁或稀疏。这里使用 multigrid 专用规则决定何时评估：
# 该超参数定义每个长周期形状内评估模型的次数。
_C.MULTIGRID.EVAL_FREQ = 3

# 无需手动指定；会自动设置并作为全局变量使用。
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

# -----------------------------------------------------------------------------
# Tensorboard 可视化选项
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# 写入摘要写入器；训练/评估期间会自动记录 loss、lr 和指标。
_C.TENSORBOARD.ENABLE = False
# 提供用于可视化的预测结果路径。
# 这是包含 [prediction_tensor, label_tensor] 的 pickle 文件。
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Tensorboard 日志目录路径。
# 默认是 cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}。
_C.TENSORBOARD.LOG_DIR = ""
# 提供类别名称到 id 映射的 json 文件路径，格式为
# 中文说明：{"class_name1": id1, "class_name2": id2, ...}。
# 如需按子集或父类别绘制混淆矩阵，必须提供该文件。
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

# 类别组到类别列表映射的 json 文件路径，格式为
# 中文说明：{"parent_class": ["child_class1", "child_class2",...], ...}。
_C.TENSORBOARD.CATEGORIES_PATH = ""

# 混淆矩阵可视化配置。
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
# 是否可视化混淆矩阵。
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
# 绘制混淆矩阵的图像尺寸。
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
# 要可视化的类别子集路径。
# 文件中用换行符分隔类别名称。
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

# 直方图可视化配置。
_C.TENSORBOARD.HISTOGRAM = CfgNode()
# 是否可视化直方图。
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
# 要绘制直方图的类别子集路径。
# 类别名称必须用换行符分隔。
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
# 对每个选定 true label，在直方图上可视化预测最多的 top-k 类。
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
# 绘制直方图的图像尺寸。
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]

# 层的权重和激活可视化配置。
# _C.TENSORBOARD.ENABLE 必须为 True。
_C.TENSORBOARD.MODEL_VIS = CfgNode()

# 如果为 False，跳过模型可视化。
_C.TENSORBOARD.MODEL_VIS.ENABLE = False

# 如果为 False，跳过模型权重可视化。
_C.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = False

# 如果为 False，跳过模型激活可视化。
_C.TENSORBOARD.MODEL_VIS.ACTIVATIONS = False

# 如果为 False，跳过输入视频可视化。
_C.TENSORBOARD.MODEL_VIS.INPUT_VIDEO = False


# 字符串列表，描述要可视化权重和激活的层名及其索引。
# 索引用来选择某一层输出激活的子集进行可视化。
# 如果未指定索引，则可视化该层输出的所有激活。
# 每个字符串中，层名和索引用空白字符分隔。
# 例如：[layer1 1,2;1,2, layer2, layer3 150,151;3,4]；
# 这表示对 `layer1` 中沿 batch 维度的每个数组 `arr`，取 arr[[1, 2], [1, 2]]。
_C.TENSORBOARD.MODEL_VIS.LAYER_LIST = []
# 在视频上绘制的 top-k 预测数量。
_C.TENSORBOARD.MODEL_VIS.TOPK_PREDS = 1
# 文本框和边界框使用的色彩映射。
_C.TENSORBOARD.MODEL_VIS.COLORMAP = "Pastel2"
# 使用 Grad-CAM 可视化视频输入的配置。
# _C.TENSORBOARD.ENABLE 必须为 True。
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM = CfgNode()
# 是否使用 Grad-CAM 技术运行可视化。
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
# Grad-CAM 使用的 CNN 层。层数量必须等于路径数量。
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = []
# 如果为 True，使用每个实例的 true labels 可视化 Grad-CAM。
# 如果为 False，使用预测分数最高的类别。
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
# 文本框和边界框使用的色彩映射。
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP = "viridis"

# 错误预测可视化配置。
# _C.TENSORBOARD.ENABLE 必须为 True。
_C.TENSORBOARD.WRONG_PRED_VIS = CfgNode()
_C.TENSORBOARD.WRONG_PRED_VIS.ENABLE = False
# 用于组织模型评估视频的文件夹标签。
_C.TENSORBOARD.WRONG_PRED_VIS.TAG = "Incorrectly classified videos."
# 要可视化的标签子集。只会可视化 true labels 落在该子集中的错误预测。
_C.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH = ""


# ---------------------------------------------------------------------------- #
# 演示选项
# ---------------------------------------------------------------------------- #
_C.DEMO = CfgNode()

# 是否以 DEMO 模式运行模型。
_C.DEMO.ENABLE = False

# 提供类别名称到 id 映射的 json 文件路径，格式为
# 中文说明：{"class_name1": id1, "class_name2": id2, ...}。
_C.DEMO.LABEL_FILE_PATH = ""

# 指定摄像头设备作为输入；如果设置，会优先于输入视频。
# 如果为 -1，则改用输入视频。
_C.DEMO.WEBCAM = -1

# 演示输入视频路径。
_C.DEMO.INPUT_VIDEO = ""
# 读取输入视频数据时使用的自定义宽度。
_C.DEMO.DISPLAY_WIDTH = 0
# 读取输入视频数据时使用的自定义高度。
_C.DEMO.DISPLAY_HEIGHT = 0
# Detectron2 目标检测模型配置路径，仅检测任务使用。
_C.DEMO.DETECTRON2_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# Detectron2 目标检测模型预训练权重路径。
_C.DEMO.DETECTRON2_WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
# Detectron2 选择预测边界框时使用的阈值。
_C.DEMO.DETECTRON2_THRESH = 0.9
# 两个连续片段之间的重叠帧数。
# 增大该值可获得更频繁的动作预测。
# 重叠帧数不能大于序列长度 `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE` 的一半。
_C.DEMO.BUFFER_SIZE = 0
# 如果指定，可视化输出会写入该路径的视频文件；否则会在窗口中显示。
_C.DEMO.OUTPUT_FILE = ""
# 写入输出视频文件时使用的每秒帧数。
# 如果未设置（-1），使用输入文件的 fps。
_C.DEMO.OUTPUT_FPS = -1
# 演示视频读取器输出的输入格式（"RGB" 或 "BGR"）。
_C.DEMO.INPUT_FORMAT = "BGR"
# 在闭区间 [keyframe_idx - CLIP_VIS_SIZE, keyframe_idx + CLIP_VIS_SIZE] 内绘制可视化帧。
_C.DEMO.CLIP_VIS_SIZE = 10
# 运行视频可视化器的进程数。
_C.DEMO.NUM_VIS_INSTANCES = 2

# 预先计算的预测框路径。
_C.DEMO.PREDS_BOXES = ""
# 是否使用多线程视频读取器运行。
_C.DEMO.THREAD_ENABLE = False
# 每隔 `DEMO.NUM_CLIPS_SKIP` + 1 个片段取一个用于预测和可视化。
# 这会通过降低预测/可视化频率来加快演示速度。
# 如果为 -1，则取最近读取的片段进行可视化。该模式仅在
# `DEMO.THREAD_ENABLE` 为 True 时支持。
_C.DEMO.NUM_CLIPS_SKIP = 0
# 真值框和标签的路径（可选）。
_C.DEMO.GT_BOXES = ""
# 视频相对于边界框文件的起始秒数。
_C.DEMO.STARTING_SECOND = 900
# 输入视频或图像文件夹的 fps。
_C.DEMO.FPS = 30
# 使用 top-k 预测或高于特定阈值的预测进行可视化。
# 选项：{"thres", "top-k"}。
_C.DEMO.VIS_MODE = "thres"
# 常见类别名称的阈值。
_C.DEMO.COMMON_CLASS_THRES = 0.7
# 非常见类别名称的阈值。如果 `_C.DEMO.COMMON_CLASS_NAMES` 为空，则不会使用。
_C.DEMO.UNCOMMON_CLASS_THRES = 0.3
# 该列表根据 AVA 数据集中各类别样本分布选择。
_C.DEMO.COMMON_CLASS_NAMES = [
    "watch (a person)",
    "talk to (e.g., self, a person, a group)",
    "listen to (a person)",
    "touch (an object)",
    "carry/hold (an object)",
    "walk",
    "sit",
    "lie/sleep",
    "bend/bow (at the waist)",
]
# 可视化的慢放倍率。视频中被可视化的部分会比正常速度慢 `_C.DEMO.SLOWMO` 倍播放。
_C.DEMO.SLOWMO = 1

def _assert_and_infer_cfg(cfg):
    """检查配置值是否合法，并根据需要推断派生配置。"""
    # BN 断言。
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN 断言。
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST 断言。
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET 断言。
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # 按 num_shards 执行学习率缩放。
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS

    # 通用断言。
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    获取默认配置的一份拷贝。
    """
    return _assert_and_infer_cfg(_C.clone())
