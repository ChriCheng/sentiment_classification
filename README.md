# 基于 PyTorch 的情感分类实验报告

## 1. 实验目的

本实验旨在完成一个英文五分类情感分类任务，基于预处理后的文本数据实现并比较两类模型：传统卷积神经网络 **TextCNN** 与预训练语言模型 **BERT**。实验围绕数据读取、模型训练、验证集选择、测试集评估与结果分析展开，最终比较不同模型在细粒度情感分类任务上的性能差异。

本实验的具体目标如下：

1. 掌握文本分类任务的基本流程；
2. 熟悉 PyTorch 下文本数据集构建、模型训练与评估方法；
3. 理解传统词向量 + CNN 方法与预训练语言模型方法的差异；
4. 分析五分类细粒度情感任务中的主要难点，并比较不同模型的优劣。

---

## 2. 实验原理

### 2.1 情感分类任务简介

情感分类（Sentiment Classification）是自然语言处理中的经典任务，目标是根据文本内容判断其情感极性。本实验采用五分类设置，将文本划分为：

- 0：very negative
- 1：negative
- 2：neutral
- 3：positive
- 4：very positive

与二分类情感分析相比，五分类任务不仅需要区分正负情感，还需要进一步区分强弱程度，因此分类边界更细，任务难度更高。

### 2.2 TextCNN 模型原理

TextCNN 是经典文本分类模型。其基本流程为：

1. 使用 `Embedding` 层将 token id 映射为稠密向量；
2. 使用多组不同卷积核尺寸提取局部 n-gram 特征；
3. 对各卷积特征执行全局最大池化；
4. 将拼接后的特征输入全连接层，输出五个类别的 logits。

项目中的 TextCNN 由一个词嵌入层、多个二维卷积层、最大池化和全连接分类层构成，卷积核尺寸为 `3/4/5`，每个卷积核提取一类局部语义模式，适合做句子级分类任务。

### 2.3 BERT 模型原理

BERT（Bidirectional Encoder Representations from Transformers）是一类基于 Transformer 编码器的预训练语言模型。它通过大规模语料预训练获得上下文相关的表示，再在下游分类任务上进行微调。

本实验中使用 `AutoModelForSequenceClassification` 加载 `bert-base-uncased`，并在五分类情感数据上进行端到端微调。相较于 TextCNN，BERT 的优势在于：

1. 能够建模双向上下文；
2. 具备更强的语义表达能力；
3. 在小到中等规模数据集上通常具有更好的迁移性能。

### 2.4 损失函数与优化方式

- **TextCNN**：使用 `CrossEntropyLoss` 作为分类损失，优化器为 Adam；
- **BERT**：使用 Hugging Face `AutoModelForSequenceClassification` 内置的交叉熵损失，优化器为 AdamW，并配合线性 warmup 学习率调度。

---

## 3. 数据集与评价指标

### 3.1 数据集说明

本项目使用预处理后的英文情感分类数据，已划分为训练集、验证集和测试集，并额外提供词表映射文件：

- `train.csv`
- `dev.csv`
- `test.csv`
- `tokens2id.csv`
- `id2tokens.csv`

数据规模如下：

| 数据划分 | 样本数 |
| -------- | -----: |
| Train    |   8544 |
| Dev      |   1101 |
| Test     |   2210 |

标签分布整体较为平衡，但 `neutral` 类别区分难度相对更高。

### 3.2 评价指标

本实验主要使用以下指标：

1. **Accuracy**：分类正确样本数占总样本数的比例；
2. **Precision / Recall / F1-score**：从类别层面衡量模型预测质量；
3. **Macro Average**：对各类别指标简单平均，更能体现类别间均衡性；
4. **Weighted Average**：按类别样本数加权平均；
5. **Confusion Matrix**：观察模型在各类别之间的混淆情况。

---

## 4. 实验环境与参数设置

### 4.1 实验环境

- Python 3.10+
- PyTorch
- pandas
- tqdm
- transformers
- scikit-learn
- CUDA（测试日志显示评测使用 `cuda` 设备）

### 4.2 TextCNN 训练参数

| 参数           |    取值 |
| -------------- | ------: |
| 模型           | TextCNN |
| `max_len`      |      48 |
| `embed_dim`    |     128 |
| `num_filters`  |     100 |
| `kernel_sizes` | 3, 4, 5 |
| `dropout`      |     0.5 |
| `batch_size`   |      64 |
| `epochs`       |      15 |
| `lr`           |    1e-3 |
| `weight_decay` |    1e-4 |
| 优化器         |    Adam |

### 4.3 BERT 微调参数

| 参数           |                         取值 |
| -------------- | ---------------------------: |
| 模型           |          `bert-base-uncased` |
| `max_len`      |                           64 |
| `batch_size`   |                           16 |
| `epochs`       |                            4 |
| `lr`           |                         2e-5 |
| `weight_decay` |                         1e-2 |
| `warmup_ratio` |                          0.1 |
| 优化器         |                        AdamW |
| 学习率调度     | Linear Warmup + Linear Decay |

---

## 5. 项目目录结构说明

当前项目目录可整理为如下结构：

```text
sentiment_classification/
|-- README.md                         # 项目说明与实验报告
|
|-- preprocessed_file/                # 预处理后的数据集目录
|   |-- train.csv                     # 训练集
|   |-- dev.csv                       # 验证集
|   |-- test.csv                      # 测试集
|   |-- tokens2id.csv                 # token 到 id 的映射
|   `-- id2tokens.csv                 # id 到 token 的映射
|
|-- checkpoints/                      # 保存训练得到的模型参数
|   |-- textcnn_best.pt               # TextCNN 最优模型
|   `-- bert_best/                    # BERT 最优模型目录
|
`-- src/                             # 核心源码目录
    |-- dataset.py                    # TextCNN 数据集与词表加载代码
    |-- model.py                      # TextCNN 模型定义
    |-- train.py                      # TextCNN 训练脚本
    |-- evaluate.py                   # TextCNN 评测脚本（含分类报告与混淆矩阵）
    |-- bert_dataset.py               # BERT 数据集与 detokenize 逻辑
    |-- bert_train.py                 # BERT 微调训练脚本
    `-- bert_evaluate.py              # BERT 评测脚本（含分类报告与混淆矩阵）
```

---

## 6. 实验流程

### 6.1 TextCNN 实验流程

1. 从 `tokens2id.csv` 读取词表；
2. 从 `train/dev/test.csv` 读取句子与标签；
3. 将句子编码为定长 token id 序列；
4. 输入 TextCNN 进行前向传播；
5. 使用交叉熵损失训练并在验证集上选择最优模型；
6. 在测试集上输出 Accuracy、分类报告与混淆矩阵。

### 6.2 BERT 实验流程

1. 使用 `bert-base-uncased` 的 tokenizer 对文本重新编码；
2. 将预处理 token 序列通过 `detokenize` 恢复为更接近自然文本的形式；
3. 使用 `AutoModelForSequenceClassification` 进行微调训练；
4. 在验证集上保存最优 checkpoint；
5. 在测试集上输出 Accuracy、分类报告与混淆矩阵。

---

## 7. 实验结果

### 7.1 测试集整体结果对比

| 方法    |  Test Loss | Test Accuracy |
| ------- | ---------: | ------------: |
| TextCNN |     1.4399 |        0.3665 |
| BERT    | **1.0272** |    **0.5751** |

相较于 TextCNN，BERT 在测试集上的准确率提升为：

- **绝对提升：+0.2086**
- **即提升约 20.86 个百分点**

说明在该五分类细粒度情感分类任务上，预训练语言模型明显优于传统卷积分类模型。

### 7.2 BERT 测试结果

#### （1）整体指标

- Test Loss：`1.0272`
- Test Accuracy：`0.5751`
- Macro Avg F1：`0.5601`
- Weighted Avg F1：`0.5706`

#### （2）各类别准确率

| 类别          | Accuracy | 正确数 / 总数 |
| ------------- | -------: | ------------: |
| very negative |   0.5161 |     144 / 279 |
| negative      |   0.6288 |     398 / 633 |
| neutral       |   0.3496 |     136 / 389 |
| positive      |   0.6490 |     331 / 510 |
| very positive |   0.6566 |     262 / 399 |

#### （3）分类报告

| label            |  precision |     recall |   f1-score |  support |
| ---------------- | ---------: | ---------: | ---------: | -------: |
| very negative    |     0.5353 |     0.5161 |     0.5255 |      279 |
| negative         |     0.5985 |     0.6288 |     0.6133 |      633 |
| neutral          |     0.4579 |     0.3496 |     0.3965 |      389 |
| positive         |     0.5658 |     0.6490 |     0.6046 |      510 |
| very positive    |     0.6650 |     0.6566 |     0.6608 |      399 |
| **accuracy**     |            |            | **0.5751** | **2210** |
| **macro avg**    | **0.5645** | **0.5600** | **0.5601** | **2210** |
| **weighted avg** | **0.5702** | **0.5751** | **0.5706** | **2210** |

### 7.3 TextCNN 测试结果

#### （1）整体指标

- Test Loss：`1.4399`
- Test Accuracy：`0.3665`
- Macro Avg F1：`0.3021`
- Weighted Avg F1：`0.3342`

#### （2）各类别准确率

| 类别          | Accuracy | 正确数 / 总数 |
| ------------- | -------: | ------------: |
| very negative |   0.1075 |      30 / 279 |
| negative      |   0.5308 |     336 / 633 |
| neutral       |   0.1388 |      54 / 389 |
| positive      |   0.6176 |     315 / 510 |
| very positive |   0.1880 |      75 / 399 |

#### （3）分类报告

| label            |  precision |     recall |   f1-score |  support |
| ---------------- | ---------: | ---------: | ---------: | -------: |
| very negative    |     0.3158 |     0.1075 |     0.1604 |      279 |
| negative         |     0.4153 |     0.5308 |     0.4660 |      633 |
| neutral          |     0.2784 |     0.1388 |     0.1852 |      389 |
| positive         |     0.3316 |     0.6176 |     0.4315 |      510 |
| very positive    |     0.4630 |     0.1880 |     0.2674 |      399 |
| **accuracy**     |            |            | **0.3665** | **2210** |
| **macro avg**    | **0.3608** | **0.3166** | **0.3021** | **2210** |
| **weighted avg** | **0.3679** | **0.3665** | **0.3342** | **2210** |

---

## 8. 结果分析

### 8.1 TextCNN 的表现特点

TextCNN 能够学习一定程度的局部模式，因此在 `negative` 和 `positive` 两类上取得了相对还可以的结果，但在极性更细的 `very negative`、`very positive` 和 `neutral` 类别上表现明显不足。说明在五分类细粒度情感任务中，仅依赖浅层词向量和局部卷积特征难以准确建模复杂语义边界。

### 8.2 BERT 的优势

BERT 明显优于 TextCNN，主要体现在以下几个方面：

1. **更强的上下文表示能力**：能够更好地理解情感触发词在句子中的上下文含义；
2. **更高的类别区分能力**：特别是在 `very positive` 与 `very negative` 这类强极性类别上提升更明显；
3. **更好的整体平衡性**：BERT 的 Macro Avg 与 Weighted Avg 均明显优于 TextCNN，说明其不仅整体精度更高，而且在多类任务中更加稳定。

### 8.3 当前任务的主要难点

从实验结果看，`neutral` 类别始终是最难识别的类别：

- TextCNN：`0.1388`
- BERT：`0.3496`

这说明中性文本与弱正、弱负文本之间存在较强重叠，分类边界不清晰，是该任务的主要难点之一。

### 8.4 方法优缺点总结

#### TextCNN

**优点：**

- 结构简单，易于实现；
- 训练速度快；
- 作为课程作业中的传统深度学习基线具有代表性。

**缺点：**

- 对上下文语义建模能力有限；
- 对细粒度情感类别区分不足；
- 在五分类任务中整体性能偏低。

#### BERT

**优点：**

- 具备强大的预训练语义表示能力；
- 在细粒度情感分类任务上显著优于传统模型；
- 测试集准确率、宏平均指标和类别均衡性均更优。

**缺点：**

- 训练和推理开销较大；
- 对显存和运行环境要求更高；
- 实现复杂度高于传统 TextCNN。

---

## 9. 结论

本实验基于 PyTorch 完成了一个英文五分类情感分类系统，实现并比较了 **TextCNN** 与 **BERT** 两类模型。

实验结果表明：

- TextCNN 在测试集上达到 **0.3665** 的准确率；
- BERT 在测试集上达到 **0.5751** 的准确率；
- BERT 相比 TextCNN 提升 **20.86 个百分点**，表现出明显优势。

因此可以得出结论：对于当前的五分类细粒度情感分类任务，**传统 TextCNN 可以作为有效基线，但最终更适合作为主模型的方案是 BERT 微调方法**。TextCNN 体现了经典深度学习文本分类思路，而 BERT 则通过预训练语言表示显著提高了分类性能。

总体来看，本实验完整实现了：

- 预处理文本数据读取；
- TextCNN 基线训练与测试；
- BERT 微调训练与测试；
- 分类报告、类别准确率与混淆矩阵分析；
- 不同模型在细粒度情感分类任务上的对比实验。

---
## 10. 其他
项目已在github开源：https://github.com/ChriCheng/sentiment_classification

## 11. 参考文献

1. Kim Y. Convolutional Neural Networks for Sentence Classification. EMNLP, 2014.
2. Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019.
3. Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017.