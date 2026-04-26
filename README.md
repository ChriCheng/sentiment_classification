# 基于 PyTorch 的情感分类实验报告

## 1. 实验目的

本实验旨在完成一个英文五分类情感分类任务，围绕电影评论数据集的情感分类问题，实现并比较三类模型：传统卷积神经网络 **TextCNN**、参考 Kim (2014) 思路实现的 **Kim-style CNN-non-static**，以及预训练语言模型 **BERT**。实验围绕数据读取、模型训练、验证集选择、测试集评估与结果分析展开，重点考察在细粒度五分类情感任务中，普通卷积模型、结合句法树短语监督的卷积模型，以及预训练语言模型之间的性能差异。

本实验的具体目标如下：

1. 掌握文本分类任务的基本流程；
2. 熟悉 PyTorch 下文本数据集构建、模型训练与评估方法；
3. 理解传统词向量 + CNN、参考论文改进的卷积模型与预训练语言模型之间的差异；
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

### 2.3 Kim-style CNN-non-static 原理

为了更贴近 Kim (2014) 在 SST-1 上的卷积实验，本实验额外实现了一个参考论文思路的卷积版本，记为 **Kim-style CNN-non-static**。该模型在结构上仍然保持单层卷积神经网络的基本框架，但在训练数据与词向量初始化方式上更接近论文：

1. 使用 `word2vec` 预训练词向量初始化嵌入层；
2. 采用 **non-static** 策略，使嵌入层在下游任务训练过程中继续更新；
3. 使用卷积核尺寸 `3/4/5`，每种卷积核输出 100 个特征图；
4. 继续使用全局最大池化与全连接分类层；
5. 在训练数据上引入 **phrase-level augmentation**，即从 Stanford Sentiment Treebank 原始句法树中抽取训练句子的所有短语节点作为附加监督样本。

其中，phrase-level augmentation 是该版本相较普通 TextCNN 最关键的改进。原始句子级训练集只有 8544 条样本，而扩展后训练集中共包含 318582 条样本，其中句子级样本 8544 条、短语级样本 310038 条。通过引入短语级监督，卷积模型可以学习到更多局部情感模式，从而更接近论文中的卷积神经网络设定。

### 2.4 BERT 模型原理

BERT（Bidirectional Encoder Representations from Transformers）是一类基于 Transformer 编码器的预训练语言模型。它通过大规模语料预训练获得上下文相关的表示，再在下游分类任务上进行微调。

本实验中使用 `AutoModelForSequenceClassification` 加载 `bert-base-uncased`，并在五分类情感数据上进行端到端微调。相较于卷积模型，BERT 的优势在于：

1. 能够建模双向上下文；
2. 具备更强的语义表达能力；
3. 在小到中等规模数据集上通常具有更好的迁移性能。

### 2.5 损失函数与优化方式

- **TextCNN**：使用 `CrossEntropyLoss` 作为分类损失，优化器为 Adam；
- **Kim-style CNN-non-static**：使用 `CrossEntropyLoss`，卷积结构与 TextCNN 类似，但使用 `Adadelta` 优化器、max-norm 约束，并配合 `word2vec` 初始化和 phrase-level 训练数据；
- **BERT**：使用 Hugging Face `AutoModelForSequenceClassification` 内置的交叉熵损失，优化器为 AdamW，并配合线性 warmup 学习率调度。

---

## 3. 数据集与评价指标

### 3.1 数据集说明

本项目主要使用预处理后的英文情感分类数据，已划分为训练集、验证集和测试集，并额外提供词表映射文件：

- `train.csv`
- `dev.csv`
- `test.csv`
- `tokens2id.csv`
- `id2tokens.csv`

句子级数据规模如下：

| 数据划分 | 样本数 |
| -------- | -----: |
| Train    |   8544 |
| Dev      |   1101 |
| Test     |   2210 |

为了复现参考论文中的卷积方法，本实验还进一步使用 Stanford Sentiment Treebank 原始文件：

- `SOStr.txt`
- `STree.txt`
- `datasetSentences.txt`
- `datasetSplit.txt`
- `dictionary.txt`
- `sentiment_labels.txt`

从中恢复出 phrase-level 训练样本，并构造新的卷积训练集：

| 数据划分        | 样本数 |
| --------------- | -----: |
| Train Sentences |   8544 |
| Train Phrases   | 310038 |
| Train Total     | 318582 |
| Dev Sentences   |   1101 |
| Test Sentences  |   2210 |

这意味着 Kim-style CNN-non-static 并不是只在句子级监督上训练，而是额外利用了大量短语级监督信号。

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
- nltk
- gensim
- CUDA（训练与评测均在 `cuda` 设备上完成）

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

### 4.3 Kim-style CNN-non-static 训练参数

| 参数           |                        取值 |
| -------------- | --------------------------: |
| 模型           |    Kim-style CNN-non-static |
| `data_dir`     |              `data/kim_sst` |
| `max_len`      |                          56 |
| `embed_dim`    |                         300 |
| `num_filters`  |                         100 |
| `kernel_sizes` |                     3, 4, 5 |
| `dropout`      |                         0.5 |
| `batch_size`   |                          50 |
| `epochs`       |                          25 |
| `lr`           |                         1.0 |
| `rho`          |                        0.95 |
| `weight_decay` |                         0.0 |
| `max_norm`     |                         3.0 |
| `patience`     |                           4 |
| 词向量初始化   | Google News `word2vec` 300d |
| 优化器         |                    Adadelta |

### 4.4 BERT 微调参数

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
|-- preprocessed_file/                # 预处理后的句子级数据集目录
|   |-- train.csv                     # 训练集
|   |-- dev.csv                       # 验证集
|   |-- test.csv                      # 测试集
|   |-- tokens2id.csv                 # token 到 id 的映射
|   `-- id2tokens.csv                 # id 到 token 的映射
|
|-- stanfordSentimentTreebank/        # Stanford Sentiment Treebank 原始文件
|   |-- SOStr.txt
|   |-- STree.txt
|   |-- datasetSentences.txt
|   |-- datasetSplit.txt
|   |-- dictionary.txt
|   `-- sentiment_labels.txt
|
|-- data/
|   `-- kim_sst/                      # 从 SST 原始文件恢复出的 phrase-level 卷积训练集
|       |-- train.csv
|       |-- dev.csv
|       |-- test.csv
|       `-- tokens2id.csv
|
|-- checkpoints/                      # 保存训练得到的模型参数
|   |-- textcnn_best.pt               # TextCNN 最优模型
|   |-- kim_cnn_non_static.pt         # Kim-style CNN-non-static 最优模型
|   `-- bert_best/                    # BERT 最优模型目录
|
|-- logs/                             # 训练与评测日志目录
|
`-- src/                              # 核心源码目录
    |-- dataset.py                    # TextCNN 数据集与词表加载代码
    |-- model.py                      # TextCNN 模型定义
    |-- train.py                      # TextCNN 训练脚本
    |-- evaluate.py                   # TextCNN / Kim-style CNN 评测脚本
    |-- bert_dataset.py               # BERT 数据集与 detokenize 逻辑
    |-- bert_train.py                 # BERT 微调训练脚本
    |-- bert_evaluate.py              # BERT 评测脚本
    |-- train_kim_cnn.py              # Kim-style CNN-non-static 训练脚本
    `-- build_sst_phrase_dataset.py   # 从 SST 原始文件构造 phrase-level 训练集
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

### 6.2 Kim-style CNN-non-static 实验流程

1. 读取 Stanford Sentiment Treebank 原始文件；
2. 根据 `SOStr.txt` 和 `STree.txt` 恢复每个训练句子的树结构与子短语；
3. 结合 `dictionary.txt` 与 `sentiment_labels.txt` 为短语节点恢复情感标签；
4. 根据 `datasetSplit.txt` 构造新的 `train/dev/test` 数据，其中训练集包含句子级样本与短语级样本，验证集和测试集只保留句子级样本；
5. 使用 `word2vec` 初始化嵌入层，并以 non-static 方式继续微调；
6. 使用 Kim 风格的卷积结构、Adadelta 优化器与 max-norm 约束训练模型；
7. 在测试集上输出 Accuracy、分类报告与混淆矩阵。

### 6.3 BERT 实验流程

1. 使用 `bert-base-uncased` 的 tokenizer 对文本重新编码；
2. 将预处理 token 序列通过 `detokenize` 恢复为更接近自然文本的形式；
3. 使用 `AutoModelForSequenceClassification` 进行微调训练；
4. 在验证集上保存最优 checkpoint；
5. 在测试集上输出 Accuracy、分类报告与混淆矩阵。

---

## 7. 实验结果

### 7.1 测试集整体结果对比

| 方法                     |  Test Loss | Test Accuracy |
| ------------------------ | ---------: | ------------: |
| TextCNN                  |     1.4399 |        0.3665 |
| Kim-style CNN-non-static |     1.3165 |        0.4330 |
| BERT                     | **1.0272** |    **0.5751** |

相较于普通 TextCNN，Kim-style CNN-non-static 在测试集上的准确率提升为：

- **绝对提升：+0.0665**
- **即提升约 6.65 个百分点**

相较于普通 TextCNN，BERT 在测试集上的准确率提升为：

- **绝对提升：+0.2086**
- **即提升约 20.86 个百分点**

这表明，在卷积神经网络设定下，引入 phrase-level augmentation 和预训练词向量能够有效提升性能；而从整体效果看，BERT 仍然显著优于卷积模型。

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

### 7.4 Kim-style CNN-non-static 测试结果

#### （1）整体指标

- Test Loss：`1.3165`
- Test Accuracy：`0.4330`
- Macro Avg F1：`0.3790`
- Weighted Avg F1：`0.4111`

#### （2）各类别准确率

| 类别          | Accuracy | 正确数 / 总数 |
| ------------- | -------: | ------------: |
| very negative |   0.1004 |      28 / 279 |
| negative      |   0.6730 |     426 / 633 |
| neutral       |   0.2725 |     106 / 389 |
| positive      |   0.5176 |     264 / 510 |
| very positive |   0.3333 |     133 / 399 |

#### （3）分类报告

| label            |  precision |     recall |   f1-score |  support |
| ---------------- | ---------: | ---------: | ---------: | -------: |
| very negative    |     0.6222 |     0.1004 |     0.1728 |      279 |
| negative         |     0.4392 |     0.6730 |     0.5315 |      633 |
| neutral          |     0.2834 |     0.2725 |     0.2779 |      389 |
| positive         |     0.4211 |     0.5176 |     0.4644 |      510 |
| very positive    |     0.6856 |     0.3333 |     0.4486 |      399 |
| **accuracy**     |            |            | **0.4330** | **2210** |
| **macro avg**    | **0.4903** | **0.3794** | **0.3790** | **2210** |
| **weighted avg** | **0.4752** | **0.4330** | **0.4111** | **2210** |

#### （4）混淆矩阵分析

从混淆矩阵看，Kim-style CNN-non-static 明显比普通 TextCNN 更容易将样本识别为 `negative` 或 `positive` 的正确类别，说明 phrase-level augmentation 的确增强了模型对局部情感模式的学习能力。但模型对于两端类别的区分仍然不够充分：

- `very negative` 召回率仅为 `0.1004`
- `very positive` 召回率为 `0.3333`

这说明即使引入了短语级监督，卷积模型在细粒度极性强弱判别上仍然存在局限。

---

## 8. 结果分析

### 8.1 TextCNN 的表现特点

TextCNN 能够学习一定程度的局部模式，因此在 `negative` 和 `positive` 两类上取得了相对还可以的结果，但在极性更细的 `very negative`、`very positive` 和 `neutral` 类别上表现明显不足。说明在五分类细粒度情感任务中，仅依赖浅层词向量和局部卷积特征难以准确建模复杂语义边界。

### 8.2 Kim-style CNN-non-static 的改进效果

Kim-style CNN-non-static 在普通 TextCNN 基础上引入了两项关键改进：一是使用 `word2vec` 预训练词向量进行 non-static 初始化，二是利用 Stanford Sentiment Treebank 的树结构恢复 phrase-level 训练样本。实验结果表明，这一改进带来了明确收益：

- 测试准确率从 `0.3665` 提升到 `0.4330`
- 宏平均 F1 从 `0.3021` 提升到 `0.3790`
- 加强了模型对 `negative`、`neutral`、`positive` 等中间类别的识别能力

这说明在卷积神经网络框架下，训练监督规模和词向量初始化方式对最终性能有明显影响。普通句子级 TextCNN 只依赖 8544 条训练句子，而 Kim-style CNN-non-static 能额外利用 31 万条以上的短语级样本，因此性能显著优于普通 TextCNN。

### 8.3 BERT 的优势

尽管 Kim-style CNN-non-static 已经明显优于普通 TextCNN，但 BERT 仍然取得了最高的测试精度。BERT 明显优于卷积模型，主要体现在以下几个方面：

1. **更强的上下文表示能力**：能够更好地理解情感触发词在句子中的上下文含义；
2. **更高的类别区分能力**：特别是在 `very positive` 与 `very negative` 这类强极性类别上提升更明显；
3. **更好的整体平衡性**：BERT 的 Macro Avg 与 Weighted Avg 均明显优于两类卷积模型，说明其不仅整体精度更高，而且在多类任务中更加稳定。

### 8.4 当前任务的主要难点

从实验结果看，`neutral` 类别始终是最难识别的类别之一：

- TextCNN：`0.1388`
- Kim-style CNN-non-static：`0.2725`
- BERT：`0.3496`

这说明中性文本与弱正、弱负文本之间存在较强重叠，分类边界不清晰，是该任务的主要难点之一。除此之外，`very negative` 与 `very positive` 这类极端类别在卷积模型中也较难稳定识别，说明五分类任务中的“强弱程度判断”本身比简单正负极性判断更难。

### 8.5 方法优缺点总结

#### TextCNN

**优点：**

- 结构简单，易于实现；
- 训练速度快；
- 作为课程作业中的传统卷积神经网络基线具有代表性。

**缺点：**

- 对上下文语义建模能力有限；
- 对细粒度情感类别区分不足；
- 在五分类任务中整体性能偏低。

#### Kim-style CNN-non-static

**优点：**

- 保持了卷积神经网络的基本形式，符合题目对卷积方法的要求；
- 通过 phrase-level augmentation 和预训练词向量显著提高了性能；
- 相比普通 TextCNN，更接近经典论文中的卷积文本分类设定。

**缺点：**

- 数据构造和训练流程更复杂；
- 对极端情感类别的识别仍然不足；
- 与 BERT 相比整体性能仍有较大差距。

#### BERT

**优点：**

- 具备强大的预训练语义表示能力；
- 在细粒度情感分类任务上显著优于传统模型；
- 测试集准确率、宏平均指标和类别均衡性均更优。

**缺点：**

- 训练和推理开销较大；
- 对显存和运行环境要求更高；
- 实现复杂度高于传统卷积模型。

---

## 9. 结论

本实验基于 PyTorch 完成了一个英文五分类情感分类系统，实现并比较了 **TextCNN**、**Kim-style CNN-non-static** 与 **BERT** 三类模型。

实验结果表明：

- TextCNN 在测试集上达到 **0.3665** 的准确率；
- Kim-style CNN-non-static 在测试集上达到 **0.4330** 的准确率；
- BERT 在测试集上达到 **0.5751** 的准确率；
- Kim-style CNN-non-static 相比普通 TextCNN 提升 **6.65 个百分点**；
- BERT 相比普通 TextCNN 提升 **20.86 个百分点**。

由此可以得出以下结论：

1. 普通 TextCNN 可以作为基础卷积神经网络基线，但在五分类细粒度情感任务中效果有限；
2. 参考 Kim (2014) 思路，引入 phrase-level augmentation 与预训练词向量后，卷积模型性能能够得到明显改善，说明卷积神经网络在情感分类任务上仍具有较强可塑性；
3. 从最终性能看，BERT 仍然显著优于卷积模型，是当前任务中效果最好的方案；
4. 如果严格按照题目“使用卷积神经网络”来完成任务，则 Kim-style CNN-non-static 是比普通 TextCNN 更有代表性的最终卷积方案。

总体来看，本实验完整实现了：

- 预处理文本数据读取；
- TextCNN 基线训练与测试；
- 基于 Stanford Sentiment Treebank 树结构恢复的 phrase-level 卷积训练集构造；
- Kim-style CNN-non-static 训练与测试；
- BERT 微调训练与测试；
- 分类报告、类别准确率与混淆矩阵分析；
- 不同模型在细粒度情感分类任务上的对比实验。

---

## 10. 其他

项目已在 GitHub 开源：`https://github.com/ChriCheng/sentiment_classification`

## 11. 参考文献

1. Kim Y. Convolutional Neural Networks for Sentence Classification. EMNLP, 2014.
2. Socher R, Perelygin A, Wu J, et al. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP, 2013.
3. Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019.
4. Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017.