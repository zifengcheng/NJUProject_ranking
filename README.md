# 作业任务：基于大语言模型的零样本文档排序

本次作业任务要求利用大语言模型（LLM）进行零样本（zero-shot）文档排序，该任务是给定一个文档在一系列文档中找到跟其最相似的。具体地，任务主要是在BM25初筛结果的基础上，利用LLM进行零样本的排序。你需要在提高排序效果的同时，考虑LLM推理的计算开销和成本之间的平衡。

---

## 一、背景要求

随着信息检索领域的发展，传统的BM25虽然在文本匹配方面具有较高效率，但在深层语义理解上存在一定局限。利用LLM对候选文档进行重排序，可以捕捉文档与查询之间复杂的语义关联，从而进一步提升排序性能。然而，由于LLM模型（例如flan-t5-large）的计算成本较高，如何在排序效果和推理开销之间进行合理权衡成为本任务的研究重点。

---

## 二、排序方法介绍

本作业中主要讨论Listwise与Pairwise两类排序方法，这两种方法采用不同的提示策略来指导LLM输出每个候选文档的相关性估计，并相应地对文档进行排序。其中pairwise需要逐次地做两个文档或者段落进行比较，从而给所有候选文档或者段落一个分数。由于其进行了广泛的比较，往往具有良好的性能但开销较大。而listwise把所有文档或者段落一次性输入，从而减轻了开销，但往往性能一般。

本作业中主要基于Listwise与Pairwise两类排序方法：

### 2.1 Listwise方法

Listwise方法基于滑动窗口的排序策略，具体描述如下：

- **滑动窗口策略**：
  - **窗口大小（window_size）**：从候选文档列表中每次取出几个文档；
  - **步长（step_size）**：窗口每次滑动时的位移值；
  - **多轮重复（num_repeat）**：从原始文档列表的末尾向前多轮进行排序，每轮将窗口内的文档提交给LLM进行比较；
- **实验中使用的计算方法**：  
每次将窗口内多个文档输入LLM，使用`compare`函数获得窗口内文档之间的排序结果。

### 2.2 Pairwise方法

Pairwise方法通过对文档成对比较来获得最终排序结果。该方法分为以下三种策略：

- **Bubble策略和Heapsort策略**：
  - 利用LLM `compare`函数来判断两两元素的顺序，实现经典bubble算法和heapsort算法；
- **AllPair策略**：
  - 对所有文档对进行比较，每一对文档比较后，采用简单的聚合方式得到最终得分：胜者得1分，败者不得分，平局得0.5分。最终根据各文档得分高低确定排序结果。

---

## 三、数据集说明

本次实验采用 **TREC DL19 Dataset**，具体情况如下：

- **数据来源**：TREC DL19 数据集；
- **输入文件**：已提供 BM25 排序后的文档结果文件 `run.msmarco-v1-passage.bm25-default.dl19.txt`；
- **文件内容**：文件包含43个查询，每个查询对应BM25得分前1000个文档及其得分，但实际实现只取前100个文档；

---

## 四、模型下载与环境配置

- **模型下载与环境配置**：  
  实验使用模型为 `flan-t5-large`，请按以下步骤进行环境配置：
  
  1. **安装必要的Python库**：
     ```bash
     pip install torch transformers faiss-cpu pyserini ir-datasets openai tiktoken accelerate
     ```
  
  2. **建议安装Java环境**（部分检索工具依赖Java）：
     ```bash
     sudo apt update
     sudo apt install openjdk-11-jdk -y
     ```
  
  3. **下载 flan-t5-large 模型**（你也可以用别的方式下载）：
     ```bash
     pip install modelscope
     modelscope download --model 'mindnlp/flan-t5-large' --local_dir 'model/flan-t5-large'
     ```

## 五、运行指令示例

参考以下Listwise方法指令运行实验：
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  run --model_name_or_path model/flan-t5-large \
      --tokenizer_name_or_path model/flan-t5-large \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.liswise.generation.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 100 \
      --scoring generation \
      --device cuda \
  listwise --window_size 4 \
           --step_size 2 \
           --num_repeat 5
```

后续可利用以下命令评估结果：
```bash
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage run.liswise.generation.txt
```

---

## 六、任务描述

作业任务分为三个任务，内容如下：

### 6.1 简单任务（必须完成，60分）
- **任务内容**：  
  - **Listwise**：完成rerank函数，实现listwise方法。
  - **Pairwise**：实现Bubble、Heapsort和AllPair三种策略，其中Bubble和Heapsort使用大模型的`compare`函数进行相邻文档或堆中元素的比较；AllPair方法中，对所有文档两两进行比较，然后使用简单聚合方式（胜者得1分，败者不得分，平局得0.5分）得到最终排序结果。
  
### 6.2 进阶任务 （可选，40分）
  设计并实现一种创新的文档重排序算法，要求：  
  1. 算法在排序性能上必须超越AllPair方法，或在保持较好排序效果的前提下显著降低计算开销；  
  2. 算法设计必须为原创，严禁抄袭已有论文方法。

---

## 七、提交要求

提交内容包括：
- **完整代码文件**：将所有代码文件打包提交，包含实现Listwise与Pairwise方法以及进阶任务的代码。
- **实验报告（PDF）**


## 八、注意事项

1. **参考文献**：

    - 如果你在实验和报告中参考了已发表的文献，请列出你所参考的相关文献。

2. 我们提供了部分代码和数据，可以基于我们的[代码]（https://github.com/zifengcheng/NJUProject_ranking）进行实现。

3. 如有疑问，请联系 chengzf@smail.nju.edu.cn。


