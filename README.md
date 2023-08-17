# Bridging the KB-Text Gap: Leveraging Structured Knowledge-aware Pre-training for KBQA


## Overview
This is the repository for our work **SKP**, which is recieved by **CIKM 2023**.

## Brief introduction
We propose a **S**tructured **K**nowledge-aware **P**re-training method (**SKP**). In the pre-training stage, we introduce two novel structured knowledge-aware tasks, guiding the model to effectively learn the implicit relationship and better representations of complex subgraphs. 
In downstream KBQA task, we further design an efficient linearization strategy and an interval attention mechanism, which assist the model to better encode complex subgraphs and shield the interference of irrelevant subgraphs during reasoning respectively.
Detailed experiments and analyses on WebQSP verify the effectiveness of SKP, especially the significant improvement in subgraph retrieval (+4.08% H@10).

## Overall Framework
<img width="733" alt="image" src="https://github.com/dongguanting/Structured-Knowledge-aware-Pretraining-for-KBQA/assets/60767110/c63e55fb-0cee-474c-8dbf-392498ad24e6">

## Result

**Main Result**
| Model                                                    |  Hits@1    | 
| --------------------- | :------:   | 
| GraftNet                                                 |  69.5     | 
| PullNet                                                    |  68.1    | 
| EMQL                                                      |  75.5     | 
| BERT-KBQA                                                |  72.9    | 
| NSM                                                       |  74.3    | 
| KGT5                                                      | 56.1 | 
| SR-NSM                                                   | 69.5| 
| EmbededKGQA                                               | 72.5| 
| DECAF(Answer only)                                        | 74.7 | 
| UniK-QA∗                                                 | 77.9 |
| SKP (ours)                                               | **79.6** | 

**In-Context Learning Result For LLMs**
Since there were very few open source large models when the article was written (2022.12), we now supplement the SKP framework with the results of **In Context Learning** when the LLMs is used as a Reader. Due to the limitation of the Max sequence length of the LLMs, for the **Topk** documents retrieved by the retriever, we select the documents with the highest semantic similarity and truncate them with 2048 tokens as the knowledge prompting for reader (about 5 documents)


| Model                                                    |  Hits@1     | 
| -------------------------------------------------------- | :------:   | 
| SKP(ChatGPT)                                              |  71.9     | 
| SKP(LLAMA)                                                |  coming soon   | 
| SKP(LLAMA2)                                                |  coming soon     | 
| SKP(ChatGLM)                                                |  coming soon     | 
| SKP(ChatGLM2)                                               |  coming soon     | 


Our Code and Dataset will be released soon!



