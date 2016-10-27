#LDAExample
Online LDA的实现，数据并行使用RDD分布式化，模型没有并行
#HDPExample
Online HDP的实现，数据和模型都没有分布式化

# Topic Modeling on Apache Spark
This package contains a set of distributed text modeling algorithms implemented on Spark, including:

- **Online LDA**: an early version of the implementation was merged into MLlib (PR #4419), and several extensions (e.g., predict) are added here

- **Gibbs sampling LDA**: the implementation is adapted from Spark PRs(#1405 and #4807) and JIRA SPARK-5556 (https://github.com/witgo/spark/tree/lda_Gibbs, https://github.com/EntilZha/spark/tree/LDA-Refactor, https://github.com/witgo/zen/tree/lda_opt/ml, etc.), with several extensions (e.g., support for MLlib interface, predict and in-place state update) added

- **Online HDP (hierarchical Dirichlet process)**: implemented based on the paper "Online Variational Inference for the Hierarchical Dirichlet Process" (Chong Wang, John Paisley and David M. Blei)
