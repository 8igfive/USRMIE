# Unsupervised Semantic Retrieval via Mutual Information Estimation

This is a PyTorch implementation of the paper *Unsupervised Semantic Retrieval via Mutual Information Estimation*.

## Requirements

+ python=3.9.13
+ numpy=1.23.1
+ tqdm=4.64.1
+ pytorch=1.12.1
+ transformers=4.18.0
+ tensorflow=2.8.2
+ tensorflow-hub=0.12.0
+ sentence-transformers=2.2.2
+ json=2.0.9

## Preprocess

1. Arrange datasets in json files as following form:
    ```json
    [
      {
        "id": "<id for sample>", 
        "query": "<query text>",
        "candidates": [
          {
              "cid": "<id for candidate>",
              "order": "<rank sequence>",
              "label": "<0 or 1>",
              "subject": "<Subject of the candidate or empty>",
              "body": "<Body of the candidate>"
          }, ...
        ]
      }, ...
    ]
    ```
2. Place datasets in the `data` folder as described in `preprocess.py/CORPUS2PATH`
3. formalize the datasets by:
    ```shell
    python preprocess.py -fc <corpus name> -dd   <dump path> -dc -dqac
    ```
4. Calculate the results of source domain function by:
    ```shell
    python preprocess.py -cm <source domain   function type> -c <corpus name> -mp <model   path>
    ```
5. Calculate the metrics for the results of source domain function by:
    ```shell
    python preprocess.py -cmx <MAP, MRR or F1> -pp   <path of results> -cr <cal range> 
    ```

## Training

1. Prepare a configuration file like the sample in `conf/wikiqa/wikiqa.yaml`
2. Train our model by:
    ```shell
    python USRMIE.py -t <qa or wsd> -cp <path to   configuration file> -dtr -dte -g <gpus to use>   -r <local rank> -wf <path to save tensorboard   information> -s <seed>
    ```