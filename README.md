SOTA
---
Performance: Highest F1 Score/ UAS(LAS)
- [Constituency Parsing with a Self-Attentive Encoder](https://aclweb.org/anthology/P18-1249)
- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)

Speed: Fastest
- [Straight to the Tree: Constituency Parsing with Neural Syntactic Distance](https://aclweb.org/anthology/P18-1108)

Our Naive Motivation
---
Span: Relation between the Constituency Tree and the Dependency Tree. 

Implementation Details
---
Self-Attentive Con Parser, Need:
- Tree Structure Load/Conversion/Expr, Oracle Design
- Self-Attentive Encoder
- Max-Span Inference
- Pretrain Model Loaded: ELMO
- Training


Biaffine Dep Parser, Need:
- Dependency Tree Load/Conversion/Expr, Oracle Design
- Bi-LSTM Encoder
- MST Inference
- Pretrain Word Embedding Loaded
- Training


Similar to the design of [Fairseq](https://github.com/pytorch/fairseq), we organize our package as:

```
Dual Parser
│   README.md
│   train.py
│   inference.py
│   preprocess.py
│
└───evaluation: for validation or testing
│   │   F1
│   │   Accuracy
│   │       │ UAS
│   │       └ LAS
│   └  ...
│
└───data: 
│   │   tree loaded or structure utils
│   │   
│   └ ...
│   
└───criterion: 
│   │   cross entropy. etc
│   │   
│   └ ...
│
└───models: 
│   │   three parser
│   │   
│   └ ...
│
└───modules: 
│   │   series of module used in models
│   │   
│   └ ...
│
└───optim: for optimizer
│   │   lr_shedule
│   │   adam...
│   └ ...
│
└───tasks: for loss computing
│   │   Constituency Parser
│   │   Dependenecy Parser
│   │   Dual Parser
│   └ ...
│
└───utlis:
│   │   command/preprocess/meters...
│   │   
│   └ ...
```

Criterion and Modules could be initialized with the fairseq's sub-modules. 