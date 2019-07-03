SOTA
---
Performance: Highest F1 Score/ UAS(LAS)
- [Constituency Parsing with a Self-Attentive Encoder](https://aclweb.org/anthology/P18-1249)
- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)

Speed: Fastest
- [Straight to the Tree: Constituency Parsing with Neural Syntactic Distance](https://aclweb.org/anthology/P18-1108)

This repo's target
---
Faster and Accurate Syntactic Parsing both on Constituency and Dependency.

Naive Motivation
---
Conversion: Dependency Tree could be converted from constituency by utilizing head rule.\
Span: where the head rule is used.  

Implementation
---
Self-Attentive Con Parser, start from scratch with:
- Phrase Structure Tree Load/Conversion/Expr, Oracle Design
- Pretrain Model Loaded: ELMO
- Multi-Head Self-Attentive Encoder
- Max-Span Tree Inference
- Training

Biaffine Dep Parser, start from scratch with:
- Dependency Relation Tree Load/Conversion/Expr, Oracle Design
- Pretrain Word Embedding Loaded
- Bi-LSTM Encoder
- MST Inference with Attention Module
- Training

FAParser's new features:
- Enhanced Pretrain Module of Word Representation
- Enhanced MST Inference with Multi-Head Attention Module
- Conditional modeling on interaction between information flows of phrase structure and dependency relation

Similar to the design of [fairseq](https://github.com/pytorch/fairseq), we organize our FAParser as:

```
FAParser
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
│   │   FAParser
│   └ ...
│
└───utlis:
│   │   command/preprocess/meters...
│   │   
│   └ ...
```

criterion,modules,optim and data could be initialized with the fairseq's sub-modules. we just need build some 
task-specific sub-packages. 