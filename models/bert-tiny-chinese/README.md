---
language:
  - zh
thumbnail: https://ckip.iis.sinica.edu.tw/files/ckip_logo.png
tags:
  - pytorch
  - lm-head
  - bert
  - zh
license: gpl-3.0
---

# CKIP BERT Tiny Chinese

This project provides traditional Chinese transformers models (including ALBERT, BERT, GPT2) and NLP tools (including word segmentation, part-of-speech tagging, named entity recognition).

這個專案提供了繁體中文的 transformers 模型（包含 ALBERT、BERT、GPT2）及自然語言處理工具（包含斷詞、詞性標記、實體辨識）。

## Homepage

- https://github.com/ckiplab/ckip-transformers

## Contributers

- [Mu Yang](https://muyang.pro) at [CKIP](https://ckip.iis.sinica.edu.tw) (Author & Maintainer)

## Usage

Please use BertTokenizerFast as tokenizer instead of AutoTokenizer.

請使用 BertTokenizerFast 而非 AutoTokenizer。

```
from transformers import (
  BertTokenizerFast,
  AutoModel,
)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('ckiplab/bert-tiny-chinese')
```

For full usage and more information, please refer to https://github.com/ckiplab/ckip-transformers.

有關完整使用方法及其他資訊，請參見 https://github.com/ckiplab/ckip-transformers 。
