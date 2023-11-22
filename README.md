# MulGT: Multi-task Graph-Transformer with Task-aware Knowledge Injection and Domain Knowledge-driven Pooling for Whole Slide Image Analysis
This repository provide the Pytorch implementation for AAAI 2023 paper "MulGT: Multi-task Graph-Transformer with Task-aware Knowledge Injection and Domain Knowledge-driven Pooling for Whole Slide Image Analysis".
Paper can be found [here](https://arxiv.org/abs/2302.10574).


## Data Download
The WSIs are downloaded from the [TCGA GDC Data Portal](https://portal.gdc.cancer.gov/).

Detailed downloading instructions can be found [here](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).

## Preprocessing
### Patch Cropping
To crop patches from the WSIs, users need to refer [DS-MIL](https://github.com/binli123/dsmil-wsi/tree/master) repository. This project uses their patch preparation code.
```
    $ python deepzoom_tiler.py -m 0 -b 20 -s 512
```

### Graph Construction
After the patch cropping finished, users can build the 8-adjacency graph by runing:
```
    $ python ./feature_extractor/build_graphs.py
```

## Run
The implementation of `MulGT` model is in  `./models/MulGT`. To run the experiments, users can use the following command:
```
    $ python main.py
```
Hyper-parameters and data path can be customized in `option.py`.


## Citation
If you use the code or results in your research, please use the following BibTeX entry.
```
@inproceedings{DBLP:conf/aaai/ZhaoWYNY23,
  author       = {Weiqin Zhao and
                  Shujun Wang and
                  Maximus Yeung and
                  Tianye Niu and
                  Lequan Yu},
  editor       = {Brian Williams and
                  Yiling Chen and
                  Jennifer Neville},
  title        = {MulGT: Multi-Task Graph-Transformer with Task-Aware Knowledge Injection
                  and Domain Knowledge-Driven Pooling for Whole Slide Image Analysis},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2023, Thirty-Fifth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February
                  7-14, 2023},
  pages        = {3606--3614},
  publisher    = {{AAAI} Press},
  year         = {2023},
  url          = {https://doi.org/10.1609/aaai.v37i3.25471},
  doi          = {10.1609/AAAI.V37I3.25471},
  timestamp    = {Mon, 04 Sep 2023 16:50:28 +0200},
  biburl       = {https://dblp.org/rec/conf/aaai/ZhaoWYNY23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
