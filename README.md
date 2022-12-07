# Cellular Composition
We developed a novel informatics system to facilitate objective and scalable diagnosis screening for serous borderline ovarian tumor (SBOT) and high-grade serous ovarian cancer (HGSOC). The system was built upon Groovy scripts and QuPath to enable interactive annotation and data exchange. Many Groovy modules are included in this repo, including parsing whole slide images annotations, cell segmentation, staining and morphological cellular feature extraction. Though the workflow is design for ovarian tissue, it can be easily extended to other digital pathology analysis.      
Please refer to our [paper](https://www.sciencedirect.com/science/article/pii/S2153353922001468) to get more details.
![framework](./doc/framework.png)
Figure 1. The proposed workflow for cellular analysis.
## Load cell classification to QuPath
After cell classification, cells can be loaded into QuPath for visualization.
![framework](./doc/cells_animation.gif)
## Cite our work
```
author = {Jiang, Jun. and Tekin, Burak. and Guo, Ruifeng. and Liu, Hongfang. and Huang, Yajue. and Wang, Chen.},
title =  {{Digital pathology-based study of cell- and tissue-level morphologic features in serous borderline ovarian tumor and high-grade serous ovarian cancer}},
journal  ={Journal of Pathology Informatics},
volume ={12},
number ={1},
pages  = {24},
doi  = {10.4103/jpi.jpi_76_20},
year  = {2021}
}
```
