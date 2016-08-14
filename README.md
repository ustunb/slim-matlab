``slim-matlab`` is a package to create scoring systems using MATLAB and the CPLEX Optimization Studio.

## Introduction

*Scoring systems* are simple models to make quick predictions by adding, subtracting and multiplying a few numbers:

![SLIM scoring system for the mushrooms dataset](https://github.com/ustunb/slim_for_matlab/blob/master/images/slim_mushroom.png)

[SLIM](http://http//arxiv.org/abs/1502.04269/) is new machine learning method for learning scoring systems. It can learn models that are fully optimized for accuracy and sparsity, and that satisfy difficult constraints **without parameter tuning** (e.g. hard limits on model size, the true positive rate, the false positive rate).

## Installation

``slim-matlab`` was developed using MATLAB 2014b and CPLEX V12.6. It may work with earlier versions of MATLAB and/or CPLEX, but this has not been tested and will not be supported.

### CPLEX 

*CPLEX* is cross-platform commercial optimization tool that can be called from MATLAB. It is freely available to students and faculty members at accredited institutions as part of the IBM Academic Initiative. To setup the latest version of CPLEX, you should:

1. Join the [IBM Academic Initiative](http://www-304.ibm.com/ibm/university/academic/pub/page/mem_join). Note that it may take up to a week to obtain approval.
2. Download *IBM ILOG CPLEX Optimization Studio V12.6.1* (or higher) from the [software catalog](https://www-304.ibm.com/ibm/university/academic/member/softwaredownload)
3. Install the file on your computer. Note mac/unix users will [need to install a .bin file](http://www-01.ibm.com/support/docview.wss?uid=swg21444285).
4. Add the CPLEX API to your MATLAB search path using ``pathtool``/``addpath`` [as shown here](http://www-01.ibm.com/support/knowledgecenter/SSSA5P_12.6.1/ilog.odms.cplex.help/CPLEX/MATLAB/topics/gs_install.html)

Please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059) if you have problems installing CPLEX.

## Citation 

If you use SLIM for academic research, please cite [our paper](http://http//arxiv.org/abs/1502.04269/)!  
     
```
@article{
    ustun2015slim,
    year = {2015},
    issn = {0885-6125},
    journal = {Machine Learning},
    doi = {10.1007/s10994-015-5528-6},
    title = {Supersparse linear integer models for optimized medical scoring systems},
    url = {http://dx.doi.org/10.1007/s10994-015-5528-6},
    publisher = { Springer US},
    author = {Ustun, Berk and Rudin, Cynthia},
    pages = {1-43},
    language = {English}
}
```

