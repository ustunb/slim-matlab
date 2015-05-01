``slim_for_matlab`` is a package to create scoring systems using MATLAB and IBM ILOG CPLEX Optimization Studio.

# Introduction

*Scoring systems* are simple classification models that allow users to make a prediction by adding, subtracting and multiplying a few meaningful numbers. See, for example, the following scoring system produced for the [mushroom dataset](http://archive.ics.uci.edu/ml/datasets/Mushroom).

![SLIM scoring system for the mushrooms dataset](https://github.com/ustunb/slim_for_matlab/blob/master/images/slim_mushroom.png)

SLIM is a method to create data-driven scoring systems (described in [this paper](http://http//arxiv.org/abs/1502.04269/)). SLIM scoring systems are fully optimized for accuracy and sparsity, and can directly satisfy multiple hard constraints **without any parameter tuning** (e.g. limits on the true positive rate, the false rate, and/or the model size).

SLIM scoring systems are typically just as (if not more) accurate and/or sparse compared to models produced by popular classification methods such as [glmnet](http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html). However, they may take more time to create since SLIM works by solving an [integer programming](http://en.wikipedia.org/wiki/Integer_programming) problem. Note that SLIM is not appropriate for large-scale datasets (N > 100'000).

# Installation

``slim_for_matlab`` was developed for MATLAB 2014b and the IBM ILOG CPLEX V12.6. It may work with earlier versions of MATLAB and/or CPLEX, but this has not been tested and will not be supported.

## Installing CPLEX 

The *IBM ILOG CPLEX Optimization Studio* is a commericial mathematical programming solver that can be called from MATLAB. It is free for students, research professionals, and faculty members via the IBM Academic Initiative.

To setup latest version of the IBM ILOG CPLEX Optimization Studio on your computer, you should:

1. Join the [IBM Academic Initiative](http://www-304.ibm.com/ibm/university/academic/pub/page/mem_join). Note that it may take up to a week to obtain approval.
2. Download *IBM ILOG CPLEX Optimization Studio V12.6.1* from the [software catalog](https://www-304.ibm.com/ibm/university/academic/member/softwaredownload)
3. Install the file on your computer. Mac/Unix users will [need to install a .bin file](http://www-01.ibm.com/support/docview.wss?uid=swg21444285).
4. Add the CPLEX MATLAB API to the MATLAB path using the ``pathtool`` or ``addpath`` [as shown here](http://www-01.ibm.com/support/knowledgecenter/SSSA5P_12.6.1/ilog.odms.cplex.help/CPLEX/MATLAB/topics/gs_install.html)

Please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059) if you have problems installing CPLEX.

# Citation 

If you use SLIM for academic research, please cite the following paper:  
     
@ARTICLE{,  
 title = "{Supersparse Linear Integer Models for Optimized Medical Scoring Systems}",  
 author = {{Ustun}, B. and {Rudin}, C.},  
 journal = {ArXiv e-prints},  
 archivePrefix = "arXiv",  
 eprint = {1502.04269},  
 primaryClass = "stat.ML",  
 keywords = {Statistics - Machine Learning, Computer Science - Discrete Mathematics, Computer Science - Learning, Statistics - Applications, Statistics - Methodology},  
 year = 2015,  
 url = {http://http://arxiv.org/abs/1502.04269/}  
}
