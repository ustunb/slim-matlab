slim_for_matlab is a package to create SLIM scoring systems using MATLAB and IBM ILOG CPLEX Optimization Studio.

# Introduction

*Scoring systems* are simple classification models that allow users to make a prediction by adding, subtracting and multiplying a few meaningful numbers. See, for example, the following scoring system produced for the [mushroom dataset](http://archive.ics.uci.edu/ml/datasets/Mushroom).

![SLIM scoring system for the mushrooms dataset](https://github.com/ustunb/slim_for_matlab/blob/master/images/slim_mushroom.png)

SLIM is a method for create scoring systems from data (see [the paper]()). This approach is computationally expensive in comparison to other classification methods. However, it can produce simple models that are very accurate, sparse, and that accomodate hard operational constraints **without any parameter tuning** (i.e. a hard limit on model size, a hard limit on class-based accuracy, and other feature-based constraints).

# Installation

slim_for_matlab was developed for MATLAB 2014b and the IBM ILOG CPLEX V12.6. The package may work with versions of MATLAB and/or CPLEX, but this has not been tested and will not be supported.

# Installing CPLEX 

The IBM ILOG CPLEX Optimization Studio is a commericial mathematical programming solver that can be called from MATLAB and can be obtained for free via the IBM Academic Initiative.

To setup latest version of the IBM ILOG CPLEX Optimization Studio on your computer, you should:

1. Join the [IBM Academic Initiative](http://www-304.ibm.com/ibm/university/academic/pub/page/mem_join). Note that it may take up to a week to obtain approval.
2. Download *IBM ILOG CPLEX Optimization Studio V12.6.1* from the [Academic Initiative software catalog](https://www-304.ibm.com/ibm/university/academic/member/softwaredownload)
3. Install the IBM ILOG CPLEX Optimization Studio. Mac/Linux users: [see here](http://www-01.ibm.com/support/docview.wss?uid=swg21444285) here install .bin files.
4. Add the CPLEX for MATLAB API to the MATLAB path using ``addpath`` [as shown here](http://www-01.ibm.com/support/knowledgecenter/SSSA5P_12.6.1/ilog.odms.cplex.help/CPLEX/MATLAB/topics/gs_install.html)

Please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059) if you have problems installing CPLEX.

# Citation and References

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
