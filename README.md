SLIM is a machine learning method for creating *scoring systems.* These are simple classification models that allow users to make a prediction by adding, subtracting and multiplying a few meaningful numbers.

![SLIM scoring system for the mushrooms dataset](https://github.com/ustunb/slim_for_matlab/blob/master/images/slim_mushroom.png)

SLIM produces scoring systems by solving an integer programming problem. See our paper [here for further details](http://arxiv.org/abs/1502.04269).

slim_for_matlab is a toolbox to create these SLIM scoring systems in MATLAB. The toolbox uses MATLAB 2014b and the IBM ILOG CPLEX V12.6. It may work with earlier versions of MATLAB and/or CPLEX, but this has not been tested and will not be supported.

Installing IBM ILOG CPLEX Optimization Studio V12.6
--------------------------------------------------------------------------------

The IBM ILOG CPLEX Optimization Studio is a commericial mathematical programming solver that can be called from MATLAB and can be obtained for free via the IBM Academic Initiative.

To setup latest version of the IBM ILOG CPLEX Optimization Studio on your computer, you should:

1. Join the [IBM Academic Initiative](http://www-304.ibm.com/ibm/university/academic/pub/page/mem_join) (may take up to 2 days)

2. Download the IBM ILOG CPLEX Optimization Studio V12.6.1 from the [software site](https://www-304.ibm.com/ibm/university/academic/member/softwaredownload)
- Make sure you choose the right version for your operating system
- Check 'Download via HTTP' if you do not wish to use the IBM Download Director

3. Install the IBM ILOG CPLEX Optimization Studio. 
- If you are unfamiliar with how to install .bin files on a Mac, [see here](http://www-01.ibm.com/support/docview.wss?uid=swg21444285)

4. Add the CPLEX for MATLAB API to the MATLAB path using 'addpath' [as shown here](http://www-01.ibm.com/support/knowledgecenter/SSSA5P_12.6.1/ilog.odms.cplex.help/CPLEX/MATLAB/topics/gs_install.html)

If you have problems setting up CPLEX, please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or ask your question on the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059).


Citations
--------------------------------------------------------------------------------

To cite SLIM, please cite:

@ARTICLE{,
   title = "{Supersparse Linear Integer Models for Optimized Medical Scoring Systems}",
   author = {{Ustun}, B. and {Rudin}, C.},
   journal = {ArXiv e-prints},
   archivePrefix = "arXiv",
   eprint = {1502.04269},
   primaryClass = "stat.ML",
   keywords = {Statistics - Machine Learning, Computer Science - Discrete Mathematics, Computer Science - Learning, Statistics - Applications, Statistics - Methodology},
   month = April,
   year = 2015,
   url = {http://http://arxiv.org/abs/1502.04269/}
}
