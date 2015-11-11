% SLIM quick start script. This script shows how to create scoring systems using SLIM
%
%Author:      Berk Ustun 
%Contact:     ustunb@mit.edu
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%% Clear 

clear all;
clear mem;
close all;
clc;
dbstop if error;

%% Load Breastcancer Dataset

repo_dir    = [cd,'/']; %run from the slim_for_matlab directory
code_dir    = [repo_dir, 'src/'];
example_dir = [repo_dir, 'examples/'];
addpath(code_dir);

data_name = 'breastcancer';
data_file = [example_dir, data_name, '_processed.mat'];
load(data_file)

%% Example 1: Quick Start

[N,P] = size(X);

%set SLIM input struct()
input                               = struct();
input.display_warnings              = true; 

%data
input.X                             = X;       %X should include a column of 1s to act as an intercept
input.X_names                       = X_names; %the intercept should have the name '(Intercept)'
input.Y                             = Y;
input.Y_name                        = Y_name;

%misclassification costs w_pos > 0 and w_neg > 0
%if w_pos and w_neg are not provided, then SLIM will use w_pos = w_neg = 1.00 
%if w_pos and w_neg are provided, then SLIM will normalize values so that w_pos + w_neg = 2.00
input.w_pos                         = 1.00;
input.w_neg                         = 1.00;

%L0 regularization parameter, C_0
%by default, SLIM imposes a penalty of C_0 if lambda_j !=0
%C_0 should be set as the % gain in accuracy required for a feature to have a non-zero coefficient
input.C_0                          = 0.01;

%coefficient set
CoefSet                            = newCoefSet(X_names);  %by default each coefficient lambda_j is an integer from -10 to 10
changeCoefSetField(CoefSet, '(Intercept)', 'C_0j', 0)             %the regularization penalty for the intercept should be set to 0 manually

%simple operational constraints (no need to set these)
%input.L0_min                        = 0;
%input.L0_max                        = P;
%input.err_min                       = 0;
%input.err_max                       = 1;
%input.pos_err_min                   = 0;
%input.pos_err_max                   = 1;
%input.neg_err_min                   = 0;
%input.neg_err_max                   = 1;

%advanced parameters (set automatically)
%input.C_1                          = NaN;
%input.M
%input.epsilon 

%createSLIM create a Cplex object, slim_IP and provides useful info in slim_info
[slim_IP, slim_info] = createSLIM(input);

%set default CPLEX solver parameters
slim_IP.Param.emphasis.mip.Cur               = 1;   %mip solver strategy
slim_IP.Param.timelimit.Cur                  = 30;  %timelimit in seconds
slim_IP.Param.randomseed.Cur                 = 0;
slim_IP.Param.threads.Cur                    = 1;   %# of threads; >1 starts a parallel solver
slim_IP.Param.output.clonelog.Cur            = 0;   %disable CPLEX's clone log
slim_IP.Param.mip.tolerances.lowercutoff.Cur = 0;
slim_IP.Param.mip.tolerances.mipgap.Cur      = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.absmipgap.Cur   = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.integrality.Cur = eps; %use maximal precision for IP solver (only recommended for testing)

%solve the SLIM IP
slim_IP.solve

%once you solve the SLIM IP, you can solve it again
slim_IP.Param.timelimit.Cur = 5;
slim_IP.solve

%get summary statistics
summary = getSLIMSummary(slim_IP, slim_info);

%% Example 2: Customize the Coefficient Set

%coefficient values can be specified in a CoefSet object
%the coefSet object is a P x 1 struct array such that:
%
% coefSet(j).name   name of feature j; must be non-empty
% coefSet(j).ub     upperbound on the coefficient for feature j
% coefSet(j).lb     lowerbound on the coefficient for feature j
%
% coefSet(j).type   either 'integer', integers between coefSet(j).lb to coefSet(j).ub
%                   or      'custom', any one of the values on coefSet(j).values
%
% coefSet(j).values a 1D array containing the set of values for coefficient j; 
%                   values should include 0; if not coefficient j cannot be dropped
%
% coefSet(j).sign   constraint on the sign of the coefficient
%                   coefSet(j).sign = 1 means coefficient j >= 0
%                   coefSet(j).sign =-1 means coefficient j <= 0
%                   coefSet(j).sign = NaN means no constraint
%
% coefSet(j).C_0j   customized regularization parameter for C_0j; if C_0j = NaN, then a global C_0 parameter is used
%

%create a default coefSet
coefSet = newCoefSet(X_names);

%modify field for all coefficients in the coefSet
coefSet = changeCoefSetField(coefSet,'all', 'ub', 5); %set upperbound to 5
coefSet = changeCoefSetField(coefSet,'all', 'lb', -5); %set lowerbound to -5

%restrict all coefficients to a custom discrete set
coefSet = changeCoefSetField(coefSet,'all', 'values', [-5,-2.5,-1,0,1,2.5,5]);  
%make sure set contains 0; otherwise,set C_0j = 0 to prevent unnecessary L0-regularization for feature j 

%modify a field for a single coefficient
coefSet = changeCoefSetField(coefSet,'(Intercept)', 'type','integer');
coefSet = changeCoefSetField(coefSet,'(Intercept)', 'C_0j',0);  %regularization penalty for the intercept should be set to 0 manually
coefSet = changeCoefSetField(coefSet,'(Intercept)', 'ub', 20);  %change ub for intercept to 20
coefSet = changeCoefSetField(coefSet,'(Intercept)', 'lb', -20); %change lb for intercept to 20

%checkCoefSet makes sure that the coefSet is proper
coefSet = checkCoefSet(coefSet);

%print the final coefficientSet for checking
coefSetTable = printCoefSet(coefSet);

%train the model
input.coefSet = coefSet;

[slim_IP, slim_info] = createSLIM(input);

%set default CPLEX solver parameters
slim_IP.Param.emphasis.mip.Cur               = 1;   %mip solver strategy
slim_IP.Param.timelimit.Cur                  = 60;  %timelimit in seconds
slim_IP.Param.randomseed.Cur                 = 0;
slim_IP.Param.threads.Cur                    = 1;   %# of threads; >1 starts a parallel solver
slim_IP.Param.output.clonelog.Cur            = 0;   %disable CPLEX's clone log
slim_IP.Param.mip.tolerances.lowercutoff.Cur = 0;
slim_IP.Param.mip.tolerances.mipgap.Cur      = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.absmipgap.Cur   = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.integrality.Cur = eps; %use maximal prec

slim_IP.solve
summary = getSLIMSummary(slim_IP, slim_info);

%% Example 3: Imposing Hard Limits on Model Size

%SLIM can set hard limits on the % of non-zero coefficients as follows

%Setup usual input struct
input                               = struct();
input.display_warnings              = true; 
input.X                             = X;      
input.X_names                       = X_names;
input.Y                             = Y;
input.Y_name                        = Y_name;
input.w_pos                         = 1.00;
input.w_neg                         = 1.00;
input.coefSet                      = newCoefSet(X_names);
changeCoefSetField(input.coefSet, '(Intercept)', 'C_0j', 0)            

%Add L0_constraints to limit classifiers to 1 to 3 features 
input.L0_min = 1; 
input.L0_max = 3; 
%note: that these constraints only apply to features such that C_0j > 0 
%so here, for example, the constraint does not apply to '(Intercept)'

%Set C0 small enough to guarantee that we hit the L0_max constraint 
w_total = input.w_pos+input.w_neg;
w_pos   = 2.00*(input.w_pos/w_total);
w_neg   = 2.00*(input.w_neg/w_total);
L0_regularized_variables = [coefSet.C_0j]~=0.0;
L0_max  = 3;
input.C_0 = 0.9*min(w_pos/N,w_neg/N)/ min(L0_max,sum(L0_regularized_variables)); %see paper for derivation

%createSLIM create a Cplex object, slim_IP and provides useful info in slim_info
[slim_IP, slim_info] = createSLIM(input);

%set default CPLEX solver parameters
slim_IP.Param.emphasis.mip.Cur               = 1;   %mip solver strategy
slim_IP.Param.timelimit.Cur                  = 60;  %timelimit in seconds
slim_IP.Param.randomseed.Cur                 = 0;
slim_IP.Param.threads.Cur                    = 1;   %# of threads; >1 starts a parallel solver
slim_IP.Param.output.clonelog.Cur            = 0;   %disable CPLEX's clone log
slim_IP.Param.mip.tolerances.lowercutoff.Cur = 0;
slim_IP.Param.mip.tolerances.mipgap.Cur      = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.absmipgap.Cur   = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.integrality.Cur = eps; %use maximal precision for IP solver (only recommended for testing)

%solve the SLIM IP
slim_IP.solve

%once you solve the SLIM IP, you can solve it again
slim_IP.Param.timelimit.Cur = 30;
slim_IP.solve

%You can now get summary statistics using the getSLIMSummary function
summary = getSLIMSummary(slim_IP, slim_info);
assert(summary.coefficients(L0_regularized_variables)>=1)
assert(summary.coefficients(L0_regularized_variables)<=3)

%% Example 4: Imposing Hard Limits on TPR/FPR

%SLIM can set hard limits on the % of non-zero coefficients as follows

%Setup usual input struct
input                               = struct();
input.display_warnings              = true; 
input.X                             = X;      
input.X_names                       = X_names;
input.Y                             = Y;
input.Y_name                        = Y_name;
input.w_pos                         = 1.00;
input.w_neg                         = 1.00;
input.coefSet                      = newCoefSet(X_names);
changeCoefSetField(input.coefSet, '(Intercept)', 'C_0j', 0)            

%Add a loss constraint to limit the negative error (= 1 - FPR)
input.neg_err_max = 0.01; 

%Set w_pos large enough to guarantee that SLIM hits the FPR constraint
N_pos = sum(Y==1);
N_neg = N-N_pos;
input.w_pos = 2*N_neg/(1+N_neg);
input.w_neg = 2-input.w_pos;

%Set C_0j small enough to make sure that we do not sacrifice accuracy (not necessary in general)
L0_regularized_variables = [coefSet.C_0j]~=0.0;
input.C_0 = 0.9*min(input.w_pos/N,input.w_neg/N)/ min(sum(L0_regularized_variables));

%createSLIM create a Cplex object, slim_IP and provides useful info in slim_info
[slim_IP, slim_info] = createSLIM(input);

%set default CPLEX solver parameters
slim_IP.Param.emphasis.mip.Cur               = 1;   %mip solver strategy
slim_IP.Param.timelimit.Cur                  = 60;  %timelimit in seconds
slim_IP.Param.randomseed.Cur                 = 0;
slim_IP.Param.threads.Cur                    = 1;   %# of threads; >1 starts a parallel solver
slim_IP.Param.output.clonelog.Cur            = 0;   %disable CPLEX's clone log
slim_IP.Param.mip.tolerances.lowercutoff.Cur = 0;
slim_IP.Param.mip.tolerances.mipgap.Cur      = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.absmipgap.Cur   = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.integrality.Cur = eps; %use maximal precision for IP solver (only recommended for testing)

%solve the SLIM IP
slim_IP.solve

%check that FPR is > input.neg_err_max
summary = getSLIMSummary(slim_IP, slim_info);
assert(summary.false_positive_rate<=0.99)

%% Imposing Hard Limits on Accuracy
%TODO

%% Finding Multiple Solutions
%TODO

%% Solving in Parallel
%TODO








