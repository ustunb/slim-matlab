function varargout = createSLIM_BinaryData(input)
%Creates a CPLEX IP object whose solution will yield a SLIM scoring system
%This version is specially designed for cases where the data are binary
%and the coefficients are integers.
%
%input is a struct with following required fields:
%
%Y          N x 1 vector of labels (-1 or 1 only)
%X          N x P matrix of features
%           Must include a column of 1s to act as the intercept
%           All features must be binary
%X_names    Px1 cell array of strings containing the names of the feature values
%           To stay safe, label the intercept as '(Intercept)'
%
%input may also contain the following optional fields:
%
%Y_name             string/cell array containing the outcome variable name
%coefConstraints    SLIMCoefficientConstraints object with variable_name matching X_names
%                   only integer constraints are permitted
%
%C_0                sparsity penalty; must be a value between [0,1]
%w_pos              misclassification cost for positive labels
%w_neg              misclassification cost for negative labels
%L0_min             min # of non-zero coefficients in scoring system
%L0_max             max # of non-zero coefficients in scoring system
%err_min            min error rate of desired scoring system
%err_max            max error rate of desired scoring system
%pos_err_min        min error rate of desired scoring system on -ve examples
%pos_err_max        max error rate of desired scoring system on +ve examples
%neg_err_min        min error rate of desired scoring system on -ve examples
%neg_err_max        max error rate of desired scoring system on +ve examples
%
%Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%% Check Inputs

%check data
assert(isfield(input,'X'), 'input must contain X matrix')
assert(isfield(input,'Y'), 'input must contain Y vector')
assert(isnumeric(input.X), 'input.X must be numeric')
assert(isnumeric(input.Y), 'input.Y must be numeric')
assert(~any(any(isnan(input.X))), 'input.X must not contain any NaN entries')
assert(all(input.Y==-1|input.Y==1), 'all elements of input.Y must be -1 or 1')
X = input.X;
Y = input.Y(:);

%create basic data variables
[N, P] = size(X);
assert(size(Y,1)== N, 'X and Y have different numbers of rows')
pos_indices = Y==1;
neg_indices = ~pos_indices;
N_pos = sum(pos_indices);
N_neg = sum(neg_indices);

if ~isfield(input, 'Y_name')
    input.Y_name = 'Y';
elseif iscell(input.Y_name)
    input.Y_name = input.Y_name{1};
end
Y_name = input.Y_name;

%check coefficient set
if isfield(input, 'X_names') && isfield(input, 'coefConstraints')
    
    coefCons = input.coefConstraints;
    X_names  = input.X_names;
    assert(isequal(X_names(:), coefCons.variable_name), 'X_names does not match input.coefConstraints.variable_names');
    
elseif isfield(input,'X_names') && ~isfield(input, 'coefConstraints')
    
    if ~(iscell(input.X_names) && length(input.X_names)==P)
        error('X_names must be cell array with P entries\n')
    else
        X_names   = input.X_names;
    end
    
    print_warning('input does not contain coefConstraints; using default coefficient constraints');
    coefCons = SLIMCoefficientConstraints(X_names);
    
elseif ~isfield(input,'X_names') &&  isfield(input,'coefConstraints')
    
    print_warning('input does not contain X_names field; extracting variable names from coefConstraints')
    coefCons  = input.coefConstraints;
    X_names   = input.coefConstraints.variable_names;
    
elseif ~isfield(input,'X_names') &&  ~isfield(input,'coefConstraints')
    
    coefCons    = SLIMCoefficientConstraints(P);
    X_names     = coefCons.variable_names;
    print_warning('input does not contain coefConstraints field; using default coefficient constraints');
    
end

%check that data is binary
only_binary_features = all(all(input.X==0|input.X==1));
no_custom_coefficients = all(strcmp(coefCons.type, 'integer'));
assert(only_binary_features, 'all entries of X should be 0 or 1');
assert(no_custom_coefficients, 'all entries of X should be 0 or 1');

%compress data
N_o     = N;
N_pos_o = N_pos;
N_neg_o = N_neg;
X_o     = X;
[X, nY_pos, nY_neg, pos_loss_con, neg_loss_con, conflict_table, iu, ix]  = compress_data(X,Y);
[N, P] = size(X);

%% Default Settings

input = setdefault(input, 'C_0', 1e-3);
input = setdefault(input, 'w_pos', 1.000);
input = setdefault(input, 'w_neg', 1.000);
input = setdefault(input, 'C_1', NaN);
input = setdefault(input, 'M', NaN);
input = setdefault(input, 'L0_min', 0);
input = setdefault(input, 'L0_max', P);
input = setdefault(input, 'err_min', 0);
input = setdefault(input, 'err_max', 1);
input = setdefault(input, 'pos_err_min', 0);
input = setdefault(input, 'pos_err_max', 1);
input = setdefault(input, 'neg_err_min', 0);
input = setdefault(input, 'neg_err_max', 1);
input = setdefault(input, 'add_intercept_constraint', false);
input = setdefault(input, 'add_conflict_constraint', false);

%% Initialize Variables used in Creating IP Constraints

lambda_lb   = coefCons.lb;
lambda_ub   = coefCons.ub;
lambda_max  = max(abs(lambda_lb), abs(lambda_ub));

signs       = coefCons.sign;
sign_pos    = signs == 1;
sign_neg    = signs == -1;

%setup class-based weights
assert(input.w_pos > 0,'w_pos must be positive');
assert(input.w_neg > 0,'w_neg must be positive');
w_pos = input.w_pos;
w_neg = input.w_neg;

%renormalize weights
if (w_pos + w_neg) ~= 2
    tot = (w_pos + w_neg);
    print_warning(sprintf('w_pos + w_neg = %1.2f\nrenormalizing so that w_pos +w_neg = 2.00', tot))
    w_pos = 2*w_pos/tot;
    w_neg = 2*w_neg/tot;
end

%bounded L0 norm
L0_min          = input.L0_min;
L0_max          = input.L0_max;
if isnan(L0_min) || isinf(L0_min), L0_min = 0; end
if isnan(L0_max) || isinf(L0_max), L0_max = P; end
if L0_min > L0_max
    error('user specified L0_min (=%d) which is greater than L0_max (=%d)\n', L0_min, L0_max);
end
L0_min          = max(ceil(L0_min),0);
L0_max          = min(floor(L0_max),P);

%total error
err_min         = input.err_min;
err_max         = input.err_max;
if isnan(err_min) || isinf(err_min), err_min = 0; end
if isnan(err_max) || isinf(err_max), err_max = 1; end
if err_min > err_max
    error('user specified err_min (=%1.2f) which is greater than err_max (=%1.2f)\n',err_min,err_max)
end
err_min         = max(ceil(N_o*err_min),0);
err_max         = min(floor(N_o*err_max),N_o);

%total pos_error
pos_err_min         = input.pos_err_min;
pos_err_max         = input.pos_err_max;
if isnan(pos_err_min) || isinf(pos_err_min), pos_err_min = 0; end
if isnan(pos_err_max) || isinf(pos_err_max), pos_err_max = 1; end
if pos_err_min > pos_err_max
    error('user specified pos_err_min (=%1.2f) which is greater than pos_err_max (=%1.2f)\n',pos_err_min,pos_err_max)
end
pos_err_min         = max(ceil(N_pos_o*pos_err_min),0);
pos_err_max         = min(floor(N_pos_o*pos_err_max),N_pos_o);

%total neg_error
neg_err_min         = input.neg_err_min;
neg_err_max         = input.neg_err_max;
if isnan(neg_err_min) || isinf(neg_err_min), neg_err_min = 0; end
if isnan(neg_err_max) || isinf(neg_err_max), neg_err_max = 1; end
if neg_err_min > neg_err_max
    error('user specified neg_err_min (=%1.2f) which is greater than neg_err_max (=%1.2f)\n',neg_err_min,neg_err_max)
end
neg_err_min         = max(ceil(N_neg_o*neg_err_min),0);
neg_err_max         = min(floor(N_neg_o*neg_err_max),N_neg_o);

%total conflicts
min_conflicts  = bsxfun(@min,nY_pos(conflict_table(:,2),:),nY_neg(conflict_table(:,3),:));
err_min        = max(sum(min_conflicts), err_min);
conflict_min   = length(min_conflicts);
conflict_max   = sum(pos_loss_con) + sum(neg_loss_con) - length(min_conflicts);

%flags for whether or not we will add contraints
add_L0_norm_constraint      = (L0_min > 0) || (L0_max < P);
add_error_constraint        = ((err_min > 0) || (err_max < N_o));
add_pos_error_constraint    = ((pos_err_min > 0) || (pos_err_max < N_pos_o));
add_neg_error_constraint    = ((neg_err_min > 0) || (neg_err_max < N_neg_o));
add_conflict_constraint     = ((conflict_min > 0) || (conflict_max < sum(pos_loss_con) + sum(neg_loss_con)));
add_intercept_constraint    = input.add_intercept_constraint;

%% IP Formulation Parameters

%Regularization Parameters
C_0     = input.C_0 .* ones(P,1);
UC_0    = coefCons.C_0j;
UC_ind  = ~isnan(UC_0);

if any(UC_ind)
    print_warning(sprintf('user requested customized regularization parameters for %d variables', sum(UC_ind)))
    C_0(UC_ind)  = UC_0(UC_ind);
    C_0 = C_0(:);
end
assert(all(C_0>=0), 'user specified negative value for C_0')

%identify variables that will have L0-regularization
L0_reg_ind  = C_0 > 0;

%adjust L0_max if there are variables with no L0 regularization
L0_max = min(L0_max, sum(L0_reg_ind));

%set L1 penalty to be as large as possible without adding regularization
L1_reg_ind = L0_reg_ind;
L1_max = sort(lambda_max(L1_reg_ind),'descend');
L1_max = sum(L1_max(1:L0_max));
if ~isnan(input.C_1)
    assert(input.C_1 > 0, 'if user supplies C_1 then it must be positive');
    C_1 = input.C_1;
else
    C_1 = 0.975.*min([w_pos/N_o, w_neg/N_o, min(C_0(L0_reg_ind))])./L1_max;
end
C_1                 = C_1.*ones(P,1);
C_1(~L1_reg_ind)    = 0;

%Loss Constraint Parameters
M        = input.M(:);

if isnan(M)
    
    Z_ub                = bsxfun(@times,X,lambda_ub');
    Z_lb                = bsxfun(@times,X,lambda_lb');
    Z_min               = bsxfun(@min,Z_lb,Z_ub);
    Z_max               = bsxfun(@max,Z_lb,Z_ub);
    Z_min_reg           = sort(Z_min(:,L0_reg_ind),2,'ascend');
    S_min_reg           = sum(Z_min_reg(:,1:L0_max),2);
    S_min_no_reg        = sum(Z_min(:,~L0_reg_ind),2);
    M                   = 1 - S_min_reg -S_min_no_reg;
    
    if any(neg_loss_con)
        Z_max_reg           = sort(Z_max(:,L0_reg_ind),2,'descend');
        S_max_reg           = sum(Z_max_reg(:,1:L0_max),2);
        S_max_no_reg        = sum(Z_max(:,~L0_reg_ind),2);
        M_neg               = S_max_reg + S_max_no_reg;
        M(neg_loss_con)     = M_neg(neg_loss_con);
    end
    
end

%% Setup Matrices for Bounded Variables
nZ_pos              = sum(pos_loss_con);
nZ_neg              = sum(neg_loss_con);
pos_errcost         = w_pos*nY_pos(pos_loss_con);
neg_errcost         = w_neg*nY_neg(neg_loss_con);
l0cost              = N_o*C_0;
l1cost              = N_o*C_1;
sparse_P_x_1_Inf    = sparse(Inf(P,1));

%Compute Indices
n_vars = 0;
lambda_pos_ind = (n_vars+1):(n_vars+P);         n_vars = n_vars+P;
lambda_neg_ind = (n_vars+1):(n_vars+P);         n_vars = n_vars+P;
alpha_pos_ind  = (n_vars+1):(n_vars+P);         n_vars = n_vars+P;
alpha_neg_ind  = (n_vars+1):(n_vars+P);         n_vars = n_vars+P;
pos_err_ind    = (n_vars+1):(n_vars+nZ_pos);    n_vars = n_vars+nZ_pos;
neg_err_ind    = (n_vars+1):(n_vars+nZ_neg);    n_vars = n_vars+nZ_neg;

%
%z_pos[1]...z_pos[nZ_pos]           = +ve errors on x[i] i = find(pos_loss_con) == nY_pos>0;
%z_neg[1]...z_neg[nL_neg]           = -ve errors on x[i] i = find(neg_loss_con) == nY_neg>0 and nY_pos==0;
%z_neg[1]...z_neg[nL_neg+1:nZ_neg]  = -ve errors on x[i] i = nY_pos>0 and nY_neg>0;

%variables  =        lambda_pos (P)         lambda_neg (P)  alpha_pos (P)   alpha_neg (P)  pos_errors (nZ_pos)     negative errors (nZ_neg))
obj                 = sparse([l1cost;       -l1cost;        l0cost;         l0cost;        pos_errcost;            neg_errcost]);
ub                  = sparse([lambda_ub;    zeros(P,1);     ones(P+P+nZ_pos+nZ_neg,1)]);  %ones(nZ_pos,1)          ones(nZ_neg,1)];
lb                  = sparse([zeros(P,1);   lambda_lb;      zeros(P+P+nZ_pos+nZ_neg,1)]); %zeros(nZ_pos,1)         zeros(nZ_neg,1)]);
ctype               = ['I'*ones(1,P),       'I'*ones(1,P),  'B'*ones(1,P+P+nZ_pos+nZ_neg)];

%nZ_pos constraints for Positive Errors
A_1                 = [sparse(X(pos_loss_con,:)),sparse(X(pos_loss_con,:)),sparse(nZ_pos,P+P),sparse(1:nZ_pos,1:nZ_pos,M(pos_loss_con),nZ_pos,nZ_pos),sparse(nZ_pos,nZ_neg)];
lhs_1               = sparse(ones(nZ_pos,1));
rhs_1               = sparse(Inf(nZ_pos,1));

%nZ_neg constraints for Negative Errors
A_2                 = [sparse(-X(neg_loss_con,:)),sparse(-X(neg_loss_con,:)),sparse(nZ_neg,P+P),sparse(nZ_neg,nZ_pos),sparse(1:nZ_neg,1:nZ_neg,M(neg_loss_con),nZ_neg,nZ_neg)];
lhs_2               = sparse(nZ_neg,1);
rhs_2               = sparse(Inf(nZ_neg,1));

%Add new constraints that directly model the conflict
nL_neg              = length(conflict_table(:,3));
A_3                 = [sparse(nL_neg,P+P+P+P),sparse(1:nL_neg,conflict_table(:,2),ones(nL_neg,1),nL_neg,nZ_pos), sparse(1:nL_neg,conflict_table(:,3),ones(nL_neg,1),nL_neg,nZ_neg)];
lhs_3               = sparse(ones(nL_neg,1));
rhs_3               = sparse(ones(nL_neg,1));

% 0-Norm LB Constraints:
%alpha_neg[j] * lb[j] < lambda_neg[j]
%0 < lambda_neg[j] - alpha_neg[j]*lb[j] < Inf
A_4                 = [sparse(P,P), speye(P), sparse(P,P), -sparse(1:P,1:P,lambda_lb), sparse(P,nZ_pos+nZ_neg)];
lhs_4               = sparse(P,1);
rhs_4               = sparse_P_x_1_Inf;

%remove constraints for any j s.t. C_0j = 0 or lambda_j > 0
% to_drop             = ~L0_reg_ind | L0_reg_ind & sign_pos;
% A_4(to_drop,:)      = [];
% lhs_4(to_drop,:)    = [];
% rhs_4(to_drop,:)    = [];

% 0-Norm UB Constraints: lambda_j < lambda_j,ub*alpha_pos[j] --> 0 < -lambda_j + lambda_j,ub*alpha_pos[j]< Inf
A_5                 = [-speye(P), sparse(P,P), sparse(1:P,1:P,lambda_ub), sparse(P,P), sparse(P,nZ_pos+nZ_neg)];
lhs_5               = sparse(P,1);
rhs_5               = sparse_P_x_1_Inf;

%remove constraints for any j s.t. C_0j = 0 or lambda_j < 0
% to_drop          = ~L0_reg_ind | L0_reg_ind & sign_neg;
% A_5(to_drop,:)   = [];
% lhs_5(to_drop,:) = [];
% rhs_5(to_drop,:) = [];

% 0-Norm RHS Constraints:
% alpha_pos[j] < lambda_pos[j]
% 0 <= lambda_pos[j] - alpha_pos[j] <= Inf
A_6                 = [speye(P), sparse(P,P), -speye(P), sparse(P,P), sparse(P,nZ_pos+nZ_neg)];
lhs_6               = sparse(P,1);
rhs_6               = sparse_P_x_1_Inf;

% to_drop             = ~L0_reg_ind;
% A_6(to_drop,:)      = [];
% lhs_6(to_drop,:)    = [];
% rhs_6(to_drop,:)    = [];

% 0-Norm RHS Constraints:
% lambda_neg[j] <= -alpha_neg[j]
% 0 <= -lambda_neg[j] -alpha_neg[j] <= Inf
A_7                 = [sparse(P,P), -speye(P), sparse(P,P), -speye(P), sparse(P,nZ_pos+nZ_neg)];
lhs_7               = sparse(P,1);
rhs_7               = sparse_P_x_1_Inf;

% to_drop          = ~L0_reg_ind;
% A_7(to_drop,:)   = [];
% lhs_7(to_drop,:) = [];
% rhs_7(to_drop,:) = [];

% 0-Norm RHS Constraints: lambda_neg[j]
% 0 <= alpha_pos[j] + alpha_neg[j] <= 1
A_8 = [sparse(P,P), sparse(P,P), speye(P), speye(P), sparse(P,nZ_pos+nZ_neg)];
lhs_8 = sparse(P,1);
rhs_8 = sparse(ones(P,1));

% to_drop          = ~L0_reg_ind;
% A_8(to_drop,:)   = [];
% lhs_8(to_drop,:) = [];
% rhs_8(to_drop,:) = [];

%merge all matrices together
A   = [A_1;A_2;A_3;A_4;A_5; A_6; A_7; A_8];
lhs = [lhs_1; lhs_2; lhs_3; lhs_4; lhs_5; lhs_6; lhs_7;lhs_8];
rhs = [rhs_1; rhs_2; rhs_3; rhs_4; rhs_5; rhs_6; rhs_7;rhs_8];

%constraint indices
m = 0;
pos_loss_constraint_ind             = (m+1):(m+size(A_1,1)); m = m+length(pos_loss_constraint_ind);
neg_loss_constraint_ind             = (m+1):(m+size(A_2,1)); m = m+length(neg_loss_constraint_ind);
neg_loss_conflict_constraint_ind    = (m+1):(m+size(A_3,1)); m = m+length(neg_loss_conflict_constraint_ind);
n_loss_constraints                  = m;

%% Hard Constraints
intercept_constraint_ind    = NaN;
L0_norm_constraint_ind      = NaN;
conflict_constraint_ind     = NaN;
error_constraint_ind        = NaN;
pos_error_constraint_ind    = NaN;
neg_error_constraint_ind    = NaN;

if add_intercept_constraint
    
    % Intercept UB Constraint #1
    % lambda_pos[1] <= -lambda_neg[2:P] --> 0 <= -lambda_pos[1] -lambda_neg[2:P]  <= Inf
    A_add       = [sparse([-1 zeros(1,P-1)]),-sparse([0 ones(1,P-1)]), sparse(1,P+P+nZ_pos+nZ_neg)];
    lhs_add     = sparse(0);
    rhs_add     = sparse(Inf);
    
    A           = [A;A_add];
    lhs         = [lhs;lhs_add];
    rhs         = [rhs;rhs_add];
    
    intercept_constraint_ind = size(A,1);
    
    % Intercept Lower Bound Constraint #1
    % 1 - lambda_pos[2:P] <= lambda_pos[1] --> 1 <= lambda_pos[1:P] <= Inf
    A_add   = [sparse(ones(1,P)), sparse(1,P), sparse(1,P+P+nZ_pos+nZ_neg)];
    lhs_add = sparse(1);
    rhs_add = sparse(Inf);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    intercept_constraint_ind = [intercept_constraint_ind, size(A,1)];
    
    % Intercept Lower Bound Constraint #2
    % 1 - lambda_pos[2:P] <= lambda_neg[1] --> 1 <= lambda_pos[2:P] + lambda_neg[1] <= Inf
    A_add   = [sparse([0 ones(1,P-1)]),sparse([1 zeros(1,P-1)]), sparse(1,P+P+nZ_pos+nZ_neg)];
    lhs_add = sparse(1);
    rhs_add = sparse(Inf);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    intercept_constraint_ind = [intercept_constraint_ind, size(A,1)];
    
end

if add_L0_norm_constraint
    
    
    A_add   = [sparse(1,P+P), sparse(L0_reg_ind(:)'), sparse(L0_reg_ind(:)'), sparse(1,nZ_pos+nZ_neg)];
    lhs_add = sparse(L0_min);
    rhs_add = sparse(L0_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    L0_norm_constraint_ind = size(A,1);
    
end

if add_conflict_constraint
    
    A_add   = [sparse(1,P+P+P+P), sparse(ones(1,nZ_pos+nZ_neg))];
    lhs_add = sparse(conflict_min);
    rhs_add = sparse(conflict_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
    conflict_constraint_ind = size(A,1);
    
end


if add_error_constraint
    
    A_add   = [sparse(1,P+P+P+P), sparse([nY_pos(pos_loss_con,:);nY_neg(neg_loss_con,:)]')];
    lhs_add = sparse(err_min);
    rhs_add = sparse(err_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
    error_constraint_ind = size(A,1);
    
end

if add_pos_error_constraint
    
    A_add   = [sparse(1,P+P+P+P), sparse(nY_pos(pos_loss_con,:)'), sparse(1,nZ_neg)];
    lhs_add = sparse(pos_err_min);
    rhs_add = sparse(pos_err_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
    pos_error_constraint_ind = size(A,1);
    
end

if add_neg_error_constraint
    
    A_add   = [sparse(1,P+P+P+P), sparse(1,nZ_pos), sparse(nY_neg(neg_loss_con,:)')];
    lhs_add = sparse(neg_err_min);
    rhs_add = sparse(neg_err_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
    neg_error_constraint_ind = size(A,1);
    
end

%% Drop Redundant Variables to avoid Potential CPLEX Issues

%drop alpha[j] if no L0_regularization
%drop lambda_pos[j] if lambda[j] \leq 0
%drop lambda_neg[j] if lambda[j] \geq 0
alpha_pos_drop_ind    = alpha_pos_ind(~L0_reg_ind);
alpha_neg_drop_ind    = alpha_neg_ind(~L0_reg_ind);
lambda_pos_drop_ind   = lambda_pos_ind(sign_pos);
lambda_neg_drop_ind   = lambda_neg_ind(sign_neg);
drop_ind              = [lambda_pos_drop_ind,lambda_neg_drop_ind,alpha_pos_drop_ind,alpha_neg_drop_ind];
drop_ind              = []; %%%% TODO

obj(drop_ind)         = [];
ctype(drop_ind)       = [];
ub(drop_ind)          = [];
lb(drop_ind)          = [];
A(:,drop_ind)         = [];

%Recompute Indices
if any(drop_ind)
    
    n_alpha_pos_dropped     = length(alpha_pos_drop_ind);
    n_alpha_neg_dropped     = length(alpha_neg_drop_ind);
    n_lambda_pos_dropped    = length(lambda_pos_drop_ind);
    n_lambda_neg_dropped    = length(lambda_neg_drop_ind);
    
    n_vars = 0;
    lambda_pos_ind = (n_vars+1):(n_vars+P-n_lambda_pos_dropped);        n_vars = n_vars+(P-n_lambda_pos_dropped);
    lambda_neg_ind = (n_vars+1):(n_vars+P-n_lambda_neg_dropped);        n_vars = n_vars+(P-n_lambda_neg_dropped);
    alpha_pos_ind  = (n_vars+1):(n_vars+P-n_alpha_pos_dropped);         n_vars = n_vars+(P-n_alpha_pos_dropped);
    alpha_neg_ind  = (n_vars+1):(n_vars+P-n_alpha_neg_dropped);         n_vars = n_vars+(P-n_alpha_neg_dropped);
    pos_err_ind    = (n_vars+1):(n_vars+nZ_pos);                        n_vars = n_vars+nZ_pos;
    neg_err_ind    = (n_vars+1):(n_vars+nZ_neg);                        n_vars = n_vars+nZ_neg;
    
end

%% Return Values

%IP Formulation
if nargout > 0
    
    IP = Cplex();
    
    Model = struct();
    Model.sense = 'minimize';
    Model.obj    = obj;
    Model.rhs    = rhs;
    Model.lhs    = lhs;
    Model.A      = A;
    Model.ub     = ub;
    Model.lb     = lb;
    Model.ctype  = char(ctype);
    
    IP.Model     = Model;
    
    varargout{1} = IP;
end

%Debug Inforamtion
if nargout == 2
    
    info = struct();
    
    %IP formulation
    info.version = 'binary';
    info.only_binary_features = only_binary_features;
    info.no_custom_coefficients = no_custom_coefficients;
    
    %default fields
    info.X              = X_o;
    info.Y              = Y;
    info.X_names        = X_names;
    info.Y_name         = Y_name;
    info.L0_min         = L0_min;
    info.L0_max         = L0_max;
    info.err_min        = err_min;
    info.err_max        = err_max;
    info.pos_err_min    = pos_err_min;
    info.pos_err_max    = pos_err_max;
    info.neg_err_min    = neg_err_min;
    info.neg_err_max    = neg_err_max;
    
    %data compression details
    info.Xa                 = X;
    info.ix                 = ix;
    info.iu                 = iu;
    info.pos_loss_con       = pos_loss_con;
    info.neg_loss_con       = neg_loss_con;
    info.nY_pos             = nY_pos;
    info.nY_neg             = nY_neg;
    info.conflict_table     = conflict_table;
    
    %IP parameters
    info.M              = M;
    info.L0_reg_ind     = L0_reg_ind;
    info.L1_reg_ind     = L1_reg_ind;
    info.C_0            = C_0;
    info.C_1            = C_1;
    
    %IP variable information
    info.n_vars                     = n_vars;
    info.indices.lambda_pos         = lambda_pos_ind;
    info.indices.lambda_neg         = lambda_neg_ind;
    info.indices.alpha_pos          = alpha_pos_ind;
    info.indices.alpha_neg          = alpha_neg_ind;
    info.indices.pos_errors         = pos_err_ind;
    info.indices.neg_errors         = neg_err_ind;
    info.alpha_pos_drop_ind         = alpha_pos_drop_ind;
    info.alpha_neg_drop_ind         = alpha_neg_drop_ind;
    info.lambda_pos_drop_ind        = lambda_pos_drop_ind;
    info.lambda_neg_drop_ind        = lambda_neg_drop_ind;
    
    %IP constraint information
    info.n_loss_constraints                 = n_loss_constraints;
    info.pos_loss_constraint_ind            = pos_loss_constraint_ind;
    info.neg_loss_constraint_ind            = neg_loss_constraint_ind;
    info.neg_loss_conflict_constraint_ind   = neg_loss_conflict_constraint_ind;
    
    %IP extra constraint infomration
    info.add_conflict_constraint    = add_conflict_constraint;
    info.add_L0_norm_constraint     = add_L0_norm_constraint;
    info.add_intercept_constraint   = add_intercept_constraint;
    info.add_error_constraint       = add_error_constraint;
    info.add_pos_error_constraint   = add_pos_error_constraint;
    info.add_neg_error_constraint   = add_neg_error_constraint;
    
    info.intercept_constraint_ind      = intercept_constraint_ind;
    info.L0_norm_constraint_ind        = L0_norm_constraint_ind;
    info.conflict_constraint_ind       = conflict_constraint_ind;
    info.error_constraint_ind          = error_constraint_ind;
    info.pos_error_constraint_ind      = pos_error_constraint_ind;
    info.pos_error_constraint_ind      = pos_error_constraint_ind;
    
    info.input = input;
    varargout{2} = info;
    
end

%% Helper Functions

    function settingfile = setdefault(settingfile,settingname,defaultvalue)
        if ~isfield(settingfile,settingname)
            settingfile.(settingname) = defaultvalue;
        end
    end

    function print_warning(msg)
        warning('SLIM:IPWarning', msg)
    end


    function [Xa, nY_pos, nY_neg, pos_loss_con, neg_loss_con, conflict_table, iu, ix] = compress_data(X,Y)
        %given a feature matrix X, returns Xa which contains no duplicate rows
        %iu and ix are transformation indices such that
        %
        %X == Xa(ix,:)
        %Xa == X(iu,:)
        %
        %nY_pos(k) is the # of points with features Xa(k,:) where Y(ix(k)) == +1
        %nY_neg(k) is the # of points with features Xa(k,:) where Y(ix(k)) == -1
        %conflict_table contains the indices of rows in Xa that are labelled as
        %+1 and -1 in the uncompressed dataset. specifically
        %
        %conflict_table = (m, nY_pos(m), nY_neg(m))
        %
        %where nY_pos(m) > 0, nY_neg(m) > 0
        
        [Xa,iu,ix]       = unique(X,'rows','stable');
        n               = length(iu);
        nY_pos          = zeros(n,1);
        nY_neg          = zeros(n,1);
        conflict_table  = zeros(n,3);
        
        i_pos = 1;i_neg = 1;
        
        for i = 1:n
            
            x_locations     = ix==i;
            nY_pos(i)       = sum(Y(x_locations)==1);
            nY_neg(i)       = sum(Y(x_locations)~=1);
            
            if (nY_pos(i)>0) && (nY_neg(i)>0);
                conflict_table(i,:) = [i,i_pos,i_neg];
                i_pos = i_pos + 1;
                i_neg = i_neg + 1;
            elseif (nY_pos(i) > 0) && (nY_neg(i)==0)
                i_pos  = i_pos + 1;
            elseif (nY_neg(i) > 0) && (nY_pos(i)==0)
                i_neg = i_neg + 1;
            end
            
        end
        
        pos_loss_con        = nY_pos > 0;
        neg_loss_con        = nY_neg > 0;
        conflict_table      = conflict_table(conflict_table(:,1)>0,:);
        
    end

end