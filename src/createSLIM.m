function varargout = createSLIM(input)
%Creates a Cplex integer program whose solution is a SLIM scoring system
%requires a working installation of IBM CPLEX
%
%input is a struct that has to contain the following required fields:
%
%X          N x P matrix of feature values (should contain a column of 1s to model an intercept)
%Y          N x 1 vector of labels (-1 or 1 only)
%X_names    P x 1 cell array of strings containing the names of the feature values
%           To stay safe, label the intercept as '(Intercept)'
%
%Y_name     string or cell array containing the outcome variable name
%
%in addition, input can also contain the following optional fields:
%
%coefSet        coefSet object (set as default coefSet if not specified)
%C_0            L0-regularization parameter; must be a value between [0,1]
%w_pos          misclassification cost for positive labels
%w_neg          misclassification cost for negative labels
%L0_min         min # of non-zero coefficients in scoring system
%L0_max         max # of non-zero coefficients in scoring system
%err_min        min error rate of desired scoring system
%err_max        max error rate of desired scoring system
%pos_err_min    min error rate of desired scoring system on -ve examples
%pos_err_max    max error rate of desired scoring system on +ve examples
%neg_err_min    min error rate of desired scoring system on -ve examples
%neg_err_max    max error rate of desired scoring system on +ve examples
%
%Author:      Berk Ustun 
%Contact:     ustunb@mit.edu
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

input = setdefault(input, 'display_warnings', true);
if input.display_warnings
    print_warning = @(warning_msg) disp([]);
else
    print_warning = @(warning_msg) fprintf('%s\n', warning_msg);
end


%% Check Inputs
%check data
assert(isfield(input,'X'), 'input must contain X matrix')
assert(isfield(input,'Y'), 'input must contain Y vector')
assert(isnumeric(input.X), 'input.X must be numeric')
assert(isnumeric(input.Y), 'input.Y must be numeric')
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
if ~isfield(input,'X_names') &&  ~isfield(input,'coefSet')
    
    print_warning(sprintf('input does not contain X_names; naming features as X1,...,X%d',P))
    X_names = arrayfun(@(x) sprintf('X%d',x),1:P,'UniformOutput',false);
    
    print_warning('input does not contain coefSet; using default coefSet');
    coefSet = newCoefSet(X_names);
    
elseif ~isfield(input,'X_names') &&  isfield(input,'coefSet')
    
    print_warning('input does not contain X_names; extracting variable names from coefSet \n')
    coefSet     = checkCoefSet(input.coefSet);
    X_names   = {coefSet(:).name}';
    
elseif isfield(input,'X_names') && ~isfield(input, 'coefSet')
    
    print_warning('input does not contain coefSet; creating default coefSet');
    
    if ~(iscell(input.X_names) && length(input.X_names)==P)
        error('X_names must be cell array with P entries\n')
    else
        X_names   = input.X_names;
    end
    coefSet = newCoefSet(X_names);
    
elseif isfield(input,'X_names') && isfield(input, 'coefSet')
    
    if ~(iscell(input.X_names) && length(input.X_names)==P)
        error('X_names must be cell array with P entries\n')
    else
        X_names   = input.X_names;
    end
    
    coefSet     = checkCoefSet(input.coefSet);
    
end

%% Default Settings

input = setdefault(input, 'C_0', 1e-3);
input = setdefault(input, 'w_pos', 1.000);
input = setdefault(input, 'w_neg', 1.000);
input = setdefault(input, 'C_1', NaN);
input = setdefault(input, 'M', NaN);
input = setdefault(input, 'epsilon', 0.001);
input = setdefault(input, 'L0_min', 0);
input = setdefault(input, 'L0_max', P);
input = setdefault(input, 'err_min', 0);
input = setdefault(input, 'err_max', 1);
input = setdefault(input, 'pos_err_min', 0);
input = setdefault(input, 'pos_err_max', 1);
input = setdefault(input, 'neg_err_min', 0);
input = setdefault(input, 'neg_err_max', 1);

%% Initialize Variables used in Creating IP Constraints

lambda_lb   = [coefSet(:).lb];
lambda_ub   = [coefSet(:).ub];
lambda_lb = lambda_lb(:);
lambda_ub = lambda_ub(:);
lambda_max  = max(abs(lambda_lb), abs(lambda_ub));

signs       = [coefSet(:).sign]';
signs       = signs(:);
sign_pos    = signs==1;
sign_neg    = signs==-1;
sign_fixed  = sign_pos | sign_neg;

%extract values for variables with a customized coefficient set;
custom_values       = cellfun(@(Lj) Lj(~isnan(Lj)), {coefSet(:).values}', 'UniformOutput',false);
custom_values       = cellfun(@(Lj) setdiff(Lj,0), custom_values, 'UniformOutput',false);

%determine variable types
types           = {coefSet(:).type}';
custom_ind      = strcmp(types,'custom');
int_type_ind    = zeros(1,P);
cts_type_ind    = zeros(1,P);
for j = 1:P
    switch types{j}
        case 'integer'
            int_type_ind(j) = 1;
        case 'custom'
            if ~isnan(coefSet(j).values)
                if all(coefSet(j).values==floor(coefSet(j).values))
                    int_type_ind(j) = 1;
                else
                    cts_type_ind(j) = 1;
                end
            end
    end
end
int_type_ind = logical(int_type_ind);
cts_type_ind = logical(cts_type_ind);


%setup class-based weights
assert(input.w_pos > 0,'w_pos must be a positive numeric value');
assert(input.w_neg > 0,'w_neg must be a positive numeric value');
w_pos = input.w_pos;
w_neg = input.w_neg;

%renormalize weights
if (w_pos+w_neg) ~=2
    tot = (w_pos + w_neg);
    print_warning(sprintf('w_pos + w_neg = %1.2f\nrenormalizing so that w_pos +w_neg = 2.00', tot))
    w_pos = 2*w_pos/tot;
    w_neg = 2*w_neg/tot;
end

%Regularization Parameters
C_0     = input.C_0 .* ones(P,1);
UC_0    = [coefSet(:).C_0j]';
UC_ind  = ~isnan(UC_0);

if any(UC_ind)
    print_warning(sprintf('user requested customized regularization parameters for %d variables', sum(UC_ind)))
    C_0(UC_ind)  = UC_0(UC_ind);
    C_0 = C_0(:);
end

%identify variables that will have L0-regularization
L0_reg_ind  = C_0 > 0;
L1_reg_ind  = L0_reg_ind;

C_1         = input.C_1;
if isnan(C_1)
    C_1   = 0.5.*min([w_pos/N,w_neg/N, min(C_0(L1_reg_ind))])./(sum(lambda_max));
end
C_1                 = C_1.*ones(P,1);
C_1(~L1_reg_ind)    = 0;

%Loss Constraint Parameters
epsilon  = input.epsilon;
M        = input.M(:);

if isnan(M)
    M = sum(abs(X*lambda_max),2)+1.1*epsilon;
end

XY  = bsxfun(@times,X,Y);

%% Hard Constraint Preprocessing

%bounded L0 norm
L0_min          = input.L0_min;
L0_max          = input.L0_max;
if isnan(L0_min) || isinf(L0_min), L0_min = 0; end
if isnan(L0_max) || isinf(L0_max), L0_max = P; end
if L0_min > L0_max
    error('user specified L0_min (=%d) which is greater than L0_max (=%d)\n',L0_min,L0_max)
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
err_min         = max(ceil(N*err_min),0);
err_max         = min(floor(N*err_max),N);

%total pos_error
pos_err_min         = input.pos_err_min;
pos_err_max         = input.pos_err_max;
if isnan(pos_err_min) || isinf(pos_err_min), pos_err_min = 0; end
if isnan(pos_err_max) || isinf(pos_err_max), pos_err_max = 1; end
if pos_err_min > pos_err_max
    error('user specified pos_err_min (=%1.2f) which is greater than pos_err_max (=%1.2f)\n',pos_err_min,pos_err_max)
end
pos_err_min         = max(ceil(N_pos*pos_err_min),0);
pos_err_max         = min(floor(N_pos*pos_err_max),N_pos);

%total neg_error
neg_err_min         = input.neg_err_min;
neg_err_max         = input.neg_err_max;
if isnan(neg_err_min) || isinf(neg_err_min), neg_err_min = 0; end
if isnan(neg_err_max) || isinf(neg_err_max), neg_err_max = 1; end
if neg_err_min > neg_err_max
    error('user specified neg_err_min (=%1.2f) which is greater than neg_err_max (=%1.2f)\n',neg_err_min,neg_err_max)
end
neg_err_min         = max(ceil(N_neg*neg_err_min),0);
neg_err_max         = min(floor(N_neg*neg_err_max),N_neg);

%flags for whether or not we will add contraints
add_L0_norm_constraint      = (L0_min > 0) || (L0_max < P);
add_err_constraint          = (err_min > 0) || (err_max < N);
add_pos_err_constraint      = (pos_err_min > 0) || (pos_err_max < N_pos);
add_neg_err_constraint      = (neg_err_min > 0) || (neg_err_max < N_neg);

%% Setup Standard Constraint Matrices

%lambda = P x 1 vector of coefficient values
%alpha  = P x 1 vector of L0-norm variables, alpha(j) = 1 if lambda_j != 0
%beta   = P x 1 vector of L1-norm variables, beta(j) = abs(lambda_j)
%error  = N x 1 vector of loss variables, error(i) = 1 if error on X(i)

%Objective
l0cost  = N*C_0;
l1cost  = N*C_1;
errcost = (w_pos*pos_indices + w_neg*neg_indices);

%variables  = lambda's (P)          alpha (P)       beta (P)        errors (N)
obj         = sparse([zeros(P,1);   l0cost;         l1cost;         errcost]);
ub          = sparse([lambda_ub;    ones(P,1) ;     lambda_max;     ones(N,1)]);
lb          = sparse([lambda_lb;    zeros(P,1);     zeros(P,1);     zeros(N,1)]);
ctype       = ['I'*ones(1,P),       'B'*ones(1,P),  'C'*ones(1,P),  'B'*ones(1,N)];

%fix coefficient types
ctype(int_type_ind)             = 'I'; %lambda
ctype(2*P+find(int_type_ind))   = 'I';
ctype(cts_type_ind)             = 'C';
ctype(2*P+find(cts_type_ind))   = 'C';

sparse_N_x_1_Inf    = sparse(Inf(N,1));
sparse_P_x_1_Inf    = sparse(Inf(P,1));
sparse_N_x_1_ones   = sparse(ones(N,1));

%Loss Constraints
%Enforce Z_1 = 1 if incorrect classification)
A_loss         = [sparse(XY),sparse(N,P+P),spdiags(M,0,N,N)];
lhs_loss       = epsilon.*sparse_N_x_1_ones;
rhs_loss       = sparse_N_x_1_Inf;

%L0-Norm LB Constraints
%actual constraint: lambda_j,lb < lambda_j
%cplex constraint:  0 < lambda_j - lambda_j,lb < Inf
A_L0_lb         = [speye(P), -sparse(diag(lambda_lb)), sparse(P,P), sparse(P,N)];
lhs_L0_lb       = sparse(P,1);
rhs_L0_lb       = sparse_P_x_1_Inf;

%drop constraints for any j s.t. C_0j = 0 or lambda_j > 0
to_drop             = ~L0_reg_ind | L0_reg_ind & sign_pos;
A_L0_lb(to_drop,:)      = [];
lhs_L0_lb(to_drop,:)    = [];
rhs_L0_lb(to_drop,:)    = [];

%L0-Norm RHS Constraints
%actual constraint: lambda_j < lambda_j,ub
%cplex constraint:  0 < -lambda_j + lambda_j,ub < Inf
A_L0_ub = [-speye(P), sparse(diag(lambda_ub)), sparse(P,P), sparse(P,N)];
lhs_L0_ub = sparse(P,1);
rhs_L0_ub = sparse_P_x_1_Inf;

%drop constraints for any j s.t. C_0j = 0 or lambda_j < 0
to_drop          = ~L0_reg_ind | L0_reg_ind & sign_neg;
A_L0_ub(to_drop,:)   = [];
lhs_L0_ub(to_drop,:) = [];
rhs_L0_ub(to_drop,:) = [];

%L1-Norm LHS Constraints
%actual constraint: -lambda_j < beta_j
%cplex constraint:  lambda_j + beta_j < Inf
A_L1_lb = [speye(P), sparse(P,P), speye(P), sparse(P,N)];
lhs_L1_lb = sparse(P,1);
rhs_L1_lb = sparse_P_x_1_Inf;

%L1-Norm RHS Constraints
%actual constraint: lambda_j < beta_j
%cplex constraint:  -lambda_j + beta_j < Inf
A_L1_ub = [-speye(P),   sparse(P,P), speye(P), sparse(P,N)];
lhs_L1_ub = sparse(P,1);
rhs_L1_ub = sparse_P_x_1_Inf;

%drop 1-norm constraints for coefficients with no regularization or fixed sign
to_drop = ~L1_reg_ind | L1_reg_ind & sign_fixed;
A_L1_lb(to_drop,:)   = [];
lhs_L1_lb(to_drop,:) = [];
rhs_L1_lb(to_drop,:) = [];

A_L1_ub(to_drop,:)   = [];
lhs_L1_ub(to_drop,:) = [];
rhs_L1_ub(to_drop,:) = [];

%merge all matrices together
A   = [A_loss;A_L0_lb;A_L0_ub;A_L1_lb;A_L1_ub];
lhs = [lhs_loss; lhs_L0_lb; lhs_L0_ub; lhs_L1_lb; lhs_L1_ub];
rhs = [rhs_loss; rhs_L0_lb; rhs_L0_ub; rhs_L1_lb; rhs_L1_ub];


%% Hard Constraints

if add_L0_norm_constraint
    
    A_add   = [sparse(1,P), sparse(ones(1,P)), sparse(1,P), sparse(1,N)];
    lhs_add = sparse(L0_min);
    rhs_add = sparse(L0_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
end


if add_err_constraint
    
    A_add   = [sparse(1,P), sparse(1,P), sparse(1,P), sparse(ones(1,N))];
    lhs_add = sparse(err_min);
    rhs_add = sparse(err_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
end

if add_neg_err_constraint
    
    A_add   = [sparse(1,P), sparse(1,P), sparse(1,P), sparse(neg_indices(:)')];
    lhs_add = sparse(neg_err_min);
    rhs_add = sparse(neg_err_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
end


if add_pos_err_constraint
    
    A_add   = [sparse(1,P), sparse(1,P), sparse(1,P), sparse(pos_indices(:)')];
    lhs_add = sparse(pos_err_min);
    rhs_add = sparse(pos_err_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
end

%% Setup 1-of-K Constraints for Coefficients Belong to a Custom Set

% identify IP variables that need to be dropped
alpha_drop_ind  = ~L0_reg_ind; %any j with C_0j = 0;
beta_drop_ind   = ~L1_reg_ind | L1_reg_ind & sign_fixed; %any j with C1j = 0 or fixed sign
n_alpha_drop    = sum(alpha_drop_ind);
n_beta_drop     = sum(beta_drop_ind);
n_to_drop       = n_alpha_drop + n_beta_drop;
n_alpha         = P - n_alpha_drop;
n_beta          = P - n_beta_drop;

if sum(custom_ind) == 0
    
    n_newcols = 0;
    
else
    
    [n_rows,n_cols] = size(A);
    
    n_custom    = sum(custom_ind);
    n_newcols   = sum(cellfun(@(x) length(x), custom_values));
    
    %initialize sparse reps
    sparse_n_custom_x_1_ones  = sparse(ones(n_custom,1));
    
    %intialize matrices for coefSet constraints
    A_vals   = sparse(n_custom,n_cols+n_newcols);
    lhs_vals = sparse(n_custom,1);
    rhs_vals = sparse(n_custom,1);
    
    A_lims    = sparse(n_custom,n_cols+n_newcols);
    lhs_lims  = sparse(n_custom,1);
    rhs_lims  = sparse_n_custom_x_1_ones;
    
    A_norm   = sparse(n_custom,n_cols+n_newcols);
    lhs_norm = sparse(n_custom,1);
    rhs_norm = sparse(n_custom,1);
    
    custom_index        = find(custom_ind);
    
    %stamp coefSet constraints
    counter = n_cols;
    drop_norm_index = [];
    
    for k = 1:n_custom
        
        j                   = custom_index(k);
        coefSetj            = custom_values{j};
        Kj                  = length(coefSetj);
        
        A_vals(k,j)         = -1;
        stamp_ind           = counter+1:(counter+Kj);
        A_vals(k,stamp_ind) = coefSetj;
        
        %set sum(l_j,k) <= 1
        A_lims(k,stamp_ind) = 1;
        
        %set \alpha_j + sum(l_j,k) == 1
        %only if there is L0-regularization
        if L0_reg_ind(j)
            A_norm(k,P+j)       = 1;
            A_norm(k,stamp_ind) = -1;
        else
            drop_norm_index = [drop_norm_index,k];
        end
        
        
        counter = counter+Kj;
    end
    
    obj     = [obj; sparse(n_newcols,1)];
    ub      = [ub;  sparse(ones(n_newcols,1))];
    lb      = [lb;  sparse(n_newcols,1)];
    ctype   = [ctype, 'B'*ones(1,n_newcols)];
    
    A_norm(drop_norm_index,:)   = [];
    lhs_norm(drop_norm_index,:) = [];
    rhs_norm(drop_norm_index,:) = [];
    
    A   = [A,sparse(n_rows,n_newcols); A_vals; A_lims; A_norm];
    lhs = [lhs;lhs_vals;lhs_lims;lhs_norm];
    rhs = [rhs;rhs_vals;rhs_lims;rhs_norm];
    
end

%% Drop Unused Variables (to avoid CPLEX issues)

%since beta_j will be dropped for j with fixed sign, apply L1 penalty to lamdba_j
obj(L1_reg_ind & sign_pos) = l1cost(L1_reg_ind & sign_pos);
obj(L1_reg_ind & sign_neg) = -l1cost(L1_reg_ind & sign_neg);

%drop beta_j for j without L1 penalty as well as j with fixed sign;
drop_ind              = [P+find(alpha_drop_ind);P+P+find(beta_drop_ind)];
obj(drop_ind)         = [];
ctype(drop_ind)       = [];
ub(drop_ind)          = [];
lb(drop_ind)          = [];
A(:,drop_ind)         = [];

%% Create CPLEX

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

%% Debug Information

if nargout == 2
    
    info = struct();
    
    info.X       = X;
    info.Y       = Y;
    info.Y_name  = Y_name;
    info.X_names = X_names;
    
    ii = 1;
    info.indices.lambdas = ii:(ii+P-1);                                    ii = ii+P;
    info.indices.alphas  = ii:(ii+n_alpha-1);                              ii = ii+n_alpha;
    info.indices.betas   = ii:(ii+n_beta-1);                               ii = ii+n_beta;
    info.indices.errors  = ii:(ii+N-1);                                    ii = ii+N;
    
    if n_newcols == 1
        %debug.indices.us      = ii;
    elseif n_newcols > 1
        info.indices.us      = ii:ii+n_newcols;
    end
    
    info.alpha_drop_ind                = alpha_drop_ind;
    info.beta_drop_ind                 = beta_drop_ind;
    info.L0_reg_ind                    = L0_reg_ind;
    info.L1_reg_ind                    = L1_reg_ind;
    
    info.C_0                           = C_0;
    info.C_1                           = C_1;
    info.M                             = M;
    info.epsilon                       = epsilon;
    
    info.add_L0_norm_constraint        = add_L0_norm_constraint;
    info.add_err_constraint            = add_err_constraint;
    info.add_pos_err_constraint        = add_pos_err_constraint;
    info.add_neg_err_constraint        = add_neg_err_constraint;
    info.L0_min                        = L0_min;
    info.L0_max                        = L0_max;
    info.err_min                       = err_min;
    info.err_max                       = err_max;
    info.pos_err_min                   = pos_err_min;
    info.pos_err_max                   = pos_err_max;
    info.neg_err_min                   = neg_err_min;
    info.neg_err_max                   = neg_err_max;
    
    info.input                          = input;
    varargout{2}                        = info;
    
end

%% Helper Functions

    function settingfile = setdefault(settingfile,settingname,defaultvalue)
        
        if ~isfield(settingfile,settingname)
            settingfile.(settingname) = defaultvalue;
        end
        
    end

end