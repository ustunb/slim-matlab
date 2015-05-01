function varargout = create_SLIM(input)

%% Check Inputs

%check for X
if ~isfield(input,'X')
    error('input does not contain X matrix\n')
elseif ~isnumeric(input.X);
    error('X must be a numeric array\n')
else
    X = input.X;
    input = rmfield(input,'X');
end

%check for Y
if ~isfield(input,'Y')
    error('input does not contain Y matrix\n')
elseif ~isnumeric(input.Y);
    error('Y must be a numeric array\n')
else
    Y = input.Y(:);
    if ~all(Y==-1|Y==1)
        error('all entries 
    input = rmfield(input,'Y');
end

%make sure sizes of X and Y match
[N,P] = size(X);
if (size(Y,1) ~= N)
    error('number of rows of X and Y do not match\n')
end

%make sure Y is binary
pos_indices = Y==1;
neg_indices = Y==-1;
N_pos = sum(pos_indices);
N_neg = sum(neg_indices);

if N~=(N_pos+N_neg)
    error('all elements of Y must either be +1 or -1\n')
end

%Lset or X_headers
if ~isfield(input,'X_headers') &&  ~isfield(input,'Lset')
    
    fprintf('input does not contain X_headers; naming features X1,...,X%d \n',P)
    X_headers = arrayfun(@(x) sprintf('X%d',x),1:P,'UniformOutput',false);
    
    fprintf('input does not contain Lset; creating default Lset \n');
    Lset = CreateLset(X_headers);
    
elseif ~isfield(input,'X_headers') &&  isfield(input,'Lset')
    
    fprintf('input does not contain X_headers; extracting from Lset \n')
    Lset        = CheckLset(input.Lset);
    X_headers   = {Lset(:).name}';
    
elseif isfield(input,'X_headers')
    
    if ~(iscell(input.X_headers) && length(input.X_headers)==P)
        error('X_headers must be cell array with P entries\n')
    else
        X_headers   = input.X_headers;
        %input       = rmfield(input,'X_headers');
    end
    
    if isfield(input,'Lset')
        Lset = CreateLset(X_headers,input.Lset);
    else
        fprintf('input does not contain Lset; creating default Lset \n');
        Lset = CreateLset(X_headers);
    end
end

%% Default Settings

input = setdefault(input, 'C_0', 1e-3);
input = setdefault(input, 'C_1', NaN);

input = setdefault(input, 'epsilon', 0.001);               %not yet implemented
input = setdefault(input, 'M', NaN);
input = setdefault(input, 'w_pos', 1.000);
input = setdefault(input, 'w_neg', 1.000);

input = setdefault(input, 'L0_min', 0);
input = setdefault(input, 'L0_max', P);
input = setdefault(input, 'err_min', 0);
input = setdefault(input, 'err_max', 1);
input = setdefault(input, 'pos_err_min', 0);
input = setdefault(input, 'pos_err_max', 1);
input = setdefault(input, 'neg_err_min', 0);
input = setdefault(input, 'neg_err_max', 1);

%lift style constriants
input = setdefault(input, 'pos_predict_max', 1);

%% Initialize Variables to Create Matrices

%get variable types
classes     = {Lset(:).class}';
custom_ind  = strcmp(classes,'custom');
bounded_ind = strcmp(classes,'bounded');

types       = {Lset(:).type}';
int_ind     = strcmp(types,'I');
real_ind    = strcmp(types,'C');
disc_ind    = int_ind | custom_ind;
cts_ind     = real_ind & bounded_ind;

%get values for custom types
values      = {Lset(:).values}';

%remove NaN entries from values
values                       = cellfun(@(Lsetj) Lsetj(~isnan(Lsetj)), values,'UniformOutput',false);
n_values                     = cellfun(@(Lsetj) length(Lsetj), values,'UniformOutput',true);
fixed_values_ind             = (n_values==1) & custom_ind;
one_of_K_ind                 = (n_values>1) & custom_ind;

values                       = cellfun(@(Lsetj) setdiff(Lsetj,0), values,'UniformOutput',false);

%make sure that type is custom iff value cell contains something other than NaN values
if any(custom_ind ~= cellfun(@(x) all(~isnan(x)),[{Lset(:).values}]','UniformOutput',true))
    error('mismatch between custom type specification and values')
end

lambda_lim  = [Lset(:).lim];
lambda_lb   = lambda_lim(1:2:2*P-1)';
lambda_ub   = lambda_lim(2:2:2*P)';
lambda_max  = max(abs(lambda_lb), abs(lambda_ub));

signs       = [Lset(:).sign]';
sign_pos    = signs==1;
sign_neg    = signs==-1;
sign_free   = isnan(signs);
sign_fixed  = sign_pos | sign_neg;

%X scales
XY              = bsxfun(@times,X,Y);
scales          = [Lset(:).scale]';
rescale_ind     = ~isnan(scales);

if any(rescale_ind)
    fprintf('Lset requires scaled and bounded coefficients; rescaling X \n');
    scales = scales(rescale_ind);
    scales = scales(:)';
    XY(:,rescale_ind) = bsxfun(@times,XY(:,~isnan(rescale_ind),scales));
end

%Weights for Imbalanced Version
w_pos = input.w_pos;
w_neg = input.w_neg;

if isnan(w_pos) && isnan(w_neg)
    
    w_pos = 2*(N_neg/N);
    w_neg = 2*(N_pos/N);
    
elseif isnan(w_neg)
    
    if ~((w_pos>=0)&&(w_pos<=2))
        error('w_pos must be in [0,2] \n')
    else
        w_neg = 2-w_pos;
    end
    
elseif isnan(w_pos)
    
    if ~((w_neg>=0)&&(w_neg<=2))
        error('w_neg must be in [0,2] \n')
    else
        w_pos = 2-w_neg;
    end
    
else %~isnan(w_pos) && ~isnan(w_neg)
    
    %check that w_pos and w_neg are
    if (w_pos<0)
        error('w_pos must be a positive scalar \n')
    end
    
    if (w_neg<0)
        error('w_neg must be a positive scalar \n')
    end
    
    tot = (w_pos + w_neg);
    
    if (w_pos+w_neg) ~=2
        fprintf('w_pos + w_neg = %1.2f \n renormalizing so that w_pos+w_neg = 2.00 \n', tot)
        w_pos = 2*w_pos/tot;
        w_neg = 2*w_neg/tot;
    end
end

%Regularization Parameters
C_0                = input.C_0 .* ones(P,1);
UC_0               = [Lset(:).C_0j]';
UC_ind             = ~isnan(UC_0);

if any(UC_ind)
    fprintf('user requested customized regularization parameters for %d variables \n',sum(UC_ind))
    C_0(UC_ind)  = UC_0(UC_ind);
    C_0 = C_0(:);
end

if any(C_0 < 0)
    error('user specified negative regularization penalities \n')
end

%identify indices of coefficiens that have regularization
L0_reg_ind  = (C_0>0);
L1_reg_ind  = disc_ind & L0_reg_ind;
no_reg_ind  = C_0==0;

C_1         = input.C_1;
if isnan(C_1)
   C_1   = 0.5.*min([w_pos/N,w_neg/N,min(C_0(L1_reg_ind))])./(sum(lambda_max));
end
C_1                 = C_1.*ones(P,1);
C_1(~L1_reg_ind)    = 0;

%Set M and Epsilon for 0-1 loss function
epsilon  = input.epsilon;
M        = input.M(:);

%quick references to sparse matrices
sparse_N_x_1_Inf    = sparse(Inf(N,1));
sparse_P_x_1_Inf    = sparse(Inf(P,1));
sparse_N_x_1_ones   = sparse(ones(N,1));

if isnan(M)
    M = sum(abs(X*lambda_max),2)+1.1*epsilon;
end

%% Sanity Checks for Hard Constraints 

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

%# of positive predictions
pos_predict_max           = input.pos_predict_max;
if isnan(pos_predict_max) || isinf(pos_predict_max), pos_predict_max = 0; end
pos_predict_max            = min(floor(N*pos_predict_max),N);


%flags for whether or not we will add contraints
add_L0_norm_constraint      = (L0_min > 0) || (L0_max < P);
add_err_constraint          = (err_min > 0) || (err_max < N);
add_pos_err_constraint      = (pos_err_min > 0) || (pos_err_max < N_pos);
add_neg_err_constraint      = (neg_err_min > 0) || (neg_err_max < N_neg);
add_pos_pred_constraint     = (pos_predict_max > 0) && (pos_predict_max < N);

%% Setup Matrices for Bounded Variables

%Objective
l0cost  = N*C_0;
l1cost  = N*C_1;
errcost = (w_pos*pos_indices + w_neg*neg_indices);

%variables  = lambda's (P)          alpha (P)       beta (P)        errors (N)
obj         = sparse([zeros(P,1);   l0cost;         l1cost;         errcost]);
ub          = sparse([lambda_ub;    ones(P,1) ;     lambda_max;     ones(N,1)]);
lb          = sparse([lambda_lb;    zeros(P,1);     zeros(P,1);     zeros(N,1)]);
ctype       = ['I'*ones(1,P),       'B'*ones(1,P),  'C'*ones(1,P),  'B'*ones(1,N)];

%change lambdas
ctype(int_ind)              = 'I';
ctype(real_ind)             = 'C';

%change betas
ctype(2*P+find(int_ind))    = 'I';
ctype(2*P+find(real_ind))   = 'C';


% Using Big M #2 (enforce Z_1 = 1 if incorrect classification)
A_1         = [sparse(XY),sparse(N,P+P),spdiags(M,0,N,N)];
lhs_1       = epsilon.*sparse_N_x_1_ones;
rhs_1       = sparse_N_x_1_Inf;


% 0-Norm LHS Constraints: lambda_j,lb < lambda_j --> 0 < lambda_j - lambda_j,lb < Inf
A_2         = [speye(P), -sparse(diag(lambda_lb)), sparse(P,P), sparse(P,N)];
lhs_2       = sparse(P,1);
rhs_2       = sparse_P_x_1_Inf;

%remove constraints for any j s.t. C_0j = 0 or lambda_j > 0 
to_drop             = ~L0_reg_ind | L0_reg_ind & sign_pos;
A_2(to_drop,:)      = [];
lhs_2(to_drop,:)    = [];
rhs_2(to_drop,:)    = [];

% 0-Norm RHS Constraints: lambda_j < lambda_j,ub --> 0 < -lambda_j + lambda_j,ub < Inf
A_3 = [-speye(P), sparse(diag(lambda_ub)), sparse(P,P), sparse(P,N)];
lhs_3 = sparse(P,1);
rhs_3 = sparse_P_x_1_Inf;

%remove constraints for any j s.t. C_0j = 0 or lambda_j < 0
to_drop          = ~L0_reg_ind | L0_reg_ind & sign_neg;
A_3(to_drop,:)   = [];
lhs_3(to_drop,:) = [];
rhs_3(to_drop,:) = [];

% 1-Norm LHS Constraints: -lambda_j < beta_j -> lambda_j + beta_j < Inf
A_4 = [speye(P), sparse(P,P), speye(P), sparse(P,N)];
lhs_4 = sparse(P,1);
rhs_4 = sparse_P_x_1_Inf;

% 1-Norm RHS Constraints: lambda_j < beta_j -> -lambda_j + beta_j < Inf
A_5 = [-speye(P),   sparse(P,P), speye(P), sparse(P,N)];
lhs_5 = sparse(P,1);
rhs_5 = sparse_P_x_1_Inf;

%remove 1-norm constraints for coefficients with no regularization or fixed sign
to_drop = ~L1_reg_ind | L1_reg_ind & sign_fixed;
A_4(to_drop,:)   = [];
lhs_4(to_drop,:) = [];
rhs_4(to_drop,:) = [];

A_5(to_drop,:)   = [];
lhs_5(to_drop,:) = [];
rhs_5(to_drop,:) = [];

%merge all matrices together
A   = [A_1;A_2;A_3;A_4;A_5];
lhs = [lhs_1; lhs_2; lhs_3; lhs_4; lhs_5];
rhs = [rhs_1; rhs_2; rhs_3; rhs_4; rhs_5];


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

if add_pos_err_constraint
    
    A_add   = [sparse(1,P), sparse(1,P), sparse(1,P), sparse(pos_indices(:)')];
    lhs_add = sparse(pos_err_min);
    rhs_add = sparse(pos_err_max);
    
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


if add_pos_pred_constraint
    
    A_add       = [sparse(1,P), sparse(1,P), sparse(1,P), sparse(Y(:)')];
    lhs_add     = sparse(N_pos - pos_predict_max);
    rhs_add     = sparse(Inf);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
end


%% Setup Matrices for Custom Values

% Identify Variables that will be dropped
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
    
    if n_cols ~= (3*P + N)
        error('error!')
    end
    
    n_custom    = sum(custom_ind);
    n_newcols   = sum(cellfun(@(x) length(x), values));
    
    %initialize sparse reps
    sparse_n_custom_x_1_ones  = sparse(ones(n_custom,1));
    
    %intialize matrices for Lset constraints
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
    
    %stamp Lset constraints
    counter = n_cols;
    drop_norm_index = [];
    
    for k = 1:n_custom
        
        j                   = custom_index(k);
        Lsetj               = values{j};
        Kj                  = length(Lsetj);
        
        A_vals(k,j)         = -1;
        stamp_ind           = counter+1:(counter+Kj);
        A_vals(k,stamp_ind) = Lsetj;
        
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

%% Changes to  Variables

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
    
    LP = Cplex();
    
    Model = struct();
    Model.sense = 'minimize';
    Model.obj    = obj;
    Model.rhs    = rhs;
    Model.lhs    = lhs;
    Model.A      = A;
    Model.ub     = ub;
    Model.lb     = lb;
    Model.ctype  = char(ctype);
    
    LP.Model     = Model;
    
    varargout{1} = LP;
    
end

%% Debug Information

if nargout == 2
    
    debug = struct();
    
    ii = 1;
    debug.indices.lambdas = ii:(ii+P-1);                                    ii = ii+P;
    debug.indices.alphas  = ii:(ii+n_alpha-1);                              ii = ii+n_alpha;
    debug.indices.betas   = ii:(ii+n_beta-1);                               ii = ii+n_beta;
    debug.indices.errors  = ii:(ii+N-1);                                    ii = ii+N;
    
    if n_newcols == 1
%         debug.indices.us      = ii;
    elseif n_newcols > 1
        debug.indices.us      = ii:ii+n_newcols;
    end
    
    debug.alpha_drop_ind                = alpha_drop_ind;
    debug.beta_drop_ind                 = beta_drop_ind;
    debug.L0_reg_ind                    = L0_reg_ind;
    debug.L1_reg_ind                    = L1_reg_ind;
    debug.fixed_values_ind              = fixed_values_ind;
    
    debug.C_0                           = C_0;
    debug.C_1                           = C_1;
    
    debug.M                             = M;
    debug.epsilon                       = epsilon;
    
    debug.add_L0_norm_constraint        = add_L0_norm_constraint;
    debug.add_err_constraint            = add_err_constraint;
    debug.add_pos_err_constraint        = add_pos_err_constraint;
    debug.add_neg_err_constraint        = add_neg_err_constraint;
    debug.L0_min                        = L0_min;
    debug.L0_max                        = L0_max;
    debug.err_min                       = err_min;
    debug.err_max                       = err_max;
    debug.pos_err_min                   = pos_err_min;
    debug.pos_err_max                   = pos_err_max;
    debug.neg_err_min                   = neg_err_min;
    debug.neg_err_max                   = neg_err_max;
    
    debug.user_input                    = input;
    
    if ~isfield(debug.user_input,'X'), debug.user_input.X = X; end
    if ~isfield(debug.user_input,'Y'), debug.user_input.Y = Y; end
    if ~isfield(debug.user_input,'X_headers'), debug.user_input.X_headers = X_headers; end
    varargout{2}                        = debug;
    
end

%% Helper Functions

    function settingfile = setdefault(settingfile,settingname,defaultvalue)
        
        if ~isfield(settingfile,settingname)
            settingfile.(settingname) = defaultvalue;
        end
        
    end

end