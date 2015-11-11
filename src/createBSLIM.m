function varargout = createBSLIM(input)

%% Check Inputs

%check for X
if ~isfield(input,'X')
    error('input does not contain X matrix\n')
elseif ~isnumeric(input.X);
    error('X must be a numeric array\n')
elseif ~all(all(input.X==0|input.X==1));
    error('Input data should be binary')
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

%% Compress Data / Lset
N_o     = N; 
N_pos_o = N_pos;
N_neg_o = N_neg;
P_o     = P;
X_o     = X;
[X,nY_pos,nY_neg,pos_loss_con,neg_loss_con,conflict_table,iu,ix]  = compress_data(X,Y);
[N,P] = size(X);

%% Default Settings

input = setdefault(input, 'C_0', NaN);
input = setdefault(input, 'use_weighted_version', false);
input = setdefault(input, 'w_pos', 1.000);
input = setdefault(input, 'w_neg', 1.000);
input = setdefault(input, 'start_lambda', zeros(1,P));

%Hard Constraints
input = setdefault(input, 'L0_min', 0);
input = setdefault(input, 'L0_max', P);
input = setdefault(input, 'err_min', 0);
input = setdefault(input, 'err_max', 1);
input = setdefault(input, 'pos_err_min', 0);
input = setdefault(input, 'pos_err_max', 1);
input = setdefault(input, 'neg_err_min', 0);
input = setdefault(input, 'neg_err_max', 1);
input = setdefault(input, 'pos_predict_max', 1);
input = setdefault(input, 'add_intercept_constraint',true);
input = setdefault(input, 'add_conflict_constraint',false);
input = setdefault(input, 'add_err_constraint',false);
input = setdefault(input, 'add_pos_err_constraint',false);
input = setdefault(input, 'add_neg_err_constraint',false);

%% Initialize Variables to Create Matrices

%get variable types
lambda_lim  = [Lset(:).lim];
lambda_lb   = lambda_lim(1:2:2*P-1)';
lambda_ub   = lambda_lim(2:2:2*P)';
lambda_max  = max(abs(lambda_lb), abs(lambda_ub));

signs       = [Lset(:).sign]';
sign_pos    = signs==1;
sign_neg    = signs==-1;

%Weights for Imbalanced Version
if ~input.use_weighted_version
    
    w_pos = 1;
    w_neg = 1;

else
    
    w_pos = input.w_pos;
    w_neg = input.w_neg;
    
    if isnan(w_pos) && isnan(w_neg)
        
        w_pos = 2*(N_neg/N_o);
        w_neg = 2*(N_pos/N_o);
        
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
    
end

%% Data Matrix Reprocessing
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
err_min        = max(sum(min_conflicts),err_min);
conflict_min   = length(min_conflicts);
conflict_max   = sum(pos_loss_con)+sum(neg_loss_con)-length(min_conflicts);

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


%flags for whether or not we will add contraints
add_L0_norm_constraint      = (L0_min > 0) || (L0_max < P);
add_err_constraint          = input.add_err_constraint && ((err_min > 0) || (err_max < N_o));
add_pos_err_constraint      = input.add_pos_err_constraint && ((pos_err_min > 0) || (pos_err_max < N_pos_o));
add_neg_err_constraint      = input.add_neg_err_constraint && ((neg_err_min > 0) || (neg_err_max < N_neg_o));
add_conflict_constraint     = input.add_conflict_constraint && ((conflict_min > 0) || (conflict_max < sum(pos_loss_con)+sum(neg_loss_con)));
add_intercept_constraint    = input.add_intercept_constraint;

%% Formulation Parameters

%L0-Regularization Parameter
if isnan(input.C_0)
    C_0     = ones(P,1).*min([w_pos/N_o,w_neg/N_o])./(L0_max+0.1);
else
    C_0     = input.C_0 .*ones(P,1);
end
UC_0        = [Lset(:).C_0j]';
UC_ind      = ~isnan(UC_0);

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
L1_reg_ind  = 1:P;

%L1-Regularization Parameter
L1_max              = sort(lambda_max,'descend');
L1_max              = sum(L1_max(1:L0_max));
C_1                 = 0.975.*min([w_pos/N_o,w_neg/N_o,min(C_0(L0_reg_ind))])./L1_max;
C_1                 = C_1.*ones(P,1);
C_1(~L1_reg_ind)    = 0;

%Compute Big M
switch L0_max
    
    case 1
        M = 2*ones(N,1);
    
    case 2
        M = 2*ones(N,1);
        
    case 3
        
        M = 3*ones(N,1);
        
    case 4
        
        M = 5*ones(N,1);
        
    otherwise
        
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

%variables  =                 lambda_pos (P) lambda_neg (P) alpha_pos (P)   alpha_neg (P)  pos_errors (nZ_pos)     negative errors (nZ_neg))
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

% A_3   = [];
% lhs_3 = [];
% rhs_3 = [];

%drop loss constraints for negative errors with Conflicts
% to_drop             = conflict_table(:,3);
% A_2(to_drop,:)      = [];
% lhs_2(to_drop,:)    = [];
% rhs_2(to_drop,:)    = [];
% 
% % %Get Big M Indices / Values (for subsequent refining)
% % % M_pos_row_ind               = [1:nZ_pos]';
% % % M_pos_col_ind               = [(P+P+P+P)+(1:nZ_pos)]';
% % % M_pos_val                   = M(pos_loss_con);
% % % M_x_pos_val                 = X(pos_loss_con,:);
% % % 
% % % M_neg_row_ind               = [nZ_pos+(1:(nZ_neg-length(to_drop)))]';
% % % M_neg_col_ind               = [(P+P+P+P+nZ_pos)+(1:nZ_neg)]';
% % % M_neg_val                   = M(neg_loss_con);
% % % M_x_neg_val                 = X(neg_loss_con,:);
% % % 
% % % M_neg_col_ind(to_drop,:)    = [];
% % % M_neg_val(to_drop,:)        = [];
% % % M_x_neg_val(to_drop,:)      = [];
% % 
% % 
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

conflict_constraint_ind     = NaN;
err_constraint_ind          = NaN;
pos_err_constraint_ind      = NaN;
neg_err_constraint_ind      = NaN;
intercept_constraint_ind    = NaN;

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
    intercept_constraint_ind = [intercept_constraint_ind,size(A,1)];
    
    % Intercept Lower Bound Constraint #2
    % 1 - lambda_pos[2:P] <= lambda_neg[1] --> 1 <= lambda_pos[2:P] + lambda_neg[1] <= Inf
    A_add   = [sparse([0 ones(1,P-1)]),sparse([1 zeros(1,P-1)]), sparse(1,P+P+nZ_pos+nZ_neg)];
    lhs_add = sparse(1);
    rhs_add = sparse(Inf);
     
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    intercept_constraint_ind = [intercept_constraint_ind,size(A,1)];
    
end

if add_L0_norm_constraint
    
    
    A_add   = [sparse(1,P+P), sparse(L0_reg_ind(:)'), sparse(L0_reg_ind(:)'), sparse(1,nZ_pos+nZ_neg)];
    lhs_add = sparse(L0_min);
    rhs_add = sparse(L0_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
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
    

if add_err_constraint
    
    A_add   = [sparse(1,P+P+P+P), sparse([nY_pos(pos_loss_con,:);nY_neg(neg_loss_con,:)]')];
    lhs_add = sparse(err_min);
    rhs_add = sparse(err_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
    err_constraint_ind = size(A,1);
    
end

if add_pos_err_constraint
    
    A_add   = [sparse(1,P+P+P+P), sparse(nY_pos(pos_loss_con,:)'), sparse(1,nZ_neg)];
    lhs_add = sparse(pos_err_min);
    rhs_add = sparse(pos_err_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
    pos_err_constraint_ind = size(A,1);
    
end

if add_neg_err_constraint
    
    A_add   = [sparse(1,P+P+P+P), sparse(1,nZ_pos), sparse(nY_neg(neg_loss_con,:)')];
    lhs_add = sparse(neg_err_min);
    rhs_add = sparse(neg_err_max);
    
    A       = [A;A_add];
    lhs     = [lhs;lhs_add];
    rhs     = [rhs;rhs_add];
    
    neg_err_constraint_ind = size(A,1);
    
end
%% Drop Redundant Variables

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

% M_row_ind             = [M_pos_row_ind;M_neg_row_ind];
% M_col_ind             = [M_pos_col_ind;M_neg_col_ind]-length(drop_ind);
% M_val                 = [M_pos_val;M_neg_val];
% M_x_values            = [M_x_pos_val;M_x_neg_val];

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

%% Gather Debug


if nargout == 2
    
    debug = struct();
    
    debug.Xu                            = X;
    debug.ix                            = ix;
    debug.iu                            = iu;
    debug.pos_loss_con                  = pos_loss_con;
    debug.neg_loss_con                  = neg_loss_con;
    debug.nY_pos                        = nY_pos;
    debug.nY_neg                        = nY_neg;
    debug.conflict_table                = conflict_table;
    
%     debug.start_lambda                  = start_lambda;
%     debug.M                             = M;
%     debug.M_val                         = M_val;
%     debug.M_row_ind                     = M_row_ind;
%     debug.M_col_ind                     = M_col_ind;
%     debug.M_x_values                    = M_x_values;  
    
    debug.indices.lambda_pos            = lambda_pos_ind;
    debug.indices.lambda_neg            = lambda_neg_ind;
    debug.indices.alpha_pos             = alpha_pos_ind;
    debug.indices.alpha_neg             = alpha_neg_ind;
    debug.indices.pos_errors            = pos_err_ind;
    debug.indices.neg_errors            = neg_err_ind;
    debug.n_vars                        = n_vars;
    debug.n_loss_constraints            = n_loss_constraints;
    debug.alpha_pos_drop_ind            = alpha_pos_drop_ind;
    debug.alpha_neg_drop_ind            = alpha_neg_drop_ind;
    debug.lambda_pos_drop_ind           = lambda_pos_drop_ind;
    debug.lambda_neg_drop_ind           = lambda_neg_drop_ind;
    debug.L0_reg_ind                    = L0_reg_ind;
    debug.L1_reg_ind                    = L1_reg_ind;
    debug.C_0                           = C_0;
    debug.C_1                           = C_1;
    
    
    debug.pos_loss_constraint_ind           = pos_loss_constraint_ind;
    debug.neg_loss_constraint_ind           = neg_loss_constraint_ind;
    debug.neg_loss_conflict_constraint_ind  = neg_loss_conflict_constraint_ind;
    
    debug.add_intercept_constraint      = add_intercept_constraint;
    debug.intercept_constraint_ind      = intercept_constraint_ind;
    
    debug.add_L0_norm_constraint        = add_L0_norm_constraint;
    
    debug.add_conflict_constraint       = add_conflict_constraint;
    debug.conflict_constraint_ind       = conflict_constraint_ind;
    
    debug.L0_min                        = L0_min;
    debug.L0_max                        = L0_max;
    
    debug.add_err_constraint            = add_err_constraint;
    debug.err_constraint_ind            = err_constraint_ind;
    if add_err_constraint
        debug.err_min                       = err_min;
        debug.err_max                       = err_max;
    end
    
    debug.add_pos_err_constraint        = add_pos_err_constraint;
    debug.pos_err_constraint_ind        = pos_err_constraint_ind;
    if add_pos_err_constraint
        debug.pos_err_min                   = pos_err_min;
        debug.pos_err_max                   = pos_err_max;
        
    end
    
    debug.add_neg_err_constraint        = add_neg_err_constraint;
    debug.neg_err_constraint_ind        = neg_err_constraint_ind;
    if add_pos_err_constraint
        debug.neg_err_min                   = neg_err_min;
        debug.neg_err_max                   = neg_err_max;
    end
    
    
    
    debug.user_input                    = input;
    
    if ~isfield(debug.user_input,'X'), debug.user_input.X = X_o; end
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

    function [Xa,nY_pos,nY_neg,pos_loss_con,neg_loss_con,conflict_table,iu,ix] = compress_data(X,Y)
        
        
        %[XY,is]             = sortrows(Xa);
        %[~, is_reverse]     = sort(is);
        %XY(is_reverse,:) == XYo;
        %all(XYo(is,:)==XY)
        %all(XYo == XY(is_reverse,:))
        
        [Xa,iu,ix]       = unique(X,'rows','stable');
        n               = length(iu);
        nY_pos          = zeros(n,1);
        nY_neg          = zeros(n,1);
        conflict_table  = zeros(n,3);
        
        i_pos = 1;
        i_neg = 1;
        
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