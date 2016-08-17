function polished_pool = runActiveSetPolishing(slim_IP, slim_info, time_limit)
%
%Run active set (AS) polishing on all solutions in the solution pool of a
%slim_IP object within a given time_limit. Returns the set of all solutions
%that can be polished in the time_limit in the `polished_pool` struct.
%Polished solutions are ordered according to increasing objective value
%(i.e., best solution is in polished_pool(1)).
%
%AS polishing returns the best SLIM scoring system for a fixed subset
%of input variables The proceudre takes as input a set of coefficients
%`lambda` and returns as output a set of polished coefficients `lambda_opt`
%with the same set of non-zero coefficients as `lambda`. Here:
%
%(lambda == 0) >= (lambda_opt == 0)
%
%
%
%Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%% check inputs
if nargin ~= 3
    error('need 3 input arguments: slim_IP, slim_info and time_limit')
end
assert(isa(slim_IP, 'Cplex'), 'argument 1 must be a Cplex object')
assert(isstruct(slim_info), 'argument 2 must be a struct')
assert(isnumeric(time_limit), 'argument 3 should be numeric')
assert(any(strcmp('MipStart', properties(slim_IP))), 'could not find a set of feasible solutions to polish');
assert(any(strcmp('indices', fieldnames(slim_info))), 'slim_info should contain indices struct()');
assert(any(strcmp('input', fieldnames(slim_info))), 'slim_info should contain input struct');
assert(time_limit >= 0.0, 'time_limit must be >= 0 seconds');

%% setup variables for active set polishing loop

if any(strcmp('L0_reg_ind',  fieldnames(slim_info)))
    L0_reg_ind = slim_info.L0_reg_ind;
else
    L0_reg_ind =  true(size(slim_info.input.X,2));
end

if strcmp(slim_info.version, 'binary')
    get_coefs_from_solution = @(sol) sol.x(slim_info.indices.lambda_pos) + sol.x(slim_info.indices.lambda_neg);
else
    get_coefs_from_solution = @(sol) sol.x(slim_info.indices.lambdas);
end

solution_pool = slim_IP.MipStart;
default_input = slim_info.input;

%polishing MIP parameter struct
polishing_IP                            = Cplex();
polishing_param                         = polishing_IP.Param;
polishing_param.parallel.Cur            = 1;
polishing_param.threads.Cur             = 1;
polishing_param.timelimit.Cur           = 60;
polishing_param.randomseed.Cur          = 0;
polishing_param.mip.display.Cur         = 0;
polishing_param.output.clonelog.Cur     = 0; %no clone log
clear polishing_IP

%% run active set polishing lop
polished_pool = struct('coefficients', [], 'objective_value', []);
remaining_time = time_limit;
start_time = tic();
for n = 1:length(solution_pool)
    
    %get input coefficients
    active_set = get_coefs_from_solution(solution_pool(n));
    active_set = active_set | ~L0_reg_ind;
    
    %setup polishing IP
    polishing_input = struct(default_input);
    polishing_input.active_set = active_set;
    [polishing_IP, polish_info] = createPolishingIP(polishing_input);
    polishing_IP.DisplayFunc = [];
    polishing_IP.Param = polishing_param;
    polishing_IP.Param.timelimit.Cur = remaining_time;
    
    %solve polishing IP
    Solution = polishing_IP.solve();
    
    %store polished solution if it is the optimal integer solution
    if (Solution.status == 101 || Solution.status == 102)
        active_coefs = Solution.x(polish_info.indices.lambda_pos) + Solution.x(polish_info.indices.lambda_neg);
        polished_coefs = zeros(length(active_set),1);
        polished_coefs(active_set) = active_coefs;
        polished_pool(n) = struct('coefficients', polished_coefs, 'objective_value', Solution.objval);
    end
    
    %compute total runtime
    remaining_time = remaining_time - toc(start_time);
    if remaining_time <= 0
        break
    end
    
end

%% order output by objective value
[~,ind] = sort([polished_pool(:).objective_value], 'ascend');
polished_pool = polished_pool(ind);

end
