function coefSet = checkCoefSet(coefSet)
%Checks the internal consistency of a CoefSet object, fixing mistakes
%fields of the coefSet object include:
%
%name:      name of the coefficient
%ub:        upperbound on coefficient set, 10 by default
%lb:        lowerbound on coefficient set, -10 by default
%
%type:      can either be 'integer' or 'custom', 'integer' by default
%           if 'integer' then coefficient will take on integer values in [lb,ub]
%           if 'custom' then coefficient will take any value specified in 'values'
%
%values:    array containing discrete values that a 'custom' coefficient can
%           take.
%
%sign:      sign of the coefficient set, =NaN by default
%           -1 means coefficient will be positive <=0
%            1 means coefficient will be negative >=0
%           0/NaN means no sign constraint
%
%           note that sign will override conflicts with lb/ub/values field,
%           so if sign=1, and [lb,ub] = [-10,10] then checkCoefSet will
%           adjust [lb,ub] to [0,10]
%
%
%C_0j:      custom feature selection parameter,
%           must be either NaN or a value in [0,1]
%           C_0j represents the minimum % accuracy that feature j must add
%           in order to be included in the SLIM classifier (i.e. have a non-zero
%
%Author:    Berk Ustun, 2015
%Reference: SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%GitHub Repo: <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>.

warning('off', 'verbose');
warning('off', 'backtrace');
print_warning = @(msg) warning(msg);

if (length(coefSet) > 1)
    
    coefSet = arrayfun(@(ss) checkCoefSet(ss), coefSet);
    
else
    
    %check coefSet fields
    coefSetFields = fieldnames(coefSet);
    assert(length(coefSetFields) == 7);
    assert(isfield(coefSet,'name'), 'coefSet missing field *name*');
    assert(isfield(coefSet,'type'),'coefSet missing field *type*');
    assert(isfield(coefSet,'ub'), 'coefSet missing field *ub*');
    assert(isfield(coefSet,'lb'),  'coefSet missing field *lb*');
    assert(isfield(coefSet,'sign'), 'coefSet missing field *sign*');
    assert(isfield(coefSet,'values'), 'coefSet missing field *values*');
    assert(isfield(coefSet,'C_0j'), 'coefSet missing field *C_0j*');
    
    %check coefSet values
    assert(~isempty(coefSet.name), 'name must be non-empty');
    assert(coefSet.lb <= coefSet.ub, 'lb must be <= ub')
    assert(strcmp('integer',coefSet.type) || strcmp('custom',coefSet.type),'type must be *integer* or *custom*');
    assert((isnan(coefSet.sign) || coefSet.sign==1 || coefSet.sign==-1), 'sign must either be NaN, +1 or -1');
    assert((isnan(coefSet.C_0j) || (coefSet.C_0j <= 1 && coefSet.C_0j>=0)),  'C_0j must either be NaN, or numeric between [0,1]');
    
        
    if strcmp(coefSet.type,'custom')
        assert(~isempty(coefSet.values), 'if coefficient uses custom set of values, then values field must be non-empty');
    end
    
    %make sure sign and limits are in agreement
    if (coefSet.lb<=0) && (coefSet.ub<=0) && isnan(coefSet.sign)
        warning_msg = sprintf('setting sign to -1 for %s since lb<=0 and ub<=0', coefSet.name);
        print_warning(warning_msg);
        coefSet.sign = -1;
    end
    
    if (coefSet.lb>=0) && (coefSet.ub>=0) && isnan(coefSet.sign)
        warning_msg = sprintf('setting sign to +1 for %s since lb>=0 and ub>=0', coefSet.name);
        print_warning(warning_msg);
        coefSet.sign = 1;
    end
    
    if coefSet.sign==1 && coefSet.lb<=0
        warning_msg = sprintf('setting lb of %s to 0 since sign = +1', coefSet.name);
        print_warning(warning_msg);
        coefSet.lb = max(coefSet.lb, 0);
    end
    
    if coefSet.sign==-1 && coefSet.ub>=0
        warning_msg = sprintf('setting ub of %s to 0 since sign = -1', coefSet.name);
        print_warning(warning_msg);
        coefSet.ub = min(coefSet.ub, 0);
    end
    
end


end