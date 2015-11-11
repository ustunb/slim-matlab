function CoefSet = newCoefSet(variables_names)
%Create a default coefSet from a P x 1 cell of variable_names
%the default coefSet restricts coefficients to integers in [-10,10]
%it can customized directly, or using the setter functions
%setCoefField(..) and setCoefSetField(...)
%
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
%           coefficient)
%
%Author:      Berk Ustun
%Contact:     ustunb@mit.edu
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

P = length(variables_names);

%Create CoefSet
default             = struct();
default.name        = '';
default.type        = 'integer';
default.ub          = 10;
default.lb          = -10;
default.sign        = NaN;
default.values      = NaN;
default.C_0j        = NaN;
CoefSet = repmat(default,P,1);
[CoefSet(:).name] = variables_names{:};

end
