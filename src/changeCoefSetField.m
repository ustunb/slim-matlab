function coefSet = changeCoefSetField(coefSet, variable_names, property_name, property_value)
%Change the properties of one or more variables in the coefSet object
%The coefSet is checked internally for consistency before it is returned
%
%coefSet        original coefSet that should be changed
%
%variable_names specify the names of the variables whose fields will be changed
%               can either be
%               'all' (i.e. all variables) or
%               a string containing a variable name,
%               or a cell array of strings with multiple variable_names;
%
%property_name  limited to be 'name','lb','ub','values','sign','C_0j'
%
%property_value limited to legal values of the fields above
%
%Author:      Berk Ustun 
%Contact:     ustunb@mit.edu
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

warning('off', 'verbose');
warning('off', 'backtrace');
print_warning = @(msg) warning(msg);

if ischar(variable_names) && all(strcmp(variable_names,'all'))
    variable_names = getCoefSetField(coefSet,'name');
end

for n = 1:length(variable_names)
    
    variable_name = variable_names{n};
    coef_index = strfind(getCoefSetField(coefSet,'name'), variable_name);
    
    if isempty(coef_index)
        error('could not find coef with name %s in coefSet', variable_name);
    end
    
    coefElement = coefSet(coef_index);
    
    switch property_name
        
        case 'name'
            
            assert(~isempty(property_value),'name must be set to a non-empty string')
            assert(ischar(property_value),'name must be a char array')
            coefElement = setfield(coefElement, 'name', property_value);
            
        case 'type'
            
            assert(ischar(property_value),'name must be a char array')
            assert(any(strcmp(property_value,{'integer','custom'})), 'type must be integer or custom')
            coefElement = setfield(coefElement, 'type' ,property_value);
            
        case 'ub'
            
            assert(property_value >= coefElement.lb, 'cannot set ub to a value less thanlb')
            coefElement = setfield(coefElement, 'ub', property_value);
            
        case 'lb'
            
            assert(property_value <= coefElement.ub, 'cannot set lb to a value greater than ub')
            coefElement = setfield(coefElement,'lb', property_value);
            
        case 'C_0j'
            
            assert(property_value >=0, 'C_0j must be a value in [0,1)')
            if property_value >= 1.00
                warning_msg = sprintf('C_0j should be a value in [0,1); using C_0j > 1 guarantees that coefficient for %s will be dropped', variable_name);
                print_warning(warning_msg);
            end
            coefElement = setfield(coefElement,property_name,property_value);
            
        case 'sign'
            
            assert(isnumeric(property_value) && (property_value==1||property_value==-1||isnan(property_value)),'sign must be 1,-1 or NaN')
            
            current_ub = coefElement.ub;
            current_lb = coefElement.lb;
            
            if isnan(property_value)
                
                coefElement = setfield(coefElement,'sign', NaN);
                
            elseif property_value == 1
                
                assert(current_ub > 0, 'if setting sign to +1 then make sure that ub > 0, otherwise coefficient will always be 0')
                if (current_lb < 0)
                    warning_msg = sprintf('setting lb for %s to 0 since sign = +1', variable_name);
                    print_warning(warning_msg);
                    coefElement = setCoefField(coefElement, variable_name, 'lb', 0, false);
                end
                
                coefElement = setfield(coefElement,'sign', 1);
                
            elseif property_value == -1
                
                assert(current_lb < 0, 'if setting sign to +1 then make sure that ub > 0, otherwise coefficient will always be 0')
                if (current_ub > 0)
                    warning_msg = sprintf('setting ub for %s to 0 since sign = -1', variable_name);
                    print_warning(warning_msg);
                    coefElement = setCoefField(coefElement, variable_name, 'ub', 0, false);
                end
                coefElement = setfield(coefElement,'sign', -1);
                
            end
            
        case 'values'
            
            assert(isnumeric(property_value), 'values must be a numeric array filled with unique values')
            
            if property_value ~= unique(property_value,'stable')
                print_warning('values vector contains duplicate elements; will only use unique elements of values')
                property_value = unique(property_value);
            end
            
            values_ub = max(property_value);
            values_lb = min(property_value);
            
            if isequal(floor(property_value), linspace(values_lb,values_ub))
                warning_msg = sprintf('custom values for %s == consecutive integers from %d to %d\n switching to an integer coefficient set for %s will improve performance',...
                    variable_name, values_ub,values_lb, variable_name);
                print_warning(warning_msg);
            end
            
            if all(property_value~=0)
                warning_msg = sprintf('custom values for %s do not include 0\n%s', ...
                    variable_name, 'setting C_0j to 0 to prevent L0-regularization for this variable');
                print_warning(warning_msg);
                coefElement = setfield(coefElement,'C_0j', 0.00);
            end
            
            current_ub = coefElement.ub;
            current_lb = coefElement.lb;
            current_type = coefElement.type;
            
            if current_ub ~= values_ub
                coefElement = setfield(coefElement,'ub', values_ub);
            end
            
            if current_lb ~= values_lb
                coefElement = setfield(coefElement,'lb', values_lb);
            end
            
            if ~strcmp(current_type,'custom')
                coefElement = setfield(coefElement,'type', 'custom');
            end
            
            coefElement = setfield(coefElement, 'values', property_value); %#ok<*SFLD>
            
        otherwise
            error('%s is not a valid coefSet field', property_name)
            
    end
    
    coefSet(coef_index) = coefElement;
    
end

coefSet = checkCoefSet(coefSet);

end
