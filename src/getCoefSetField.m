function property_values = getCoefSetField(coefSet,property_name)
%Get all values for a certain property from a coefSet object
%
%coefSet        coefSet containing the values
%property_name  name of the property whose value we want to get
%               valid properties are 'name','lb','ub','type', 'values','sign','C_0j'
%
%Author:      Berk Ustun 
%Contact:     ustunb@mit.edu
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

switch property_name
    
    case 'name'
        property_values = {coefSet.name}';
        
    case 'type'
        property_values = {coefSet.type}';
        
    case 'ub'
        property_values = [coefSet.ub];
        
    case 'lb'
        property_values = [coefSet.lb];
        
    case 'values'
        property_values = {coefSet.values}';
            
    case 'sign'
        property_values = [coefSet.sign]';
        
    case 'C_0j'
        property_values = [coefSet.C_0j]';
        
    otherwise
        error('%s is not a valid field', property_name)
        
end
