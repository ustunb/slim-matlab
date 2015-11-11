function coefTable  = printCoefSet(coefSet)
%Prints a table containing the fields of a coefSet object
%
%coefSet        coefSet containing the values
%property_name  name of the property whose value we want to get
%               valid properties are 'name','lb','ub','type', 'values','sign','C_0j'
%
%Author:      Berk Ustun 
%Contact:     ustunb@mit.edu
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

headers = {'name','type','lb','ub','sign','values','C_0j'};
info = [{coefSet(:).name}',{coefSet(:).type}',{coefSet(:).lb}',{coefSet(:).ub}', {coefSet(:).sign}',{coefSet(:).values}',{coefSet(:).C_0j}'];
coefTable = [headers;info];
disp(coefTable)

end