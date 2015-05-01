function varargout = print_Lset(Lset)


lims    = cell2mat({Lset(:).lim}');
lb      = num2cell(lims(:,1));
ub      = num2cell(lims(:,2));
headers = {'name','class','type','lb','ub','sign','values','C_0j'};

info = [{Lset(:).name}',{Lset(:).modeltype}',{Lset(:).class}',{Lset(:).type}',lb,ub,{Lset(:).sign}',{Lset(:).values}',{Lset(:).C_0j}'];

Lset_table = [headers;info];
disp(Lset_table)

if nargout == 1
    varargout{1} = Lset_table;
end

end