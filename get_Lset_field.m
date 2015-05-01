function fvals = get_Lset_field(Lset,field_name)

switch fname
    
    case 'name'
        fvals = {Lset.name}';
      
    case 'modeltype'
        fvals = {Lset.modeltype}';
        
    case 'class'
        fvals = {Lset.class}';
        
    case 'type'
        fvals = {Lset.type}';
        
    case 'lim'
        fvals = reshape([Lset.lim],2,length([Lset.lim]')/2)';
    
    case 'ub'
        lim = [Lset.lim];
        fvals = lim(2:2:end)';
    
    case 'lb'
        lim = [Lset.lim];
        fvals = lim(1:2:end-1)';
        
    case 'values'
        fvals = {Lset.values}';
    
    case 'nvalues'
        fvals = cellfun(@(x) length(x(~isnan(x))), {Lset.values}');
    
    case 'sign'
        fvals = [Lset.sign]';
    
    case 'scale'
        fvals = [Lset.scale]';
    
    case 'C_0j'
        fvals = [Lset.C_0j]';
        
        
    otherwise
        error('Unsupported field')

end
