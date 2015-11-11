function value = load_or_setdefault(setting_name,default_value,run_name)

if nargin==2
    run_name='';
end

try
    
    if ~isempty(run_name) && run_name(end)~='_'
        setting_file = [run_name,'_',setting_name,'.txt'];
    else
        setting_file = [setting_name,'.txt'];
    end
    
    
    if iscellstr(default_value)
        
        value = csvimport(setting_file,'Delimited',' ');
        
    elseif ischar(default_value)
        
        value = csvimport(setting_file,'Delimited',' ');
        value = value{1};
        
    elseif isnumeric(default_value)
        
        value = load(setting_file,'-ascii');
        
    elseif islogical(default_value)
        
        try
            value = load(setting_file,'-ascii');
            value = value>0;
        catch
            
            value = csvimport(setting_file,'Delimited',' ');
            if strcmpi(value,'true')
                value = true;
            elseif strcmpi(value,'false')
                value=false;
            end
            
        end
        
    else
        
        value = default_value;
        
    end
    
catch
    
    value = default_value;
    
end

end