function s = CheckLset(s)

if (length(s) > 1)
    
    s = arrayfun(@(ss) CheckLset(ss),s);
    
    %make sure every modeltype is the same
    if length(unique(GetLset(s,'modeltype')))~=1
        error('Lset contains multiple modeltypes \n')
    end
    

else
    
    if ~isfield(s,'modeltype')
        
        s = set_default_or_empty(s,'modeltype','slim');
        
        switch s.modeltype
            case {'slim','mnrules','tilm','pilm'}
            otherwise
                error('unsupported modeltype \n')
        end
        
    else
        
        s.modeltype = 'slim';
        
    end
    
    switch s.modeltype
        
        
        case 'mnrules'
            
            %name
            %modeltype
            %class
            s.class     = NaN;
            s.type      = 'I';
            s.lim       = [0,1];
            s.values    = NaN;
            s           = set_default_or_empty(s,'sign',NaN);
            s.scale     = NaN;
            s           = set_default_or_empty(s,'C_0j',NaN);
            
            
        case 'slim'
            
            
            if isfield(s,'values')
                if ~isempty(s.values) && all(~isnan(s.values))
                    s.class = 'custom';
                else
                    s.class = 'bounded';
                end
            else
                s.class = 'bounded';
            end
            
            switch s.class
                
                case 'bounded'
                    
                    s = set_default_or_empty(s,'sign',NaN);
                    s = set_default_or_empty(s,'lim',[-100,100]);
                    s = set_default_or_empty(s,'scale',NaN);
                    s = set_default_or_empty(s,'values',NaN);
                    s = set_default_or_empty(s,'C_0j',NaN);
                    
                    %make sure that the left limit < right limit
                    if length(s.lim)~=2
                        error('lim must be 2 x 1 numeric')
                    end
                    
                    s.lim = sort(s.lim);
                    
                    low     = s.lim(1);
                    high    = s.lim(2);
                    
                    %make sure sign and limits are in agreement
                    if isnan(s.sign) || isempty(s.sign)
                        if (low<=0) && (high<=0);
                            s.sign = -1;
                        elseif (low>=0) && (high>=0);
                            s.sign = 1;
                        end
                    elseif s.sign == 1
                        low  = max(low,0);
                    elseif s.sign == -1
                        high = min(0,high);
                    end
                    s.lim = [low,high];
                    
                    %infer type
                    if ~isfield(s,'type') || isempty(s.type) || isnan(s.type)
                        if all(floor([low,high])==ceil([low,high]))
                            s = set_default_or_empty(s,'type','I');
                        else
                            s = set_default_or_empty(s,'type','C');
                        end
                    end
                    
                    switch s.type
                        
                        case {'I','C'}
                            
                        otherwise
                            error('variable types must be I or C')
                    end
                    
                    
                case 'custom'
                    
                    s = set_default_or_empty(s,'C_0j',NaN);
                    
                    if ~isfield(s,'values')
                        error('custom Lset should contain a field of custom values')
                    end
                    
                    %use set of values to determine limits
                    values  = s.values;
                    values  = unique(values); %remove duplicate values;
                    values  = values(~isnan(values));
                    
                    %make sure sign and values are in agreement
                    if isfield(s,'sign') && ~isempty(s.sign) && ~isnan(s.sign)
                        agree_in_sign = (s.sign==sign(values)|(values==0));
                        values(~agree_in_sign) = [];
                        s.values = values;
                    end
                    
                    %use values to reset lower/upper limits
                    low     = min(values);
                    high    = max(values);
                    
                    s.lim   = [low,high];
                    s = set_default_or_empty(s,'scale',NaN);
                    
                    %check type
                    if all(floor(values)==ceil(values))
                        s.type = 'I';
                    else
                        s.type = 'C';
                    end
                    
                    switch s.type
                        
                        case {'I','C'}
                            
                        otherwise
                            error('variable types must be I or C')
                    end
                    
                    %user signs on limits to see if you can specify sign
                    if (low<0) && (high>0)
                        s.sign = NaN;
                    elseif (low<=0) && (high<=0);
                        s.sign = -1;
                    elseif (low>=0) && (high>=0);
                        s.sign = +1;
                    end
                    
            end
            
            
        
            
    end
    
    
    fs              = fieldnames(s);
    s_name_ind      = find(strcmp(fs,'name'));
    s_modeltype_ind = find(strcmp(fs,'modeltype'));
    s_class_ind     = find(strcmp(fs,'class'));
    s_type_ind      = find(strcmp(fs,'type'));
    s_lim_ind       = find(strcmp(fs,'lim'));
    s_values_ind    = find(strcmp(fs,'values'));
    s_sign_ind      = find(strcmp(fs,'sign'));
    s_scale_ind     = find(strcmp(fs,'scale'));
    s_C_0j_ind      = find(strcmp(fs,'C_0j'));
    
    s_perm          = [s_name_ind, s_modeltype_ind, s_class_ind,s_type_ind,s_lim_ind,s_values_ind,s_sign_ind,s_scale_ind,s_C_0j_ind];
    
    extra_fs        = setdiff(1:length(fs),s_perm);
    
    if ~isempty(extra_fs)
        
        fprintf('field %s is not recognized and will be dropped \n', fs{extra_fs});
        s               = rmfield(s,fs(extra_fs));
        fs              = fieldnames(s);
        s_name_ind      = find(strcmp(fs,'name'));
        s_modeltype_ind = find(strcmp(fs,'modeltype'));
        s_class_ind     = find(strcmp(fs,'class'));
        s_type_ind      = find(strcmp(fs,'type'));
        s_lim_ind       = find(strcmp(fs,'lim'));
        s_values_ind    = find(strcmp(fs,'values'));
        s_sign_ind      = find(strcmp(fs,'sign'));
        s_scale_ind     = find(strcmp(fs,'scale'));
        s_C_0j_ind      = find(strcmp(fs,'C_0j'));
        s_perm          = [s_name_ind, s_modeltype_ind, s_class_ind,s_type_ind,s_lim_ind,s_values_ind,s_sign_ind,s_scale_ind,s_C_0j_ind];
    
    end
    
    s               = orderfields(s,[s_perm]);
    
end




%% Helper Functions


    function settingfile = set_default_or_empty(settingfile,settingname,defaultvalue)
        
        if ~isfield(settingfile,settingname)
            settingfile.(settingname) = defaultvalue;
        elseif isfield(settingfile,settingname) && isempty(settingfile.(settingname))
            settingfile.(settingname) = defaultvalue;
        end
        
    end


end