function Lset = create_Lset(variables_names,varargin)

%creates Lset from a P x 1 cell containing variable names (X_headers)
%does MINIMAL error checking

%Lset fields include:
%
%class:  'bounded' or 'custom'           | default: bounded
%lim:   [low,high]                      | default: [-100,100]
%sign: -1 (neg), 1 (pos), NaN (neg/pos) | default: []
%scale: scaling value                   | default: []
%values: arary containing discrete vals | default: []
%C_0j: custom reg parameter             | default: [];

%Optional inputs

%1 x 1 struct containing the default values for each field
%OR
%Q x 1 struct array containing default values for variables specified in X_headers

%MAKE SURE THAT DEFAULT STRUCT CONTAINS ALL FIELDS!!!!

P = length(variables_names);

%Create Lset
if nargin == 1
    
    default             = struct();
    default.name        = '';
    default.class       = 'bounded';
    default.type        = 'I';
    default.lim         = [-100,100];
    default.scale       = NaN;
    default.sign        = NaN;
    default.values      = NaN;
    default.C_0j        = NaN;
    Lset = repmat(default,P,1);
    [Lset(:).name] = variables_names{:};
    
elseif ((nargin==2) && (isstruct(varargin{1})))
    
    if length(varargin{1}) == 1
        
        default = varargin{1};
        default = check_coef_set(default); %check all the fields;
        Lset = repmat(default,P,1);   %return
        [Lset(:).name] = variables_names{:};
        
    elseif length(varargin{1}) > 1
        
        %parse user values
        given_sets       = varargin{1};
        
        %use the default Lset for each coefficient
        if ~isfield(given_sets,'name')
            error('user provided Lset is missing name \n')
        end
        
        given_names = {given_sets(:).name}';
        
        if any(strcmp('default',given_names))
            def_ind = find(strcmp('default',given_names));
            default = CheckLset(given_sets(def_ind));
            given_sets(def_ind) = [];
        else
            default             = struct();
            default.name        = '';
            default.class       = 'bounded';
            default.type        = 'I';
            default.lim         = [-100,100];
            default.values      = NaN;
            default.sign        = NaN;
            default.scale       = NaN;
            default.C_0j        = NaN;
        end
        Lset                = repmat(default,P,1);
        [Lset(:).name]      = variables_names{:};
        
        
        user_sets      = CreateEmptyLset();
        counter        = 1;
        
        for k = 1:length(given_sets)
            %make sure name of user_sets matches a variable in X_headers
            if  any(strcmp(given_sets(k).name,variables_names))
                user_sets(counter) = check_coef_set(given_sets(k));
                counter = counter + 1;
                % else
                %sprintf('user provided an Lset for variable %s \n whose name does not exist in the list of variables \n this variable will use the default set', user_sets(k).name)
            end
        end
        ind            = cellfun(@(x) find(strcmp(x,variables_names)),{user_sets(:).name},'UniformOutput',true);
        Lset(ind)      = user_sets;
        
    else
        
        error('user provided empty struct array \n')
        
    end
    
else
    
    error('user provided wrong number of inputs \n')
    
end

%Reorder Lset fields so that variable name is at the top

fields          = fieldnames(Lset);
name_ind        = find(strcmp(fields,'name'));
class_ind       = find(strcmp(fields,'class'));
type_ind        = find(strcmp(fields,'type'));
lim_ind         = find(strcmp(fields,'lim'));
values_ind      = find(strcmp(fields,'values'));
sign_ind        = find(strcmp(fields,'sign'));
scale_ind       = find(strcmp(fields,'scale'));
C_0j_ind        = find(strcmp(fields,'C_0j'));
perm            = [name_ind, class_ind,type_ind,lim_ind,values_ind,sign_ind,scale_ind,C_0j_ind];
Lset            = orderfields(Lset,perm);


%% Helper Function


    function settingfile = set_default_or_empty(settingfile,settingname,defaultvalue)
        
        if ~isfield(settingfile,settingname)
            settingfile.(settingname) = defaultvalue;
        elseif isfield(settingfile,settingname) && isempty(settingfile.(settingname))
            settingfile.(settingname) = defaultvalue;
        end
        
    end

end
