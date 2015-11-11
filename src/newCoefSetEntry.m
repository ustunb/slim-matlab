function s = newCoefSetEntry()
%
%create a single entry for a CoefSet object (for easy building
%
s               = struct();
s.name          = '';
s.type          = 'integer';
s.ub            = 10;
s.lb            = -10;
s.sign          = NaN;
s.values        = NaN;
s.C_0j          = NaN;

end
