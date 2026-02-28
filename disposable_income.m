format long g  % avoids scientific notation

%% ===== Overall (used only for demographic multipliers) =====
inc_all = 104207;
exp_all = 78535;
r0 = exp_all / inc_all;

%% ===== Regions =====
inc_region = struct('Northeast',115770,'Midwest',97104,'South',93814,'West',120365);
exp_region = struct('Northeast',83692 ,'Midwest',74707,'South',70376,'West',92625);

regions = fieldnames(inc_region);
r_region = struct();
for i = 1:numel(regions)
    k = regions{i};
    r_region.(k) = exp_region.(k) / inc_region.(k);
end

%% ===== Ages =====
inc_age = struct('Under25',48514,'Age25_34',102494,'Age35_44',128285,'Age45_54',141121, ...
                 'Age55_64',121571,'Age65_74',75460,'Age75plus',56028);
exp_age = struct('Under25',47283,'Age25_34',74475,'Age35_44',91229,'Age45_54',100327, ...
                 'Age55_64',84946,'Age65_74',65354,'Age75plus',55834);

ages = fieldnames(inc_age);
r_age = struct();
for i = 1:numel(ages)
    k = ages{i};
    r_age.(k) = exp_age.(k) / inc_age.(k);
end

%% ===== Family structure =====
inc_fam = struct('MarriedTotal',144988,'MarriedOnly',120105,'MarriedKids',168471, ...
                 'OtherMarried',148763,'OneParent',61118,'Single',67075);
exp_fam = struct('MarriedTotal',101455,'MarriedOnly',88687,'MarriedKids',113585, ...
                 'OtherMarried',103121,'OneParent',61857,'Single',56587);

fams = fieldnames(inc_fam);
r_fam = struct();
for i = 1:numel(fams)
    k = fams{i};
    r_fam.(k) = exp_fam.(k) / inc_fam.(k);
end

%% ===== Income bracket data (to estimate income elasticity beta) =====
% Bracket means (Income, Expenditure)
income_br = [  7637  22443  34984  44824  59582  83888 121852 171847 322142 ]';
expend_br = [ 32118  36368  46246  50375  59378  71369  89727 117378 170722 ]';

% Log-log regression: ln(E) = a + beta ln(I)
X = [ones(size(income_br)) log(income_br)];
y = log(expend_br);
b = X \ y;            % OLS coefficients [a; beta]
a = b(1);
beta = b(2);

% Convert to power-law: E = alpha * S^beta
alpha = exp(a);

% (Optional) quick check: implied spending rate decreases with S if beta < 1
fprintf('Estimated income elasticity beta = %.6f\n', beta);
fprintf('Alpha = %.6f\n\n', alpha);

%% ===== Display rates (unchanged) =====
fprintf('Overall spending rate r0 = %.6f\n\n', r0);

disp('Region rates:');
for i = 1:numel(regions)
    k = regions{i};
    fprintf('  %s: %.6f\n', k, r_region.(k));
end
fprintf('\n');

disp('Age rates:');
for i = 1:numel(ages)
    k = ages{i};
    fprintf('  %s: %.6f\n', k, r_age.(k));
end
fprintf('\n');

disp('Family rates:');
for i = 1:numel(fams)
    k = fams{i};
    fprintf('  %s: %.6f\n', k, r_fam.(k));
end
fprintf('\n');

%% ===== NEW MODEL =====
% Base expected expenditure from salary using the power-law fit:
%   E_base(S) = alpha * S^beta
%
% Then apply demographic multipliers on the *rate* (relative to overall):
%   mult = (r_region/r0) * (r_age/r0) * (r_fam/r0)
% Final:
%   E_hat = E_base(S) * mult
%   D_hat = S - E_hat

predictDisposable = @(S, region, age, family) ...
    S - (alpha * (S.^beta)) * ...
    ((r_region.(region)/r0) * (r_age.(age)/r0) * (r_fam.(family)/r0));

% If you want to avoid extreme cases (optional cap):
% capMult = 1.5; % example
% predictDisposable = @(S, region, age, family) ...
%     S - (alpha * (S.^beta)) * ...
%     min(capMult, ((r_region.(region)/r0) * (r_age.(age)/r0) * (r_fam.(family)/r0)));

%% ===== Example Set 1: Several single predictions =====
format long g  % avoids scientific notation

ex1 = predictDisposable(75000,'South','Age45_54','MarriedOnly');
fprintf('Ex1 (75k, South, 45-54, MarriedOnly) Disposable = %.2f\n', ex1);

ex2 = predictDisposable(50000,'Northeast','Under25','Single');
fprintf('Ex2 (50k, Northeast, Under25, Single) Disposable = %.2f\n', ex2);

ex3 = predictDisposable(120000,'West','Age35_44','MarriedKids');
fprintf('Ex3 (120k, West, 35-44, MarriedKids) Disposable = %.2f\n', ex3);

ex4 = predictDisposable(35000,'Midwest','Age65_74','OtherMarried');
fprintf('Ex4 (35k, Midwest, 65-74, OtherMarried) Disposable = %.2f\n', ex4);

ex5 = predictDisposable(200000,'Northeast','Age55_64','MarriedTotal');
fprintf('Ex5 (200k, Northeast, 55-64, MarriedTotal) Disposable = %.2f\n\n', ex5);
%% ===== Example Set 2: Loop over salaries for one demographic profile =====
format long g

S_list = [30000 50000 75000 100000 150000 250000];
region = 'South';
age    = 'Age45_54';
family = 'MarriedOnly';

fprintf('Profile: %s | %s | %s\n', region, age, family);
for S = S_list
    D = predictDisposable(S, region, age, family);
    fprintf('  Salary=%9.0f -> Disposable=%12.2f\n', S, D);
end
fprintf('\n');

%% ===== Example Set 3: Compare regions (same salary + age + family) =====
format long g

S = 85000;
age = 'Age35_44';
family = 'Single';

fprintf('Compare regions @ Salary=%.0f, Age=%s, Family=%s\n', S, age, family);
for i = 1:numel(regions)
    reg = regions{i};
    D = predictDisposable(S, reg, age, family);
    fprintf('  %-10s -> Disposable=%12.2f\n', reg, D);
end
fprintf('\n');
%% ===== Example Set 4: Compare ages (same salary + region + family) =====
format long g

S = 65000;
region = 'Midwest';
family = 'OneParent';

fprintf('Compare ages @ Salary=%.0f, Region=%s, Family=%s\n', S, region, family);
for i = 1:numel(ages)
    akey = ages{i};
    D = predictDisposable(S, region, akey, family);
    fprintf('  %-10s -> Disposable=%12.2f\n', akey, D);
end
fprintf('\n');
%% ===== Example Set 5: Compare family structures (same salary + region + age) =====
format long g

S = 90000;
region = 'West';
age = 'Age25_34';

fprintf('Compare family @ Salary=%.0f, Region=%s, Age=%s\n', S, region, age);
for i = 1:numel(fams)
    fkey = fams{i};
    D = predictDisposable(S, region, age, fkey);
    fprintf('  %-12s -> Disposable=%12.2f\n', fkey, D);
end
fprintf('\n');
%% ===== Example Set 6: Build a table of results (nice for exporting/plotting) =====
format long g

S_list = (30000:15000:120000)';
region = 'Northeast';
age    = 'Age25_34';
family = 'Single';

D_list = zeros(size(S_list));
for j = 1:numel(S_list)
    D_list(j) = predictDisposable(S_list(j), region, age, family);
end

T = table(S_list, D_list, 'VariableNames', {'Salary','Disposable'});
disp(T);
fprintf('\n');
%% ===== Example Set 7: What-if: show multiplier breakdown explicitly =====
% This uses your rates already computed.

format long g

S = 75000;
region = 'South';
age    = 'Age45_54';
family = 'MarriedOnly';

E_base = alpha * (S.^beta);
mult   = (r_region.(region)/r0) * (r_age.(age)/r0) * (r_fam.(family)/r0);
E_hat  = E_base * mult;
D_hat  = S - E_hat;

fprintf('Breakdown @ Salary=%.0f | %s | %s | %s\n', S, region, age, family);
fprintf('  E_base = %.2f\n', E_base);
fprintf('  mult   = %.6f\n', mult);
fprintf('  E_hat  = %.2f\n', E_hat);
fprintf('  D_hat  = %.2f\n\n', D_hat);
%% ===== Example Set 8: Simple input validation wrapper (prevents typos) =====
% Use this instead of calling predictDisposable directly.

predictDisposableSafe = @(S, region, age, family) ...
    validateAndPredict(S, region, age, family, regions, ages, fams, predictDisposable);

% Local function at bottom of script (MATLAB allows local functions in scripts):
function D = validateAndPredict(S, region, age, family, regions, ages, fams, predictDisposable)
    if ~(isnumeric(S) && isscalar(S) && S > 0)
        error('Salary S must be a positive scalar number.');
    end
    if ~any(strcmp(region, regions))
        error('Unknown region "%s". Valid: %s', region, strjoin(regions, ', '));
    end
    if ~any(strcmp(age, ages))
        error('Unknown age "%s". Valid: %s', age, strjoin(ages, ', '));
    end
    if ~any(strcmp(family, fams))
        error('Unknown family "%s". Valid: %s', family, strjoin(fams, ', '));
    end
    D = predictDisposable(S, region, age, family);
end
%% ===== Example Set 9: Sensitivity around one variable (salary perturbation) =====
format long g

S0 = 80000;
region = 'South';
age    = 'Age35_44';
family = 'MarriedKids';

dS = [-20000 -10000 -5000 0 5000 10000 20000];
fprintf('Salary sensitivity around %.0f (%s | %s | %s)\n', S0, region, age, family);
for k = 1:numel(dS)
    S = S0 + dS(k);
    D = predictDisposable(S, region, age, family);
    fprintf('  Salary=%9.0f -> Disposable=%12.2f\n', S, D);
end
fprintf('\n');