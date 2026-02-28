format long g  % avoids scientific notation

%% ========================================================================
%  Disposable Income Model (Q1) - FIXED demographic multipliers
%  Key fix: demographic multipliers use "residual ratios"
%     m_g = Ebar_g / (alpha * Sbar_g^beta)
%  instead of raw spending rates r_g = Ebar_g / Sbar_g (which can double-count income effects)
%% ========================================================================

%% ===== Overall (used for centering demographic multipliers) =====
inc_all = 104207;
ess_all = 72052;

% (Optional) raw overall spending rate, mostly for reporting
r0 = ess_all / inc_all;

%% ===== Regions =====
inc_region = struct('Northeast',115770,'Midwest',97104,'South',93814,'West',120365);
ess_region = struct('Northeast',77535,'Midwest',67748,'South',65065,'West',84256);
regions = fieldnames(inc_region);

%% ===== Ages =====
inc_age = struct('Under25',48514,'Age25_34',102494,'Age35_44',128285,'Age45_54',141121, ...
                 'Age55_64',121571,'Age65_74',75460,'Age75plus',56028);
ess_age = struct('Under25',44057,'Age25_34',68832,'Age35_44',84322,'Age45_54',91359, ...
                 'Age55_64',78195,'Age65_74',59655,'Age75plus',50436);
ages = fieldnames(inc_age);

%% ===== Family structure =====
inc_fam = struct('MarriedTotal',144988,'MarriedOnly',120105,'MarriedKids',168471, ...
                 'OtherMarried',148763,'OneParent',61118,'Single',67075);
ess_fam = struct('MarriedTotal',92951,'MarriedOnly',80046,'MarriedKids',104727, ...
                 'OtherMarried',97291,'OneParent',58162,'Single',51986);
fams = fieldnames(inc_fam);

%% ===== Income bracket data (to estimate income elasticity beta) =====
income_br = [  7637  22443  34984  44824  59582  83888 121852 171847 322142 ]';
ess_br    = [ 30214  34517  43459  47664  55157  66064  83346 105686 152382 ]';
% income_br = [ 22443  34984  44824  59582  83888 121852 171847 322142 ]';
% ess_br    = [ 34517  43459  47664  55157  66064  83346 105686 152382 ]';

% Fit: log(E) = a + beta*log(S)  =>  E = alpha*S^beta, alpha=exp(a)
X = [ones(size(income_br)) log(income_br)];
y = log(ess_br);
b = X \ y;
a = b(1);
beta = b(2);
alpha = exp(a);

fprintf('Estimated essential-income elasticity beta = %.6f\n', beta);
fprintf('Alpha = %.6f\n\n', alpha);

%% ===== FIXED demographic multipliers (residual ratios) =====
% Centering constant (overall residual level). Often near 1, but we compute it.
m0 = ess_all / (alpha * (inc_all^beta));

% Region residual multipliers
m_region = struct();
for i = 1:numel(regions)
    k = regions{i};
    m_region.(k) = ess_region.(k) / (alpha * (inc_region.(k)^beta));
end

% Age residual multipliers
m_age = struct();
for i = 1:numel(ages)
    k = ages{i};
    m_age.(k) = ess_age.(k) / (alpha * (inc_age.(k)^beta));
end

% Family residual multipliers
m_fam = struct();
for i = 1:numel(fams)
    k = fams{i};
    m_fam.(k) = ess_fam.(k) / (alpha * (inc_fam.(k)^beta));
end

%% ===== Display (optional) =====
fprintf('Overall raw spending rate r0 = %.6f\n', r0);
fprintf('Overall residual level m0    = %.6f\n\n', m0);

disp('Region residual multipliers (m_region):');
for i = 1:numel(regions)
    k = regions{i};
    fprintf('  %s: %.6f\n', k, m_region.(k));
end
fprintf('\n');

disp('Age residual multipliers (m_age):');
for i = 1:numel(ages)
    k = ages{i};
    fprintf('  %s: %.6f\n', k, m_age.(k));
end
fprintf('\n');

disp('Family residual multipliers (m_fam):');
for i = 1:numel(fams)
    k = fams{i};
    fprintf('  %s: %.6f\n', k, m_fam.(k));
end
fprintf('\n');

%% ===== TAX MODEL (simple) =====
taxParams = struct();

taxParams.fed.single.brk = [11000 44725 95375 182100 231250 578125 inf];
taxParams.fed.single.r   = [0.10  0.12  0.22  0.24   0.32   0.35   0.37];

taxParams.fed.mfj.brk = [22000 89450 190750 364200 462500 693750 inf];
taxParams.fed.mfj.r   = [0.10  0.12  0.22   0.24   0.32   0.35   0.37];

taxParams.fed.hoh.brk = [15700 59850 95350 182100 231250 578100 inf];
taxParams.fed.hoh.r   = [0.10  0.12  0.22  0.24   0.32   0.35   0.37];

taxParams.std.single = 13850;
taxParams.std.mfj    = 27700;
taxParams.std.hoh    = 20800;

taxParams.fica.ss_rate = 0.062;
taxParams.fica.med_rate = 0.0145;
taxParams.fica.ss_wage_base = 168600;
taxParams.fica.addl_med_rate = 0.009;
taxParams.fica.addl_med_thresh.single = 200000;
taxParams.fica.addl_med_thresh.mfj    = 250000;
taxParams.fica.addl_med_thresh.hoh    = 200000;

taxParams.stateEff = struct('Northeast',0.045,'Midwest',0.035,'South',0.025,'West',0.040);

estimateTaxes = @(S, region, family) estimateTotalTax(S, region, filingFromFamily(family), taxParams);

%% ===== NEW MODEL (FIXED multipliers) =====
% Weights for how strongly each demographic dimension influences essentials
wR = 0.40; wA = 0.25; wF = 0.35;

% Income-only baseline essentials
E_base = @(S) alpha * (S.^beta);

% Weighted geometric blend of residual multipliers, centered at overall m0
multiplier = @(region, age, family) exp( ...
      wR * log(m_region.(char(region))/m0) ...
    + wA * log(m_age.(char(age))/m0) ...
    + wF * log(m_fam.(char(family))/m0) );

% Predicted essentials with demographics
E_hat = @(S, region, age, family) E_base(S) .* multiplier(region, age, family);

% Disposable income
predictDisposable = @(S, region, age, family) max(-10000000, ...
    S - E_hat(S, region, age, family) - estimateTaxes(S, region, family));

%% ===== Example Set 1: Several single predictions =====
ex1 = predictDisposable(75000,'South','Age45_54','MarriedOnly');
fprintf('Ex1 (75k, South, 45-54, MarriedOnly) Disposable = %.2f\n', ex1);

ex2 = predictDisposable(50000,'Northeast','Under25','Single');
fprintf('Ex2 (50k, Northeast, Under25, Single) Disposable = %.2f\n', ex2);

ex3 = predictDisposable(120000,'West','Age35_44','MarriedKids');
fprintf('Ex3 (120k, West, 35-44, MarriedKids) Disposable = %.2f\n', ex3);

ex4 = predictDisposable(35000,'Midwest','Age65_74','OtherMarried');
fprintf('Ex4 (35k, Midwest, 65-74, OtherMarried) Disposable = %.2f\n', ex4);

ex5 = predictDisposable(450000,'Northeast','Age45_54','MarriedTotal');
fprintf('Ex5 (450k, Northeast, 45-54, MarriedTotal) Disposable = %.2f\n\n', ex5);

%% ===== Example Set 7: Breakdown including taxes =====
S = 120000;
region = 'West';
age    = 'Age45_54';
family = 'MarriedKids';

E0 = E_base(S);

mr = m_region.(region)/m0;
ma = m_age.(age)/m0;
mf = m_fam.(family)/m0;

logMult = wR*log(mr) + wA*log(ma) + wF*log(mf);
mult    = exp(logMult);

Epred = E0 * mult;
That  = estimateTaxes(S, region, family);
Draw  = S - Epred - That;
Dhat  = max(0, Draw);

fprintf('Breakdown @ Salary=%.0f | %s | %s | %s\n', S, region, age, family);
fprintf('  E_base = %.2f\n', E0);
fprintf('  mult   = %.6f\n', mult);
fprintf('  E_hat  = %.2f\n', Epred);
fprintf('  Taxes  = %.2f\n', That);
fprintf('  D_raw  = %.2f\n', Draw);
fprintf('  D_hat  = %.2f\n\n', Dhat);

%% ===== Tax debug =====
fprintf('--- Tax debug ---\n');
fprintf('Ex1 taxes: %.2f\n', estimateTaxes(75000,'South','MarriedOnly'));
fprintf('Ex2 taxes: %.2f\n', estimateTaxes(50000,'Northeast','Single'));
fprintf('Ex3 taxes: %.2f\n', estimateTaxes(120000,'West','MarriedKids'));
fprintf('Ex4 taxes: %.2f\n', estimateTaxes(35000,'Midwest','OtherMarried'));
fprintf('Ex5 taxes: %.2f\n', estimateTaxes(200000,'Northeast','MarriedTotal'));

%% ========================================================================
%  EXAMPLES: show how age, family, income, and region change disposable income
%  (Drop this block AFTER predictDisposable is defined.)
%% ========================================================================

format long g   % avoids scientific notation for disp() output

% ---------- Baseline profile ----------
S0      = 100000;
region0 = 'Northeast';
age0    = 'Age35_44';
family0 = 'MarriedTotal';

fprintf('\n=== BASELINE ===\n');
D0 = predictDisposable(S0, region0, age0, family0);
fprintf('Baseline: S=%g | %s | %s | %s  =>  Disposable=%g\n\n', S0, region0, age0, family0, D0);

% ---------- 1) Vary INCOME only ----------
fprintf('=== VARY INCOME (holding region/age/family fixed) ===\n');
S_list = [25000 35000 50000 75000 100000 150000 200000 300000 450000]';
for i = 1:numel(S_list)
    S = S_list(i);
    D = predictDisposable(S, region0, age0, family0);
    fprintf('S=%g | %s | %s | %s  =>  Disposable=%g\n', S, region0, age0, family0, D);
end
fprintf('\n');

% ---------- 2) Vary REGION only ----------
fprintf('=== VARY REGION (holding income/age/family fixed) ===\n');
regions_list = {'Northeast','Midwest','South','West'};
for i = 1:numel(regions_list)
    region = regions_list{i};
    D = predictDisposable(S0, region, age0, family0);
    fprintf('S=%g | %-9s | %s | %s  =>  Disposable=%g\n', S0, region, age0, family0, D);
end
fprintf('\n');

% ---------- 3) Vary AGE only ----------
fprintf('=== VARY AGE (holding income/region/family fixed) ===\n');
ages_list = {'Under25','Age25_34','Age35_44','Age45_54','Age55_64','Age65_74','Age75plus'};
for i = 1:numel(ages_list)
    age = ages_list{i};
    D = predictDisposable(S0, region0, age, family0);
    fprintf('S=%g | %s | %-9s | %s  =>  Disposable=%g\n', S0, region0, age, family0, D);
end
fprintf('\n');

% ---------- 4) Vary FAMILY only ----------
fprintf('=== VARY FAMILY (holding income/region/age fixed) ===\n');
families_list = {'Single','OneParent','MarriedOnly','MarriedKids','OtherMarried','MarriedTotal'};
for i = 1:numel(families_list)
    family = families_list{i};
    D = predictDisposable(S0, region0, age0, family);
    fprintf('S=%g | %s | %s | %-12s =>  Disposable=%g\n', S0, region0, age0, family, D);
end
fprintf('\n');

% ---------- 5) Side-by-side "personas" (realistic combos) ----------
fprintf('=== PERSONA EXAMPLES (mixing age/family/region/income) ===\n');
cases = {
    35000,  'South',     'Under25',   'Single'
    55000,  'Midwest',   'Age25_34',  'Single'
    70000,  'Northeast', 'Age25_34',  'OneParent'
    95000,  'West',      'Age35_44',  'MarriedOnly'
    120000, 'West',      'Age35_44',  'MarriedKids'
    145000, 'Northeast', 'Age45_54',  'MarriedKids'
    80000,  'South',     'Age55_64',  'OtherMarried'
    60000,  'Midwest',   'Age65_74',  'MarriedOnly'
    50000,  'Northeast', 'Age75plus', 'Single'
    300000, 'West',      'Age45_54',  'MarriedTotal'
    450000, 'Northeast', 'Age45_54',  'MarriedTotal'
};
for i = 1:size(cases,1)
    S      = cases{i,1};
    region = cases{i,2};
    age    = cases{i,3};
    family = cases{i,4};
    D      = predictDisposable(S, region, age, family);
    fprintf('%2d) S=%g | %-9s | %-9s | %-12s => Disposable=%g\n', i, S, region, age, family, D);
end
fprintf('\n');

% ---------- 6) Mini "grid" (shows interactions) ----------
fprintf('=== MINI GRID (interaction: income x region x age x family) ===\n');
S_grid     = [50000 75000 100000 125000 150000 175000 200000];
region_grid = {'Northeast','South','West', 'Midwest'};
age_grid    = {'Under25','Age25_34', 'Age35_44','Age45_54', 'Age55_64', 'Age65_74'};
family_grid = {'Single','MarriedKids','OneParent', 'MarriedOnly'};

minD = inf; maxD = -inf;
minRow = []; maxRow = [];

row = 0;
for si = 1:numel(S_grid)
for ri = 1:numel(region_grid)
for ai = 1:numel(age_grid)
for fi = 1:numel(family_grid)
    row = row + 1;

    S      = S_grid(si);
    region = region_grid{ri};
    age    = age_grid{ai};
    family = family_grid{fi};

    D = predictDisposable(S, region, age, family);

    % print all rows (small grid: 3*3*3*3 = 81)
    fprintf('%2d) S=%g | %-9s | %-9s | %-11s => D=%g\n', row, S, region, age, family, D);

    if D < minD
        minD = D; minRow = {S, region, age, family};
    end
    if D > maxD
        maxD = D; maxRow = {S, region, age, family};
    end
end
end
end
end

fprintf('\n--- GRID SUMMARY ---\n');
fprintf('Min D=%g at S=%g | %s | %s | %s\n', minD, minRow{1}, minRow{2}, minRow{3}, minRow{4});
fprintf('Max D=%g at S=%g | %s | %s | %s\n', maxD, maxRow{1}, maxRow{2}, maxRow{3}, maxRow{4});
fprintf('\n');

%% ========================================================================
%  FIGURE #1: Disposable income vs. Salary (threshold curve)
%  FIGURE #2: Component breakdown vs. Salary (Salary vs Taxes vs Essentials)
%  (Drop this AFTER predictDisposable is defined.)
%% ========================================================================

% Choose a baseline demographic profile (edit as desired)
regionP = 'Northeast';
ageP    = 'Age35_44';
familyP = 'MarriedTotal';

% Salary range for plots
Svec = (20000:2000:450000)';   % column vector

% Preallocate
Dvec  = zeros(size(Svec));
Tvec  = zeros(size(Svec));
Evec  = zeros(size(Svec));

% Evaluate model across salary range
for i = 1:numel(Svec)
    S = Svec(i);
    Tvec(i) = estimateTaxes(S, regionP, familyP);
    Evec(i) = E_hat(S, regionP, ageP, familyP);
    Dvec(i) = S - Tvec(i) - Evec(i);     % raw disposable (can be negative)
end

% Find approximate break-even point (where D crosses 0)
idx = find(Dvec >= 0, 1, 'first');
S_break = NaN;
if ~isempty(idx) && idx > 1
    % linear interpolation between the two nearest points
    S1 = Svec(idx-1); D1 = Dvec(idx-1);
    S2 = Svec(idx);   D2 = Dvec(idx);
    S_break = S1 + (0 - D1) * (S2 - S1) / (D2 - D1);
end

%% ---------------- FIGURE #1: D vs Salary ----------------
figure(1); clf;
plot(Svec, Dvec, 'LineWidth', 2);
hold on;
yline(0, '--', 'LineWidth', 1.5);

% Mark break-even salary if it exists in range
if ~isnan(S_break)
    xline(S_break, ':', 'LineWidth', 1.5);
    text(S_break, 0, sprintf('  break-even \\approx $%0.0f', S_break), ...
        'VerticalAlignment','bottom','Interpreter','tex');
end

grid on;
xlabel('Salary S ($)');
ylabel('Disposable Income D ($)');
title(sprintf('Disposable Income vs Salary (%s | %s | %s)', regionP, ageP, familyP));
set(gca,'XLim',[min(Svec) max(Svec)]);

% Optional: improve tick formatting (no scientific notation)
xtickformat('%,.0f');
ytickformat('%,.0f');

%% ---------------- FIGURE #2: Components vs Salary ----------------
% This figure explains WHY many D values are negative:
% taxes + essentials can exceed salary at low/moderate incomes.

figure(2); clf;
plot(Svec, Svec, 'LineWidth', 2);        % Salary line (y = S)
hold on;
plot(Svec, Tvec, 'LineWidth', 2);        % Taxes
plot(Svec, Evec, 'LineWidth', 2);        % Essentials
plot(Svec, Tvec + Evec, 'LineWidth', 2); % Total burden

grid on;
xlabel('Salary S ($)');
ylabel('Dollars ($)');
title(sprintf('Model Components vs Salary (%s | %s | %s)', regionP, ageP, familyP));
legend({'Salary (S)','Taxes T(S)','Essentials \hat{E}','Taxes + Essentials'}, ...
       'Location','northwest');

set(gca,'XLim',[min(Svec) max(Svec)]);
xtickformat('%,.0f');
ytickformat('%,.0f');

% Optional: visually highlight region where D<0 (burden > salary)
% Shade where (Taxes + Essentials) exceeds Salary
mask = (Tvec + Evec) > Svec;
if any(mask)
    % Find contiguous segment(s) - simplest: shade from start to first non-mask
    first = find(mask, 1, 'first');
    last  = find(mask, 1, 'last');
    x1 = Svec(first); x2 = Svec(last);

    yl = ylim;
    patch([x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], ...
        [0.9 0.9 0.9], 'FaceAlpha',0.15, 'EdgeColor','none');
    uistack(findobj(gca,'Type','line'),'top'); % keep lines on top
    text((x1+x2)/2, yl(2)*0.95, 'Region where Taxes + Essentials > Salary (D<0)', ...
        'HorizontalAlignment','center');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ===== LOCAL FUNCTIONS (must be at the END of the script) =====
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function filing = filingFromFamily(family)
    family = char(family);
    switch family
        case 'Single'
            filing = 'single';
        case 'OneParent'
            filing = 'hoh';
        case {'MarriedTotal','MarriedOnly','MarriedKids','OtherMarried'}
            filing = 'mfj';
        otherwise
            error('Unknown family "%s" for filing status mapping.', family);
    end
end

function T = estimateTotalTax(S, region, filing, taxParams)
    region = char(region);
    filing = char(filing);

    stdDed  = taxParams.std.(filing);
    taxable = max(0, S - stdDed);

    brk = taxParams.fed.(filing).brk;
    r   = taxParams.fed.(filing).r;
    fed = progressiveTax(taxable, brk, r);

    ss  = taxParams.fica.ss_rate  * min(S, taxParams.fica.ss_wage_base);
    med = taxParams.fica.med_rate * S;

    thresh  = taxParams.fica.addl_med_thresh.(filing);
    addlMed = taxParams.fica.addl_med_rate * max(0, S - thresh);

    fica = ss + med + addlMed;

    state = taxParams.stateEff.(region) * S;

    T = fed + fica + state;
end

function tax = progressiveTax(x, brk, r)
    tax = 0;
    prev = 0;
    for i = 1:numel(brk)
        upper = brk(i);
        amt = max(0, min(x, upper) - prev);
        tax = tax + amt * r(i);
        prev = upper;
        if x <= upper
            break;
        end
    end
end