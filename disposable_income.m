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
predictDisposable = @(S, region, age, family) max(0, ...
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
S = 150000;
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