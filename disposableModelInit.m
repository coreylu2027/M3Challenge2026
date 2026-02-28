function M = disposableModelInit()
%DISPOSABLEMODELINIT builds model constants once

    format long g

    % ===== Overall =====
    inc_all = 104207;
    ess_all = 72052;

    % ===== Regions =====
    inc_region = struct('Northeast',115770,'Midwest',97104,'South',93814,'West',120365);
    ess_region = struct('Northeast',77535,'Midwest',67748,'South',65065,'West',84256);
    regions = fieldnames(inc_region);

    % ===== Ages =====
    inc_age = struct('Under25',48514,'Age25_34',102494,'Age35_44',128285,'Age45_54',141121, ...
                     'Age55_64',121571,'Age65_74',75460,'Age75plus',56028);
    ess_age = struct('Under25',44057,'Age25_34',68832,'Age35_44',84322,'Age45_54',91359, ...
                     'Age55_64',78195,'Age65_74',59655,'Age75plus',50436);
    ages = fieldnames(inc_age);

    % ===== Family =====
    inc_fam = struct('MarriedTotal',144988,'MarriedOnly',120105,'MarriedKids',168471, ...
                     'OtherMarried',148763,'OneParent',61118,'Single',67075);
    ess_fam = struct('MarriedTotal',92951,'MarriedOnly',80046,'MarriedKids',104727, ...
                     'OtherMarried',97291,'OneParent',58162,'Single',51986);
    fams = fieldnames(inc_fam);

    % ===== Income bracket fit =====
    income_br = [  7637  22443  34984  44824  59582  83888 121852 171847 322142 ]';
    ess_br    = [ 30214  34517  43459  47664  55157  66064  83346 105686 152382 ]';

    X = [ones(size(income_br)) log(income_br)];
    y = log(ess_br);
    b = X \ y;
    a = b(1);
    beta = b(2);
    alpha = exp(a);

    % ===== Residual multipliers =====
    m0 = ess_all / (alpha * (inc_all^beta));

    m_region = struct();
    for i = 1:numel(regions)
        k = regions{i};
        m_region.(k) = ess_region.(k) / (alpha * (inc_region.(k)^beta));
    end

    m_age = struct();
    for i = 1:numel(ages)
        k = ages{i};
        m_age.(k) = ess_age.(k) / (alpha * (inc_age.(k)^beta));
    end

    m_fam = struct();
    for i = 1:numel(fams)
        k = fams{i};
        m_fam.(k) = ess_fam.(k) / (alpha * (inc_fam.(k)^beta));
    end

    % ===== Tax params =====
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

    % ===== Weights =====
    M = struct();
    M.alpha = alpha;
    M.beta = beta;
    M.m0 = m0;
    M.m_region = m_region;
    M.m_age = m_age;
    M.m_fam = m_fam;
    M.taxParams = taxParams;

    M.wR = 0.40; M.wA = 0.25; M.wF = 0.35;
end