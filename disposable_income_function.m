function D = predictDisposable(S, region, age, family)
%PREDICTDISPOSABLE Disposable income model
% Usage:
%   D = predictDisposable(75000,'South','Age45_54','MarriedOnly');

    persistent M
    if isempty(M)
        M = disposableModelInit();   % build alpha,beta,multipliers,taxParams once
    end

    E0   = M.alpha .* (S.^M.beta);
    mult = exp( ...
        M.wR*log(M.m_region.(region)/M.m0) + ...
        M.wA*log(M.m_age.(age)/M.m0) + ...
        M.wF*log(M.m_fam.(family)/M.m0) );

    Ehat = E0 .* mult;

    filing = filingFromFamily(family);
    T = estimateTotalTax(S, region, filing, M.taxParams);

    D = max(-1e7, S - Ehat - T);
end