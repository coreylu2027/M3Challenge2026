from math import exp, log


# =========================
# MODEL CONSTANTS (stored once)
# =========================

# ===== Overall =====
INC_ALL = 104207.0
ESS_ALL = 72052.0

# ===== Regions =====
INC_REGION = {
    "Northeast": 115770.0,
    "Midwest": 97104.0,
    "South": 93814.0,
    "West": 120365.0,
}
ESS_REGION = {
    "Northeast": 77535.0,
    "Midwest": 67748.0,
    "South": 65065.0,
    "West": 84256.0,
}

# ===== Ages =====
INC_AGE = {
    "Under25": 48514.0,
    "Age25_34": 102494.0,
    "Age35_44": 128285.0,
    "Age45_54": 141121.0,
    "Age55_64": 121571.0,
    "Age65_74": 75460.0,
    "Age75plus": 56028.0,
}
ESS_AGE = {
    "Under25": 44057.0,
    "Age25_34": 68832.0,
    "Age35_44": 84322.0,
    "Age45_54": 91359.0,
    "Age55_64": 78195.0,
    "Age65_74": 59655.0,
    "Age75plus": 50436.0,
}

# ===== Family =====
INC_FAM = {
    "MarriedTotal": 144988.0,
    "MarriedOnly": 120105.0,
    "MarriedKids": 168471.0,
    "OtherMarried": 148763.0,
    "OneParent": 61118.0,
    "Single": 67075.0,
}
ESS_FAM = {
    "MarriedTotal": 92951.0,
    "MarriedOnly": 80046.0,
    "MarriedKids": 104727.0,
    "OtherMarried": 97291.0,
    "OneParent": 58162.0,
    "Single": 51986.0,
}

# ===== Income bracket fit =====
INCOME_BR = [7637, 22443, 34984, 44824, 59582, 83888, 121852, 171847, 322142]
ESS_BR = [30214, 34517, 43459, 47664, 55157, 66064, 83346, 105686, 152382]


def fit_power_model():
    xs = [log(x) for x in INCOME_BR]
    ys = [log(y) for y in ESS_BR]
    n = len(xs)

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    beta = sxy / sxx
    a = mean_y - beta * mean_x
    alpha = exp(a)
    return alpha, beta


ALPHA, BETA = fit_power_model()

# ===== Residual multipliers =====
M0 = ESS_ALL / (ALPHA * (INC_ALL ** BETA))

M_REGION = {
    k: ESS_REGION[k] / (ALPHA * (INC_REGION[k] ** BETA))
    for k in INC_REGION
}
M_AGE = {
    k: ESS_AGE[k] / (ALPHA * (INC_AGE[k] ** BETA))
    for k in INC_AGE
}
M_FAM = {
    k: ESS_FAM[k] / (ALPHA * (INC_FAM[k] ** BETA))
    for k in INC_FAM
}

# ===== Tax parameters =====
FED = {
    "single": {
        "brk": [11000, 44725, 95375, 182100, 231250, 578125, float("inf")],
        "r":   [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
    },
    "mfj": {
        "brk": [22000, 89450, 190750, 364200, 462500, 693750, float("inf")],
        "r":   [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
    },
    "hoh": {
        "brk": [15700, 59850, 95350, 182100, 231250, 578100, float("inf")],
        "r":   [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
    },
}

STD = {"single": 13850, "mfj": 27700, "hoh": 20800}

FICA = {
    "ss_rate": 0.062,
    "med_rate": 0.0145,
    "ss_wage_base": 168600,
    "addl_med_rate": 0.009,
    "addl_med_thresh": {"single": 200000, "mfj": 250000, "hoh": 200000},
}

STATE_EFF = {"Northeast": 0.045, "Midwest": 0.035, "South": 0.025, "West": 0.040}

# Weights
WR, WA, WF = 0.40, 0.25, 0.35


# =========================
# MAIN FUNCTION
# =========================

def predict_disposable(S, region, age, family):
    S = float(S)

    E0 = ALPHA * (S ** BETA)

    mult = exp(
        WR * log(M_REGION[region] / M0)
        + WA * log(M_AGE[age] / M0)
        + WF * log(M_FAM[family] / M0)
    )

    Ehat = E0 * mult

    # Tax calculated but not subtracted (matches MATLAB behavior)
    filing = filing_from_family(family)
    T = estimate_total_tax(S, region, filing)

    return max(-1e7, S - Ehat - T)


def filing_from_family(family):
    if family == "Single":
        return "single"
    if family == "OneParent":
        return "hoh"
    if family in {"MarriedTotal", "MarriedOnly", "MarriedKids", "OtherMarried"}:
        return "mfj"
    raise ValueError(f'Unknown family "{family}".')


def estimate_total_tax(S, region, filing):
    std_ded = STD[filing]
    taxable = max(0, S - std_ded)

    brk = FED[filing]["brk"]
    r = FED[filing]["r"]
    fed = progressive_tax(taxable, brk, r)

    ss = FICA["ss_rate"] * min(S, FICA["ss_wage_base"])
    med = FICA["med_rate"] * S
    thresh = FICA["addl_med_thresh"][filing]
    addl_med = FICA["addl_med_rate"] * max(0, S - thresh)

    fica = ss + med + addl_med
    state = STATE_EFF[region] * S

    return fed + fica + state


def progressive_tax(x, brk, r):
    tax = 0
    prev = 0
    for upper, rate in zip(brk, r):
        amt = max(0, min(x, upper) - prev)
        tax += amt * rate
        prev = upper
        if x <= upper:
            break
    return tax
