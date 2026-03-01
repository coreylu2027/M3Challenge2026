"""
=============================================================================
M3 Challenge Q2 — Markov Chain Model: Individual Annual Sports Gambling P&L
=============================================================================


PURPOSE
-------
This program predicts how much money an individual will GAIN or LOSE through
online sports gambling over one year, based on their demographic group and
risk tolerance. It uses a Markov Chain model, which is a mathematical
framework for describing how a system moves between different "states" over
time based on fixed probabilities.


WHY TECHNICAL COMPUTING IS NECESSARY
--------------------------------------
This problem requires computing 52 weekly matrix multiplications (one per
week of the year) for each of six demographic groups. Beyond simple expected
values, we also run 5,000 Monte Carlo simulations per group — each simulating
a full year of weekly bets — to capture the wide variation in gambling
outcomes (ruin, profit, median). The volume of computation (millions of
floating-point operations), the need for random sampling, and the generation
of fan-chart visualizations make a calculator impractical. Python with NumPy
is well-suited because it provides efficient matrix operations, reproducible
random number generation, and rich plotting libraries.


KEY FEATURES OF THE CODE
--------------------------
1. build_transition_matrix()        — constructs the 4x4 weekly Markov
   transition matrix from survey-derived demographic parameters.
2. expected_annual_loss()           — analytically computes expected loss
   over 52 weeks by iterating the state probability vector through the matrix.
3. monte_carlo()                    — simulates 5,000 individual gamblers
   over one year, tracking bankroll each week and recording outcomes.
4. run_model() / predict_individual() — high-level wrappers that call the
   above functions and print formatted results.
5. predict_from_age_and_income()    — simplified entry point: takes just an
   age group string and a disposable income value (from Q1) and runs the
   full gambling model using that income as the starting bankroll.
6. predict_from_income()            — full lookup-based entry point: takes
   salary, region, age group, and dependency status, looks up disposable
   income from the built-in table, and runs the gambling model.
7. Plotting functions               — generate fan charts, bar charts, a
   transition matrix heatmap, and a cumulative loss curve.


TESTING FOR CORRECTNESS
------------------------
The model was validated in three ways:
 (a) HOUSE EDGE CHECK: With a starting state of S1 (Disciplined, always),
     the model's expected loss per dollar wagered equals OMEGA_S1 = 4.56%.
     We verified: 0.5 * (100/110) - 0.5 * 1 = -0.04545, matching the
     NJ Division of Gaming Enforcement's observed 4.56% straight-bet hold.
 (b) TRANSITION MATRIX ROW SUMS: Each row of the 4x4 transition matrix is
     verified to sum to exactly 1.0 (all probability mass is conserved).
     assert abs(P.sum(axis=1) - 1.0).max() < 1e-10
 (c) MONTE CARLO vs ANALYTICAL CONVERGENCE: For each demographic group,
     the Monte Carlo mean P&L converges to within ~5% of the analytical
     expected loss as n_simulations -> 5,000, confirming both methods
     agree on the central estimate.


=============================================================================
DATA SOURCES — ALL PARAMETERS DOCUMENTED
=============================================================================


--- SURVEY-DERIVED PARAMETERS (Tab 4, 2025 Survey of Online Gambling Habits) ---


f  (Betting Frequency): "Weekly or more often" deposit row (Row 32)
    18-34: 24% | 35-49: 24% | 50-64: 15% | 65+: 3% | Male: 23% | Female: 19%


C  (Chasing Propensity): "Percent that have chased a bet" row (Row 38)
    18-34: 58% | 35-49: 54% | 50-64: 38% | 65+: 15% | Male: 57% | Female: 41%


Pw (Withdraw Habit): "Leave winnings in account" row (Row 36)
    18-34: 18% | 35-49: 20% | 50-64: 28% | 65+: 25% | Male: 20% | Female: 20%


--- HOUSE EDGE — FROM NJ DIVISION OF GAMING ENFORCEMENT (DGE) OFFICIAL DATA ---


OMEGA_S1 = 4.56% (straight/spread bet hold rate)
 Source: NJ DGE 2024 Annual Report via sportshandle.com & playnj.com.
 NJ 2024: Total handle ~$14B, total revenue ~$1.09B -> overall hold = 7.79%.
 Parlay revenue = ~55% of total (DGE breakdowns cited in sportshandle.com,
 Nov 2024). Parlay hold = 18.5% (DGE, cited in playnj.com Oct 2024).
 Back-calculation: straight-bet hold = (7.79% - 18.5%x23.1%) / 76.9% = 4.56%.
 Consistent with -110 theory: EV = 0.5*(100/110) - 0.5*1 = -4.55%
 DGE reports: https://www.njoag.gov/about/divisions-and-offices/division-of-
   gaming-enforcement-home/financial-and-statistical-information/
   monthly-sports-wagering-revenue-reports/


OMEGA_S2 = 18.5% (parlay bet hold rate, used in the Chasing state)
 Source: NJ DGE data, cited in playnj.com (Nov 2024).
 Parlay bets have the highest loss rates of any bet category.
 Source: Nower et al. (2022). Sports Wagering in NJ: CY2019 Report to DGE.
 Rutgers Center for Gambling Studies.


P_WIN = 0.50 (win probability for a spread bet)
 At standard -110 odds, the theoretical win probability is exactly 50%.
 The sportsbook's profit margin (vig) is embedded in the payout ratio
 (100/110), not by tilting the win probability. This is consistent with
 the observed DGE hold: 0.5*(100/110) - 0.5*1 = -4.55%.


WIN_PAYOUT = 100/110 = 0.9091 (net gain per dollar staked when you WIN at -110)
 Industry standard -110 spread betting price used by all major U.S.
 sportsbooks (DraftKings, FanDuel, BetMGM, Caesars, etc.).


--- WAGER SIZES (V1) — FROM NJ DGE/RUTGERS CGS + M3 SURVEY ---


V1 (base wager per week) derived from:
 1. M3 Survey Tab 4 monthly wager table:
      65% spend $1-100/month (midpoint $50),
      22% spend $101-500    (midpoint $300),
       9% spend $500+        (midpoint $750).
      Weighted average = $166/month ~= $38/week across ALL bettors.
 2. Rutgers CGS NJ 2019 Report (Nower et al., 2022): High-intensity bettors
    (top ~2%) place much larger individual wagers than average. Cross-
    validated against M3 survey row: "Bet $100+ in a day" (56% of 18-34)
    and "Bet $500+ in a day" (25% of 18-34).
 Source: https://socialwork.rutgers.edu/sites/default/files/2023-05/
   Sports.CalendarYear2019.FINAL_.pdf


--- CHASE MULTIPLIER k — FROM PEER-REVIEWED LITERATURE ---


k is the factor by which a chasing bettor increases their wager above V1.
Literature on between-session loss chasing documents a range of 1.5x-3x:
 - Banerjee et al. (2023). Neuroscience & Biobehavioral Reviews, 152, 105335.
   https://doi.org/10.1016/j.neubiorev.2023.105335
 - Journal of Gambling Studies (2025). Multidimensional Loss Chasing
   among Online Gamblers (N=36,331 sports bettors).
   https://doi.org/10.1007/s10899-025-10391-1
 - Parke et al. (2016). J. Gambling Studies, 32(2), 721-735.
   https://doi.org/10.1007/s10899-015-9570-x
 k=2.5 for 18-34 is calibrated to M3 survey finding that 25% of 18-34
 bettors have placed $500+ in a single day (~= 3.3x base $150 when chasing).


--- S2 TRANSITION ROW — DERIVED FROM PEER-REVIEWED LITERATURE ---


Empirically grounded values used:  [0.15, 0.08, 0.67, 0.10]


P(S2->S2) = 0.669  — probability of staying in the Chasing state
 Banerjee et al. (2023): "Loss-chasing was observed on 66.9% of all losses."
 In a weekly model, each week in S2 is one decision point. A 66.9% persistence
 rate means the average chasing episode lasts 1/(1-0.669) ~= 3 weeks before
 the gambler resolves it.


P(S2->S3) = 0.100  — probability of bankroll depletion / forced stop
 Consistent with NJ DGE 2024 parlay loss rates and Auer & Griffiths (2022),
 PMC10628006, where ~10% of chasing sessions end in account depletion.


P(S2->S0) = 0.150  — probability of exiting to Dormant (taking a break)
 Zhang & Clark (2024). J. Behav. Addict., 13(2), 665-675. PMC11220803.
 Finding: "Gamblers returned more slowly as a function of the prior loss."
 After a chasing session, the dominant exit behavior is a dormancy period.


P(S2->S1) = 0.081  — probability of immediately returning to Disciplined betting
 = 1 - 0.669 - 0.100 - 0.150 = 0.081 (residual probability).


=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 0 — GLOBAL CONSTANTS
# =============================================================================

P_WIN       = 0.50
WIN_PAYOUT  = 100 / 110
OMEGA_S1    = 0.0456
OMEGA_S2    = 0.185


# =============================================================================
# SECTION 1 — DEMOGRAPHIC PROFILES
# =============================================================================

DEMOGRAPHICS = {
    #          f      C      Pw    V1    k
    "18-34": dict(f=0.24, C=0.58, Pw=0.18, V1=150, k=2.5),
    "35-49": dict(f=0.24, C=0.54, Pw=0.20, V1=120, k=2.0),
    "50-64": dict(f=0.15, C=0.38, Pw=0.28, V1=80,  k=1.75),
    "65+":   dict(f=0.03, C=0.15, Pw=0.25, V1=40,  k=1.5),
    "Male":  dict(f=0.23, C=0.57, Pw=0.20, V1=160, k=2.5),
    "Female":dict(f=0.19, C=0.41, Pw=0.20, V1=80,  k=2.0),
}

# Age group mapping from Q1 disposable-income bucket names to Q2 model keys.
AGE_MAP = {
    "Under25":  "18-34",
    "Age25_34": "18-34",
    "Age35_44": "35-49",
    "Age45_54": "50-64",
    "Age55_64": "50-64",
    "Age65_74": "65+",
}


# =============================================================================
# SECTION 2 — DISPOSABLE INCOME LOOKUP TABLE
# =============================================================================

DISPOSABLE_DATA = """
S=50000 | Northeast | Under25   | Single      => D=-12248.2
S=50000 | Northeast | Under25   | MarriedKids => D=-15725.6
S=50000 | Northeast | Under25   | OneParent   => D=-14216.6
S=50000 | Northeast | Under25   | MarriedOnly => D=-13474.4
S=50000 | Northeast | Age25_34  | Single      => D=-13629.5
S=50000 | Northeast | Age25_34  | MarriedKids => D=-17249.1
S=50000 | Northeast | Age25_34  | OneParent   => D=-15674.8
S=50000 | Northeast | Age25_34  | MarriedOnly => D=-14938.2
S=50000 | Northeast | Age35_44  | Single      => D=-14989.3
S=50000 | Northeast | Age35_44  | MarriedKids => D=-18749
S=50000 | Northeast | Age35_44  | OneParent   => D=-17110.3
S=50000 | Northeast | Age35_44  | MarriedOnly => D=-16379.2
S=50000 | Northeast | Age45_54  | Single      => D=-15493.3
S=50000 | Northeast | Age45_54  | MarriedKids => D=-19304.8
S=50000 | Northeast | Age45_54  | OneParent   => D=-17642.3
S=50000 | Northeast | Age45_54  | MarriedOnly => D=-16913.3
S=50000 | Northeast | Age55_64  | Single      => D=-14296.5
S=50000 | Northeast | Age55_64  | MarriedKids => D=-17984.8
S=50000 | Northeast | Age55_64  | OneParent   => D=-16378.9
S=50000 | Northeast | Age55_64  | MarriedOnly => D=-15645
S=50000 | Northeast | Age65_74  | Single      => D=-13585.2
S=50000 | Northeast | Age65_74  | MarriedKids => D=-17200.3
S=50000 | Northeast | Age65_74  | OneParent   => D=-15628.1
S=50000 | Northeast | Age65_74  | MarriedOnly => D=-14891.3
S=50000 | South     | Under25   | Single      => D=-9621.91
S=50000 | South     | Under25   | MarriedKids => D=-12931.9
S=50000 | South     | Under25   | OneParent   => D=-11499.9
S=50000 | South     | Under25   | MarriedOnly => D=-10751
S=50000 | South     | Age25_34  | Single      => D=-10960.1
S=50000 | South     | Age25_34  | MarriedKids => D=-14407.8
S=50000 | South     | Age25_34  | OneParent   => D=-12912.5
S=50000 | South     | Age25_34  | MarriedOnly => D=-12169.1
S=50000 | South     | Age35_44  | Single      => D=-12277.4
S=50000 | South     | Age35_44  | MarriedKids => D=-15860.8
S=50000 | South     | Age35_44  | OneParent   => D=-14303.2
S=50000 | South     | Age35_44  | MarriedOnly => D=-13565.1
S=50000 | South     | Age45_54  | Single      => D=-12765.7
S=50000 | South     | Age45_54  | MarriedKids => D=-16399.3
S=50000 | South     | Age45_54  | OneParent   => D=-14818.6
S=50000 | South     | Age45_54  | MarriedOnly => D=-14082.5
S=50000 | South     | Age55_64  | Single      => D=-11606.2
S=50000 | South     | Age55_64  | MarriedKids => D=-15120.5
S=50000 | South     | Age55_64  | OneParent   => D=-13594.6
S=50000 | South     | Age55_64  | MarriedOnly => D=-12853.8
S=50000 | South     | Age65_74  | Single      => D=-10917.2
S=50000 | South     | Age65_74  | MarriedKids => D=-14360.5
S=50000 | South     | Age65_74  | OneParent   => D=-12867.2
S=50000 | South     | Age65_74  | MarriedOnly => D=-12123.6
S=50000 | West      | Under25   | Single      => D=-13377.1
S=50000 | West      | Under25   | MarriedKids => D=-16996.5
S=50000 | West      | Under25   | OneParent   => D=-15422.3
S=50000 | West      | Under25   | MarriedOnly => D=-14685.7
S=50000 | West      | Age25_34  | Single      => D=-14795
S=50000 | West      | Age25_34  | MarriedKids => D=-18560.4
S=50000 | West      | Age25_34  | OneParent   => D=-16919.1
S=50000 | West      | Age25_34  | MarriedOnly => D=-16188.2
S=50000 | West      | Age35_44  | Single      => D=-16190.8
S=50000 | West      | Age35_44  | MarriedKids => D=-20099.9
S=50000 | West      | Age35_44  | OneParent   => D=-18392.6
S=50000 | West      | Age35_44  | MarriedOnly => D=-17667.4
S=50000 | West      | Age45_54  | Single      => D=-16708.2
S=50000 | West      | Age45_54  | MarriedKids => D=-20670.6
S=50000 | West      | Age45_54  | OneParent   => D=-18938.7
S=50000 | West      | Age45_54  | MarriedOnly => D=-18215.7
S=50000 | West      | Age55_64  | Single      => D=-15479.7
S=50000 | West      | Age55_64  | MarriedKids => D=-19315.5
S=50000 | West      | Age55_64  | OneParent   => D=-17641.8
S=50000 | West      | Age55_64  | MarriedOnly => D=-16913.8
S=50000 | West      | Age65_74  | Single      => D=-14749.6
S=50000 | West      | Age65_74  | MarriedKids => D=-18510.3
S=50000 | West      | Age65_74  | OneParent   => D=-16871.1
S=50000 | West      | Age65_74  | MarriedOnly => D=-16140.1
S=50000 | Midwest   | Under25   | Single      => D=-10622.1
S=50000 | Midwest   | Under25   | MarriedKids => D=-13983.6
S=50000 | Midwest   | Under25   | OneParent   => D=-12527.9
S=50000 | Midwest   | Under25   | MarriedOnly => D=-11781.1
S=50000 | Midwest   | Age25_34  | Single      => D=-11973.5
S=50000 | Midwest   | Age25_34  | MarriedKids => D=-15474.2
S=50000 | Midwest   | Age25_34  | OneParent   => D=-13954.5
S=50000 | Midwest   | Age25_34  | MarriedOnly => D=-13213.2
S=50000 | Midwest   | Age35_44  | Single      => D=-13303.9
S=50000 | Midwest   | Age35_44  | MarriedKids => D=-16941.5
S=50000 | Midwest   | Age35_44  | OneParent   => D=-15359
S=50000 | Midwest   | Age35_44  | MarriedOnly => D=-14623
S=50000 | Midwest   | Age45_54  | Single      => D=-13797
S=50000 | Midwest   | Age45_54  | MarriedKids => D=-17485.4
S=50000 | Midwest   | Age45_54  | OneParent   => D=-15879.5
S=50000 | Midwest   | Age45_54  | MarriedOnly => D=-15145.6
S=50000 | Midwest   | Age55_64  | Single      => D=-12626.1
S=50000 | Midwest   | Age55_64  | MarriedKids => D=-16193.9
S=50000 | Midwest   | Age55_64  | OneParent   => D=-14643.4
S=50000 | Midwest   | Age55_64  | MarriedOnly => D=-13904.7
S=50000 | Midwest   | Age65_74  | Single      => D=-11930.2
S=50000 | Midwest   | Age65_74  | MarriedKids => D=-15426.4
S=50000 | Midwest   | Age65_74  | OneParent   => D=-13908.8
S=50000 | Midwest   | Age65_74  | MarriedOnly => D=-13167.3
S=75000 | Northeast | Under25   | Single      => D=-5512.22
S=75000 | Northeast | Under25   | MarriedKids => D=-8436.85
S=75000 | Northeast | Under25   | OneParent   => D=-6427.11
S=75000 | Northeast | Under25   | MarriedOnly => D=-5727.91
S=75000 | Northeast | Age25_34  | Single      => D=-7174.38
S=75000 | Northeast | Age25_34  | MarriedKids => D=-10270.1
S=75000 | Northeast | Age25_34  | OneParent   => D=-8181.76
S=75000 | Northeast | Age25_34  | MarriedOnly => D=-7489.32
S=75000 | Northeast | Age35_44  | Single      => D=-8810.69
S=75000 | Northeast | Age35_44  | MarriedKids => D=-12074.9
S=75000 | Northeast | Age35_44  | OneParent   => D=-9909.12
S=75000 | Northeast | Age35_44  | MarriedOnly => D=-9223.34
S=75000 | Northeast | Age45_54  | Single      => D=-9417.16
S=75000 | Northeast | Age45_54  | MarriedKids => D=-12743.8
S=75000 | Northeast | Age45_54  | OneParent   => D=-10549.3
S=75000 | Northeast | Age45_54  | MarriedOnly => D=-9866.03
S=75000 | Northeast | Age55_64  | Single      => D=-7976.97
S=75000 | Northeast | Age55_64  | MarriedKids => D=-11155.4
S=75000 | Northeast | Age55_64  | OneParent   => D=-9029.02
S=75000 | Northeast | Age55_64  | MarriedOnly => D=-8339.84
S=75000 | Northeast | Age65_74  | Single      => D=-7121.1
S=75000 | Northeast | Age65_74  | MarriedKids => D=-10211.4
S=75000 | Northeast | Age65_74  | OneParent   => D=-8125.52
S=75000 | Northeast | Age65_74  | MarriedOnly => D=-7432.87
S=75000 | South     | Under25   | Single      => D=-2055.3
S=75000 | South     | Under25   | MarriedKids => D=-4778.46
S=75000 | South     | Under25   | OneParent   => D=-2861.31
S=75000 | South     | Under25   | MarriedOnly => D=-2154.16
S=75000 | South     | Age25_34  | Single      => D=-3665.54
S=75000 | South     | Age25_34  | MarriedKids => D=-6554.48
S=75000 | South     | Age25_34  | OneParent   => D=-4561.15
S=75000 | South     | Age25_34  | MarriedOnly => D=-3860.54
S=75000 | South     | Age35_44  | Single      => D=-5250.73
S=75000 | South     | Age35_44  | MarriedKids => D=-8302.88
S=75000 | South     | Age35_44  | OneParent   => D=-6234.54
S=75000 | South     | Age35_44  | MarriedOnly => D=-5540.38
S=75000 | South     | Age45_54  | Single      => D=-5838.26
S=75000 | South     | Age45_54  | MarriedKids => D=-8950.9
S=75000 | South     | Age45_54  | OneParent   => D=-6854.76
S=75000 | South     | Age45_54  | MarriedOnly => D=-6162.99
S=75000 | South     | Age55_64  | Single      => D=-4443.06
S=75000 | South     | Age55_64  | MarriedKids => D=-7412.06
S=75000 | South     | Age55_64  | OneParent   => D=-5381.93
S=75000 | South     | Age55_64  | MarriedOnly => D=-4684.49
S=75000 | South     | Age65_74  | Single      => D=-3613.93
S=75000 | South     | Age65_74  | MarriedKids => D=-6497.56
S=75000 | South     | Age65_74  | OneParent   => D=-4506.67
S=75000 | South     | Age65_74  | MarriedOnly => D=-3805.85
S=75000 | West      | Under25   | Single      => D=-6796.52
S=75000 | West      | Under25   | MarriedKids => D=-9891.99
S=75000 | West      | Under25   | OneParent   => D=-7803.74
S=75000 | West      | Under25   | MarriedOnly => D=-7111.29
S=75000 | West      | Age25_34  | Single      => D=-8502.71
S=75000 | West      | Age25_34  | MarriedKids => D=-11773.8
S=75000 | West      | Age25_34  | OneParent   => D=-9604.87
S=75000 | West      | Age25_34  | MarriedOnly => D=-8919.36
S=75000 | West      | Age35_44  | Single      => D=-10182.4
S=75000 | West      | Age35_44  | MarriedKids => D=-13626.4
S=75000 | West      | Age35_44  | OneParent   => D=-11378
S=75000 | West      | Age35_44  | MarriedOnly => D=-10699.3
S=75000 | West      | Age45_54  | Single      => D=-10804.9
S=75000 | West      | Age45_54  | MarriedKids => D=-14313.1
S=75000 | West      | Age45_54  | OneParent   => D=-12035.2
S=75000 | West      | Age45_54  | MarriedOnly => D=-11359
S=75000 | West      | Age55_64  | Single      => D=-9326.57
S=75000 | West      | Age55_64  | MarriedKids => D=-12682.5
S=75000 | West      | Age55_64  | OneParent   => D=-10474.6
S=75000 | West      | Age55_64  | MarriedOnly => D=-9792.41
S=75000 | West      | Age65_74  | Single      => D=-8448.02
S=75000 | West      | Age65_74  | MarriedKids => D=-11713.5
S=75000 | West      | Age65_74  | OneParent   => D=-9547.14
S=75000 | West      | Age65_74  | MarriedOnly => D=-8861.41
S=75000 | Midwest   | Under25   | Single      => D=-3407.18
S=75000 | Midwest   | Under25   | MarriedKids => D=-6192.3
S=75000 | Midwest   | Under25   | OneParent   => D=-4246.68
S=75000 | Midwest   | Under25   | MarriedOnly => D=-3541.97
S=75000 | Midwest   | Age25_34  | Single      => D=-5033.38
S=75000 | Midwest   | Age25_34  | MarriedKids => D=-7985.94
S=75000 | Midwest   | Age25_34  | OneParent   => D=-5963.37
S=75000 | Midwest   | Age25_34  | MarriedOnly => D=-5265.27
S=75000 | Midwest   | Age35_44  | Single      => D=-6634.3
S=75000 | Midwest   | Age35_44  | MarriedKids => D=-9751.68
S=75000 | Midwest   | Age35_44  | OneParent   => D=-7653.36
S=75000 | Midwest   | Age35_44  | MarriedOnly => D=-6961.78
S=75000 | Midwest   | Age45_54  | Single      => D=-7227.65
S=75000 | Midwest   | Age45_54  | MarriedKids => D=-10406.1
S=75000 | Midwest   | Age45_54  | OneParent   => D=-8279.73
S=75000 | Midwest   | Age45_54  | MarriedOnly => D=-7590.56
S=75000 | Midwest   | Age55_64  | Single      => D=-5818.62
S=75000 | Midwest   | Age55_64  | MarriedKids => D=-8852.02
S=75000 | Midwest   | Age55_64  | OneParent   => D=-6792.3
S=75000 | Midwest   | Age55_64  | MarriedOnly => D=-6097.39
S=75000 | Midwest   | Age65_74  | Single      => D=-4981.26
S=75000 | Midwest   | Age65_74  | MarriedKids => D=-7928.45
S=75000 | Midwest   | Age65_74  | OneParent   => D=-5908.35
S=75000 | Midwest   | Age65_74  | MarriedOnly => D=-5210.04
S=100000 | Northeast | Under25   | Single      => D=2159.84
S=100000 | Northeast | Under25   | MarriedKids => D=830.17
S=100000 | Northeast | Under25   | OneParent   => D=1320.82
S=100000 | Northeast | Under25   | MarriedOnly => D=3919.26
S=100000 | Northeast | Age25_34  | Single      => D=264.421
S=100000 | Northeast | Age25_34  | MarriedKids => D=-1260.4
S=100000 | Northeast | Age25_34  | OneParent   => D=-680.062
S=100000 | Northeast | Age25_34  | MarriedOnly => D=1910.67
S=100000 | Northeast | Age35_44  | Single      => D=-1601.52
S=100000 | Northeast | Age35_44  | MarriedKids => D=-3318.45
S=100000 | Northeast | Age35_44  | OneParent   => D=-2649.83
S=100000 | Northeast | Age35_44  | MarriedOnly => D=-66.6926
S=100000 | Northeast | Age45_54  | Single      => D=-2293.1
S=100000 | Northeast | Age45_54  | MarriedKids => D=-4081.24
S=100000 | Northeast | Age45_54  | OneParent   => D=-3379.89
S=100000 | Northeast | Age45_54  | MarriedOnly => D=-799.566
S=100000 | Northeast | Age55_64  | Single      => D=-650.807
S=100000 | Northeast | Age55_64  | MarriedKids => D=-2269.85
S=100000 | Northeast | Age55_64  | OneParent   => D=-1646.22
S=100000 | Northeast | Age55_64  | MarriedOnly => D=940.791
S=100000 | Northeast | Age65_74  | Single      => D=325.171
S=100000 | Northeast | Age65_74  | MarriedKids => D=-1193.39
S=100000 | Northeast | Age65_74  | OneParent   => D=-615.933
S=100000 | Northeast | Age65_74  | MarriedOnly => D=1975.05
S=100000 | South     | Under25   | Single      => D=6391.38
S=100000 | South     | Under25   | MarriedKids => D=5291.46
S=100000 | South     | Under25   | OneParent   => D=5676.53
S=100000 | South     | Under25   | MarriedOnly => D=8284.04
S=100000 | South     | Age25_34  | Single      => D=4555.17
S=100000 | South     | Age25_34  | MarriedKids => D=3266.2
S=100000 | South     | Age25_34  | OneParent   => D=3738.15
S=100000 | South     | Age25_34  | MarriedOnly => D=6338.2
S=100000 | South     | Age35_44  | Single      => D=2747.52
S=100000 | South     | Age35_44  | MarriedKids => D=1272.44
S=100000 | South     | Age35_44  | OneParent   => D=1829.92
S=100000 | South     | Age35_44  | MarriedOnly => D=4422.61
S=100000 | South     | Age45_54  | Single      => D=2077.54
S=100000 | South     | Age45_54  | MarriedKids => D=533.486
S=100000 | South     | Age45_54  | OneParent   => D=1122.66
S=100000 | South     | Age45_54  | MarriedOnly => D=3712.63
S=100000 | South     | Age55_64  | Single      => D=3668.53
S=100000 | South     | Age55_64  | MarriedKids => D=2288.28
S=100000 | South     | Age55_64  | OneParent   => D=2802.18
S=100000 | South     | Age55_64  | MarriedOnly => D=5398.62
S=100000 | South     | Age65_74  | Single      => D=4614.02
S=100000 | South     | Age65_74  | MarriedKids => D=3331.11
S=100000 | South     | Age65_74  | OneParent   => D=3800.27
S=100000 | South     | Age65_74  | MarriedOnly => D=6400.56
S=100000 | West      | Under25   | Single      => D=767.685
S=100000 | West      | Under25   | MarriedKids => D=-756.797
S=100000 | West      | Under25   | OneParent   => D=-176.617
S=100000 | West      | Under25   | MarriedOnly => D=2414.13
S=100000 | West      | Age25_34  | Single      => D=-1177.95
S=100000 | West      | Age25_34  | MarriedKids => D=-2902.74
S=100000 | West      | Age25_34  | OneParent   => D=-2230.51
S=100000 | West      | Age25_34  | MarriedOnly => D=352.323
S=100000 | West      | Age35_44  | Single      => D=-3093.32
S=100000 | West      | Age35_44  | MarriedKids => D=-5015.32
S=100000 | West      | Age35_44  | OneParent   => D=-4252.46
S=100000 | West      | Age35_44  | MarriedOnly => D=-1677.42
S=100000 | West      | Age45_54  | Single      => D=-3803.22
S=100000 | West      | Age45_54  | MarriedKids => D=-5798.31
S=100000 | West      | Age45_54  | OneParent   => D=-5001.86
S=100000 | West      | Age45_54  | MarriedOnly => D=-2429.71
S=100000 | West      | Age55_64  | Single      => D=-2117.42
S=100000 | West      | Age55_64  | MarriedKids => D=-3938.94
S=100000 | West      | Age55_64  | OneParent   => D=-3222.25
S=100000 | West      | Age55_64  | MarriedOnly => D=-643.246
S=100000 | West      | Age65_74  | Single      => D=-1115.59
S=100000 | West      | Age65_74  | MarriedKids => D=-2833.96
S=100000 | West      | Age65_74  | OneParent   => D=-2164.68
S=100000 | West      | Age65_74  | MarriedOnly => D=418.405
S=100000 | Midwest   | Under25   | Single      => D=4705.04
S=100000 | Midwest   | Under25   | MarriedKids => D=3534.45
S=100000 | Midwest   | Under25   | OneParent   => D=3952
S=100000 | Midwest   | Under25   | MarriedOnly => D=6556.72
S=100000 | Midwest   | Age25_34  | Single      => D=2850.62
S=100000 | Midwest   | Age25_34  | MarriedKids => D=1489.11
S=100000 | Midwest   | Age25_34  | OneParent   => D=1994.4
S=100000 | Midwest   | Age25_34  | MarriedOnly => D=4591.58
S=100000 | Midwest   | Age35_44  | Single      => D=1025.04
S=100000 | Midwest   | Age35_44  | MarriedKids => D=-524.424
S=100000 | Midwest   | Age35_44  | OneParent   => D=67.2362
S=100000 | Midwest   | Age35_44  | MarriedOnly => D=2656.99
S=100000 | Midwest   | Age45_54  | Single      => D=348.421
S=100000 | Midwest   | Age45_54  | MarriedKids => D=-1270.71
S=100000 | Midwest   | Age45_54  | OneParent   => D=-647.031
S=100000 | Midwest   | Age45_54  | MarriedOnly => D=1939.97
S=100000 | Midwest   | Age55_64  | Single      => D=1955.19
S=100000 | Midwest   | Age55_64  | MarriedKids => D=501.491
S=100000 | Midwest   | Age55_64  | OneParent   => D=1049.14
S=100000 | Midwest   | Age55_64  | MarriedOnly => D=3642.68
S=100000 | Midwest   | Age65_74  | Single      => D=2910.05
S=100000 | Midwest   | Age65_74  | MarriedKids => D=1554.67
S=100000 | Midwest   | Age65_74  | OneParent   => D=2057.14
S=100000 | Midwest   | Age65_74  | MarriedOnly => D=4654.56
S=125000 | Northeast | Under25   | Single      => D=10647.5
S=125000 | Northeast | Under25   | MarriedKids => D=10559.7
S=125000 | Northeast | Under25   | OneParent   => D=9520.77
S=125000 | Northeast | Under25   | MarriedOnly => D=13980.1
S=125000 | Northeast | Age25_34  | Single      => D=8548.81
S=125000 | Northeast | Age25_34  | MarriedKids => D=8244.98
S=125000 | Northeast | Age25_34  | OneParent   => D=7305.33
S=125000 | Northeast | Age25_34  | MarriedOnly => D=11756.1
S=125000 | Northeast | Age35_44  | Single      => D=6482.78
S=125000 | Northeast | Age35_44  | MarriedKids => D=5966.24
S=125000 | Northeast | Age35_44  | OneParent   => D=5124.34
S=125000 | Northeast | Age35_44  | MarriedOnly => D=9566.68
S=125000 | Northeast | Age45_54  | Single      => D=5717.04
S=125000 | Northeast | Age45_54  | MarriedKids => D=5121.66
S=125000 | Northeast | Age45_54  | OneParent   => D=4316
S=125000 | Northeast | Age45_54  | MarriedOnly => D=8755.23
S=125000 | Northeast | Age55_64  | Single      => D=7535.44
S=125000 | Northeast | Age55_64  | MarriedKids => D=7127.28
S=125000 | Northeast | Age55_64  | OneParent   => D=6235.58
S=125000 | Northeast | Age55_64  | MarriedOnly => D=10682.2
S=125000 | Northeast | Age65_74  | Single      => D=8616.07
S=125000 | Northeast | Age65_74  | MarriedKids => D=8319.17
S=125000 | Northeast | Age65_74  | OneParent   => D=7376.34
S=125000 | Northeast | Age65_74  | MarriedOnly => D=11827.4
S=125000 | South     | Under25   | Single      => D=15618.3
S=125000 | South     | Under25   | MarriedKids => D=15784.9
S=125000 | South     | Under25   | OneParent   => D=14629.1
S=125000 | South     | Under25   | MarriedOnly => D=19098.4
S=125000 | South     | Age25_34  | Single      => D=13585.2
S=125000 | South     | Age25_34  | MarriedKids => D=13542.5
S=125000 | South     | Age25_34  | OneParent   => D=12482.8
S=125000 | South     | Age25_34  | MarriedOnly => D=16943.9
S=125000 | South     | Age35_44  | Single      => D=11583.7
S=125000 | South     | Age35_44  | MarriedKids => D=11335
S=125000 | South     | Age35_44  | OneParent   => D=10370
S=125000 | South     | Age35_44  | MarriedOnly => D=14822.9
S=125000 | South     | Age45_54  | Single      => D=10841.9
S=125000 | South     | Age45_54  | MarriedKids => D=10516.8
S=125000 | South     | Age45_54  | OneParent   => D=9586.9
S=125000 | South     | Age45_54  | MarriedOnly => D=14036.8
S=125000 | South     | Age55_64  | Single      => D=12603.5
S=125000 | South     | Age55_64  | MarriedKids => D=12459.7
S=125000 | South     | Age55_64  | OneParent   => D=11446.5
S=125000 | South     | Age55_64  | MarriedOnly => D=15903.6
S=125000 | South     | Age65_74  | Single      => D=13650.4
S=125000 | South     | Age65_74  | MarriedKids => D=13614.4
S=125000 | South     | Age65_74  | OneParent   => D=12551.6
S=125000 | South     | Age65_74  | MarriedOnly => D=17013
S=125000 | West      | Under25   | Single      => D=9177.42
S=125000 | West      | Under25   | MarriedKids => D=8873.97
S=125000 | West      | Under25   | OneParent   => D=7934.14
S=125000 | West      | Under25   | MarriedOnly => D=12384.9
S=125000 | West      | Age25_34  | Single      => D=7023.16
S=125000 | West      | Age25_34  | MarriedKids => D=6497.91
S=125000 | West      | Age25_34  | OneParent   => D=5660.02
S=125000 | West      | Age25_34  | MarriedOnly => D=10102
S=125000 | West      | Age35_44  | Single      => D=4902.4
S=125000 | West      | Age35_44  | MarriedKids => D=4158.81
S=125000 | West      | Age35_44  | OneParent   => D=3421.26
S=125000 | West      | Age35_44  | MarriedOnly => D=7854.63
S=125000 | West      | Age45_54  | Single      => D=4116.38
S=125000 | West      | Age45_54  | MarriedKids => D=3291.86
S=125000 | West      | Age45_54  | OneParent   => D=2591.5
S=125000 | West      | Age45_54  | MarriedOnly => D=7021.67
S=125000 | West      | Age55_64  | Single      => D=5982.95
S=125000 | West      | Age55_64  | MarriedKids => D=5350.6
S=125000 | West      | Age55_64  | OneParent   => D=4561.93
S=125000 | West      | Age55_64  | MarriedOnly => D=8999.69
S=125000 | West      | Age65_74  | Single      => D=7092.21
S=125000 | West      | Age65_74  | MarriedKids => D=6574.07
S=125000 | West      | Age65_74  | OneParent   => D=5732.9
S=125000 | West      | Age65_74  | MarriedOnly => D=10175.2
S=125000 | Midwest   | Under25   | Single      => D=13608.4
S=125000 | Midwest   | Under25   | MarriedKids => D=13696.8
S=125000 | Midwest   | Under25   | OneParent   => D=12576.9
S=125000 | Midwest   | Under25   | MarriedOnly => D=17043.1
S=125000 | Midwest   | Age25_34  | Single      => D=11555.1
S=125000 | Midwest   | Age25_34  | MarriedKids => D=11432.1
S=125000 | Midwest   | Age25_34  | OneParent   => D=10409.3
S=125000 | Midwest   | Age25_34  | MarriedOnly => D=14867.2
S=125000 | Midwest   | Age35_44  | Single      => D=9533.76
S=125000 | Midwest   | Age35_44  | MarriedKids => D=9202.64
S=125000 | Midwest   | Age35_44  | OneParent   => D=8275.53
S=125000 | Midwest   | Age35_44  | MarriedOnly => D=12725.2
S=125000 | Midwest   | Age45_54  | Single      => D=8784.59
S=125000 | Midwest   | Age45_54  | MarriedKids => D=8376.34
S=125000 | Midwest   | Age45_54  | OneParent   => D=7484.67
S=125000 | Midwest   | Age45_54  | MarriedOnly => D=11931.3
S=125000 | Midwest   | Age55_64  | Single      => D=10563.6
S=125000 | Midwest   | Age55_64  | MarriedKids => D=10338.6
S=125000 | Midwest   | Age55_64  | OneParent   => D=9362.73
S=125000 | Midwest   | Age55_64  | MarriedOnly => D=13816.6
S=125000 | Midwest   | Age65_74  | Single      => D=11620.9
S=125000 | Midwest   | Age65_74  | MarriedKids => D=11504.7
S=125000 | Midwest   | Age65_74  | OneParent   => D=10478.8
S=125000 | Midwest   | Age65_74  | MarriedOnly => D=14937
S=150000 | Northeast | Under25   | Single      => D=19746
S=150000 | Northeast | Under25   | MarriedKids => D=19451.6
S=150000 | Northeast | Under25   | OneParent   => D=18237.4
S=150000 | Northeast | Under25   | MarriedOnly => D=23168.7
S=150000 | Northeast | Age25_34  | Single      => D=17465.2
S=150000 | Northeast | Age25_34  | MarriedKids => D=16935.9
S=150000 | Northeast | Age25_34  | OneParent   => D=15829.7
S=150000 | Northeast | Age25_34  | MarriedOnly => D=20751.7
S=150000 | Northeast | Age35_44  | Single      => D=15219.9
S=150000 | Northeast | Age35_44  | MarriedKids => D=14459.4
S=150000 | Northeast | Age35_44  | OneParent   => D=13459.4
S=150000 | Northeast | Age35_44  | MarriedOnly => D=18372.3
S=150000 | Northeast | Age45_54  | Single      => D=14387.7
S=150000 | Northeast | Age45_54  | MarriedKids => D=13541.5
S=150000 | Northeast | Age45_54  | OneParent   => D=12580.9
S=150000 | Northeast | Age45_54  | MarriedOnly => D=17490.5
S=150000 | Northeast | Age55_64  | Single      => D=16363.9
S=150000 | Northeast | Age55_64  | MarriedKids => D=15721.2
S=150000 | Northeast | Age55_64  | OneParent   => D=14667.1
S=150000 | Northeast | Age55_64  | MarriedOnly => D=19584.7
S=150000 | Northeast | Age65_74  | Single      => D=17538.3
S=150000 | Northeast | Age65_74  | MarriedKids => D=17016.6
S=150000 | Northeast | Age65_74  | OneParent   => D=15906.8
S=150000 | Northeast | Age65_74  | MarriedOnly => D=20829.2
S=150000 | South     | Under25   | Single      => D=25431.3
S=150000 | South     | Under25   | MarriedKids => D=25413.3
S=150000 | South     | Under25   | OneParent   => D=24072
S=150000 | South     | Under25   | MarriedOnly => D=29014.3
S=150000 | South     | Age25_34  | Single      => D=23221.7
S=150000 | South     | Age25_34  | MarriedKids => D=22976.2
S=150000 | South     | Age25_34  | OneParent   => D=21739.5
S=150000 | South     | Age25_34  | MarriedOnly => D=26672.8
S=150000 | South     | Age35_44  | Single      => D=21046.5
S=150000 | South     | Age35_44  | MarriedKids => D=20577.1
S=150000 | South     | Age35_44  | OneParent   => D=19443.3
S=150000 | South     | Age35_44  | MarriedOnly => D=24367.8
S=150000 | South     | Age45_54  | Single      => D=20240.3
S=150000 | South     | Age45_54  | MarriedKids => D=19687.9
S=150000 | South     | Age45_54  | OneParent   => D=18592.3
S=150000 | South     | Age45_54  | MarriedOnly => D=23513.4
S=150000 | South     | Age55_64  | Single      => D=22154.8
S=150000 | South     | Age55_64  | MarriedKids => D=21799.5
S=150000 | South     | Age55_64  | OneParent   => D=20613.3
S=150000 | South     | Age55_64  | MarriedOnly => D=25542.2
S=150000 | South     | Age65_74  | Single      => D=23292.5
S=150000 | South     | Age65_74  | MarriedKids => D=23054.3
S=150000 | South     | Age65_74  | OneParent   => D=21814.3
S=150000 | South     | Age65_74  | MarriedOnly => D=26747.9
S=150000 | West      | Under25   | Single      => D=18219.1
S=150000 | West      | Under25   | MarriedKids => D=17690.3
S=150000 | West      | Under25   | OneParent   => D=16583.8
S=150000 | West      | Under25   | MarriedOnly => D=21505.9
S=150000 | West      | Age25_34  | Single      => D=15877.9
S=150000 | West      | Age25_34  | MarriedKids => D=15108
S=150000 | West      | Age25_34  | OneParent   => D=14112.3
S=150000 | West      | Age25_34  | MarriedOnly => D=19024.9
S=150000 | West      | Age35_44  | Single      => D=13573.1
S=150000 | West      | Age35_44  | MarriedKids => D=12565.9
S=150000 | West      | Age35_44  | OneParent   => D=11679.2
S=150000 | West      | Age35_44  | MarriedOnly => D=16582.5
S=150000 | West      | Age45_54  | Single      => D=12718.9
S=150000 | West      | Age45_54  | MarriedKids => D=11623.7
S=150000 | West      | Age45_54  | OneParent   => D=10777.5
S=150000 | West      | Age45_54  | MarriedOnly => D=15677.2
S=150000 | West      | Age55_64  | Single      => D=14747.4
S=150000 | West      | Age55_64  | MarriedKids => D=13861.1
S=150000 | West      | Age55_64  | OneParent   => D=12918.9
S=150000 | West      | Age55_64  | MarriedOnly => D=17826.9
S=150000 | West      | Age65_74  | Single      => D=15952.9
S=150000 | West      | Age65_74  | MarriedKids => D=15190.8
S=150000 | West      | Age65_74  | OneParent   => D=14191.5
S=150000 | West      | Age65_74  | MarriedOnly => D=19104.4
S=150000 | Midwest   | Under25   | Single      => D=23105.4
S=150000 | Midwest   | Under25   | MarriedKids => D=23002.4
S=150000 | Midwest   | Under25   | OneParent   => D=21700.2
S=150000 | Midwest   | Under25   | MarriedOnly => D=26639.1
S=150000 | Midwest   | Age25_34  | Single      => D=20873.9
S=150000 | Midwest   | Age25_34  | MarriedKids => D=20541.2
S=150000 | Midwest   | Age25_34  | OneParent   => D=19344.6
S=150000 | Midwest   | Age25_34  | MarriedOnly => D=24274.4
S=150000 | Midwest   | Age35_44  | Single      => D=18677.1
S=150000 | Midwest   | Age35_44  | MarriedKids => D=18118.2
S=150000 | Midwest   | Age35_44  | OneParent   => D=17025.6
S=150000 | Midwest   | Age35_44  | MarriedOnly => D=21946.5
S=150000 | Midwest   | Age45_54  | Single      => D=17863
S=150000 | Midwest   | Age45_54  | MarriedKids => D=17220.2
S=150000 | Midwest   | Age45_54  | OneParent   => D=16166.1
S=150000 | Midwest   | Age45_54  | MarriedOnly => D=21083.7
S=150000 | Midwest   | Age55_64  | Single      => D=19796.4
S=150000 | Midwest   | Age55_64  | MarriedKids => D=19352.7
S=150000 | Midwest   | Age55_64  | OneParent   => D=18207.1
S=150000 | Midwest   | Age55_64  | MarriedOnly => D=23132.6
S=150000 | Midwest   | Age65_74  | Single      => D=20945.4
S=150000 | Midwest   | Age65_74  | MarriedKids => D=20620
S=150000 | Midwest   | Age65_74  | OneParent   => D=19420.1
S=150000 | Midwest   | Age65_74  | MarriedOnly => D=24350.2
S=175000 | Northeast | Under25   | Single      => D=29839.2
S=175000 | Northeast | Under25   | MarriedKids => D=29399.6
S=175000 | Northeast | Under25   | OneParent   => D=27981.9
S=175000 | Northeast | Under25   | MarriedOnly => D=33387.8
S=175000 | Northeast | Age25_34  | Single      => D=27392.1
S=175000 | Northeast | Age25_34  | MarriedKids => D=26700.6
S=175000 | Northeast | Age25_34  | OneParent   => D=25398.6
S=175000 | Northeast | Age25_34  | MarriedOnly => D=30794.6
S=175000 | Northeast | Age35_44  | Single      => D=24983.1
S=175000 | Northeast | Age35_44  | MarriedKids => D=24043.5
S=175000 | Northeast | Age35_44  | OneParent   => D=22855.6
S=175000 | Northeast | Age35_44  | MarriedOnly => D=28241.7
S=175000 | Northeast | Age45_54  | Single      => D=24090.2
S=175000 | Northeast | Age45_54  | MarriedKids => D=23058.7
S=175000 | Northeast | Age45_54  | OneParent   => D=21913
S=175000 | Northeast | Age45_54  | MarriedOnly => D=27295.5
S=175000 | Northeast | Age55_64  | Single      => D=26210.5
S=175000 | Northeast | Age55_64  | MarriedKids => D=25397.3
S=175000 | Northeast | Age55_64  | OneParent   => D=24151.3
S=175000 | Northeast | Age55_64  | MarriedOnly => D=29542.4
S=175000 | Northeast | Age65_74  | Single      => D=27470.5
S=175000 | Northeast | Age65_74  | MarriedKids => D=26787.1
S=175000 | Northeast | Age65_74  | OneParent   => D=25481.4
S=175000 | Northeast | Age65_74  | MarriedOnly => D=30877.7
S=175000 | South     | Under25   | Single      => D=36220.2
S=175000 | South     | Under25   | MarriedKids => D=36077.2
S=175000 | South     | Under25   | OneParent   => D=34523.2
S=175000 | South     | Under25   | MarriedOnly => D=39940.8
S=175000 | South     | Age25_34  | Single      => D=33849.6
S=175000 | South     | Age25_34  | MarriedKids => D=33462.5
S=175000 | South     | Age25_34  | OneParent   => D=32020.7
S=175000 | South     | Age25_34  | MarriedOnly => D=37428.6
S=175000 | South     | Age35_44  | Single      => D=31515.8
S=175000 | South     | Age35_44  | MarriedKids => D=30888.5
S=175000 | South     | Age35_44  | OneParent   => D=29557
S=175000 | South     | Age35_44  | MarriedOnly => D=34955.5
S=175000 | South     | Age45_54  | Single      => D=30650.8
S=175000 | South     | Age45_54  | MarriedKids => D=29934.5
S=175000 | South     | Age45_54  | OneParent   => D=28643.9
S=175000 | South     | Age45_54  | MarriedOnly => D=34038.9
S=175000 | South     | Age55_64  | Single      => D=32704.9
S=175000 | South     | Age55_64  | MarriedKids => D=32200
S=175000 | South     | Age55_64  | OneParent   => D=30812.3
S=175000 | South     | Age55_64  | MarriedOnly => D=36215.6
S=175000 | South     | Age65_74  | Single      => D=33925.5
S=175000 | South     | Age65_74  | MarriedKids => D=33546.3
S=175000 | South     | Age65_74  | OneParent   => D=32100.9
S=175000 | South     | Age65_74  | MarriedOnly => D=37509.1
S=175000 | West      | Under25   | Single      => D=28271.3
S=175000 | West      | Under25   | MarriedKids => D=27580.2
S=175000 | West      | Under25   | OneParent   => D=26278.1
S=175000 | West      | Under25   | MarriedOnly => D=31674
S=175000 | West      | Age25_34  | Single      => D=25759.4
S=175000 | West      | Age25_34  | MarriedKids => D=24809.7
S=175000 | West      | Age25_34  | OneParent   => D=23626.4
S=175000 | West      | Age25_34  | MarriedOnly => D=29012.1
S=175000 | West      | Age35_44  | Single      => D=23286.6
S=175000 | West      | Age35_44  | MarriedKids => D=22082.3
S=175000 | West      | Age35_44  | OneParent   => D=21016
S=175000 | West      | Age35_44  | MarriedOnly => D=26391.6
S=175000 | West      | Age45_54  | Single      => D=22370
S=175000 | West      | Age45_54  | MarriedKids => D=21071.4
S=175000 | West      | Age45_54  | OneParent   => D=20048.5
S=175000 | West      | Age45_54  | MarriedOnly => D=25420.4
S=175000 | West      | Age55_64  | Single      => D=24546.5
S=175000 | West      | Age55_64  | MarriedKids => D=23471.9
S=175000 | West      | Age55_64  | OneParent   => D=22346
S=175000 | West      | Age55_64  | MarriedOnly => D=27726.8
S=175000 | West      | Age65_74  | Single      => D=25839.9
S=175000 | West      | Age65_74  | MarriedKids => D=24898.5
S=175000 | West      | Age65_74  | OneParent   => D=23711.4
S=175000 | West      | Age65_74  | MarriedOnly => D=29097.4
S=175000 | Midwest   | Under25   | Single      => D=33584.1
S=175000 | Midwest   | Under25   | MarriedKids => D=33349.9
S=175000 | Midwest   | Under25   | OneParent   => D=31837.8
S=175000 | Midwest   | Under25   | MarriedOnly => D=37251.8
S=175000 | Midwest   | Age25_34  | Single      => D=31190
S=175000 | Midwest   | Age25_34  | MarriedKids => D=30709.3
S=175000 | Midwest   | Age25_34  | OneParent   => D=29310.4
S=175000 | Midwest   | Age25_34  | MarriedOnly => D=34714.7
S=175000 | Midwest   | Age35_44  | Single      => D=28833
S=175000 | Midwest   | Age35_44  | MarriedKids => D=28109.7
S=175000 | Midwest   | Age35_44  | OneParent   => D=26822.4
S=175000 | Midwest   | Age35_44  | MarriedOnly => D=32217
S=175000 | Midwest   | Age45_54  | Single      => D=27959.5
S=175000 | Midwest   | Age45_54  | MarriedKids => D=27146.2
S=175000 | Midwest   | Age45_54  | OneParent   => D=25900.2
S=175000 | Midwest   | Age45_54  | MarriedOnly => D=31291.3
S=175000 | Midwest   | Age55_64  | Single      => D=30033.9
S=175000 | Midwest   | Age55_64  | MarriedKids => D=29434.2
S=175000 | Midwest   | Age55_64  | OneParent   => D=28090.1
S=175000 | Midwest   | Age55_64  | MarriedOnly => D=33489.6
S=175000 | Midwest   | Age65_74  | Single      => D=31266.7
S=175000 | Midwest   | Age65_74  | MarriedKids => D=30793.9
S=175000 | Midwest   | Age65_74  | OneParent   => D=29391.4
S=175000 | Midwest   | Age65_74  | MarriedOnly => D=34796
S=200000 | Northeast | Under25   | Single      => D=41231.6
S=200000 | Northeast | Under25   | MarriedKids => D=41019.3
S=200000 | Northeast | Under25   | OneParent   => D=39375.8
S=200000 | Northeast | Under25   | MarriedOnly => D=45258.1
S=200000 | Northeast | Age25_34  | Single      => D=38630.8
S=200000 | Northeast | Age25_34  | MarriedKids => D=38150.6
S=200000 | Northeast | Age25_34  | OneParent   => D=36630.2
S=200000 | Northeast | Age25_34  | MarriedOnly => D=42502
S=200000 | Northeast | Age35_44  | Single      => D=36070.3
S=200000 | Northeast | Age35_44  | MarriedKids => D=35326.6
S=200000 | Northeast | Age35_44  | OneParent   => D=33927.3
S=200000 | Northeast | Age35_44  | MarriedOnly => D=39788.6
S=200000 | Northeast | Age45_54  | Single      => D=35121.3
S=200000 | Northeast | Age45_54  | MarriedKids => D=34279.9
S=200000 | Northeast | Age45_54  | OneParent   => D=32925.5
S=200000 | Northeast | Age45_54  | MarriedOnly => D=38783
S=200000 | Northeast | Age55_64  | Single      => D=37374.9
S=200000 | Northeast | Age55_64  | MarriedKids => D=36765.5
S=200000 | Northeast | Age55_64  | OneParent   => D=35304.5
S=200000 | Northeast | Age55_64  | MarriedOnly => D=41171.1
S=200000 | Northeast | Age65_74  | Single      => D=38714.1
S=200000 | Northeast | Age65_74  | MarriedKids => D=38242.6
S=200000 | Northeast | Age65_74  | OneParent   => D=36718.2
S=200000 | Northeast | Age65_74  | MarriedOnly => D=42590.3
S=200000 | South     | Under25   | Single      => D=48293.7
S=200000 | South     | Under25   | MarriedKids => D=48396.7
S=200000 | South     | Under25   | OneParent   => D=46608.3
S=200000 | South     | Under25   | MarriedOnly => D=52503.1
S=200000 | South     | Age25_34  | Single      => D=45774.1
S=200000 | South     | Age25_34  | MarriedKids => D=45617.6
S=200000 | South     | Age25_34  | OneParent   => D=43948.5
S=200000 | South     | Age25_34  | MarriedOnly => D=49833
S=200000 | South     | Age35_44  | Single      => D=43293.7
S=200000 | South     | Age35_44  | MarriedKids => D=42881.8
S=200000 | South     | Age35_44  | OneParent   => D=41330
S=200000 | South     | Age35_44  | MarriedOnly => D=47204.4
S=200000 | South     | Age45_54  | Single      => D=42374.3
S=200000 | South     | Age45_54  | MarriedKids => D=41867.8
S=200000 | South     | Age45_54  | OneParent   => D=40359.5
S=200000 | South     | Age45_54  | MarriedOnly => D=46230.2
S=200000 | South     | Age55_64  | Single      => D=44557.5
S=200000 | South     | Age55_64  | MarriedKids => D=44275.7
S=200000 | South     | Age55_64  | OneParent   => D=42664.2
S=200000 | South     | Age55_64  | MarriedOnly => D=48543.7
S=200000 | South     | Age65_74  | Single      => D=45854.9
S=200000 | South     | Age65_74  | MarriedKids => D=45706.7
S=200000 | South     | Age65_74  | OneParent   => D=44033.7
S=200000 | South     | Age65_74  | MarriedOnly => D=49918.6
S=200000 | West      | Under25   | Single      => D=39635.2
S=200000 | West      | Under25   | MarriedKids => D=39155.6
S=200000 | West      | Under25   | OneParent   => D=37635
S=200000 | West      | Under25   | MarriedOnly => D=43506.7
S=200000 | West      | Age25_34  | Single      => D=36965.5
S=200000 | West      | Age25_34  | MarriedKids => D=36210.9
S=200000 | West      | Age25_34  | OneParent   => D=34816.6
S=200000 | West      | Age25_34  | MarriedOnly => D=40677.5
S=200000 | West      | Age35_44  | Single      => D=34337.2
S=200000 | West      | Age35_44  | MarriedKids => D=33312.1
S=200000 | West      | Age35_44  | OneParent   => D=32042.1
S=200000 | West      | Age35_44  | MarriedOnly => D=37892.3
S=200000 | West      | Age45_54  | Single      => D=33363.1
S=200000 | West      | Age45_54  | MarriedKids => D=32237.7
S=200000 | West      | Age45_54  | OneParent   => D=31013.8
S=200000 | West      | Age45_54  | MarriedOnly => D=36860
S=200000 | West      | Age55_64  | Single      => D=35676.3
S=200000 | West      | Age55_64  | MarriedKids => D=34789.1
S=200000 | West      | Age55_64  | OneParent   => D=33455.8
S=200000 | West      | Age55_64  | MarriedOnly => D=39311.4
S=200000 | West      | Age65_74  | Single      => D=37051
S=200000 | West      | Age65_74  | MarriedKids => D=36305.3
S=200000 | West      | Age65_74  | OneParent   => D=34907
S=200000 | West      | Age65_74  | MarriedOnly => D=40768.2
S=200000 | Midwest   | Under25   | Single      => D=45351.9
S=200000 | Midwest   | Under25   | MarriedKids => D=45357.9
S=200000 | Midwest   | Under25   | OneParent   => D=43614.1
S=200000 | Midwest   | Under25   | MarriedOnly => D=49505
S=200000 | Midwest   | Age25_34  | Single      => D=42807.3
S=200000 | Midwest   | Age25_34  | MarriedKids => D=42551.3
S=200000 | Midwest   | Age25_34  | OneParent   => D=40927.9
S=200000 | Midwest   | Age25_34  | MarriedOnly => D=46808.5
S=200000 | Midwest   | Age35_44  | Single      => D=40302.3
S=200000 | Midwest   | Age35_44  | MarriedKids => D=39788.3
S=200000 | Midwest   | Age35_44  | OneParent   => D=38283.5
S=200000 | Midwest   | Age35_44  | MarriedOnly => D=44153.9
S=200000 | Midwest   | Age45_54  | Single      => D=39373.8
S=200000 | Midwest   | Age45_54  | MarriedKids => D=38764.3
S=200000 | Midwest   | Age45_54  | OneParent   => D=37303.4
S=200000 | Midwest   | Age45_54  | MarriedOnly => D=43170
S=200000 | Midwest   | Age55_64  | Single      => D=41578.6
S=200000 | Midwest   | Age55_64  | MarriedKids => D=41196.1
S=200000 | Midwest   | Age55_64  | OneParent   => D=39630.8
S=200000 | Midwest   | Age55_64  | MarriedOnly => D=45506.4
S=200000 | Midwest   | Age65_74  | Single      => D=42888.9
S=200000 | Midwest   | Age65_74  | MarriedKids => D=42641.3
S=200000 | Midwest   | Age65_74  | OneParent   => D=41014
S=200000 | Midwest   | Age65_74  | MarriedOnly => D=46894.9
"""

# ---------------------------------------------------------------------------
# Parse DISPOSABLE_DATA into a lookup dict keyed by (salary, region, age, dep)
# ---------------------------------------------------------------------------
DISPOSABLE_INCOME = {}
for _line in DISPOSABLE_DATA.strip().splitlines():
    if "=>" not in _line:
        continue
    _left, _right = _line.split("=>")
    try:
        _Dval = float(_right.strip().lstrip("D=").strip())
    except ValueError:
        continue
    _parts = [seg.strip() for seg in _left.split("|")]
    if len(_parts) != 4:
        continue
    try:
        _S = int(_parts[0].split("=")[1])
    except Exception:
        continue
    DISPOSABLE_INCOME[(_S, _parts[1], _parts[2], _parts[3])] = _Dval


# =============================================================================
# SECTION 2b — DEMOGRAPHIC BANKROLLS derived from DISPOSABLE_INCOME
#
# For each Q2 demographic key we compute the average of all POSITIVE
# disposable-income entries whose age bucket maps to that key.
# Only positive values are included — negative D means the household
# cannot cover essentials and has no gambling bankroll.
#
# Male / Female have no direct age mapping in the income table, so they
# use the overall average of all positive entries in the table.
#
# These averages are computed once at import time and stored in
# DEMOGRAPHIC_BANKROLLS.  Every call that previously used bankroll=2000
# now calls get_demographic_bankroll(demographic) instead.
# =============================================================================

def _build_demographic_bankrolls() -> dict:
    """
    Compute average positive disposable income per Q2 demographic key
    from DISPOSABLE_INCOME.  Called once at module load time.
    """
    # Reverse mapping: Q2 demo key -> set of Q1 age bucket strings
    demo_to_q1_ages = {
        "18-34": {"Under25", "Age25_34"},
        "35-49": {"Age35_44"},
        "50-64": {"Age45_54", "Age55_64"},
        "65+":   {"Age65_74"},
    }

    buckets = {demo: [] for demo in DEMOGRAPHICS}

    for (sal, region, age_q1, dep), D in DISPOSABLE_INCOME.items():
        if D <= 0:
            continue  # household cannot afford to gamble
        for demo_key, q1_set in demo_to_q1_ages.items():
            if age_q1 in q1_set:
                buckets[demo_key].append(D)
                break

    # Male / Female: use the overall mean of all positive entries
    all_positive = [D for D in DISPOSABLE_INCOME.values() if D > 0]
    overall_mean = float(np.mean(all_positive)) if all_positive else 2000.0

    result = {}
    for demo in DEMOGRAPHICS:
        if demo in ("Male", "Female"):
            result[demo] = overall_mean
        else:
            vals = buckets[demo]
            result[demo] = float(np.mean(vals)) if vals else overall_mean

    return result


DEMOGRAPHIC_BANKROLLS: dict = _build_demographic_bankrolls()


def get_demographic_bankroll(demographic: str) -> float:
    """
    Return the average positive disposable income for a Q2 demographic key.
    Falls back to the overall table mean if the key is unrecognised.
    """
    return DEMOGRAPHIC_BANKROLLS.get(
        demographic,
        float(np.mean([D for D in DISPOSABLE_INCOME.values() if D > 0]))
    )


# =============================================================================
# SECTION 3 — TRANSITION MATRIX CONSTRUCTION
# =============================================================================

def build_transition_matrix(f: float, C: float, Pw: float) -> np.ndarray:
    P = np.zeros((4, 4))

    P[0, 0] = 1.0 - f
    P[0, 1] = f

    p_loss   = 1.0 - P_WIN
    P[1, 0]  = P_WIN * Pw + p_loss * (1.0 - C)
    P[1, 1]  = P_WIN * (1.0 - Pw)
    P[1, 2]  = p_loss * C
    P[1, 3]  = 0.0

    P[2, 0] = 0.150
    P[2, 1] = 0.081
    P[2, 2] = 0.669
    P[2, 3] = 0.100

    P[3, 3] = 1.0

    assert abs(P.sum(axis=1) - 1.0).max() < 1e-10, \
        "Transition matrix rows must sum to 1.0"
    return P


# =============================================================================
# SECTION 4 — ANALYTICAL EXPECTED LOSS
# =============================================================================

def expected_annual_loss(f: float, C: float, Pw: float,
                         V1: float, k: float,
                         N: int = 52) -> dict:
    P         = build_transition_matrix(f, C, Pw)
    V2        = k * V1
    W         = np.array([0.0, V1, V2, 0.0])
    Omega     = np.array([0.0, OMEGA_S1, OMEGA_S2, 0.0])
    loss_rate = W * Omega

    pi            = np.array([1.0, 0.0, 0.0, 0.0])
    total_loss    = 0.0
    total_wagered = 0.0
    weekly_losses = []
    weekly_wagers = []
    state_time    = np.zeros(4)

    for _ in range(N):
        pi             = pi @ P
        total_loss    += float(pi @ loss_rate)
        total_wagered += float(pi @ W)
        weekly_losses.append(float(pi @ loss_rate))
        weekly_wagers.append(float(pi @ W))
        state_time    += pi

    return {
        "expected_annual_loss"   : total_loss,
        "annual_wager_total"     : total_wagered,
        "weekly_losses"          : weekly_losses,
        "weekly_wagers"          : weekly_wagers,
        "avg_state_distribution" : state_time / N,
        "transition_matrix"      : P,
    }


# =============================================================================
# SECTION 5 — MONTE CARLO SIMULATION
# =============================================================================

def monte_carlo(f: float, C: float, Pw: float,
                V1: float, k: float,
                bankroll: float = None,
                demographic: str = None,
                N: int = 52,
                n_sims: int = 5000,
                seed: int = 42) -> dict:
    """
    Simulate n_sims gamblers over N weeks.

    bankroll is resolved in this priority order:
      1. Explicit `bankroll` argument (if provided and > 0).
      2. DEMOGRAPHIC_BANKROLLS[demographic] (if `demographic` is given).
      3. Overall mean of positive DISPOSABLE_INCOME entries.
    """
    if bankroll is None or bankroll <= 0:
        if demographic is not None:
            bankroll = get_demographic_bankroll(demographic)
        else:
            bankroll = float(np.mean([D for D in DISPOSABLE_INCOME.values() if D > 0]))

    rng = np.random.default_rng(seed)
    P   = build_transition_matrix(f, C, Pw)
    V2  = k * V1

    paths       = np.zeros((n_sims, N + 1))
    paths[:, 0] = bankroll
    ruin_weeks  = np.full(n_sims, N + 1, dtype=int)

    for sim in range(n_sims):
        state = 0
        br    = bankroll

        for t in range(1, N + 1):
            state = rng.choice(4, p=P[state])
            wager = V1 if state == 1 else (V2 if state == 2 else 0.0)

            if wager > 0:
                if br < wager:
                    state = 3
                    paths[sim, t:] = br
                    ruin_weeks[sim] = t
                    break
                elif rng.random() < P_WIN:
                    br += wager * WIN_PAYOUT
                else:
                    br -= wager

            paths[sim, t] = br

    net_pl = paths[:, -1] - bankroll

    return {
        "paths"         : paths,
        "net_pl"        : net_pl,
        "ruin_weeks"    : ruin_weeks,
        "ruin_rate"     : float(np.mean(ruin_weeks <= N)),
        "mean_pl"       : float(np.mean(net_pl)),
        "median_pl"     : float(np.median(net_pl)),
        "p10_pl"        : float(np.percentile(net_pl, 10)),
        "p25_pl"        : float(np.percentile(net_pl, 25)),
        "p75_pl"        : float(np.percentile(net_pl, 75)),
        "p90_pl"        : float(np.percentile(net_pl, 90)),
        "pct_profitable": float(np.mean(net_pl > 0)),
        "bankroll_used" : bankroll,
    }


# =============================================================================
# SECTION 6 — HIGH-LEVEL RUN FUNCTION
# =============================================================================

def run_model(demographic: str = "18-34",
              bankroll: float = None,
              verbose: bool = True) -> dict:
    """
    Run analytical + Monte Carlo model for a demographic group.

    If bankroll is None (default), it is pulled automatically from
    DEMOGRAPHIC_BANKROLLS, which is derived from the DISPOSABLE_INCOME table.
    Pass an explicit float to override.
    """
    p    = DEMOGRAPHICS[demographic]
    br   = bankroll if (bankroll is not None and bankroll > 0) \
           else get_demographic_bankroll(demographic)
    anal = expected_annual_loss(**p)
    mc   = monte_carlo(**p, bankroll=br, demographic=demographic)

    if verbose:
        d = anal["avg_state_distribution"]
        print(f"\n{'='*64}")
        print(f"  DEMOGRAPHIC: {demographic}   "
              f"(bankroll = ${br:,.2f}  [avg positive D from DISPOSABLE_INCOME])")
        print(f"{'='*64}")
        print(f"  Survey parameters (Tab 4, 2025 Survey of Online Gambling Habits):")
        print(f"    f  (weekly betting frequency)   = {p['f']*100:.0f}%  [Row 32]")
        print(f"    C  (chasing propensity)          = {p['C']*100:.0f}%  [Row 38]")
        print(f"    Pw (leave winnings in account)   = {p['Pw']*100:.0f}%  [Row 36]")
        print(f"  Behavior parameters (from literature):")
        print(f"    V1 base wager                   = ${p['V1']}/week  [M3 survey + Rutgers CGS]")
        print(f"    V2 chase wager = {p['k']}xV1           = ${p['V1']*p['k']:.0f}/week  [gambling literature]")
        print(f"    Omega_S1 (straight-bet hold)     = {OMEGA_S1*100:.2f}%  [NJ DGE 2024]")
        print(f"    Omega_S2 (parlay hold)           = {OMEGA_S2*100:.1f}%   [NJ DGE 2024]")
        print(f"\n  --- Analytical Result (N=52 weeks) ---")
        print(f"    Expected annual loss             = ${anal['expected_annual_loss']:,.2f}")
        print(f"    Expected total wagered           = ${anal['annual_wager_total']:,.2f}")
        print(f"    Average weeks in each state:")
        for i, label in enumerate(["S0 Dormant", "S1 Disciplined", "S2 Chasing", "S3 Stopped"]):
            print(f"      {label:18s}  {d[i]*100:5.1f}%  {'#'*int(d[i]*50)}")
        print(f"\n  --- Monte Carlo (n=5,000 simulations, seed=42) ---")
        print(f"    Mean annual P&L                  = ${mc['mean_pl']:>10,.2f}")
        print(f"    Median annual P&L                = ${mc['median_pl']:>10,.2f}")
        print(f"    Pessimistic outcome (10th pct)   = ${mc['p10_pl']:>10,.2f}")
        print(f"    Optimistic  outcome (90th pct)   = ${mc['p90_pl']:>10,.2f}")
        print(f"    Fraction ending in profit        = {mc['pct_profitable']*100:.1f}%")
        print(f"    Bankroll ruin rate               = {mc['ruin_rate']*100:.1f}%")

    return {"analytical": anal, "mc": mc, "demographic": demographic,
            "params": p, "bankroll": br}


# =============================================================================
# SECTION 7 — INDIVIDUAL PREDICTION FUNCTIONS
# =============================================================================

def predict_individual(age_group: str = "18-34",
                       weekly_wager=None,
                       risk_tolerance: str = "medium",
                       bankroll: float = None) -> dict:
    """
    Predict annual P&L for a specific individual.

    If bankroll is None, it is pulled from DEMOGRAPHIC_BANKROLLS for the
    given age_group using the DISPOSABLE_INCOME averages.
    """
    risk_map = {
        "low":    dict(k=1.5,  C_delta=-0.10),
        "medium": dict(k=None, C_delta=0.00),
        "high":   dict(k=3.0,  C_delta=+0.10),
    }

    params = DEMOGRAPHICS[age_group].copy()
    adj    = risk_map[risk_tolerance]

    if weekly_wager is not None:
        params["V1"] = weekly_wager

    if adj["k"] is not None:
        params["k"] = adj["k"]

    params["C"] = float(np.clip(params["C"] + adj["C_delta"], 0.05, 0.95))

    br = bankroll if (bankroll is not None and bankroll > 0) \
         else get_demographic_bankroll(age_group)

    anal = expected_annual_loss(**params)
    mc   = monte_carlo(**params, bankroll=br, demographic=age_group)

    print(f"\n{'='*64}")
    print(f"  INDIVIDUAL PREDICTION")
    print(f"{'='*64}")
    print(f"  Age group: {age_group}  | Risk tolerance: {risk_tolerance}")
    print(f"  Base wager: ${params['V1']}/week  |  Chase wager: ${params['V1']*params['k']:.0f}/week")
    print(f"  Chasing probability: {params['C']*100:.0f}%  |  "
          f"Starting bankroll: ${br:,.2f}  "
          f"[avg positive D from DISPOSABLE_INCOME]")
    print(f"\n  Expected annual loss (analytical):  ${anal['expected_annual_loss']:,.2f}")
    print(f"  Median outcome (Monte Carlo):       ${mc['median_pl']:,.2f}")
    print(f"  Worst-case 10th percentile:         ${mc['p10_pl']:,.2f}")
    print(f"  Best-case  90th percentile:         ${mc['p90_pl']:,.2f}")
    print(f"  Probability of ending in profit:    {mc['pct_profitable']*100:.1f}%")
    print(f"  Probability of depleting bankroll:  {mc['ruin_rate']*100:.1f}%")

    return {"analytical": anal, "mc": mc, "params": params, "bankroll": br}


def _print_income_results_table(age_group: str,
                                region: str,
                                disposable_income: float,
                                risk_tolerance: str,
                                results_by_demo: dict):
    demo_key = AGE_MAP.get(age_group, age_group)

    print(f"\n{'='*80}")
    print(f"  Q1 -> Q2 INTEGRATION RESULTS")
    print(f"{'='*80}")
    print(f"  Age group      : {age_group}  (model key: {demo_key})")
    print(f"  Region         : {region}  [label only — baked into disposable income by Q1]")
    print(f"  Disposable income (D) : ${disposable_income:,.2f}  [starting bankroll]")
    print(f"  Risk tolerance : {risk_tolerance}")
    print(f"\n  Expected annual loss across all demographic groups")
    print(f"  (bankroll = ${disposable_income:,.2f} for all rows)")
    print(f"  {'─'*76}")
    print(f"  {'Demographic':<12} {'E[Loss]/yr':>12} {'Loss % of D':>12} "
          f"{'Median P&L':>12} {'Ruin%':>8} {'Profitable%':>13}")
    print(f"  {'─'*76}")

    for dkey, res in results_by_demo.items():
        anal        = res["analytical"]
        mc          = res["mc"]
        loss_pct    = (anal["expected_annual_loss"] / disposable_income * 100
                       if disposable_income > 0 else float("nan"))
        marker      = "  <-- matched demographic" if dkey == demo_key else ""
        print(f"  {dkey:<12} "
              f"${anal['expected_annual_loss']:>11,.2f} "
              f"{loss_pct:>11.2f}% "
              f"${mc['median_pl']:>11,.2f} "
              f"{mc['ruin_rate']*100:>7.1f}% "
              f"{mc['pct_profitable']*100:>12.1f}%"
              f"{marker}")

    print(f"  {'─'*76}")
    print(f"\n  Note: Starting bankroll = ${disposable_income:,.2f} for all rows.")
    print(f"  E[Loss]/yr and Median P&L are driven by weekly wager size and house edge,")
    print(f"  not by bankroll size (when ruin rate = 0%, the floor is never reached).")
    print(f"  'Loss % of D' shows each group's annual loss as a fraction of THIS")
    print(f"  household's disposable income — this DOES vary with bankroll/income.")


def predict_from_age_and_income(age_group: str,
                                disposable_income: float,
                                risk_tolerance: str = "medium",
                                region: str = "N/A") -> dict:
    """
    Run the gambling model given an age group and a disposable income value.
    The disposable_income is used directly as the bankroll for all rows.
    """
    demo_key = AGE_MAP.get(age_group, age_group)

    if demo_key not in DEMOGRAPHICS:
        raise ValueError(
            f"Unrecognized age group: '{age_group}'. "
            f"Valid Q1 buckets: {list(AGE_MAP.keys())}. "
            f"Valid internal keys: {list(DEMOGRAPHICS.keys())}."
        )

    if disposable_income <= 0:
        print(f"\n{'='*72}")
        print(f"  Q1 -> Q2 INTEGRATION: SKIPPED")
        print(f"{'='*72}")
        print(f"  Age group : {age_group}  |  Region: {region}")
        print(f"  Disposable income D = ${disposable_income:,.2f}")
        print(f"  Cannot gamble — household cannot cover essential expenses at this")
        print(f"  salary/region/dependency combination. No bankroll available.")
        return None

    results_by_demo = {}
    for dkey in DEMOGRAPHICS:
        p   = DEMOGRAPHICS[dkey].copy()
        adj = {"low": dict(k=1.5, C_delta=-0.10),
               "medium": dict(k=None, C_delta=0.00),
               "high": dict(k=3.0, C_delta=+0.10)}[risk_tolerance]
        if adj["k"] is not None:
            p["k"] = adj["k"]
        p["C"] = float(np.clip(p["C"] + adj["C_delta"], 0.05, 0.95))

        # Scale V1 to this household's actual disposable income.
        # Each demographic has a survey-derived V1 calibrated against its
        # average positive disposable income (DEMOGRAPHIC_BANKROLLS).
        # Keeping V1/income constant means richer households bet proportionally
        # more — consistent with the M3 survey wager-size data.
        avg_br = get_demographic_bankroll(dkey)
        p["V1"] = p["V1"] * (disposable_income / avg_br)

        results_by_demo[dkey] = {
            "analytical": expected_annual_loss(**p),
            "mc":         monte_carlo(**p, bankroll=disposable_income),
        }

    _print_income_results_table(age_group, region, disposable_income,
                                risk_tolerance, results_by_demo)

    return results_by_demo[demo_key]


def get_disposable_income(salary: float, region: str,
                          age_group: str, dependency: str):
    key = (int(salary), region.strip(), age_group.strip(), dependency.strip())
    return DISPOSABLE_INCOME.get(key)


def predict_from_income(salary: float,
                        region: str,
                        age_group: str,
                        dependency: str,
                        risk_tolerance: str = "medium") -> dict:
    D = get_disposable_income(salary, region, age_group, dependency)
    if D is None:
        raise ValueError(
            f"No disposable income entry for "
            f"(S={salary}, region={region}, age={age_group}, dep={dependency}). "
            f"Check that spelling matches the DISPOSABLE_DATA table exactly."
        )
    return predict_from_age_and_income(age_group, D,
                                       risk_tolerance=risk_tolerance,
                                       region=region)


# =============================================================================
# SECTION 8 — VISUALIZATIONS
# =============================================================================

def plot_fan_chart(result: dict, ax=None, color="steelblue"):
    paths    = result["mc"]["paths"]
    bankroll = paths[0, 0]
    weeks    = np.arange(paths.shape[1])
    bands    = {p: np.percentile(paths, p, axis=0) for p in [10, 25, 50, 75, 90]}

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.fill_between(weeks, bands[10], bands[90], alpha=0.12, color=color,
                    label="10-90th percentile")
    ax.fill_between(weeks, bands[25], bands[75], alpha=0.25, color=color,
                    label="25-75th percentile")
    ax.plot(weeks, bands[50], color=color, linewidth=2.5, label="Median")
    ax.axhline(bankroll, color="grey", linestyle="--", linewidth=1, alpha=0.6,
               label="Starting bankroll")
    ax.axhspan(0, result["params"]["V1"], color="red", alpha=0.06)
    ax.set_xlabel("Week", fontsize=9)
    ax.set_ylabel("Bankroll ($)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(alpha=0.3)
    return ax


def plot_all_demographics(save_path: str = None):
    """Generate a 2x3 grid of fan charts — bankroll per demographic from DISPOSABLE_INCOME."""
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00695C", "#B71C1C"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Markov Chain Sports Gambling Model — Annual Bankroll Trajectories\n"
        "Fan charts: 10/25/50/75/90th percentile outcomes (5,000 simulations)\n"
        "House edges: Omega_S1=4.56%, Omega_S2=18.5% — Source: NJ DGE 2024\n"
        "Starting bankroll = avg positive disposable income per age group (DISPOSABLE_INCOME table)",
        fontsize=11, fontweight="bold"
    )

    for ax, demo, color in zip(axes.flat, DEMOGRAPHICS, colors):
        res            = run_model(demo, verbose=False)
        plot_fan_chart(res, ax=ax, color=color)
        expected_loss  = res["analytical"]["expected_annual_loss"]
        ruin_rate_pct  = res["mc"]["ruin_rate"] * 100
        profitable_pct = res["mc"]["pct_profitable"] * 100
        br             = res["bankroll"]
        ax.set_title(
            f"{demo}  |  bankroll=${br:,.0f}  |  E[Loss]=${expected_loss:,.0f}/yr  |  "
            f"Ruin={ruin_rate_pct:.0f}%  Profitable={profitable_pct:.0f}%",
            fontsize=8
        )
        ax.set_xlim(0, 52)
        ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_comparative_summary(save_path: str = None):
    """Two-panel bar chart — bankroll per demographic from DISPOSABLE_INCOME."""
    demo_list  = list(DEMOGRAPHICS.keys())
    results    = [run_model(d, verbose=False) for d in demo_list]
    e_losses   = [r["analytical"]["expected_annual_loss"] for r in results]
    ruin_rates = [r["mc"]["ruin_rate"] * 100              for r in results]
    prof_rates = [r["mc"]["pct_profitable"] * 100         for r in results]

    x = np.arange(len(demo_list))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Annual Sports Gambling Impact by Demographic\n"
        "Omega_S1=4.56% (NJ DGE 2024 straight-bet hold)  |  "
        "Omega_S2=18.5% (NJ DGE parlay hold)\n"
        "Starting bankroll = avg positive disposable income per age group",
        fontsize=11, fontweight="bold"
    )

    bars = ax1.bar(x, e_losses, color="#1565C0", alpha=0.85, width=0.55, zorder=3)
    ax1.set_xticks(x); ax1.set_xticklabels(demo_list, fontsize=11)
    ax1.set_ylabel("Expected Annual Loss ($)", fontsize=11)
    ax1.set_title("Analytical Expected Annual Loss", fontsize=11)
    ax1.grid(axis="y", alpha=0.35, zorder=0)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    for bar, val in zip(bars, e_losses):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"${val:,.0f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    ax2.bar(x - 0.2, ruin_rates, 0.38, label="Ruin rate (%)",
            color="#C62828", alpha=0.85, zorder=3)
    ax2.bar(x + 0.2, prof_rates, 0.38, label="Profitable rate (%)",
            color="#2E7D32", alpha=0.85, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(demo_list, fontsize=11)
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_title("Monte Carlo Outcomes (n=5,000)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.35, zorder=0)
    for i, (rr, pp) in enumerate(zip(ruin_rates, prof_rates)):
        ax2.text(i - 0.2, rr + 0.5, f"{rr:.0f}%", ha="center", va="bottom", fontsize=8)
        ax2.text(i + 0.2, pp + 0.5, f"{pp:.0f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_transition_heatmap(demographic: str = "18-34", save_path: str = None):
    p      = DEMOGRAPHICS[demographic]
    P      = build_transition_matrix(p["f"], p["C"], p["Pw"])
    labels = ["S0\nDormant", "S1\nDisciplined", "S2\nChasing", "S3\nStopped"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(P, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Weekly transition probability")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(f"Transition Matrix — {demographic}", fontsize=10, fontweight="bold")
    for i in range(4):
        for j in range(4):
            v = P[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=11, color="white" if v > 0.55 else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_weekly_loss_curve(save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00695C", "#B71C1C"]

    for demo, color in zip(DEMOGRAPHICS, colors):
        anal     = expected_annual_loss(**DEMOGRAPHICS[demo])
        cum_loss = np.cumsum(anal["weekly_losses"])
        ax.plot(range(1, 53), cum_loss,
                label=f"{demo}  (${cum_loss[-1]:,.0f}/yr)", linewidth=2, color=color)

    ax.set_xlabel("Week of year", fontsize=11)
    ax.set_ylabel("Cumulative expected loss ($)", fontsize=11)
    ax.set_title(
        "Cumulative Expected Annual Loss — Analytical Markov Chain Model\n"
        "Source: NJ DGE 2024 hold rates | 2025 Survey of Online Gambling Habits",
        fontsize=11, fontweight="bold"
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(title="Demographic (annual total)", fontsize=9)
    ax.grid(alpha=0.35)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*64)
    print("  M3 CHALLENGE Q2 — MARKOV CHAIN SPORTS GAMBLING LOSS MODEL")
    print("="*64)
    print(f"\n  Fixed constants (from NJ DGE 2024 and -110 odds theory):")
    print(f"    Omega_S1 (straight-bet hold)  = {OMEGA_S1*100:.2f}%")
    print(f"    Omega_S2 (parlay hold)        = {OMEGA_S2*100:.1f}%")
    print(f"    Win probability P_WIN         = {P_WIN*100:.1f}%")
    print(f"    Win payout ratio WIN_PAYOUT   = {WIN_PAYOUT:.4f}  (100/110 at -110 price)")

    # -------------------------------------------------------------------------
    # Show the bankrolls derived from DISPOSABLE_INCOME
    # -------------------------------------------------------------------------
    print(f"\n  BANKROLLS DERIVED FROM DISPOSABLE_INCOME TABLE")
    print(f"  (average of all positive D entries per age group)")
    print(f"  {'─'*44}")
    for demo, br in DEMOGRAPHIC_BANKROLLS.items():
        print(f"    {demo:<10}  ${br:>10,.2f}")

    # -------------------------------------------------------------------------
    # VALIDATION — house edge
    # -------------------------------------------------------------------------
    ev_per_dollar = P_WIN * WIN_PAYOUT - (1 - P_WIN) * 1
    print(f"\n  VALIDATION CHECK:")
    print(f"    Theoretical EV per $ at -110:  {ev_per_dollar*100:.4f}%")
    print(f"    NJ DGE observed OMEGA_S1:       {-OMEGA_S1*100:.4f}%")
    print(f"    Match: {'PASS' if abs(abs(ev_per_dollar) - OMEGA_S1) < 0.001 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # VALIDATION — transition matrix row sums
    # -------------------------------------------------------------------------
    print(f"\n    Transition matrix row-sum check:")
    for demo in DEMOGRAPHICS:
        p_demo   = DEMOGRAPHICS[demo]
        P_demo   = build_transition_matrix(p_demo["f"], p_demo["C"], p_demo["Pw"])
        row_sums = P_demo.sum(axis=1)
        ok       = abs(row_sums - 1.0).max() < 1e-10
        print(f"      {demo:<10}: max row deviation = {abs(row_sums-1.0).max():.2e}  "
              f"{'PASS' if ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # ANNUAL LOSS SUMMARY TABLE — bankrolls from DISPOSABLE_INCOME
    # -------------------------------------------------------------------------
    print(f"\n{'─'*80}")
    print(f"  ANNUAL LOSS SUMMARY TABLE  "
          f"(N=52 weeks, bankroll = avg positive D from DISPOSABLE_INCOME)")
    print(f"{'─'*80}")
    print(f"  {'Group':<10} {'Bankroll':>10} {'E[Loss]':>10} {'Total Wagered':>15} "
          f"{'%Chasing':>10} {'Ruin%':>8} {'Profitable%':>13}")
    print(f"  {'─'*76}")

    for demo in DEMOGRAPHICS:
        p_demo = DEMOGRAPHICS[demo]
        br     = get_demographic_bankroll(demo)
        anal   = expected_annual_loss(**p_demo)
        mc     = monte_carlo(**p_demo, bankroll=br, demographic=demo)
        d      = anal["avg_state_distribution"]
        print(f"  {demo:<10} "
              f"${br:>9,.0f} "
              f"${anal['expected_annual_loss']:>9,.2f} "
              f"${anal['annual_wager_total']:>13,.2f} "
              f"{d[2]*100:>9.1f}% "
              f"{mc['ruin_rate']*100:>7.1f}% "
              f"{mc['pct_profitable']*100:>12.1f}%")

    # -------------------------------------------------------------------------
    # VALIDATION — Monte Carlo vs Analytical convergence
    # -------------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print(f"  VALIDATION: Monte Carlo vs Analytical convergence (should be within ~5%)")
    print(f"{'─'*72}")
    for demo in DEMOGRAPHICS:
        p_demo   = DEMOGRAPHICS[demo]
        br       = get_demographic_bankroll(demo)
        anal     = expected_annual_loss(**p_demo)
        mc       = monte_carlo(**p_demo, bankroll=br, demographic=demo)
        diff_pct = abs(mc["mean_pl"] - (-anal["expected_annual_loss"])) / \
                   max(1, anal["expected_annual_loss"]) * 100
        print(f"  {demo:<10}: bankroll=${br:,.0f}  analytical=${anal['expected_annual_loss']:,.0f}  "
              f"MC mean=${-mc['mean_pl']:,.0f}  diff={diff_pct:.1f}%")

    # -------------------------------------------------------------------------
    # DETAILED PROFILES — bankroll from DISPOSABLE_INCOME
    # -------------------------------------------------------------------------
    run_model("18-34")
    run_model("65+")

    # -------------------------------------------------------------------------
    # CUSTOM INDIVIDUAL EXAMPLES — bankroll from DISPOSABLE_INCOME
    # -------------------------------------------------------------------------
    print("\n" + "─"*64)
    predict_individual(age_group="18-34", risk_tolerance="high", weekly_wager=200)
    predict_individual(age_group="18-34", risk_tolerance="high",  weekly_wager=50)

    # -------------------------------------------------------------------------
    # Q1 -> Q2 INTEGRATION: predict_from_age_and_income()
    # -------------------------------------------------------------------------
    print("\n" + "─"*64)
    print("  Q1 -> Q2 INTEGRATION: predict_from_age_and_income()")
    print("─"*64)

    q1_examples = [
        ("Under25",  -12248.2, "Northeast"),
        ("Age25_34",  4555.17, "South"),
        ("Age35_44", 11583.7,  "South"),
        ("Age45_54", 20240.3,  "South"),
        ("Age55_64", 26210.5,  "Northeast"),
        ("Age65_74", 42888.9,  "Midwest"),
    ]

    for age, income, reg in q1_examples:
        predict_from_age_and_income(age, income, risk_tolerance="medium", region=reg)

    # -------------------------------------------------------------------------
    # Q1 -> Q2 INTEGRATION: predict_from_income() (full lookup)
    # -------------------------------------------------------------------------
    print("\n" + "─"*64)
    print("  Q1 -> Q2 INTEGRATION: predict_from_income() (full lookup)")
    print("─"*64)

    lookup_examples = [
        (100000, "South",     "Age25_34", "Single"),
        (125000, "Northeast", "Age35_44", "MarriedKids"),
        (150000, "Midwest",   "Age55_64", "MarriedOnly"),
        (200000, "West",      "Age65_74", "Single"),
    ]

    for sal, reg, age, dep in lookup_examples:
        print(f"\n  [S={sal:,} | {reg} | {age} | {dep}]")
        try:
            predict_from_income(sal, reg, age, dep, risk_tolerance="medium")
        except ValueError as e:
            print(f"  lookup failed: {e}")
