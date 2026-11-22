from params import *
import lp as L
COV0 = 0.92
POL = {
 'A0 Incumbent rules':         dict(fc='naive', da='rule',  price='tou', rt='rule', sell_re=False, pi=0.0, cov=COV0),
 'A1 +ML forecasts':           dict(fc='ml',    da='rule',  price='tou', rt='rule', sell_re=False, pi=0.0, cov=COV0),
 'A2 +Deterministic MPC':      dict(fc='ml',    da='det',   price='tou', rt='mpc',  sell_re=False, pi=3000.0),
 'A3 +Stochastic CVaR plan':   dict(fc='ml',    da='stoch', price='tou', rt='mpc',  sell_re=False, pi=3000.0),
 'A4 +Dynamic pricing':        dict(fc='ml',    da='stoch', price='opt', rt='mpc',  sell_re=False, pi=3000.0),
 'A5 +Hub steering':           dict(fc='ml',    da='stoch', price='opt', rt='mpc',  sell_re=False, pi=3000.0, steer=True),
 'A6 +RE monetise & packs':    dict(fc='ml',    da='stoch', price='opt', rt='mpc',  sell_re=True,  pi=3000.0, steer=True, rebal=True),
}
HW_SIC = L.HW(ETA_DC_SIC, ETA_SW_SIC)
FULL_NAME = 'A6 +RE monetise & packs'
SIC_NAME = 'A7 +SiC charger hardware'
