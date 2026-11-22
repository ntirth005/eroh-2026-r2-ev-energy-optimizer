"""Central assumptions for the EV charging + swapping network digital twin (all INR, hourly).
Every number here is an ASSUMPTION (indicative of Indian market conditions) and is listed in the report's
assumption table. Demand scale / grid limits are CALIBRATED so that the *baseline* operation reproduces the
case's current KPIs (reliability ~78 %, losses ~10 %)."""
import numpy as np

SHORT = ['DEL', 'MUM', 'BLR', 'HYD', 'MAA']
NAME = {'DEL': 'Delhi NCR', 'MUM': 'Mumbai MMR', 'BLR': 'Bengaluru', 'HYD': 'Hyderabad', 'MAA': 'Chennai'}

# ---- network (totals match the case: 1,200 MW; 60/40 DC/swap; 600 MWh storage) -----------------------------
# bess = stationary battery MWh (240 total), pack = swap-pack inventory MWh (360 total)
CFG = {
    'DEL': dict(cap=300, bess=60, pack=90, discom=4.5, lat=28.6, Tm=25.0, Ta=10.5, Td=6.0, solar=1.00,
                cloud_base=0.25, cloud_amp=0.35, monsoon=(6.5, 9.5), windy=(5.5, 8.5), price_f=1.00,
                dc_mw=0.0, sw_mw=0.0, grid=0.0),
    'MUM': dict(cap=280, bess=56, pack=84, discom=5.9, lat=19.1, Tm=27.5, Ta=3.0, Td=3.5, solar=0.40,
                cloud_base=0.30, cloud_amp=0.55, monsoon=(6.0, 9.5), windy=(5.5, 9.0), price_f=1.05,
                dc_mw=0.0, sw_mw=0.0, grid=0.0),
    'BLR': dict(cap=260, bess=52, pack=78, discom=5.0, lat=13.0, Tm=24.0, Ta=3.0, Td=5.0, solar=0.50,
                cloud_base=0.30, cloud_amp=0.35, monsoon=(6.0, 10.5), windy=(5.5, 9.0), price_f=1.00,
                dc_mw=0.0, sw_mw=0.0, grid=0.0),
    'HYD': dict(cap=200, bess=40, pack=60, discom=6.0, lat=17.4, Tm=27.0, Ta=6.0, Td=6.0, solar=0.80,
                cloud_base=0.25, cloud_amp=0.35, monsoon=(6.0, 10.0), windy=(5.5, 9.0), price_f=0.97,
                dc_mw=0.0, sw_mw=0.0, grid=0.0),
    'MAA': dict(cap=160, bess=32, pack=48, discom=6.3, lat=13.1, Tm=29.0, Ta=4.0, Td=4.0, solar=0.30,
                cloud_base=0.30, cloud_amp=0.35, monsoon=(10.0, 12.5), windy=(5.5, 9.0), price_f=0.95,
                dc_mw=0.0, sw_mw=0.0, grid=0.0),
}
DC_SHARE, SW_SHARE = 0.60, 0.40
RESERVE = 0.05                      # 5 % of capacity / pack inventory uncommitted
RE_SHARE = 0.35                     # share of energy from open-access RE

# ---- demand (calibrated; mean MW of *requested* energy at reference price in 2025) ---------------------------
# Filled in by calibrate.py -> data/calib.json ; defaults here are first guesses
DEFAULT_CALIB = {
    'DEL': dict(dc_mw=62.0, sw_mw=8.0, grid=105.0),
    'MUM': dict(dc_mw=62.0, sw_mw=7.5, grid=100.0),
    'BLR': dict(dc_mw=48.0, sw_mw=6.5, grid=80.0),
    'HYD': dict(dc_mw=34.0, sw_mw=5.0, grid=62.0),
    'MAA': dict(dc_mw=24.0, sw_mw=4.0, grid=50.0),
}
GROWTH = 0.20                       # yearly demand growth (EV adoption), history shaped by this

# ---- segments (share of energy, own-price elasticity, deferrable share) ------------------------------------
SEG_DC = dict(retail=(0.55, -0.75), ride=(0.30, -0.40), logi=(0.15, -0.20))
SEG_SW = dict(retail=(0.10, -0.50), ride=(0.50, -0.30), logi=(0.40, -0.15))
RHO_TRUE = 0.30                     # share of price-suppressed volume recaptured in next 1-4 h
RECAP_W = np.array([0.4, 0.3, 0.2, 0.1])

# ---- prices & economics (INR) ------------------------------------------------------------------------------
P_DC, P_SW = 17.5, 38.0             # reference tariff Rs/kWh (swap price includes pack-as-a-service)
BAND = 0.20                         # +-20 % of previous-day average tariff
VAR_OPEX = 0.20                     # payment fees 2 %, site-host revenue share 15 %, variable O&M 3 %
SW_LOGISTICS = 1.2                  # Rs/kWh swapped: rebalancing trucks, riders, rack handling
RE_PPA = 4.2                        # Rs/kWh landed, pay-as-generated (take-or-pay)
DAM_ADDER = 0.9                     # Rs/kWh transmission, SLDC, exchange fees
TRANS_LOSS = 0.03                   # ISTS/STU loss on exchange energy
STANDBY = 0.10                      # unscheduled DISCOM draw limited to 10 % of sanctioned load (feeder / DSM limits)
UNSCHED_PREM = 1.25                 # unscheduled DISCOM draw = 1.25 x tariff
SELL_FACTOR = 0.75                  # surplus sold at 0.75 x DAM price (DSM / RTM)
TOD = np.array([0.8] * 6 + [1.0] * 12 + [1.2] * 4 + [1.0] * 2)   # DISCOM ToD multiplier by hour
PRICE_CAP = 20.0                    # highest IEX DAM price in the data (Rs/kWh), used to clip forecasts
PRICE_CAP_NOW = 10.0                # current exchange cap, used for stress shocks
FIXED_OPEX_KW = 0.0                 # Rs per kW installed per year -> calibrated (staff, leases, O&M, network)

# ---- physics -----------------------------------------------------------------------------------------------
ETA_DC = 0.935                     # DC fast charger conversion (baseline hardware, avg over load range)
ETA_SW = 0.925                     # swap-rack AC/DC charger
ETA_DC_SIC, ETA_SW_SIC = 0.960, 0.950   # after SiC retrofit
DIST_LOSS = 0.024                   # cables, transformers, aux (cooling) at site bus
BESS_ETA = 0.955                    # one-way (91 % round trip)
BESS_SD = 0.00010                   # per hour self-discharge + standby
BESS_SOC_MIN, BESS_SOC_MAX = 0.15, 0.95   # incl. 5 % reserve
BESS_C = 0.5                        # power = 0.5C
PACK_SD = 0.00045                   # per hour idle loss of full racked packs (~1 %/day incl. BMS/thermal)
PACK_TURN = 7.0                     # mean hours a swapped pack stays out
PACK_EMPTY, PACK_FULL = 0.10, 0.95  # SoC window of racked packs
PEAK_FACTOR = 0.70                  # intra-hour peaking -> usable fraction of nameplate
CHARGER_AVAIL = 0.95                # nameplate minus reserve share

# ---- battery wear (Rs) -- cycle lives & heat sensitivity are FITTED from the state-of-health records (degradation.py)
import json as _json, os as _os
_deg = {}
_f = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data', 'deg.json')
if not _os.path.exists(_f):
    _f = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'deg.json')
if _os.path.exists(_f):
    _deg = _json.load(open(_f))
BESS_CAPEX_KWH = 14000.0
BESS_CYCLES = _deg.get('bess_cycles', 6000.0)       # equivalent full cycles to 80 % state of health at 25 C
PACK_CAPEX_KWH = 15000.0
PACK_CYCLES = _deg.get('pack_cycles', 2500.0)
HEAT_WEAR = _deg.get('heat_wear', 0.030)            # extra cycle-wear per degC above 25 C
PACK_CAL_RS_MWH_H = 220.0           # calendar-ageing cost of holding packs near full SoC, Rs per MWh-hour


def bess_wear_rs_per_mwh(T):
    """Rs per MWh discharged: capex / equivalent-full-cycle life x temperature multiplier (1.0 at 25C)."""
    return BESS_CAPEX_KWH * 1000.0 / BESS_CYCLES * (1.0 + HEAT_WEAR * np.maximum(T - 25.0, 0.0))


def pack_cycle_rs_per_mwh():
    return PACK_CAPEX_KWH * 1000.0 / PACK_CYCLES


def derate(T):
    """Charger power derate above 38C."""
    return 1.0 - 0.012 * np.maximum(T - 38.0, 0.0)

# ---- spatial mismatch inside a cluster (20 hubs / stations, lognormal demand dispersion) ------------------------
SIG_MAX = 0.70                      # max share of hot-hub overflow that a price/credit incentive can redirect
DELTA0 = 0.10                       # incentive scale: share redirected = SIG_MAX * (1 - exp(-incentive / DELTA0))
#   incentive = loyalty credit as a fraction of the tariff, paid only on redirected energy (chosen by the optimiser)
STEER_GRID = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
BETA_REBAL = 0.60                   # share of rack inventory re-positioned in proportion to forecast station demand
TURN_FRAC = 0.30                    # share of racked energy a station network can hand out in one hour (bay throughput)
PACK_REBAL_RS_MWH = 1500.0          # Rs per MWh swapped, trucks + riders for rebalancing
