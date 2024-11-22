# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:59:36 2024

@author: gaut729
"""

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import math
np.random.seed(0)

solar_data = pd.read_excel('solar_profile.xlsx', engine='openpyxl')
wind_data = pd.read_excel('wind_profile_NY.xlsx', engine='openpyxl')
class RTSSystem:
    def __init__(self):
        self.MTTF = [2940, 2940, 2940, 2940, 2940, 450, 450, 450, 450, 1980, 1980, 1980, 1980, 1980, 1980, 1960, 1960, 1960, 1960, 1200, 1200, 1200, 960, 960, 960, 960, 950, 950, 950, 1150, 1100, 1100]
        self.MTTR = [60, 60, 60, 60, 60, 50, 50, 50, 50, 20, 20, 20, 20, 20, 20, 40, 40, 40, 40, 50, 50, 50, 40, 40, 40, 40, 50, 50, 50, 100, 150, 150]
        self.GM = [12, 12, 12, 12, 12, 20, 20, 20, 20, 50, 50, 50, 50, 50, 50, 76, 76, 76, 76, 100, 100, 100, 155, 155, 155, 155, 197, 197, 197, 350, 400, 400]

def generate_load_profile():
    Weeks = [86.2, 90, 87.8, 83.4, 88, 84.1, 83.2, 80.6, 74, 73.7, 71.5, 72.7, 70.4, 75, 72.1, 80, 75.4, 83.7, 87, 88, 85.6, 81.1, 90, 88.7, 89.6, 86.1, 75.5, 81.6, 80.1, 88, 72.2, 77.6, 80, 72.9, 72.6, 70.5, 78, 69.5, 72.4, 72.4, 74.3, 74.4, 80, 88.1, 88.5, 90.9, 94, 89, 94.2, 97, 100, 95.2]
    Days = [93, 100, 98, 96, 94, 77, 75]
    Wkdy1to8and44to52 = [67, 63, 60, 59, 59, 60, 74, 86, 95, 96, 96, 95, 95, 95, 93, 94, 99, 100, 100, 96, 91, 83, 73, 63]
    Wknd1to8and44to52 = [78, 72, 68, 66, 64, 65, 66, 70, 80, 88, 90, 91, 90, 88, 87, 87, 91, 100, 99, 97, 94, 92, 87, 81]
    Wkdy18to30 = [64, 60, 58, 56, 56, 58, 64, 76, 87, 95, 99, 100, 99, 100, 100, 97, 96, 96, 93, 92, 92, 93, 87, 72]
    Wknd18to30 = [74, 70, 66, 65, 64, 62, 62, 66, 81, 86, 91, 93, 93, 92, 91, 91, 92, 94, 95, 95, 100, 93, 88, 80]
    Wkdy9to17and31to43 = [63, 62, 60, 58, 59, 65, 72, 85, 95, 99, 100, 99, 93, 92, 90, 88, 90, 92, 96, 98, 96, 90, 80, 70]
    Wknd9to17and31to43 = [75, 73, 69, 66, 65, 65, 68, 74, 83, 89, 92, 94, 91, 90, 90, 86, 85, 88, 92, 100, 97, 95, 90, 85]

    H = np.zeros(8760)
    n = 0
    for k in range(52):
        for j in range(7):
            for i in range(24):
                n += 1
                if (k < 8 or k >= 44):
                    if j < 5:
                        H[n-1] = round(Wkdy1to8and44to52[i] * Days[j] * Weeks[k] * 2850 / 1000000)
                    else:
                        H[n-1] = round(Wknd1to8and44to52[i] * Days[j] * Weeks[k] * 2850 / 1000000)
                elif k >= 18 and k <= 30:
                    if j < 5:
                        H[n-1] = round(Wkdy18to30[i] * Days[j] * Weeks[k] * 2850 / 1000000)
                    else:
                        H[n-1] = round(Wknd18to30[i] * Days[j] * Weeks[k] * 2850 / 1000000)
                else:
                    if j < 5:
                        H[n-1] = round(Wkdy9to17and31to43[i] * Days[j] * Weeks[k] * 2850 / 1000000)
                    else:
                        H[n-1] = round(Wknd9to17and31to43[i] * Days[j] * Weeks[k] * 2850 / 1000000)

    H[8736:8760] = H[8712:8736]  # Adjust last 24 hours
    return H

# # Define Renewable Energy Profile generation
# def generate_wind_profile(seed=None):
#     if seed is not None:
#         np.random.seed(seed)  # For reproducibility with variation
#     capacity_factor = 0.35  # Average capacity factor
#     wind_profile = np.clip(np.random.normal(capacity_factor, 0.15, 8760), 0, 1) 
#     return wind_profile

# def generate_pv_profile(seed=None):
#     if seed is not None:
#         np.random.seed(seed)  # For reproducibility with variation
#     daily_pattern = np.maximum(0, np.sin(np.linspace(0, 2 * np.pi, 24)))  # Peak at noon
#     pv_profile = np.tile(daily_pattern, 365)
#     return pv_profile


# Energy Storage System class
class EnergyStorageSystem:
    def __init__(self, duration=2, power=200, efficiency=0.5):
        self.capacity = power * duration # Maximum storage capacity in MWh
        self.power = power  # Maximum charge/discharge power in MW
        self.efficiency = efficiency # Round-trip efficiency
        self.storage = self.capacity  # Initial storage level, assumed full

    def charge(self, amount):
        chrg_power = min(self.power, amount)
        potential_storage = self.storage + chrg_power * self.efficiency
        self.storage = min(self.capacity, potential_storage)

    def discharge(self, amount):
        potential_storage = self.storage - amount
        actual_discharge_power = min(amount, self.storage)
        self.storage = max(0, potential_storage)
        return actual_discharge_power

casedata = RTSSystem()
L = generate_load_profile()
# wind_profile = generate_wind_profile()
# pv_profile = generate_pv_profile()
# ess = EnergyStorageSystem()

lam = 1.0 / np.array(casedata.MTTF)
mu = 1.0 / np.array(casedata.MTTR)
G = np.array(casedata.GM)
N = len(G)

# User can enable/disable wind, PV, and storage
#enable_wind = False
# enable_pv = False
#enable_storage = False
#enable_wind = True
#enable_pv = True
# enable_storage = True

def calculate_reliability_metrics(duration=2, x=0, enable_pv=False,enable_wind = False,enable_storage = False):
    ess = EnergyStorageSystem(duration=duration)
    z = np.random.rand(N)
    t = np.floor(-np.log(z) / lam)
    status = np.ones(N)
    T = 0
    T1 = min(t)
    Tdown = 0
    TENS = 0
    down_transition = 0.0
    sys_status = 1  # Initial system status
    factor = 2850*x
    
    Yr = 1000  # Number of years to simulate
    
    if enable_pv==True and enable_wind==True and enable_storage==False: 
        pv_fraction= 2
        wind_fraction = 2
            
    elif enable_pv==True and enable_wind==False and enable_storage==False: 
        pv_fraction= 1
        wind_fraction = math.inf
    
    elif enable_pv==False and enable_wind==True and enable_storage==False:
        pv_fraction= math.inf
        wind_fraction = 1
    
    elif enable_pv==False and enable_wind== False and enable_storage==False:
        pv_fraction= math.inf
        wind_fraction = math.inf
    
    elif enable_pv==True and enable_wind== True and enable_storage==True:
        pv_fraction= 2
        wind_fraction = 2

    elif enable_pv==False and enable_wind== True and enable_storage==True:
        pv_fraction= math.inf
        wind_fraction = 1

    elif enable_pv==True and enable_wind== False and enable_storage==True:
        pv_fraction= 1
        wind_fraction = math.inf


    for year in range(Yr):
        # wind_profile = generate_wind_profile(seed=year)*factor/2
        # pv_profile = generate_pv_profile(seed=year)*factor/2

        
        pv_profile = solar_data.solar_profile*factor/pv_fraction
        wind_profile = wind_data.wind_profile*factor/wind_fraction

        #print(year, wind_profile.shape)
        while T < 8760 * (year + 1):
            tmin = min(t)
            t -= tmin
            idx = np.where(t == 0)[0]
            for i in idx:
                z = np.random.rand()
                status[i] = abs(status[i] - 1)
                if status[i] == 1:
                    t[i] = np.floor(-np.log(z) / lam[i])
                else:
                    t[i] = np.floor(-np.log(z) / mu[i])
            Thermal_gen = np.dot(G, status)
            TotalGen = np.dot(G, status)
            
            if enable_wind:
                TotalGen += wind_profile[int(T1) % 8760]
    
            if enable_pv:
                TotalGen += pv_profile[int(T1) % 8760]
    
            for j in range(int(tmin)):
                hr = int(T1 % 8760)  # Ensure hr is an integer
                if hr == 0:
                    hr = 8760
                Load = L[hr-1]
                
                if TotalGen >= Load:
                    sys_status = 1
                    if enable_storage:
                        ess.charge(TotalGen - Load-Thermal_gen)
                else:
                    if sys_status == 1:
                        down_transition += 1
                    sys_status = 0
                    if enable_storage and ess.storage > 0:
                        discharge = min(ess.power, Load - TotalGen)
                        discharge_power = ess.discharge(discharge)
                        TotalGen += discharge_power
                    if TotalGen < Load:
                        Tdown += 1
                        TENS += (Load - TotalGen)
                T1 += 1
            T += tmin
        
    
    # Obtain annual metrics
    LOLP = Tdown / T
    LOLF = down_transition / T
    EDNS = TENS / T
    LOLE = Tdown/Yr
    EENS = TENS/Yr
    
    print(f"LOLP: {LOLP}")
    print(f"LOLF: {LOLF}")
    print(f"EDNS: {EDNS}")
    print(f"LOLE: {LOLE} hours/yr")
    print(f"EENS: {EENS} MWh/yr")
    
if __name__ == "__main__":
    
    for case in range(6):
        x=case/10
        calculate_reliability_metrics(duration=5, x=x,enable_pv=True,enable_wind = False,enable_storage = True )
    