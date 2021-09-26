"""

@author: hannantahir

"""

import numpy as np
import csv

''' path to preference matrix file '''
pref_mat_dir = '/SPECIFY PATH TO CODE DIRECTORY/intra_hospital/'

''' gerneral simulation_parameters '''
time_step = 5 # in minutes
num_steps_per_day = int(60*24/time_step)
sim_time = 427 # in days
max_iter = int(num_steps_per_day * sim_time)
#max_iter = 1
colonized_patient_arrival_day = 30
colonization_arrival_day_simtime = colonized_patient_arrival_day * num_steps_per_day
# patient agents related

''' Patient and hospital related '''
perc_hosp_room_vaccant_initially =20 # Percent of room empty at the start of simulation
patient_avg_arrival_rate = 70 # Daily patient arrival rate
highrisk_arrival_prob = 0.3668
prob_colonized_arrivals_ICU = 0.03
prob_colonized_arrivals_nonICU = 0.00
prob_colonized_arrivals = 0.05 # probability colonized patient arrivals in full hospital
prob_hidden_colonized_arrivals_ICU = 0.00
prob_hidden_colonized_arrivals_nonICU = 0.00
prob_col_to_inf = 0.012 #per day
room_num = [22,42,38,36,35,25,23,135,22,18,\
            17,14,13,12,12,9,9,7,7,6,\
            6,5,5,4,3,3,3,3,3,3,\
            2,2,2,2]

 
''' LOS related '''
los_lambda_lowrisk = 0.221
los_lambda_highrisk = 0.150
los_max_days = 100
los_increase_infected = 3 # in days


''' Movement related parameters estimated from actual hospital data'''
daily_discharge_rates = [3.014,0.155,1.527,0.562,0.485,0.494,0.190,0.867,0.183,0.105,\
                         0.201,0.405,0.972,0.159,0.171,0.052,0.070,1.218,0.419,0.101,\
                         1.581,0.155,0.075,0.330,0.073,0.040,0.588,0.073,0.052,0.115,\
                         0.009,0.009,0.019,0.000]
daily_discharge_prob = [ b / m for b,m in zip(daily_discharge_rates, room_num)] # This will divide discharge rate / room_num
movement_rate_highrisk = 0.0308
movement_rate_lowrisk = 0.0236
prob_return_to_prev_ward = 0.81
# Load movement preference matrix
with open(pref_mat_dir+'pivot_table_normalized.csv', newline='') as csvfile:
    pref_mat = list(csv.reader(csvfile))
pref_mat = list(np.float_(pref_mat))    
    

'''Transmission parameters '''
states = np.array([0,1,2,3,4])  # disease states: 0 susceptible, 1 colonized, 2 hidden colnoized, 3 infected, 4 colonized in hospital
alpha_nonICU = 0.0 ## enviornmental contribution
alpha_ICU = 0.0 ## enviornmental contribution
beta_nonICU=0.25
beta_ICU=0.25  

''' interventions '''
contact_isoloation_effectiveness = 0.5 ##(0.9 means contribution in tranmission is reduced by 90%, range 0.0 - 1.0. 1 means no role of infected individuals towards tranmission)
