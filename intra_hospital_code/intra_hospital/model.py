'''
@author: hannantahir

'''
from intra_hospital import params
import random
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
from mesa import Model
import matplotlib.pyplot as plt
from intra_hospital.agents import *
from intra_hospital.schedule import RandomActivationByType


class IntraHospModel(Model):
    """A model with some number of agents."""
    verbose = False  # Print-monitoring
    ''' Specify path to model directory where run.py exist '''
    results_dir = '/SPECIFY PATH TO CODE DIRECTORY/simulations/'

    ### output dataframes for patients
    data_patient_risk_count = pd.DataFrame(columns=['lowrisk','highrisk'])
    data_patient_state_count = pd.DataFrame(columns=['SUSCEPTIBLE','COLONIZED','HIDDEN_COLONIZED','INFECTED','COLONIZEDINHOSP'])
    data_patient_movement = pd.DataFrame(columns=['id','start_time','end_time','ward_from','ward_to','risk'])
    data_hospital_los = pd.DataFrame(columns=['id','los','risk'])
    ward_num_array = [item for item in range(1, len(params.room_num)+1)]
    cols = ['w' + str(x) for x in ward_num_array] # creates columns names such as w1, w2 etc.
    data_prev_per_ward_transmission_only = pd.DataFrame(columns = cols)
    data_col_inf_per_ward = pd.DataFrame(columns = cols)
    data_total_patient_count = pd.DataFrame(columns = cols)
    data_total_prev_per_ward = pd.DataFrame(columns = cols)
    data_transmission_count = pd.DataFrame(columns = ['trans_count'])
    
    available_rooms = list() # list of rooms that are available
    rooms_occupied = list() # list of rooms that patients occupy

    
    def __init__(self):
        '''
        Model initialization. Patients are admitted to hospital rooms
        '''
        self.schedule = RandomActivationByType(self)
        self.newuniqueID = 0
        self.room_num = params.room_num
        self.patient_max = sum(self.room_num)
        self.num_steps_per_day = params.num_steps_per_day

        ### risk groups and disease state related
        self.highrisk_arrival_prob = params.highrisk_arrival_prob
        self.risk_scores = [0,1]  # 0 is for low risk, 1 is for highrisk
        self.arrival_prob_by_risk = [1-self.highrisk_arrival_prob, self.highrisk_arrival_prob]
        self.prob_colonized_arrivals_ICU = params.prob_colonized_arrivals_ICU
        self.prob_hidden_colonized_arrivals_ICU = params.prob_hidden_colonized_arrivals_ICU
        self.prob_colonized_arrivals_nonICU = params.prob_colonized_arrivals_nonICU
        self.prob_colonized_arrivals = params.prob_colonized_arrivals
        self.prob_hidden_colonized_arrivals_nonICU = params.prob_hidden_colonized_arrivals_nonICU
        self.states = params.states
        self.states_prob_ICU = [1-(self.prob_colonized_arrivals_ICU + self.prob_hidden_colonized_arrivals_ICU), self.prob_colonized_arrivals_ICU, self.prob_hidden_colonized_arrivals_ICU, 0.00, 0]
        self.states_prob_nonICU = [1-(self.prob_colonized_arrivals_nonICU + self.prob_hidden_colonized_arrivals_nonICU), self.prob_colonized_arrivals_nonICU, self.prob_hidden_colonized_arrivals_nonICU, 0.00, 0]
        self.states_prob_allwards = [1-self.prob_colonized_arrivals, self.prob_colonized_arrivals, 0.00, 0.00, 0] # without considering hidden colonized in the model
        self.prob_col_to_inf = params.prob_col_to_inf
        
        ## movement matrix related
        self.pref_mat = params.pref_mat
        self.movement_rate_lowrisk = params.movement_rate_lowrisk
        self.movement_rate_highrisk = params.movement_rate_highrisk
        self.daily_discharge_prob = params.daily_discharge_prob
        
        ## transmission related
        self.alpha_nonICU = params.alpha_nonICU
        self.alpha_ICU = params.alpha_ICU
        self.beta_nonICU = params.beta_nonICU
        self.beta_ICU = params.beta_ICU
        
        self.rooms_per_ward = list()
        self.number_of_new_agents = 0
        self.add_colonized_patient_flag = False
        self.num_agents_to_move = [0] * len(self.room_num)
        self.prob_return_to_prev_ward = params.prob_return_to_prev_ward
        self.total_transmissions = 0

        ### create rooms in the hospitals with ward number. e.g 1.10 refers to ward 1, room # 10
        for ward, val in enumerate(self.room_num, start=1):
            for room in range(1,val+1):
                num_room = str(str(ward)+"."+str(room))
                self.available_rooms.append(num_room)     
        random.shuffle(self.available_rooms)  # shuffle room list randomly
        
        ### generate samples from LOS 
        #### LOS for low risk
        self.elements_lowrisk = np.arange(1,params.los_max_days,0.1)
        self.weights_lowrisk = params.los_lambda_lowrisk*np.exp(-params.los_lambda_lowrisk*self.elements_lowrisk)
        self.weights_lowrisk /= self.weights_lowrisk.sum() # normalize the weight to 100% otherwise random.choice will give error

       ### LOS for high risk
        self.elements_highrisk = np.arange(1,params.los_max_days,0.1)
        self.weights_highrisk = params.los_lambda_highrisk*np.exp(-params.los_lambda_highrisk*self.elements_highrisk)
        self.weights_highrisk /= self.weights_highrisk.sum()

        rooms_free = int(len(self.available_rooms) * params.perc_hosp_room_vaccant_initially / 100)
        copy_available_rooms = self.available_rooms[:]
        for idx, val in enumerate(copy_available_rooms):
            if len(self.available_rooms) > rooms_free:
                ward_room_num = val.split('.') # split the string to get room and ward number
                ward_room_num = [int(x) for x in ward_room_num] ## make ward_room_num list into integers
                los_realtime, los_sim_time, initial_state, movement_flag, infec_los_couter = 0.00, 0, 0, 0, 0
                
                risk = np.random.choice(self.risk_scores, p=self.arrival_prob_by_risk)
                if risk == 0: ## risk == 0 mean low risk entry
                    los_realtime = abs(np.random.choice(self.elements_lowrisk, p=self.weights_lowrisk))
                    los_sim_time = int(los_realtime * self.num_steps_per_day)
                    num_movements = int(np.random.poisson(self.movement_rate_lowrisk*los_realtime))
                    patient = Patient(self.newuniqueID, self, initial_state, ward_room_num[0], ward_room_num[1], los_sim_time, los_sim_time, risk, num_movements, movement_flag,infec_los_couter, initial_state, None, False)
                    self.schedule.add(patient)
                    self.available_rooms.remove(val) #
                    self.newuniqueID += 1
                
                if risk == 1: ## risk == 1 means high risk entry
                    los_realtime = abs(np.random.choice(self.elements_highrisk, p=self.weights_highrisk))
                    los_sim_time = int(los_realtime * self.num_steps_per_day)
                    num_movements = int(np.random.poisson(self.movement_rate_highrisk*los_realtime))
                    patient = Patient(self.newuniqueID, self, initial_state, ward_room_num[0], ward_room_num[1], los_sim_time, los_sim_time, risk, num_movements, movement_flag,infec_los_couter, initial_state, None, False)
                    self.schedule.add(patient)
                    self.available_rooms.remove(val) #
                    self.newuniqueID += 1
        
        del copy_available_rooms[:] # This will make the copy avaialable rooms list empty

        ### make a list of rooms occupied
        for pat in self.schedule.agents_by_type[Patient]:
            self.rooms_occupied.append([pat.ward,pat.room])

        if not self.available_rooms:
            print('No room is available')
        
        print('Initial number of patients = ',self.schedule.get_type_count(Patient))
        print('Initial number lowrisk patients = ',self.count_patient_by_risk(0))
        print('Initial number highrisk patients = ',self.count_patient_by_risk(1))


    def new_patient_arrivals(self):
        '''
        Daily patient arrival into the hospital based on average daily arrival rate
        '''
        if self.available_rooms:
            if self.schedule.time == params.colonization_arrival_day_simtime: 
                self.add_colonized_patient_flag = True
            rate = params.patient_avg_arrival_rate / self.num_steps_per_day ## rate per simulation step
            self.number_of_new_agents = self.number_of_new_agents + int(np.random.poisson(rate))
            if self.available_rooms:
                while self.available_rooms and self.number_of_new_agents > 0 and (self.schedule.get_type_count(Patient)+1) < self.patient_max+1:
                    copy_available_rooms = self.available_rooms[:]
                    for idx, val in enumerate(copy_available_rooms):
                        ward_room_num = val.split('.') # split the string to get room and ward number
                        los_realtime, los_sim_time, initial_state, movement_flag, infec_los_couter = 0.00, 0, 0, 0, 0
                        pat_isolated = False
                        ### to add percent colonized in all wards
                        if self.schedule.time > params.colonization_arrival_day_simtime:
                            initial_state = np.random.choice(self.states, p=self.states_prob_allwards)
                            if ward_room_num[0] == '1': ## check if patient is going to be admitted to ward 1
                                if initial_state == 1: ## check if state is colonized. check this at admission and isolate patient
                                    pat_isolated = True                                
                        else:
                            initial_state = 0
   
                        risk = np.random.choice(self.risk_scores, p=self.arrival_prob_by_risk)
                        if risk == 0: # Low risk
                            los_realtime = abs(np.random.choice(self.elements_lowrisk, p=self.weights_lowrisk))
                            los_sim_time = int(los_realtime * self.num_steps_per_day)
                            num_movements = int(np.random.poisson(self.movement_rate_lowrisk*los_realtime))
                            patient = Patient(self.newuniqueID, self, initial_state, ward_room_num[0], ward_room_num[1], los_sim_time, los_sim_time, risk, num_movements, movement_flag,infec_los_couter,initial_state, None, pat_isolated)
                            
                        if risk == 1: # High risk
                            los_realtime = abs(np.random.choice(self.elements_highrisk, p=self.weights_highrisk))
                            los_sim_time = int(los_realtime * self.num_steps_per_day)
                            num_movements = int(np.random.poisson(self.movement_rate_highrisk*los_realtime))
                            patient = Patient(self.newuniqueID, self, initial_state, ward_room_num[0], ward_room_num[1], los_sim_time, los_sim_time, risk, num_movements, movement_flag, infec_los_couter,initial_state, None, pat_isolated)
                        self.number_of_new_agents -= 1
                        self.schedule.add(patient)
                        self.newuniqueID += 1
                        self.available_rooms.remove(val) ## remove the room from available rooms list
                        self.rooms_occupied.append([ward_room_num[0], ward_room_num[1]])
                    del copy_available_rooms[:]
            else:
                pass
 

    def remove_agents_on_discharge(self):
        '''
        When patient length of stay is equal zero, patient discharge from the hospital.
        patient related data is added to respective dataframes before removing them from the agent list
        '''        
        agent_list_copy = self.schedule.agents_by_type[Patient][:] # [:] makes a copy of the list
        count = 0         
        for pat in agent_list_copy:
            if pat.los_remaining <= 0:
                ## to write LOS data for median and mean computation
                self.data_hospital_los = self.data_hospital_los.append({'id':pat.unique_id, 'los':(pat.los/self.num_steps_per_day),'risk':pat.risk}, ignore_index=True)
                self.data_patient_movement.loc[(self.data_patient_movement["id"] == pat.unique_id) & (self.data_patient_movement["end_time"].isnull()), "end_time"] = self.schedule.time
                room_num = pat.room
                ward_num = pat.ward
                num_room = str(str(ward_num)+"."+str(room_num))
                self.schedule.remove(pat)
                self.available_rooms.append(num_room)
                # remove room and ward informtion from occupied room list
                for i, j in enumerate(self.rooms_occupied):
                    if j[0] == ward_num and j[1] == room_num:
                        del self.rooms_occupied[i] 



    def data_write(self):
        '''
        This writes data to relevant dataframes at a specifiec time interval (e.g. every day).
        '''
        pat_count_per_ward = 0
        # to write colonization and prevalence data per ward per day
        pat_count = [0]*len(self.room_num) # count patient per ward
        inf_col_count = [0]*len(self.room_num) # this stores number of colonized and infected per ward per day
        total_prev_count = [0]*len(self.room_num) # to write total prevalence of colonized and infected per ward per day including tranmission
        inf_col_count_inhosp = [0]*len(self.room_num) # to write prevalence of colonized and infected who became colonized during hospital stay per ward per day
        for ward in range(1,len(self.room_num)+1):
            pat_count_per_ward = self.count_patient_per_ward(ward)
            pat_count[ward-1] = pat_count_per_ward
            inf_col_count[ward-1] = self.count_infected_colonized_by_ward(ward)
            if pat_count_per_ward != 0:
                total_prev_count[ward-1] = self.count_infected_colonized_by_ward(ward)/pat_count_per_ward
                inf_col_count_inhosp[ward-1] = self.count_infected_colonized_in_hosp_by_ward(ward)/pat_count_per_ward
            else:
                total_prev_count[ward-1] = 0.0
                inf_col_count_inhosp[ward-1] = 0.0
                
                
        self.data_total_patient_count.loc[len(self.data_total_patient_count), :] = pat_count
        self.data_col_inf_per_ward.loc[len(self.data_total_patient_count), :] = inf_col_count
        self.data_total_prev_per_ward.loc[len(self.data_total_patient_count), :] = total_prev_count
        self.data_prev_per_ward_transmission_only.loc[len(self.data_total_patient_count), :] = inf_col_count_inhosp

        num_risks = 2
        risk_count = [0]*num_risks
        for risk in range(0,num_risks):
            risk_count[risk] = self.count_patient_by_risk(risk)

        self.data_patient_risk_count.loc[len(self.data_patient_risk_count), :] = risk_count
        
        num_states = 5
        state_count = [0]*num_states
        for state in range(0,num_states):
            state_count[state] = self.count_patient_by_state(state)

        self.data_patient_state_count.loc[len(self.data_patient_state_count), :] = state_count


    def write_and_plot(self):
            '''
            This writes output dataframes as excel files or as png.
            '''
            #### to create network of movements, graphml edge list file can be loaded in Gephi to visualize the network
            self.data_patient_movement.to_csv (self.results_dir+'movement_data/movement_data_incl_ward_LOS.csv', index = None, header=True)            
            self.data_patient_movement.to_excel(self.results_dir+'movement_data.xlsx')
            df_movement=self.data_patient_movement.groupby(['ward_from', 'ward_to']).size().reset_index(name = "weight")
            df_movement.to_excel(self.results_dir+'movement_data/movement_data_with_weights.xlsx')
            G_df_movement = nx.from_pandas_dataframe(df_movement,source='ward_from', target='ward_to', edge_attr=["weight"], create_using=nx.DiGraph())
            nx.write_graphml(G_df_movement,self.results_dir+'movement_simulation.graphml')
        
            self.data_total_patient_count.to_excel(self.results_dir+'df_col_inf/data_total_patient_count.xlsx')
            self.data_col_inf_per_ward.to_excel(self.results_dir+'df_col_inf/data_col_inf_per_ward.xlsx')
            self.data_total_prev_per_ward.to_excel(self.results_dir+'df_col_inf/data_total_prev_per_ward.xlsx')
            self.data_prev_per_ward_transmission_only.to_excel(self.results_dir+'df_col_inf/data_prev_per_ward_transmission_only.xlsx')
            self.data_hospital_los.to_csv(self.results_dir+'df_col_inf/LOS_admission_risk.csv')
            
            self.data_transmission_count.loc[len(self.data_transmission_count), :] = self.total_transmissions
            self.data_transmission_count.to_csv(self.results_dir+'df_col_inf/total_tranmissions_count.csv')            

##### to plot admission LOS as box plot and for every rik group
#            
            los_df_lr = self.data_hospital_los[self.data_hospital_los['risk'] == 0]
            los_df_lr.reset_index(inplace = True)
            los_df_hr = self.data_hospital_los[self.data_hospital_los['risk'] == 1]
            los_df_hr.reset_index(inplace = True)
            los_df = pd.DataFrame()
            los_df['Complete'] = self.data_hospital_los['los']
            los_df['Low Risk'] = los_df_lr['los']
            los_df['High Risk'] = los_df_hr['los']
            my_pal = {"Complete": "white","Low Risk": "white", "High Risk":"white"}
            sns.set_style("whitegrid")
            plt.tight_layout()
            meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
            sns.boxplot(data=los_df, palette=my_pal, fliersize=1, linewidth=2, meanprops=meanlineprops, meanline = True, showmeans=True, showfliers=False)
            plt.tight_layout()
            plt.ylabel('Mean LOS in days')
            plt.savefig(self.results_dir+'admission_los.png', dpi=600,bbox_inches="tight")
            plt.close()


######


###### new code for patient movements based on discharge probability
            
    def movement_based_on_disch_prob(self):
        '''
        ### This  takes cares of the patient movements inside the hospital. This uses parameters that were estimated from the actual hospital data
        '''

        if self.schedule.time%self.num_steps_per_day == 0:
            if len(self.num_agents_to_move) != len(self.daily_discharge_prob):
                raise ValueError('number of wards and movement probabilities list are not of same length')
            else:
                for ward_num in range(0,len(self.room_num)):
                    self.num_agents_to_move[ward_num] = self.num_agents_to_move[ward_num] + (self.daily_discharge_prob[ward_num] * self.room_num[ward_num])

        for ward_num in range(0,len(self.room_num)):
            if self.num_agents_to_move[ward_num] >= 1:
               agents_id_list = self.patient_ids_by_ward(ward_num+1)
               ## Pick up a random patient id to be moved
               if agents_id_list:
                   random_id = random.choice(agents_id_list)
               else: 
                   random_id = 0

               for pat in self.schedule.agents_by_type[Patient]:
                   ## check if patient ID matches with the randomy selected ID
                   if pat.unique_id == random_id:
                       current_room, current_ward = 0, 0
                       # check if patient has not previously been moved to a ward. This information is necessary as most patients go back to the wards where they came from (Depending on a given probability)
                       if pat.prev_ward is None:
                           if self.available_rooms:
                                current_room = pat.room
                                current_ward = pat.ward
                                # find patient next ward preference based on preferance matrix
                                ward_list = np.arange(1,len(self.room_num)+1)
                                next_ward_pref = np.array(self.pref_mat)[int(current_ward)-1,:] # this selects the pref matrix row for current ward
                                next_ward = np.random.choice(ward_list, p=next_ward_pref)
                                # Once future ward is selected, Look for available rooms in that ward
                                poss_rooms = [s for s in self.available_rooms if s.split('.')[0] == str(next_ward)]
                                if poss_rooms:
                                    # here we just pick the first entry in the room list. We could pick randomly but these list lists are shuffled occasionally anyway. 
                                    selected_room = poss_rooms[0]
                                    new_ward_room_num = poss_rooms[0].split('.') # split the string to get room and ward number
                                    # Below condition will exclude self loops by not moving patient in the same ward
                                    if current_ward != new_ward_room_num[0]:
                                        pat.ward = new_ward_room_num[0]
                                        pat.room = new_ward_room_num[1]
                                        ### screened and isolated in and out patients from ICU ward 1
                                        if int(current_ward) == 1 or int(pat.ward) == 1:
                                            if pat.state == State.COLONIZED or pat.state == State.COLONIZEDINHOSP or pat.state == State.INFECTED:
                                                pat.patient_isolated = True ## Patient is put on contact isolation

                                        pat.movement_flag = 0
                                        pat.num_movements -= 1
                                        selec_room_index=self.available_rooms.index(selected_room) ## returns the index of selected room in the room list
                                        self.available_rooms.pop(selec_room_index) # remove selected room from the available room list
                                        num_room = str(str(current_ward)+"."+str(current_room))
                                        pat.prev_ward = current_ward
                                        self.available_rooms.append(num_room)
                                        self.data_patient_movement.loc[(self.data_patient_movement["id"] == pat.unique_id) & (self.data_patient_movement["end_time"].isnull()) & (self.data_patient_movement["ward_to"].isnull()), ["end_time","ward_to"]] = [self.schedule.time, pat.ward]
                                        self.data_patient_movement = self.data_patient_movement.append({'id':pat.unique_id,  'start_time':self.schedule.time,'end_time':None,'ward_from':pat.ward,'ward_to':None,'risk':pat.risk}, ignore_index=True)
                                        self.num_agents_to_move[ward_num] -= 1

                                if not poss_rooms:
                                    pass
                            
                           else:
                                pass              
                       
                       # if patient has previously been moved, patient might return to the previous ward based on a probability. 
                       elif pat.prev_ward:
                           if self.available_rooms:
                                current_room = pat.room
                                current_ward = pat.ward
                                if random.random() <= self.prob_return_to_prev_ward:
                                    rooms_in_prev_ward = [s for s in self.available_rooms if s.split('.')[0] == str(pat.prev_ward)]
                                    if rooms_in_prev_ward:
                                        selected_room = rooms_in_prev_ward[0]
                                        new_ward_room_num = rooms_in_prev_ward[0].split('.') # split the string to get room and ward number
                            
                                        if current_ward != new_ward_room_num[0]: # this will exclude self loops by moving by not moving patient in the same ward
                                            pat.ward = new_ward_room_num[0]
                                            pat.room = new_ward_room_num[1]
                                            ### screen and isolated in and out from ICU ward 1
                                            if int(current_ward) == 1 or int(pat.ward) == 1:
                                                if pat.state == State.COLONIZED or pat.state == State.COLONIZEDINHOSP or pat.state == State.INFECTED:
                                                    pat.patient_isolated = True

                                            pat.movement_flag = 0
                                            pat.num_movements -= 1
                                            selec_room_index=self.available_rooms.index(selected_room)
                                            self.available_rooms.pop(selec_room_index) # remove selected room from the available room list
                                            num_room = str(str(current_ward)+"."+str(current_room))
                                            pat.prev_ward = current_ward
                                            self.available_rooms.append(num_room)
                                            self.data_patient_movement.loc[(self.data_patient_movement["id"] == pat.unique_id) & (self.data_patient_movement["end_time"].isnull()) & (self.data_patient_movement["ward_to"].isnull()), ["end_time","ward_to"]] = [self.schedule.time, pat.ward]
                                            self.data_patient_movement = self.data_patient_movement.append({'id':pat.unique_id,  'start_time':self.schedule.time,'end_time':None,'ward_from':pat.ward,'ward_to':None,'risk':pat.risk}, ignore_index=True)
                                            self.num_agents_to_move[ward_num] -= 1
                                            
                                    if not rooms_in_prev_ward:
                                        pass
                                
                                # find patient preference based on preferance matrix
                                else:    
                                    ward_list = np.arange(1,len(self.room_num)+1)
                                    next_ward_pref = np.array(self.pref_mat)[int(current_ward)-1,:] # this selects the pref matrix row for current ward
                                    next_ward = np.random.choice(ward_list, p=next_ward_pref)
                                    # Once future ward is selected, Look for available rooms in that ward
                                    poss_rooms = [s for s in self.available_rooms if s.split('.')[0] == str(next_ward)]
                                    if poss_rooms:
                                        # here we just pick the first entry in the room list. We could pick randomly but these list lists are shuffled occasionally anyway. 
                                        selected_room = poss_rooms[0]
                                        new_ward_room_num = poss_rooms[0].split('.') # split the string to get room and ward number
                                        # Below condition will exclude self loops by not moving patient in the same ward
                                        if current_ward != new_ward_room_num[0]:
                                            pat.ward = new_ward_room_num[0]
                                            pat.room = new_ward_room_num[1]
                                            ### screened and isolated in and out patients from ICU ward 1
                                            if int(current_ward) == 1 or int(pat.ward) == 1:
                                                if pat.state == State.COLONIZED or pat.state == State.COLONIZEDINHOSP or pat.state == State.INFECTED:
                                                    pat.patient_isolated = True ## Patient is put on contact isolation
    
                                            pat.movement_flag = 0
                                            pat.num_movements -= 1
                                            selec_room_index=self.available_rooms.index(selected_room) ## returns the index of selected room in the room list
                                            self.available_rooms.pop(selec_room_index) # remove selected room from the available room list
                                            num_room = str(str(current_ward)+"."+str(current_room))
                                            pat.prev_ward = current_ward
                                            self.available_rooms.append(num_room)
                                            self.data_patient_movement.loc[(self.data_patient_movement["id"] == pat.unique_id) & (self.data_patient_movement["end_time"].isnull()) & (self.data_patient_movement["ward_to"].isnull()), ["end_time","ward_to"]] = [self.schedule.time, pat.ward]
                                            self.data_patient_movement = self.data_patient_movement.append({'id':pat.unique_id,  'start_time':self.schedule.time,'end_time':None,'ward_from':pat.ward,'ward_to':None,'risk':pat.risk}, ignore_index=True)
                                            self.num_agents_to_move[ward_num] -= 1
    
                                    if not poss_rooms:
                                        pass
                           else:
                                pass              


######## 
    def transmission(self):
        '''
        This takes care of the transmission event based on given alpha and beta parameters.
        '''
            # make a list of zeros containing Force of infection (FOI) for every ward
        foi = [0]*(len(self.room_num)+1)
        for ward_num in range(1,len(self.room_num)+1):
            if self.count_patient_per_ward(ward_num):
                if ward_num == 1:
                    foi[ward_num] = self.alpha_ICU + self.beta_ICU*((self.count_nonisolated_colonized_infec_by_ward(ward_num) + (1-params.contact_isoloation_effectiveness)*self.count_isolated_colonized_infec_by_ward(ward_num))/self.count_patient_per_ward(ward_num))
                else:
                    foi[ward_num] = self.alpha_nonICU + self.beta_nonICU*((self.count_nonisolated_colonized_infec_by_ward(ward_num) + (1-params.contact_isoloation_effectiveness)*self.count_isolated_colonized_infec_by_ward(ward_num))/self.count_patient_per_ward(ward_num))
            else: 
                foi[ward_num] = 0.000
        # use foi from index 1, index zero just contains 0 value.    
        for pat in self.schedule.agents_by_type[Patient]:
            if pat.state == State.SUSCEPTIBLE:
                pat_ward = int(pat.ward)
                risk_prob = foi[pat_ward] # this risk_prob should be between range 0 - 1 for bernaulli trail otherwise it throws an error.
                if risk_prob > 1.0:
                    risk_prob = 1.0
                elif risk_prob < 0:
                    risk_prob = 0.0
                # do the bernaoulli trial to see if transmission is successful or not.
                b_trial = self.get_bernoulli_trial(risk_prob)
                if b_trial == 1:
                    self.total_transmissions += 1
                    pat.state = State.COLONIZEDINHOSP
                    pat.prev_colonization_state = 4

    def colonized_to_infected(self):
        '''
        This function takes care of progression from colonized to infected. Infected patients LOS is then increased.
        '''

        for pat in self.schedule.agents_by_type[Patient]:
            if pat.state == State.COLONIZED or pat.state == State.COLONIZEDINHOSP:
                b_trial = self.get_bernoulli_trial(self.prob_col_to_inf)
                if b_trial == 1:
                    pat.prev_colonization_state = pat.state.value
                    pat.state = State.INFECTED
                    pat.los_remaining = pat.los_remaining + (self.num_steps_per_day*params.los_increase_infected)
                    pat.infec_los_couter = self.num_steps_per_day*params.los_increase_infected
                    pat.los = pat.los + (self.num_steps_per_day*params.los_increase_infected)


    def infected_to_colonized(self):
        '''
        Patients return to colonized state after infected duration is over. They remain colonized throughout their hospital stay
        '''

        for pat in self.schedule.agents_by_type[Patient]:
            if pat.state == State.INFECTED and pat.infec_los_couter == 0: 
                if pat.prev_colonization_state == 1:
                    pat.state = State.COLONIZED
                elif pat.prev_colonization_state == 4:
                    pat.state = State.COLONIZEDINHOSP
                pat.infec_los_couter = 0
                    

        
    def get_bernoulli_trial(self, p):
        ''' do bernoulli trial based on the given probability '''
        return np.random.binomial(1, p) # when n = 1, random.binomial(n,p) is similar to bernoulli distribution, so we use n=1 here in the return method. 


    def count_patient_by_risk(self, risk):
        ''' count patients by risk group '''
        count = 0
        for pat in self.schedule.agents_by_type[Patient]:
            if pat.risk == risk:
                count += 1
        return(count)
    
    def count_patient_by_state(self, state):
        ''' count patients by disease state '''
        count = 0
        for pat in self.schedule.agents_by_type[Patient]:
            if pat.state.value == state:

                count += 1
        return(count)

    def count_patient_per_ward(self, ward):
        ''' count patients by ward '''
        pat_count = 0
        for pat in self.schedule.agents_by_type[Patient]:
            if int(pat.ward) == ward:
                pat_count += 1
        return(pat_count)

    def count_nonisolated_colonized_infec_by_ward(self, ward):
        ''' count non-isolated positive patients by ward '''
        col_count = 0
        for pat in self.schedule.agents_by_type[Patient]:
            if int(pat.ward) == ward:                
                if (pat.patient_isolated is False) and ((pat.state == State.COLONIZED) or (pat.state == State.COLONIZEDINHOSP) or (pat.state == State.INFECTED)):
                    col_count += 1
        return(col_count)

    def count_isolated_colonized_infec_by_ward(self, ward):
        ''' count isolated positive patients by ward '''
        col_count = 0
        for pat in self.schedule.agents_by_type[Patient]:
            if int(pat.ward) == ward:                
                if (pat.patient_isolated is True) and ((pat.state == State.COLONIZED) or (pat.state == State.COLONIZEDINHOSP) or (pat.state == State.INFECTED)):
                    col_count += 1
        return(col_count)

    def count_infected_colonized_by_ward(self, ward):
        ''' count positive patients by ward '''
        inf_count = 0
        for pat in self.schedule.agents_by_type[Patient]:
            if int(pat.ward) == ward:                
                if (pat.state == State.COLONIZED) or (pat.state == State.INFECTED) or (pat.state == State.COLONIZEDINHOSP):
                    inf_count += 1
        return(inf_count)

    def count_infected_colonized_in_hosp_by_ward(self, ward):
        ''' count positive patients by ward who became colonized during the hospital stay '''
        count = 0
        for pat in self.schedule.agents_by_type[Patient]:
            if int(pat.ward) == ward:                
                if (pat.state == State.INFECTED and pat.prev_colonization_state == 4) or (pat.state == State.COLONIZEDINHOSP):
                    count += 1
        return(count)
    
    def patient_ids_by_ward(self,ward):
        ''' list of patient ids by ward '''
        id_list = []
        for pat in self.schedule.agents_by_type[Patient]:
            if int(pat.ward) == ward:
                id_list.append(pat.unique_id)
        return(id_list)
        
        
    def step(self):
        '''
        Below methods will be called at every step or at a fixed interval (every hour, every day etc. ) or at a fixed moment (e.g. last step of the simulation)
        '''

        self.schedule.step()
        self.remove_agents_on_discharge()
        self.movement_based_on_disch_prob()
        self.infected_to_colonized()
        self.new_patient_arrivals()

        if self.schedule.time%self.num_steps_per_day == 0:
            self.transmission()
            self.colonized_to_infected()
            self.data_write()

        if self.schedule.time == params.max_iter:
            self.write_and_plot()
