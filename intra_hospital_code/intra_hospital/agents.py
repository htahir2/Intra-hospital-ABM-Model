"""
    
    @author: hannantahir
    
"""

from enum import Enum
from mesa import Agent

class State(Enum):
    SUSCEPTIBLE = 0
    COLONIZED = 1
    HIDDENCOLONIZED = 2
    INFECTED = 3
    COLONIZEDINHOSP = 4
    
class Patient(Agent):
    """ A patient agent with all its attributes."""
    def __init__(self, unique_id, model, initial_state, ward, room, los, los_remaining, risk, num_movements, movement_flag,infec_los_couter, prev_colonization_state, prev_ward, patient_isolated):
        super().__init__(unique_id, model)
        if initial_state == 0:
            self.state = State.SUSCEPTIBLE 
        elif initial_state == 1:
            self.state = State.COLONIZED ## colonized but not symptomatically infected 
        elif initial_state == 2:
            self.state = State.HIDDENCOLONIZED ## This is colonized but not detected. This is not used in the current model
        elif initial_state == 3:
            self.state = State.INFECTED ## symptomatically infected state
        elif initial_state == 4:
            self.state = State.COLONIZEDINHOSP ## This is just to keep track of transmissions happended in the hospital. Otherwise no difference from colonized state    
           
        self.unique_id = unique_id ## This is the unique id of the patient
        self.ward = int(ward) ## This stores ward number of the patient
        self.room = int(room) ## This stores room number of the patient
        self.los = int(los) ## Length of stay (LOS)
        self.los_remaining = int(los_remaining) ## This LOS counter is being counted down every step. Once zero, patient is discharged from the hospital
        self.risk = risk ## 0 means low risk, 1 means high risk
        self.num_movements = int(num_movements) ## how many movements patient will make
        self.movement_flag = int(movement_flag)
        self.infec_los_couter = int(infec_los_couter) # counter that counts down infection duration        
        self.prev_colonization_state = int(prev_colonization_state) # this keep track of either patient was colonized at arrival (1) or during hospital stay (4), same state is assigned back after a patient is recovered from infection and becomes colonized
        self.prev_ward = prev_ward ## Patient previous ward from where patient was moved
        self.patient_isolated = patient_isolated ## keep track if patient was on contact isolation.
        

    ''' patient infection duration count down'''
    def infec_los_coutdown(self):        
        if self.state.value == 3:
            if self.infec_los_couter > 0:
                self.infec_los_couter -= 1
            elif self.infec_los_couter <= 0:
                self.infec_los_couter = 0

    ''' patient length of stay count down '''
    def LOS_countdown(self):       
        if self.los_remaining <= 0:
            self.los_remaining = 0
        elif self.los_remaining > 0:
            self.los_remaining -= 1

                  
    def step(self):
        ''' methods thats are called at every time step for every agent '''
        self.LOS_countdown()
#        self.movement_time_countdown()
        self.infec_los_coutdown()
