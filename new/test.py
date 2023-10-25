import serial
import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control
    
       
        
#Describing Fuzzy System            
flame = control.Antecedent(np.arange(0,1024,1),'flame')
sd = control.Antecedent(np.arange(0,1024,1), 'sd')
fire = control.Consequent(np.arange(0,101,1), 'fire')

#Defining Fuzzyficaion System (Membership Functions)
flame['Negative'] = fuzzy.trapmf(flame.universe, [0,0,200,500])
flame['Far'] = fuzzy.trapmf(flame.universe,[350,440,600,650])
flame['Large'] = fuzzy.trapmf(flame.universe,[500,650,1023,1023])

sd['Negative'] = fuzzy.trapmf(sd.universe, [0,0,250,500])
sd['Low'] = fuzzy.trapmf(sd.universe, [400,400,600,700])
sd['High'] = fuzzy.trapmf(sd.universe, [550,800,1023,1023])

fire.automf(2, names=['No Fire','Fire'])

#Describing Fuzzy Inference System using Rules
from skfuzzy.control import ControlSystemSimulation, Rule

rule1 = Rule(sd['Negative'] & flame['Negative'], fire['No Fire'])
rule2 = Rule(sd['Negative'] & flame['Far'], fire['Fire'])
rule3 = Rule(sd['Negative'] & flame['Large'], fire['Fire'])
rule4 = Rule(sd['Low'] & flame['Negative'], fire['No Fire'])
rule5 = Rule(sd['Low'] & flame['Far'], fire['Fire'])
rule6 = Rule(sd['Low'] & flame['Large'], fire['Fire'])
rule7 = Rule(sd['High'] & flame['Negative'], fire['No Fire'])
rule8 = Rule(sd['High'] & flame['Far'], fire['Fire'])
rule9 = Rule(sd['High'] & flame['Large'], fire['Fire'])

rule_book = control.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9])
system = ControlSystemSimulation(rule_book)

system.input['flame'] = 000
system.input['sd'] = 100
system.compute()
print(system.output['fire'])
