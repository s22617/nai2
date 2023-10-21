"""
==========================================
Fuzzy Control Systems: The Diesel Problem
==========================================


In this project, the described fuzzy control system is a fuzzy control system which models the relationship between
global diesel price and polish diesel price.
When the final price is set, few things are considered:
- the global price of diesel
- the excise duty
- the exchange rate

We would formulate this problem as:

* Antecedents (Inputs)
   - `global price`
      * Universe (ie, crisp value range): How expensive is diesel worldwide on a scale of 0 to 10?
      * Fuzzy set (ie, fuzzy value range): poor, average, good
   - `excise duty`
      * Universe: How severe is the excise duty, on a scale of 0 to 10?
      * Fuzzy set: poor, average, good
   - `exchange rate`
      * Universe: How profitable is the current exchange rate, on a scale of 0 to 10?
      * Fuzzy set: poor, average, good
* Consequents (Outputs)
   - `final price`
      * Universe: What would the final price be compared to the global price, on a scale of 0 to 10?
      * Fuzzy set: low, medium, high
* Rules
    1. IF the global price is "poor" (not expensive) OR the excise_duty is "poor" (low) OR the exchange rate is "poor"
    (profitable), THEN the final price will be low
    2. IF the global price is "average", THEN the final price will be medium
    3. IF the global price is "good" (expensive) OR the excise duty is "good" (high) OR the exchange rate is "good"
    (not profitable), THEN the final price will be high.
* Usage
   - If I tell this controller that:
      * the global price is 1.30, and
      * the excise duty is 1.59, and
      * the exchange rate is 4.21
   - it would predict the diesel price in Poland compared to the global diesel price would be:
      * 3.5798.

Project created by:
        Kajetan Welc
        Daniel Wirzba
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

global_price = ctrl.Antecedent(np.arange(0., 10.01, 0.01), 'global_price')  # in $
excise_duty = ctrl.Antecedent(np.arange(0., 10.01, 0.01), 'excise_duty')  # in zl
exchange_rate = ctrl.Antecedent(np.arange(0., 10.01, 0.01), 'exchange_rate')  # zl to $
final_price = ctrl.Consequent(np.arange(0., 10.01, 0.01), 'final_price')

global_price.automf(3)
excise_duty.automf(3)
exchange_rate.automf(3)

final_price['low'] = fuzz.trimf(final_price.universe, [0., 0., 5.])
final_price['medium'] = fuzz.trimf(final_price.universe, [0., 5., 10.])
final_price['high'] = fuzz.trimf(final_price.universe, [5., 10., 10.])

"""
RULES:
1. IF the global price is "poor" (not expensive) OR the excise_duty is "poor" (low) OR the exchange rate is "poor"
    (profitable), THEN the final price will be low
2. IF the global price is "average", THEN the final price will be medium
3. IF the global price is "good" (expensive) OR the excise duty is "good" (high) OR the exchange rate is "good"
    (not profitable), THEN the final price will be high.

Most people would agree on these rules, but the rules are fuzzy. Mapping the
imprecise rules into a defined, actionable tip is a challenge. This is the
kind of task at which fuzzy logic excels.
"""

rule1 = ctrl.Rule(global_price['good'] | excise_duty['good'] | exchange_rate['good'], final_price['high'])
rule2 = ctrl.Rule(global_price['average'], final_price['medium'])
rule3 = ctrl.Rule(excise_duty['poor'] | global_price['poor'] | exchange_rate['poor'], final_price['low'])

rule1.view()

oil_purchase_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

oil_purchase = ctrl.ControlSystemSimulation(oil_purchase_ctrl)

"""
CURRENT AVERAGE PRICE OF DIESEL AROUND THE WORLD: 1.30$
CURRENT EXCISE RATE FOR CAR PETROL IS 1529 PLN PER 1000 LITERS, WHICH IS: 1.529zl PER LITER
CURRENT EXCHANGE RATE IS 4,21zl to 1$
"""

oil_purchase.input['global_price'] = 1.30
oil_purchase.input['excise_duty'] = 1.59
oil_purchase.input['exchange_rate'] = 4.21

oil_purchase.compute()

print(oil_purchase.output['final_price'])
final_price.view(sim=oil_purchase)
