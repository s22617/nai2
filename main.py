"""
==========================================
Fuzzy Control Systems: The Tipping Problem
==========================================

The 'tipping problem' is commonly used to illustrate the power of fuzzy logic
principles to generate complex behavior from a compact, intuitive set of
expert rules.

The Diesel Problem
-------------------

Let's create a fuzzy control system which models how much you'd pay for diesel per liter.
When the final price is set, few things are considered: the service and food quality,
- the global price of diesel
- the excise duty
- the exchange rate

We would formulate this problem as:

* Antecednets (Inputs)
   - `global price`
      * Universe (ie, crisp value range): How expensive is diesel worldwide on a scale of 0 to 10?
      * Fuzzy set (ie, fuzzy value range): poor, acceptable, amazing
   - `excise duty`
      * Universe: How severe is the excise duty, on a scale of 0 to 10?
      * Fuzzy set: bad, decent, great
   - `exchange rate`
      * Universe: How profitable is the current exchange rate, on a scale of 0 to 10?
      * Fuzzy set: bad, decent, great
* Consequents (Outputs)
   - `final price`
      * Universe: What would the final price be, on a scale of 0 to 10$
      * Fuzzy set: low, medium, high
* Rules
    1. IF the global price is low (not expensive) OR the excise_duty is low (low) OR the exchange rate is good, THEN the final price will be low
    2. IF the global price is average, THEN the final price will be medium
    3. IF the global price is good (expensive) OR the excise duty is good (high) OR the exchange rate is poor, THEN the final price will be high.
* Usage
   - If I tell this controller that:
      * the global price is 9.8, and
      * the excise duty is 6.5, and
      * the exchange rate is 1
   - it would predict the diesel price would be:
      * ? $.


Creating the Tipping Controller Using the skfuzzy control API
-------------------------------------------------------------

We can use the `skfuzzy` control system API to model this.  First, let's
define fuzzy variables
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# CURRENT AVERAGE PRICE OF DIESEL AROUND THE WORLD: 1.30$

global_price = ctrl.Antecedent(np.arange(0., 10.01, 0.01), 'global_price') # in $
excise_duty = ctrl.Antecedent(np.arange(0., 10.01, 0.01), 'excise_duty') # in %
exchange_rate = ctrl.Antecedent(np.arange(0., 10.01, 0.01), 'exchange_rate') # $ to zl
final_price = ctrl.Consequent(np.arange(0., 10.01, 0.01), 'final_price')

# Auto-membership function population is possible with .automf(3, 5, or 7)
global_price.automf(3)
excise_duty.automf(3)
exchange_rate.automf(3)

final_price['low'] = fuzz.trimf(final_price.universe, [0., 0., 5.])
final_price['medium'] = fuzz.trimf(final_price.universe, [0., 5., 10.])
final_price['high'] = fuzz.trimf(final_price.universe, [5., 10., 10.])

# #global_price['average'].view()
# excise_duty.view()
# exchange_rate.view()
# final_price.view()

"""
1. If the global price is low (not expensive) OR the excise_duty is low (low) OR the exchange rate is good, then the final price will be low
2. If the global price is average, then the final price will be medium
3. If the global price is good (expensive) OR the excise duty is good (high) OR the exchange rate is poor, then the final price will be high.

Most people would agree on these rules, but the rules are fuzzy. Mapping the
imprecise rules into a defined, actionable tip is a challenge. This is the
kind of task at which fuzzy logic excels.
"""

rule1 = ctrl.Rule(global_price['good'] | excise_duty['good'] | exchange_rate['poor'], final_price['high'])
rule2 = ctrl.Rule(global_price['average'], final_price['medium'])
rule3 = ctrl.Rule(excise_duty['poor'] | global_price['poor'] | exchange_rate['good'], final_price['low'])

rule1.view()

oil_purchase_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

oil_purchase = ctrl.ControlSystemSimulation(oil_purchase_ctrl)

oil_purchase.input['global_price'] = 10
oil_purchase.input['excise_duty'] = 10
oil_purchase.input['exchange_rate'] = 1

oil_purchase.compute()

print(oil_purchase.output['final_price'])
final_price.view(sim=oil_purchase)
