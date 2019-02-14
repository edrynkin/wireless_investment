import numpy as np
import single_type_investment as sti
import pandas as pd

zipdata = pd.read_csv("zipdata.csv")
states = pd.unique(zipdata['state'])
M = len(states)
markets_list = []
for m in range(M):
    zipcodes = []
    subdata = zipdata.loc[zipdata.state == states[m]]
    Z = subdata.shape[0]
    for z in range(Z):
        zipcode = sti.Zipcode(subdata['TRI'].iloc[z], subdata['Total.population'].iloc[z])
        zipcodes.append(zipcode)
    markets_list.append(sti.Market(zipcodes))
markets = sti.Markets(markets_list)

T = 50
J = 3
agg_data = pd.read_csv("agg_data_minimum.csv")
dynamic_pars = {'alpha_0': np.array([[5.16,  3.42, 10.41]]), 'alpha_tri': 0.08, 'alpha_pop': 0.90, 'sigma': 0.26}
costs_list = []
encoding = {'ATT': 0, 'STM': 1, 'Verizon': 2}
agg_data.carrier = [encoding[item] for item in agg_data.carrier]
agg_data.year = agg_data.year - 2009
Tmax = max(agg_data.year)
for t in range(T):
    mc = np.array(agg_data.mc.loc[agg_data.year == min(t, Tmax)]).reshape(1,-1)
    static_pars = {'static_mc' : mc}
    cost = sti.Cost(static_pars, dynamic_pars)
    costs_list.append(cost)
costs = sti.Costs(costs_list)

alpha = 0.046
beta = 1.98
xi_data = pd.read_csv("xi_data_minimum.csv")
xi = np.zeros((M, J))
demands_list = []
xi_data.carrier = [encoding[item] for item in xi_data.carrier]
xi_data.year = xi_data.year - 2009
for t in range(T):
    subdata = xi_data.loc[xi_data.year == min(t, Tmax)].pivot(index='state',columns='carrier',values='xi')
    for m in range(M):
        xi[m,] = np.array(subdata.loc[states[m]])
    demand = sti.Demand(alpha, beta, xi, delta_q = 0.16)
    demands_list.append(demand)
demands = sti.Demands(demands_list)

model = sti.Model(markets, demands, costs, [np.zeros((M, J)) for t in range(T)])
model.find_eqm(verbose=True, learning_rate=0.2)