import numpy as np
import single_type_investment as sti

Z = 10
tri = 2
pop = 3
M = 5
markets_list = []
for m in range(M):
    zipcodes = []
    for z in range(Z):
        zipcodes.append(sti.Zipcode(tri, pop))
    markets_list.append(sti.Market(zipcodes))
markets = sti.Markets(markets_list)

T = 20
J = 4
static_pars = {'static_mc' : np.array([[10]*J])}
dynamic_pars = {'alpha_0': np.array([[10]*J]), 'alpha_tri': 0.05, 'alpha_pop': 0.95, 'sigma': 0.2}
cost = sti.Cost(static_pars, dynamic_pars)
costs = sti.Costs([cost for t in range(T)])

alpha = 0.045
beta = 2
xi = np.zeros((M, J))
demand = sti.Demand(alpha, beta, xi)
demands = sti.Demands([demand for t in range(T)])

model = sti.Model(markets, demands, costs, [np.zeros((M, J)) for t in range(T)])
model.find_eqm()