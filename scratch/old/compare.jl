using ACEhamiltonians.Structure: Params





data_info = Data([path], [(1, 2)])

r_cut = repeat([20.0], 9)
max_deg = repeat([8], 9)
order = repeat([2], 9)
λ = repeat([1e-7], 9)
reg_type = 2
parameters = Params(r_cut, max_deg, order, λ, reg_type, "LSQR")

MWH, MWS, data_whole = params2wmodels(data_info, parameters, parameters)