N_PROCS = 10

using Distributed 
addprocs(N_PROCS-1) 
@everywhere __PARALLEL__ = true   

@everywhere using GridInterpolations
@everywhere using LocalApproximationValueIteration
@everywhere using LocalFunctionApproximation
@everywhere using DiscreteValueIteration
@everywhere using Random
@everywhere using NearestNeighbors
@everywhere using Distances

## For Loading Files from JLD2
@everywhere Core.eval(Main, :(using Distances))
@everywhere Core.eval(Main, :(using NearestNeighbors))
@everywhere Core.eval(Main, :(using Random))
@everywhere Core.eval(Main, :(using LocalApproximationValueIteration))
@everywhere Core.eval(Main, :(using LocalFunctionApproximation))
@everywhere Core.eval(Main, :(using DiscreteValueIteration))
## function to include
@everywhere include("../src/helperFunctions.jl")
@everywhere include("../src/setupMDP.jl")
@everywhere include("../src/simFunctions.jl")
@everywhere include("../src/genPathFunctions.jl")
## helpers to include while profiling
@everywhere waypoints = FileIO.load("../assets/waypoints.jld2"); # enu from cape canaveral

# simulation speeds
@everywhere speeds = [245.0, 247.5, 250.0, 252.5, 255.0]

# times for 50 randomly generated paths
@everywhere time_dicts = [
	Dict(100.0=>[5675.0, 6185.0],110.0=>[5440.0, 5825.0],90.0=>[5845.0, 6325.0]),
	Dict(100.0=>[9465.0, 10100.0],110.0=>[9265.0, 9815.0]),
	Dict(70.0=>[2945.0, 3195.0],90.0=>[2695.0, 2880.0]),
	Dict(170.0=>[425.0, 1720.0],160.0=>[1195.0, 1695.0],180.0=>[50.0, 1660.0],190.0=>[100.0, 710.0]),
	Dict(520.0=>[2135.0, 4065.0],530.0=>[2190.0, 3575.0],470.0=>[2670.0, 5215.0],420.0=>[5140.0, 5600.0],510.0=>[2175.0, 4410.0],430.0=>[4495.0, 5585.0],500.0=>[2265.0, 4745.0],460.0=>[2805.0, 5375.0],440.0=>[3460.0, 5505.0],450.0=>[2960.0, 5450.0],490.0=>[2370.0, 4945.0],480.0=>[2500.0, 5125.0]),
	Dict(50.0=>[4695.0, 4980.0],60.0=>[4665.0, 4950.0],70.0=>[4450.0, 4810.0]),
	Dict(310.0=>[10370.0, 11540.0],320.0=>[9515.0, 11645.0]),
	Dict(310.0=>[7840.0, 8290.0],290.0=>[8755.0, 9625.0],300.0=>[7665.0, 9710.0]),
	Dict(120.0=>[3460.0, 3950.0],140.0=>[2460.0, 2780.0],130.0=>[3035.0, 3470.0]),
	Dict(100.0=>[2490.0, 3100.0],120.0=>[1405.0, 1865.0],130.0=>[730.0, 915.0],110.0=>[1610.0, 2780.0]),
	Dict(100.0=>[1025.0, 1310.0],110.0=>[825.0, 1060.0]),
	Dict(120.0=>[2435.0, 2780.0],130.0=>[1655.0, 1840.0],110.0=>[2810.0, 3215.0]),
	Dict(320.0=>[10120.0, 12025.0],360.0=>[9445.0, 10805.0],330.0=>[9860.0, 11975.0]),
	Dict(160.0=>[9600.0, 11710.0],170.0=>[9635.0, 11630.0],180.0=>[9730.0, 10745.0],150.0=>[10640.0, 11655.0]),
	Dict(170.0=>[2600.0, 3750.0],160.0=>[3210.0, 3630.0],180.0=>[2000.0, 3785.0],190.0=>[2050.0, 3265.0]),
	Dict(240.0=>[1625.0, 2800.0],250.0=>[1040.0, 2735.0]),
	Dict(310.0=>[13700.0, 15010.0],320.0=>[12850.0, 15115.0]),
	Dict(200.0=>[1280.0, 2005.0],170.0=>[2290.0, 2795.0],180.0=>[1505.0, 2935.0],190.0=>[1150.0, 2905.0]),
	Dict(120.0=>[1955.0, 2310.0],110.0=>[2390.0, 2960.0]),
	Dict(520.0=>[1305.0, 3990.0],530.0=>[1125.0, 3835.0],470.0=>[2735.0, 4770.0],560.0=>[685.0, 2995.0],540.0=>[975.0, 3605.0],550.0=>[830.0, 3320.0],510.0=>[1430.0, 4170.0],580.0=>[875.0, 1895.0],500.0=>[1695.0, 4275.0],460.0=>[3205.0, 4815.0],440.0=>[4110.0, 5045.0],590.0=>[465.0, 1315.0],570.0=>[575.0, 2500.0],450.0=>[3870.0, 4930.0],490.0=>[1915.0, 4545.0],480.0=>[2525.0, 4590.0]),
	Dict(100.0=>[2625.0, 2935.0],110.0=>[2400.0, 2715.0]),
	Dict(80.0=>[1515.0, 1665.0]),
	Dict(240.0=>[9950.0, 10645.0],250.0=>[9055.0, 10725.0],260.0=>[8695.0, 10435.0]),
	Dict(160.0=>[6465.0, 7855.0],140.0=>[7245.0, 8365.0],130.0=>[7660.0, 8300.0],150.0=>[6430.0, 8350.0]),
	Dict(100.0=>[1500.0, 1855.0],120.0=>[710.0, 885.0],110.0=>[1375.0, 1565.0]),
	Dict(60.0=>[20500.0, 21455.0],80.0=>[19690.0, 21025.0],70.0=>[20410.0, 21355.0]),
	Dict(220.0=>[8805.0, 10845.0],210.0=>[9625.0, 10845.0],230.0=>[8865.0, 10130.0]),
	Dict(220.0=>[9205.0, 10265.0],230.0=>[8260.0, 10270.0],240.0=>[8300.0, 9585.0]),
	Dict(310.0=>[5250.0, 6075.0],320.0=>[4200.0, 6235.0]),
	Dict(120.0=>[5270.0, 5585.0],110.0=>[5705.0, 6230.0]),
	Dict(100.0=>[3560.0, 3895.0],110.0=>[3280.0, 3595.0],90.0=>[3680.0, 4070.0]),
	Dict(330.0=>[9350.0, 10535.0],360.0=>[8870.0, 11385.0],350.0=>[7495.0, 10105.0],340.0=>[7830.0, 10440.0]),
	Dict(130.0=>[2010.0, 2265.0],110.0=>[3060.0, 3570.0]),
	Dict(120.0=>[6545.0, 6970.0],90.0=>[7760.0, 8315.0]),
	Dict(100.0=>[4965.0, 5370.0],90.0=>[5210.0, 5485.0]),
	Dict(100.0=>[1070.0, 1325.0]),
	Dict(160.0=>[4245.0, 6140.0],170.0=>[4280.0, 6060.0],180.0=>[4375.0, 5175.0],150.0=>[5290.0, 6085.0]),
	Dict(120.0=>[1570.0, 1830.0],110.0=>[2015.0, 2465.0]),
	Dict(200.0=>[4290.0, 6525.0],100.0=>[10110.0, 10700.0],140.0=>[6060.0, 7880.0],210.0=>[4170.0, 6310.0],190.0=>[4465.0, 6750.0],170.0=>[5540.0, 6900.0],160.0=>[6485.0, 6910.0],180.0=>[4705.0, 6905.0],130.0=>[6860.0, 8180.0],150.0=>[6120.0, 7065.0],220.0=>[4270.0, 5290.0],110.0=>[9855.0, 10390.0],90.0=>[10200.0, 10910.0],120.0=>[6290.0, 10700.0]),
	Dict(100.0=>[4280.0, 4665.0],110.0=>[4060.0, 4435.0]),
	Dict(100.0=>[750.0, 1060.0],110.0=>[515.0, 695.0],90.0=>[920.0, 1200.0]),
	Dict(200.0=>[1645.0, 3395.0],210.0=>[1660.0, 3230.0],190.0=>[2870.0, 3380.0]),
	Dict(230.0=>[2275.0, 2875.0],240.0=>[1100.0, 2860.0],250.0=>[1130.0, 2295.0]),
	Dict(130.0=>[3905.0, 4320.0],110.0=>[5080.0, 5650.0]),
	Dict(120.0=>[6940.0, 7535.0],130.0=>[6400.0, 6980.0]),
	Dict(100.0=>[7565.0, 8085.0],110.0=>[7375.0, 7880.0]),
	Dict(310.0=>[10060.0, 11830.0],290.0=>[10475.0, 12940.0],300.0=>[10130.0, 12725.0],280.0=>[11940.0, 12970.0]),
	Dict(120.0=>[3545.0, 3910.0],110.0=>[4010.0, 4475.0]),
	Dict(230.0=>[1495.0, 1635.0],240.0=>[465.0, 1765.0],250.0=>[100.0, 1580.0]),
	Dict(60.0=>[4770.0, 5065.0],80.0=>[4570.0, 4915.0],70.0=>[4615.0, 5005.0],90.0=>[3715.0, 4755.0])
]

# enter lambda value
@everywhere lambda_value = 0.00005 
@everywhere lambda_string = "00005" 

# function to load policies, setup path and mdp 
@everywhere function get_policies(anom_times::Array{Float64,1}, path_num::String, lambda_value::Float64)
	policies = Array{LocalApproximationValueIterationPolicy{LocalGIFunctionApproximator{RectangleGrid{4}},MersenneTwister},1}(undef, length(anom_times))
	for i = 1:length(anom_times)
	    policies[i] = load_policy("path_"*path_num*"_"*lambda_string*"_"*string(anom_times[i])*"_policy")
	end

	sample_policy = policies[1]
	path = sample_policy.mdp.path_string
	locations = path_locations(path, waypoints)
	path_es, path_ns = path_locations_to_es_ns(locations)
	mdp = MeterMDP(path_string=sample_policy.mdp.path_string, path=sample_policy.mdp.path, ts_launch=sample_policy.mdp.ts_launch, 
	       p_follow_action=sample_policy.mdp.p_follow_action, ts_anom=sample_policy.mdp.ts_anom, lambda=sample_policy.mdp.lambda, n_debris_profiles=50)
	return policies, path_es, path_ns, mdp
end

# figure out number of launch time and speed combos
@everywhere function count_combos(launch_times, speeds)
	possible_combos = 0
	for launch_array in launch_times
    	possible_combos += length(launch_array)
	end
	return possible_combos * length(speeds)
end

# write results to a file 
@everywhere function write_results(path_num, lambda_string, anom_times, sims_t_anom, unsafe_by_t_anom, metering_by_t_anom, delta_time, h_unsafe_by_t_anom, h_metering_by_t_anom, na_metering, na_delta_time)
	open("results_"*path_num*"_"*lambda_string*"_.txt", "w") do f
    	write(f, "$path_num\n")
    	write(f, "$anom_times\n")
    	write(f, "$sims_t_anom\n")
    	write(f, "$unsafe_by_t_anom\n")
    	write(f, "$metering_by_t_anom\n")
    	write(f, "$delta_time\n")
    	write(f, "$h_unsafe_by_t_anom\n")
    	write(f, "$h_metering_by_t_anom\n")
    	write(f, "$na_metering\n")
    	write(f, "$na_delta_time\n")
	end
end

# function to run simulations
@everywhere function run_simulation(path_num, lambda_string, possible_combos, anom_times, launch_times, speeds, path_es, path_ns, policies, mdp)
	# set seed so repeatable
	Random.seed!(1234)
	# setup empty metrics
	sims_t_anom = zeros(length(anom_times))
	delta_time = zeros(length(anom_times))
	unsafe_by_t_anom = zeros(length(anom_times))
	metering_by_t_anom = zeros(length(anom_times))
	h_delta_time = zeros(length(anom_times))
	h_unsafe_by_t_anom = zeros(length(anom_times))
	h_metering_by_t_anom = zeros(length(anom_times))
	na_metering = 0
	na_delta_time = 0
	println(possible_combos*2)
	# run for 2x possible combos
	for i = 1:possible_combos*2
		if i%100 == 0 || i == 1
			println(i, " out of ", possible_combos*2)
		end
	    anom_idxs = collect(1:length(anom_times))
	    # select random time of anomaly
	    anom_idx = rand(anom_idxs)
	    sims_t_anom[anom_idx] += 1
	    anom_t = anom_times[anom_idx]
	    # select random time of launch
	    launch_t = rand(launch_times[anom_idx])
	    # select random initial speed
	    speed = rand(speeds)
	    # generate path varient
	    new_es, new_ns, wps = generate_path_varient(path_es, path_ns)
	    # nominal simulation
	    nominal_time = sim_single_path_var_nominal(new_es, new_ns, wps, speed)
	    # anomaly mdp simulation
	    unsafe, metered, mdp_time = sim_single_path_var_response(new_es, new_ns, wps, anom_idx, anom_t, launch_t, speed, policies, mdp, 1., 1.)
	    # update safety metric
	    if unsafe
	        unsafe_by_t_anom[anom_idx] += 1
	    end
	    # update metered metric
	    if metered
	        metering_by_t_anom[anom_idx] += 1
	    end
	    # update time metric
	    if !unsafe
	    	delta_time[anom_idx] += abs(nominal_time-mdp_time)
	    end
		# no anomaly mdp simulation 
    	metered, no_anom_time = sim_single_path_var_no_anom(new_es, new_ns, wps, anom_idx, anom_t, launch_t, speed, policies, mdp, 1., 1.)
	    # udpated metered metric
	    if metered
	    	na_metering += 1
	    end
	    # update time metric
	    na_delta_time += abs(nominal_time-no_anom_time)
	    # heuristic simulation
	    unsafe, metered, h_time = sim_single_path_var_heuristic(new_es, new_ns, wps, anom_idx, anom_t, launch_t, speed, mdp, 1., 1.)
    	# update safety metric
    	if unsafe
        	h_unsafe_by_t_anom[anom_idx] += 1
    	end
    	# updated metered metric
    	if metered
        	h_metering_by_t_anom[anom_idx] += 1
    	end
    	# update time metric
		if !unsafe
			h_delta_time[anom_idx] += abs(nominal_time-h_time)
        end
	end
	# process time metrics
	for j = 1:length(anom_times)
		if sims_t_anom[j] > 0
			delta_time[j] = delta_time[j]/(sims_t_anom[j]-unsafe_by_t_anom[j])
			h_delta_time[j] = h_delta_time[j]/(sims_t_anom[j]-h_unsafe_by_t_anom[j])
		end
	end
	# update the no anomaly time
	na_delta_time = na_delta_time/possible_combos*2
	# write results
	write_results(path_num, lambda_string, anom_times, sims_t_anom, unsafe_by_t_anom, metering_by_t_anom, delta_time, h_unsafe_by_t_anom, h_metering_by_t_anom, na_metering, na_delta_time)
end

# path 1-50
path_nums = [1]

# run simulations in parallel
a = @distributed for path in path_nums
	# setup the params for this path
	path_num = string(path)
	cur_dict = time_dicts[path]
	anom_times = collect(keys(cur_dict))
	launch_times = Array{Array{Float64,1},1}()
	for at in anom_times
		push!(launch_times, collect(cur_dict[at][1]:5.0:cur_dict[at][2]))
	end

	# calculate number of combos
	possible_combos = count_combos(launch_times, speeds)
	# load policies, setup path and mdp 
	policies, path_es, path_ns, mdp = get_policies(anom_times, path_num, lambda_value)
	# run the simulations
	run_simulation(path_num, lambda_string, possible_combos, anom_times, launch_times, speeds, path_es, path_ns, policies, mdp)
end

# get the parallel results
fetch(a)
