## Load Dependencies for POMDPs
@everywhere using POMDPs
@everywhere using DiscreteValueIteration
@everywhere using POMDPModelTools
@everywhere using POMDPPolicies
@everywhere using GridInterpolations
@everywhere using LocalFunctionApproximation
@everywhere using LocalApproximationValueIteration
@everywhere using Random
@everywhere using StaticArrays
## Load Dependencies for functions
using PGFPlots
using Colors
using DelimitedFiles
using FileIO
using Geodesy
using HDF5
using Interact
using JLD
using JLD2
using NearestNeighbors
using Parameters
using PGFPlots
## For Loading Files from JLD2
Core.eval(Main, :(using Distances))
Core.eval(Main, :(using NearestNeighbors))
Core.eval(Main, :(using Random))
Core.eval(Main, :(using LocalApproximationValueIteration))
Core.eval(Main, :(using LocalFunctionApproximation))
Core.eval(Main, :(using DiscreteValueIteration))
## function to include
include("../src/helperFunctions.jl")
include("../src/setupMDP.jl")

## for two waypoints (coarse), find locations of the aircraft at least every time step 
## until a piece of debris is encountered. return true when within threshold of a 
## piece of debris. If never within the threshold of a piece of debris return false and the final
## aircraft location and time.
function array_seg_locations_with_deb(seg_start::Int64, seg_start_time::Float64, sim_mdp::MeterMDP, 
    time_step::Float64, speed::Float64, anom::Float64, launch::Float64)
    debris_encounter = false
    leg_distance = segment_distance(sim_mdp.path[seg_start], sim_mdp.path[seg_start+1])
    leg_time = leg_distance/speed
    at_time_step = time_step-seg_start_time%time_step
    if at_time_step == time_step
        at_time_step = 0.
    end
    for travel_time = at_time_step:time_step:leg_time
        location = update_point(sim_mdp, seg_start, seg_start+1, speed, travel_time)
        debris_time = travel_time+seg_start_time-launch
    	# only care about debris after an anomaly could have occured
    	if debris_time > anom && !debris_encounter
        	debris_encounter = bool_active_debris(anom, debris_time, sim_mdp.debris_time_threshold, location, sim_mdp.debris_dist_threshold, sim_mdp.debris.debris_dict, sim_mdp.debris.debris_time_KD_tree)
    	end
    end
    return debris_encounter, seg_start_time+leg_time
end

## for two waypoints (coarse), find locations of the aircraft at least every time step 
## until a piece of debris is encountered. return true when within threshold of a 
## piece of debris. If never within the threshold of a piece of debris return false and the final
## aircraft location and time.
function array_seg_locations_with_deb(start_location::Tuple{Float64,Float64}, 
    end_location::Tuple{Float64,Float64}, seg_start_time::Float64, sim_mdp::MeterMDP, 
    time_step::Float64, speed::Float64, anom::Float64, launch::Float64)
    debris_encounter = false
    leg_distance = segment_distance(start_location, end_location)
    leg_time = leg_distance/speed
    at_time_step = time_step-seg_start_time%time_step
    if at_time_step == time_step
        at_time_step = 0.
    end
    for travel_time = at_time_step:time_step:leg_time
        location = update_point(start_location, end_location, speed, travel_time)
        debris_time = travel_time+seg_start_time-launch
        # only care about debris after an anomaly could have occured
        if debris_time > anom && !debris_encounter
            debris_encounter = bool_active_debris(anom, debris_time, sim_mdp.debris_time_threshold, location, sim_mdp.debris_dist_threshold, sim_mdp.debris.debris_dict, sim_mdp.debris.debris_time_KD_tree)
            if debris_encounter
                return debris_encounter, seg_start_time+leg_time
            end
        end
    end
    return debris_encounter, seg_start_time+leg_time
end

## update the time after a segment leg
function update_seg_time(seg_start::Int64, seg_start_time::Float64, sim_mdp::MeterMDP, 
    speed::Float64)
    leg_distance = segment_distance(sim_mdp.path[seg_start], sim_mdp.path[seg_start+1])
    leg_time = leg_distance/speed
    return seg_start_time+leg_time
end

## update the time after a segment leg
function update_seg_time(point1::Tuple{Float64,Float64}, point2::Tuple{Float64,Float64}, 
    seg_start_time::Float64, speed::Float64)
    leg_distance = segment_distance(point1, point2)
    leg_time = leg_distance/speed
    return seg_start_time+leg_time
end

## simulate a nominal varient path - record the time for metrics 
function sim_single_path_var_nominal(new_es::Array{Float64,1}, new_ns::Array{Float64,1}, 
    wps::Array{Int64,1}, speed::Float64)
    seg_start_time = 0.
    # setup the path at this speed
    for i = 1:length(wps)-1
        for j = wps[i]:wps[i+1]-1
            start_location = (new_es[j], new_ns[j])
            end_location = (new_es[j+1], new_ns[j+1])
            seg_start_time = update_seg_time(start_location, end_location, seg_start_time, speed)
        end
    end
    return seg_start_time # how long does it take with no adjustments?
end

## how to break a tie (no speed change, small speed change, large speed change)
function policy_tie_breaker(qs)
    q_max = qs[3] # defualt action = 0
    q_i = 3 # default action = 0
    check_index_order = [2, 4, 1, 5]
    for i in check_index_order
        if qs[i] > q_max
            q_max = qs[i]
            q_i = i
        end
    end
    return q_i
end

## during post processing, used to find the optimal action at a given state when there is
## more than one policy to consider
function multi_policy_action(state::MeterState, policies)
    acts = actions(policies[1].mdp)
    n_acts = length(actions(policies[1].mdp))
    n_policies = length(policies)
    qs = ones(n_acts)*Inf
    ## want optimal, so start with min and find max
    for i = 1:n_acts
        for j = 1:n_policies
            if state.t_launch in policies[j].mdp.ts_launch
                if (state.pt, state.t_flight) in policies[j].mdp.ind_ts_flight # new change
                    q_now = value(policies[j], state, acts[i])
                    if q_now < qs[i]
                        qs[i] = q_now
                    end
                end
            end
        end
    end
    if length(unique(qs)) < length(qs)
        acts_i = policy_tie_breaker(qs)
        return acts[acts_i]
    else
        return acts[argmax(qs)]
    end
end

## get the action when there is a tie
function action_with_tie(policy::LocalApproximationValueIterationPolicy, s)
    mdp = policy.mdp
    best_a_idx = -1
    max_util = -Inf
    # order actions in the desired tie braking order
    sub_aspace = [:constant, :slower, :faster, :slowest, :fastest]
    discount_factor = discount(mdp)
    u = 0.0
    for a in sub_aspace
        
        iaction = actionindex(mdp, a)

        if s.t_flight < mdp.point_start_times[s.pt]
            new_s = MeterState(s.pt, mdp.point_start_times[s.pt], s.dt, s.t_launch, s.t_anom)
            # s.t_flight = mdp.point_start_times[s.pt]
            u = value(policy, new_s, a)
        elseif s.t_flight > mdp.point_end_times[s.pt]
            new_s = MeterState(s.pt, mdp.point_end_times[s.pt], s.dt, s.t_launch, s.t_anom)
            # s.t_flight = mdp.point_end_times[s.pt]
            u = value(policy, new_s, a)
        else
            u = value(policy, s, a)
        end

        if u > max_util
            max_util = u
            best_a_idx = iaction
        end
    end

    return policy.action_map[best_a_idx]
end

## simulate a single path varient
function sim_single_path_var_response(new_es::Array{Float64,1}, new_ns::Array{Float64,1}, 
    wps::Array{Int64,1}, anom_idx::Int, anom::Float64, launch::Float64, speed::Float64, 
    policies::Array{LocalApproximationValueIterationPolicy{LocalGIFunctionApproximator{RectangleGrid{4}},MersenneTwister},1}, 
    mdp::MeterMDP, sim_time_step::Float64, sim_pilot_response::Float64)
    seg_start_time = 0.
    # setup empty metrics
    path_metered = false
    # setup the path at this speed
    for i = 1:mdp.path_length-1
        if seg_start_time < anom # dont see an anomaly until it occurs
            s = MeterState(i, seg_start_time, speed, launch, -1.)
            a = multi_policy_action(s, policies)
        else
            s = MeterState(i, seg_start_time, speed, launch, anom)
            a = action_with_tie(policies[anom_idx], s)
        end
        if a != :constant
            path_metered = true
        end

        #if rand(1)[1] < sim_pilot_response
        speed += mdp.action_deltas[a]

        #end
        for j = wps[i]:wps[i+1]-1
            start_location = (new_es[j], new_ns[j])
            end_location = (new_es[j+1], new_ns[j+1])
            seg_debris, seg_end_time = array_seg_locations_with_deb(start_location, end_location, 
                seg_start_time, mdp, sim_time_step, speed, anom, launch)
            if seg_debris
                return true, path_metered, seg_end_time # interact with debris, was it metered
            end
            seg_start_time = seg_end_time
        end
    end
    return false, path_metered, seg_start_time # no debris interaction, was it metered, time
end

## during heuristic simulation, check if provided speed has path that intersects
## debris safety threshold
function heuristic_var_check_speed(wp_start::Int, wp_end::Int, new_es::Array{Float64,1}, 
    new_ns::Array{Float64,1}, seg_start_time::Float64, mdp::MeterMDP, sim_time_step::Float64, 
    speed::Float64, anom::Float64, launch::Float64)
    seg_debris = false
    for i = wp_start:wp_end-1
        start_location = (new_es[i], new_ns[i])
        end_location = (new_es[i+1], new_ns[i+1])
        seg_debris, seg_end_time = array_seg_locations_with_deb(start_location, end_location, 
            seg_start_time, mdp, sim_time_step, speed, anom, launch)
        if seg_debris
            return seg_debris, seg_end_time
        end
        seg_start_time = seg_end_time
    end
    return seg_debris, seg_start_time # no debris - or would have already been returned
end

## simulate a single path with the hueristic
function sim_single_path_var_heuristic(new_es::Array{Float64,1}, new_ns::Array{Float64,1}, 
    wps::Array{Int64,1}, anom_id::Int, anom::Float64, launch::Float64, speed::Float64, 
    mdp::MeterMDP, sim_time_step::Float64, sim_pilot_response::Float64)

    seg_start_time = 0.
    # setup empty metrics
    path_metered = false

    for i = 1:mdp.path_length-1
        if seg_start_time < anom 
            for j = wps[i]:wps[i+1]-1
                start_location = (new_es[j], new_ns[j])
                end_location = (new_es[j+1], new_ns[j+1])
                seg_debris, seg_end_time = array_seg_locations_with_deb(start_location, end_location, 
                    seg_start_time, mdp, sim_time_step, speed, anom, launch)
                if seg_debris
                    return true, path_metered, seg_start_time # interact with debris, was it metered
                end
                seg_start_time = seg_end_time
            end
        else
            debris_reroute = false
            seg_start_t = seg_start_time

            for j = wps[i]:wps[i+1]-1
                start_location = (new_es[j], new_ns[j])
                end_location = (new_es[j+1], new_ns[j+1])
                seg_debris, seg_end_time = array_seg_locations_with_deb(start_location, end_location, 
                    seg_start_t, mdp, sim_time_step, speed, anom, launch)
                if seg_debris
                    debris_reroute = true
                    break
                else
                    seg_start_t = seg_end_time
                end
            end
            if !debris_reroute 
                seg_start_time = seg_start_t
            else
                if speed+mdp.speed_step <= mdp.max_speed
                    seg_debris, seg_end_time = heuristic_var_check_speed(wps[i], wps[i+1], new_es, 
                            new_ns, seg_start_time, mdp, sim_time_step, speed+mdp.speed_step, anom, launch)
                    if !seg_debris
                        seg_start_time = seg_end_time
                        path_metered = true
                        speed += mdp.speed_step
                        debris_reroute = false
                    end
                end
                if debris_reroute && speed-mdp.speed_step >= mdp.min_speed
                    seg_debris, seg_end_time = heuristic_var_check_speed(wps[i], wps[i+1], new_es, 
                        new_ns, seg_start_time, mdp, sim_time_step, speed-mdp.speed_step, anom, launch)
                    if !seg_debris
                        seg_start_time = seg_end_time
                        path_metered = true
                        speed += -mdp.speed_step
                        debris_reroute = false
                    end
                end
                if debris_reroute && speed+2*mdp.speed_step <= mdp.max_speed
                    seg_debris, seg_end_time = heuristic_var_check_speed(wps[i], wps[i+1], new_es, 
                        new_ns, seg_start_time, mdp, sim_time_step, speed+2*mdp.speed_step, anom, launch)
                    if !seg_debris
                        seg_start_time = seg_end_time
                        path_metered = true
                        speed += 2*mdp.speed_step
                        debris_reroute = false
                    end
                end
                if debris_reroute && speed-2*mdp.speed_step >= mdp.min_speed
                    seg_debris, seg_end_time = heuristic_var_check_speed(wps[i], wps[i+1], new_es, 
                        new_ns, seg_start_time, mdp, sim_time_step, speed-2*mdp.speed_step, anom, launch)
                    if !seg_debris
                        seg_start_time = seg_end_time
                        path_metered = true
                        speed += -2*mdp.speed_step
                        debris_reroute = false
                    end
                end
                if debris_reroute
                    return true, path_metered, seg_start_time
                end
            end
        end
    end
    return false, path_metered, seg_start_time
end

## simulate a path when there is no anomaly
function sim_single_path_var_no_anom(new_es::Array{Float64,1}, new_ns::Array{Float64,1}, 
    wps::Array{Int64,1}, anom_idx::Int, anom::Float64, launch::Float64, speed::Float64, 
    policies::Array{LocalApproximationValueIterationPolicy{LocalGIFunctionApproximator{RectangleGrid{4}},MersenneTwister},1}, 
    mdp::MeterMDP, sim_time_step::Float64, sim_pilot_response::Float64)
    seg_start_time = 0.
    # setup empty metrics
    path_metered = false
    # setup the path at this speed
    for i = 1:mdp.path_length-1
        s = MeterState(i, seg_start_time, speed, launch, -1.)
        a = multi_policy_action(s, policies)
        if a != :constant
            path_metered = true
        end
        speed += mdp.action_deltas[a]

        for j = wps[i]:wps[i+1]-1
            start_location = (new_es[j], new_ns[j])
            end_location = (new_es[j+1], new_ns[j+1])
            seg_debris, seg_end_time = array_seg_locations_with_deb(start_location, end_location, 
                seg_start_time, mdp, sim_time_step, speed, anom, launch)
            seg_start_time = seg_end_time
        end

    end
    return path_metered, seg_start_time # was it metered, time
end

