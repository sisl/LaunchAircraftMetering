using POMDPs
using DiscreteValueIteration
using POMDPModelTools
using POMDPPolicies
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using Random
using StaticArrays
## Load Dependencies for MY functions
using DelimitedFiles
using Distributions
using FileIO
using Geodesy
using HDF5
using Interact
using JLD
using JLD2
using NearestNeighbors
using Parameters

include("helperFunctions.jl")

# STATE SPACE definition
struct MeterState
    pt::Int            # way-point
    t_flight::Float64  # time into flight
    dt::Float64        # current speed [m/s]
    t_launch::Float64  # time into flight launch starts
    t_anom::Float64    # time anomaly occurs after launch starts
end

# setup MDP struct
@with_kw struct MeterMDP <: MDP{MeterState, Symbol}
    # state space information
    # string of way-points
    path_string::Vector{String}
    # Array of way-point location tuples     
    path::Array{Tuple{Float64,Float64}}
    # number of way-points
    path_length::Int = length(path)
    # [m] total distance of path
    path_distance::Float64 = path_distance(path)
    
    # action space information
    min_speed::Float64 = 245. # [m/s]
    speed_step::Float64 = 2.5 # [m/s]
    max_speed::Float64 = 255. # [m/s]
    dts::Vector{Float64} = collect(min_speed:speed_step:max_speed)  # [m/2]
    
    # action information
    action_deltas::Dict{Symbol,Real} = Dict([(:fastest, 2*speed_step), (:faster, speed_step), 
                    (:constant, 0), (:slower, -1.0*speed_step), (:slowest, -2.0*speed_step)])

    # reward function information
    # scaling between velocity rewards and debris rewards
    lambda::Float64 = 500.0
    # speed rewards
    # reward for going faster or slower
    r_er_change::Float64 = -0.5
    # reward for going fastest of slowest
    r_est_change::Float64 = -1.0
    # reward for going faster or slower than aircraft limits
    r_speed_limits::Float64 = -Inf
    # debris rewards 
    # reward for being within the threshold of debris
    r_within_debris::Float64 = -1.0
    # threshold allowed between debris and plane 
    debris_dist_threshold::Float64 = 152.0*10  
    
    # debris information
    # number of debris profiles
    n_debris_profiles = 25
    # debris
    debris::Debris = Debris(n_debris_profiles, LaunchSiteInfo())
    # time around when debris passes it is active
    debris_time_threshold::Float64 = 20.0
    # discretization of anomalies from debris modeling  
    debris_anomaly_step::Float64 = 10.0
    # time of anomaly around time of anomaly active
    debris_anomoly_active::Float64 = 10.0
    
    # time information 
    # time increments
    time_step::Float64 = 5.0 # [s] 
    max_time::Float64 = path_distance/min_speed  # [m]
    # reachable set start, in 
    point_start_times::Vector{Float64} = # [s]
        time_setup(path, path_length, time_step, max_speed, min_speed)[1]
    # reachable set end, in 
    point_end_times::Vector{Float64} = # [s]
        time_setup(path, path_length, time_step, max_speed, min_speed)[2]
    # length of reachable set in time_steps
    point_time_length::Vector{Float64} =
        time_setup(path, path_length, time_step, max_speed, min_speed)[3]
    # times launch can occur during flight
    ts_launch::Vector{Float64} = collect() # [s]
    # times after launch an anomaly can occur 
    ts_anom::Vector{Float64} = vcat(-1) # [s] 
    
    # transition function information
    # probability pilot will follow action
    p_follow_action::Float64 = 0.5 
    # probability of anomaly 
    p_anomaly::Float64 = 1/(length(ts_anom)-1)
    
    # for indexing
    path_ind::Vector{Int64} = collect(1:length(path))
    a_ind::Dict{Symbol, Int} = Dict(:slowest=>1, :slower=>2, :constant=>3, :faster=>4, :fastest=>5)
    dts_ind::Dict{Float64, Int} = Dict(val => ind for (ind, val) in enumerate(dts))
    ts_flight_ind::Dict{Tuple{Int, Float64}, Int} = ts_flight_ind_dict(path_ind, point_start_times, time_step, point_end_times, point_time_length)
    ind_ts_flight::Array{Tuple{Int, Float64}} = ind_ts_flight_array(ts_flight_ind)
    ts_launch_ind::Dict{Float64, Int} = Dict(val => ind for (ind, val) in enumerate(ts_launch))
    ts_anom_ind::Dict{Float64, Int} = Dict(val => ind for (ind, val) in enumerate(ts_anom))
    
    state_size::Tuple{Int, Int, Int, Int} = (sum(point_time_length), length(dts), length(ts_launch), length(ts_anom))
end

## STATE SPACE definition
function POMDPs.states(mdp::MeterMDP)
    s = MeterState[]
    # loop over all states
    for pt = reverse(mdp.path_ind)
        for t_flight = reverse(mdp.point_start_times[pt]:mdp.time_step:mdp.point_end_times[pt])
            for dt = mdp.dts, t_launch = mdp.ts_launch, t_anom = mdp.ts_anom
                push!(s,MeterState(pt,t_flight,dt,t_launch,t_anom))
            end
        end
    end
    return s
end

# ACTION SPACE definition
POMDPs.actions(mdp::MeterMDP) = (:slowest, :slower, :constant, :faster, :fastest)

# TRANSITION FUNCTION definition
function POMDPs.transition(mdp::MeterMDP, state::MeterState, action::Symbol)
    # setup arrays for the next states and associated probabilities
    new_states = MeterState[]
    probs = Float64[]
    
    # check if terminal point
    if state.pt == mdp.path_ind[end]
        return SparseCat([state], [1.0])
    end
    
    # update way-point
    new_way_point = state.pt+1
    
    # time into flight launch occurs does not change
    new_time_launch = state.t_launch
    
    # find distance between way-points 
    leg_dist = segment_distance(mdp.path[state.pt], mdp.path[new_way_point])
    # leg_distance is used just for timing so add uncertainty
    leg_dists = [leg_dist, leg_dist*1.01, leg_dist*1.015, leg_dist*1.02]
    leg_dist_probs = [0.4, 0.3, 0.2, 0.1]
    
    # update speed and time based on action
    if action == :constant  # assume pilot always responds to constant velocity
        new_speed = state.dt
        # update time into flight based on action
        for i = 1:length(leg_dists)
            new_time = state.t_flight + leg_dists[i]/new_speed
            # update time of anomaly
            new_time_anom, prob_new_time_anom = update_anom_times(mdp, state, new_time)
            # update full states and probabilities
            for anomID = 1:length(new_time_anom)
                push!(new_states, MeterState(new_way_point, new_time, new_speed, new_time_launch, new_time_anom[anomID]))
                push!(probs, prob_new_time_anom[anomID]*leg_dist_probs[i])
            end
        end
    else  # update speed and time based on action - if speed update is greater than max_speed 
          # or less than min_speed, aircraft deterministically stays going at current speed
        new_speed = [state.dt]
        action_speed = state.dt + mdp.action_deltas[action]
        if action_speed < mdp.max_speed && action_speed > mdp.min_speed
            push!(new_speed, action_speed)
        end
        # listens, does not listen
        for i = 1:length(leg_dists)
            new_time = state.t_flight .+ (leg_dists[i] ./ new_speed)
            if length(new_speed) > 1
                response_probs = [mdp.p_follow_action, 1.0-mdp.p_follow_action] 
            else
                response_probs = [1.0]
            end
            # update time of anomaly
            for speedID in 1:length(new_speed)
                new_time_anom, prob_new_time_anom = update_anom_times(mdp, state, new_time[speedID])
                for anomID in 1:length(new_time_anom)
                    # update full states and probabilities
                    push!(new_states, MeterState(new_way_point, new_time[speedID], new_speed[speedID], new_time_launch, new_time_anom[anomID]))
                    push!(probs, prob_new_time_anom[anomID]*response_probs[speedID]*leg_dist_probs[i])
                end
            end
        end
    end
    # return new states and probabilities
    return SparseCat(new_states, probs)
end

# REWARD FUNCTION definition
function POMDPs.reward(mdp::MeterMDP, state::MeterState, action::Symbol, nextState::MeterState)
    # check if at terminal state
    if state.pt == mdp.path_ind[end] 
        if action == :constant
            return 0.0
        else 
            return -1.0
        end
    end
    
    # reward for action
    # check within speed limit
    if state.dt + mdp.action_deltas[action] > mdp.max_speed || state.dt + mdp.action_deltas[action] < mdp.min_speed
        return mdp.r_speed_limits
    end
    
    # award within speed limit rewards
    action_reward = 0.
    if action == :faster || action == :slower        # er reward
        action_reward = mdp.r_er_change
    elseif action == :fastest || action == :slowest  # est reward
        action_reward = mdp.r_est_change
    end
    
    # reward for debris
    # if within t_anom that produce debris that falls to earth
    if nextState.t_anom <= mdp.debris.time_anom_ends && nextState.t_anom >= mdp.debris.time_anom_starts
        # cycle over potential pass through times
        for launch_vehicle_time = (state.t_flight-state.t_launch):mdp.time_step:(nextState.t_flight-state.t_launch+mdp.time_step)
            # check that launch vehicle time is in a range of when an anomaly could have occured
            if launch_vehicle_time >= nextState.t_anom
                travel_time = launch_vehicle_time-(state.t_flight-state.t_launch)
                location = update_point(mdp, state.pt, nextState.pt, nextState.dt, travel_time)
                # see if there is active debris
                for anom = nextState.t_anom-mdp.debris_anomoly_active:mdp.debris_anomaly_step:nextState.t_anom+mdp.debris_anomoly_active
                    if anom in mdp.ts_anom
                        if bool_active_debris(anom, launch_vehicle_time, mdp.debris_time_threshold, location, mdp.debris_dist_threshold, mdp.debris.debris_dict, mdp.debris.debris_time_KD_tree)
                            return mdp.lambda*action_reward + mdp.r_within_debris 
                        end
                    end
                end
            end
        end
    end
    # no debris reward 
    return mdp.lambda*action_reward
end

## Solver Helper Functions
# number of states
POMDPs.n_states(mdp::MeterMDP) = prod(mdp.state_size)
# number of actions
POMDPs.n_actions(mdp::MeterMDP) = length(POMDPs.actions(mdp::MeterMDP))
# discount
POMDPs.discount(mdp::MeterMDP) = 1.

# function to obtain state index
function POMDPs.stateindex(mdp::MeterMDP, state::MeterState)
    t_flight = mdp.ts_flight_ind[(state.pt, state.t_flight)]
    dt = mdp.dts_ind[state.dt]
    t_launch = mdp.ts_launch_ind[state.t_launch]
    t_anom = mdp.ts_anom_ind[state.t_anom]
    return LinearIndices(mdp.state_size)[t_flight, dt, t_launch, t_anom]
end

# function to obtain action index
POMDPs.actionindex(mdp::MeterMDP, act::Symbol) = mdp.a_ind[act]

# required POMDPs.jl function
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::MeterState, mdp::MeterMDP)
    time_offset = s.t_flight%mdp.time_step
    base_index = mdp.ts_flight_ind[(s.pt, s.t_flight-time_offset)] # floor(s.t_flight))]
    actual_index = base_index + time_offset/mdp.time_step
    return SVector{4,Float64}(actual_index, s.dt, s.t_launch, s.t_anom)
end

# required POMDPs.jl function
function POMDPs.convert_s(::Type{MeterState}, v::AbstractVector{Float64}, mdp::MeterMDP) 
    int_v1 = round(Int, v[1])
    return MeterState(mdp.ind_ts_flight[int_v1][1], round(mdp.ind_ts_flight[int_v1][2], digits = 1), v[2], v[3], v[4])
end
