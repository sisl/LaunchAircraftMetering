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
using FileIO
using Geodesy
using HDF5
using JLD2
using NearestNeighbors
using Parameters

## Launch Location Information
@with_kw struct LaunchSiteInfo
	# cape canaveral 
    latRef::Float64 = 28.6
    longRef::Float64 = -80.6
    altRef::Float64 = 76.2
    flightAlt::Float64 = 10688.
end

## Location structure
struct Location
    e::Float64  # east location coordinate
    n::Float64  # north location coordinate
end

## Debris Setup
function read_in_debris(n_debris_profiles::Int, launch_site_info::LaunchSiteInfo)
    debris = []
    for i = 1:n_debris_profiles
        debris_filename = string("../assets/debrisProfile_",string(i),".txt")
        debris_profile = readdlm(debris_filename, ',', Float64, '\n') # read in raw info
        if i == 1
            debris = debris_profile
        else
            debris = unique(vcat(debris, debris_profile), dims=1)
        end
    end
    # debris is output long, lat
    debris = hcat(debris[:,1],debris[:,5],debris[:,2],debris[:,3]) 
    # only keep uniques debris
    debris = unique(debris, dims=1)
    for i = 1:size(debris)[1]
        # during previous processing the time was divided by 10
        debris[i,1] = debris[i,1]*10  
        # update from long and lat to ENU coordinates
        enu_location = ENU(LLA(debris[i,4],debris[i,3],launch_site_info.flightAlt),LLA(launch_site_info.latRef,launch_site_info.longRef,launch_site_info.altRef),wgs84)
        # update to east and north
        debris[i,3] = enu_location[1] 
        debris[i,4] = enu_location[2] 
    end
    # keep unique and sort for future use
    debris = unique(debris, dims=1)
    debris = sortslices(debris, dims=1)
    return debris
end

## to limit search time, find a maximum and minumem time
## for when debris passes through the threshold (post launch)
function start_end_times(debris::Array{Float64,2})
    anom_times = unique(debris[1:size(debris)[1]])
    return minimum(anom_times), maximum(anom_times)
end

## make a dictionary of debris information and dictionary of debris KD trees
function make_debris_dictionary(debris::Array{Float64,2}, time_anom_starts::Float64, 
    time_anom_ends::Float64)
    debris_dict = Dict{Tuple{Float64,Float64},KDTree{SArray{Tuple{2},Float64,1,2},Euclidean,Float64}}()
    debris_time_KD_tree = Dict{Float64,KDTree{SArray{Tuple{1},Float64,1,1},Euclidean,Float64}}()
    for at = time_anom_starts:10.:time_anom_ends
        ss_at = searchsorted(debris[:,1], at)
        debris_at = debris[ss_at,:]
        # anomaly time to KD tree of pass through times
        debris_time_KD_tree[at] = KDTree(Array(transpose(debris_at[:,2])))
        for pt in unique(debris_at[:,2])
            # anomaly and pass through time to KD tree of locations
            debris_pt = searchsorted(debris_at[:,2], pt)
            debris_dict[(at,pt)] = KDTree(Array(transpose(debris_at[debris_pt,3:4])))
        end
    end
    return debris_dict, debris_time_KD_tree
end

## return if based on time and location if within distance threshold of potential debris
function bool_active_debris(anom_time::Float64, cur_time::Float64, 
    time_threshold::Float64, location::Location, distance_threshold::Float64, 
    debris_dict::Dict{Tuple{Float64,Float64},KDTree{SArray{Tuple{2},Float64,1,2},Euclidean,Float64}}, 
    debris_time_KD_tree::Dict{Float64,KDTree{SArray{Tuple{1},Float64,1,1},Euclidean,Float64}})
    # check time
    time_idxs = inrange(debris_time_KD_tree[anom_time], [cur_time], time_threshold, true)
    for i in time_idxs
        time_current_debris = debris_dict[(anom_time, debris_time_KD_tree[anom_time].data[i][1])]
        # check location
        location_idxs = inrange(time_current_debris, [location.e;location.n], distance_threshold, true)
        if length(location_idxs) > 0
            return true
        end
    end
    return false  
end

## return if location comes within distance threshold of potential debris
function bool_debris(location::Location, distance_threshold::Float64, 
    debris_kd_tree::Dict{Float64,KDTree{SArray{Tuple{1},Float64,1,1},Euclidean,Float64}})
    # check location
    location_idxs = inrange(debris_kd_tree, [location.e;location.n], distance_threshold, true)
    if length(location_idxs) > 0
        return true
    end
    return false  
end

## Debris structure
struct Debris
    debris::Array{Float64,2}
    time_anom_starts::Float64
    time_anom_ends::Float64
    debris_dict::Dict{Tuple{Float64,Float64},KDTree{SArray{Tuple{2},Float64,1,2},Euclidean,Float64}}
    debris_time_KD_tree::Dict{Float64,KDTree{SArray{Tuple{1},Float64,1,1},Euclidean,Float64}}
end

## read in debris and process into structure 
function Debris(n_debris_profiles::Int64, launch_site_info::LaunchSiteInfo)
    debris = read_in_debris(n_debris_profiles, launch_site_info)
    time_anom_starts, time_anom_ends = start_end_times(debris)
    debris_dict, debris_time_KD_tree = make_debris_dictionary(debris, time_anom_starts, time_anom_ends)
    return Debris(debris, time_anom_starts, time_anom_ends, debris_dict, debris_time_KD_tree)
end

## This function takes the user input string of way points and translates it into an 
## array of way point locations
function path_locations(path::Vector{String}, waypoints::Dict{String,Any})
    locations = Array{Tuple{Float64,Float64}}(undef, length(path)) # x and y
    for i = 1:length(path)
        locations[i] = waypoints[path[i]]
    end
    return locations
end 

## This function finds the distance between two points
function segment_distance(point1::Tuple{Float64,Float64}, point2::Tuple{Float64,Float64})
    return hypot((point1[1]-point2[1]),(point1[2]-point2[2]))
end

## This function finds the path distance for a series of points
function path_distance(path_locations::Array{Tuple{Float64,Float64}})
    distance = 0
    for i = 2:length(path_locations)
        distance += segment_distance(path_locations[i-1], path_locations[i])
    end
    return distance
end

## helper function for MDP transition function
## updates anomaly times
function update_anom_times(mdp, state, new_time::Float64)
    new_time_anom = []
    prob_new_time_anom = []
    if state.t_anom != -1.  # if an anomaly has already occured, do not update
        return [state.t_anom], [1.]
    else
        # check if in a time when an anomaly can occur
        for time_anom in mdp.ts_anom[2:end]
            if time_anom >= state.t_flight-state.t_launch && time_anom <= new_time-state.t_launch
                push!(new_time_anom, time_anom)
                push!(prob_new_time_anom, mdp.p_anomaly)  
            end
        end
        # add entry for no anomaly!
        push!(new_time_anom, -1.)
        if length(prob_new_time_anom) > 0 
            push!(prob_new_time_anom, 1. - sum(prob_new_time_anom)) # chance for no anomaly
        else
            push!(prob_new_time_anom, 1.)
        end
    end
    return new_time_anom, prob_new_time_anom
end

## update point based on travel time and speed (using MDP structure)
function update_point(mdp, point1::Int, point2::Int, speed::Float64, travel_time::Float64)
    x1, y1 = mdp.path[point1]
    x2, y2 = mdp.path[point2]
    angle = atan((y2-y1),(x2-x1))
    x1 += speed*cos(angle)*travel_time
    y1 += speed*sin(angle)*travel_time
    return Location(x1, y1)
end

## update point based on travel time and speed
function update_point(point1::Tuple{Float64,Float64}, point2::Tuple{Float64,Float64}, speed::Float64, travel_time::Float64)
    x1, y1 = point1[1], point1[2]
    x2, y2 = point2[1], point2[2]
    angle = atan((y2-y1),(x2-x1))
    x1 += speed*cos(angle)*travel_time
    y1 += speed*sin(angle)*travel_time
    return Location(x1, y1)
end

## find potential times for each waypoint
function time_setup(path::Array{Tuple{Float64,Float64}}, path_length::Int, time_step::Float64, max_speed::Float64, min_speed::Float64)
    point_start_times = zeros(path_length)
    point_end_times = zeros(path_length)
    point_time_length = ones(path_length)
    for i = 2:path_length
        # find the distance of this path part
        path_part = path_distance(path[i-1:i])
        # update the start time with path_part
        point_start_times[i] = point_start_times[i-1] + path_part/max_speed
        # round to time step interval
        point_start_times[i] = point_start_times[i] - point_start_times[i]%time_step
        # add a buffer to the path length (for when aircraft don't follow the nominal trajectory)
        path_part = 1.025*path_part
        # update end times with buffered path_part
        point_end_times[i] = point_end_times[i-1] + path_part/min_speed
        # round to time step interval
        point_end_times[i] = point_end_times[i] - point_end_times[i]%time_step + time_step
        # find how many time steps
        point_time_length[i] += (point_end_times[i]-point_start_times[i])/time_step
    end
    return point_start_times, point_end_times, point_time_length
end

## create dictionary waypoint to times times to index
function ts_flight_ind_dict(path_ind::Vector{Int64}, point_start_times::Vector{Float64}, time_step::Float64, 
    point_end_times::Vector{Float64}, point_time_length::Vector{Float64})
    ts_flight_ind = Dict{Tuple{Int,Float64}, Int}()
    for i in reverse(path_ind)
        for j = point_end_times[i]:-time_step:point_start_times[i]
            ts_flight_ind[(i,j)] = round(Int, 1 + sum(point_time_length[i+1:end]) + (point_end_times[i] - j)/time_step)
        end
    end
    return ts_flight_ind
end

## create dictionary of index to waypoint
function ind_ts_flight_array(ts_flight_ind::Dict{Tuple{Int, Float64}, Int})
    ind_ts_flight = Array{Tuple{Int, Float64}}(undef,length(keys(ts_flight_ind)))
    for key in keys(ts_flight_ind)
        ind_ts_flight[ts_flight_ind[key]] = key
    end
    return ind_ts_flight
end

## function to save policy
function save_policy(policy::LocalApproximationValueIterationPolicy{LocalGIFunctionApproximator{RectangleGrid{4}},MersenneTwister}, 
    identifying_string::String)
    JLD2.jldopen("../results/interp_"*identifying_string*".jld2", "w") do file
        file["interp"] = policy.interp
    end
    JLD2.jldopen("../results/mdp_"*identifying_string*".jld2", "w") do file
        file["mdp"] = policy.mdp
    end
end

## function to load a saved policy
function load_policy(identifying_string::String)
    p_action_map = [:slowest, :slower, :constant, :faster, :fastest]
    p_interp = FileIO.load("../results/interp_"*identifying_string*".jld2")["interp"]
    p_mdp = FileIO.load("../results/mdp_"*identifying_string*".jld2")["mdp"]
    p_is_mdp_generative = false
    p_n_generative_samples = 0
    p_rng = [0xa9780953, 0x2c902242, 0xa8e4dfe4, 0xc8522861]
    return LocalApproximationValueIterationPolicy(p_interp, p_action_map, p_mdp, p_is_mdp_generative,
                                                  p_n_generative_samples, MersenneTwister(p_rng))
end

## find the start and end times for MDP input
function find_various_start_end_times(paths, waypoints, min_speed::Float64, max_speed::Float64, speed_step::Float64,
    time_step::Float64, distance_threshold::Float64, debris::Array{Float64,2})
    ps = length(paths)
    speeds = min_speed:speed_step:max_speed
    for p = 1:length(paths)
        println(paths[p])
        path_points = path_locations(paths[p], waypoints)
        a_times = Array{Float64,1}()
        d_times = Array{Float64,1}()
        p_times = Array{Float64,1}()
        for speed in speeds
            t_total = 0.
            for i = 1:length(path_points)-1
                leg_distance = segment_distance(path_points[i], path_points[i+1])
                for timing = t_total:time_step:t_total+leg_distance/speed+time_step
                    location = update_point(path_points[i], path_points[i+1], speed, timing-t_total)
                    for j = 1:size(debris)[1]
                        if hypot((debris[j,3]-location.e), (debris[j,4]-location.n)) < distance_threshold
                            push!(a_times, debris[j,1])
                            push!(d_times, round(debris[j,2]))
                            push!(p_times, round(timing))
                        end
                    end
                end
                t_total += leg_distance/speed
            end
        end
        times = hcat(a_times, d_times, p_times)
        times = unique(times, dims=1)
        dict_offset_times = Dict{Float64,Array{Float64,1}}()
        for t in unique(times[:,1])
            times_d = Array{Float64,1}()
            times_p = Array{Float64,1}()
            for i = 1:size(times)[1]
                if times[i,1] == t
                    push!(times_d, times[i,2])
                    push!(times_p, times[i,3])
                end
            end    
            min_offset, max_offset = find_mdp_inputs(t, minimum(times_d), maximum(times_d), 
                minimum(times_p), maximum(times_p))
            dict_offset_times[t] = [min_offset, max_offset]
        end
        println(dict_offset_times)
    end
end

## determine if path interacts with launch vehicle
function find_launch_vehicle_start_end_times(paths, waypoints, min_speed::Float64, max_speed::Float64, 
    time_step::Float64, distance_threshold::Float64, debris::Array{Float64,2})
    ps = length(paths)
    speeds = min_speed:2.5:max_speed
    for p = 1:ps
        println("path: ", p)
        println(paths[p])
        path_points = path_locations(paths[p], waypoints)
        for speed in speeds
            t_total = 0.
            for i = 1:length(path_points)-1
                leg_distance = segment_distance(path_points[i], path_points[i+1])
                for timing = t_total:time_step:t_total+leg_distance/speed+time_step
                    location = update_point(path_points[i], path_points[i+1], speed, timing-t_total)
                    lv_intersect_e, lv_intersect_n = 107.82, 44.4101
                    if hypot((lv_intersect_e-location.e), (lv_intersect_n-location.n)) < distance_threshold
                        println(timing)
                    end
                end
                t_total += leg_distance/speed
            end
        end
    end
end

