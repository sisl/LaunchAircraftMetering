using Distributions
using GridInterpolations
using LinearAlgebra
using Random
using StaticArrays
using NearestNeighbors

## generate a random path from the waypoints
function generate_path(points_kd_tree, point_data, max_path_length, k)   
    length_waypoints = size(point_data)[2]
    path_es = zeros(max_path_length)
    path_ns = zeros(max_path_length)
    for i = 1:max_path_length
        if i == 1
            # pick a random way point as the first way point
            rand_idx = rand(1:length_waypoints)
            path_es[i] = point_data[1,rand_idx]
            path_ns[i] = point_data[2,rand_idx]
        elseif i == 2
            # find k points closest to previous point
            ides, dists = knn(points_kd_tree, [path_es[i-1],path_ns[i-1]], k+1, true)
            # pick a random one as the next point
            new_point_idx = rand(ides[2:k+1])
            path_es[i] = point_data[1,new_point_idx]
            path_ns[i] = point_data[2,new_point_idx]
        else
            # angle between previous points
            prev_angle = atan(path_ns[i-1]-path_ns[i-2], path_es[i-1]-path_es[i-2])
            # find k points closest to previous point
            ides, dists = knn(points_kd_tree, [path_es[i-1],path_ns[i-1]], k+1, true)
            new_point_ides = shuffle(ides[2:k+1])
            # cycle through them to find a new_point that is within 90 degrees of previous angle
            new_point = false
            j = 1
            while !new_point && j <= length(new_point_ides)
                xj = point_data[1, new_point_ides[j]]
                yj = point_data[2, new_point_ides[j]]
                cur_angle = atan(yj-path_ns[i-1], xj-path_es[i-1])
                if abs(prev_angle-cur_angle) <= pi/4
                    path_es[i] = point_data[1,new_point_ides[j]]
                    path_ns[i] = point_data[2,new_point_ides[j]]
                    new_point = true
                else
                    j += 1
                end
            end
            if !new_point
                return path_es[1:i-1], path_ns[1:i-1]
            end
        end
    end
    return path_es, path_ns
end 

## check that path comes within a distance threshold of debris 
function check_path_hits_debris(debris_kd_tree, path_es, path_ns, time_step, distance_threshold, min_speed)
    # check between each segment
    for i = 1:length(path_es)-1
        seg_dist = segment_distance((path_es[i], path_ns[i]), (path_es[i+1], path_ns[i+1]))
        for travel_time = 0:time_step:seg_dist/min_speed 
            location = update_point((path_es[i], path_ns[i]), (path_es[i+1], path_ns[i+1]), min_speed, travel_time)
            # debris check
            if bool_debris(location, distance_threshold, debris_kd_tree)
                # exit as soon as true
                return true
            end
        end
    end
    return false
end

## gather the waypoint names
function generate_path_labels(es, ns, waypoint_names)
    labels = Array{String}(undef, length(es))
    for i = 1:length(es)
        labels[i] = waypoint_names[[es[i], ns[i]]]
    end
    return labels
end

## generate the mu and sigma for the path variants
function generate_mu_sig(ee_1, ee_end, nn_1, nn_end, diag_val)    
    middle_distance = segment_distance((ee_1, nn_1), (ee_end, nn_end))
    middle_cuts = Int(floor(middle_distance/100000.)) + 1
    ee_middle = collect(range(ee_1, length=middle_cuts+2, stop=ee_end))[2:end-1]
    nn_middle = collect(range(nn_1, length=middle_cuts+2, stop=nn_end))[2:end-1]
    end_point = length(ee_middle)
    mu = vcat(ee_middle, nn_middle)
    one_off_diag_val = diag_val * .5
    two_off_diag_val = diag_val * .25
    # create diag matrix with diag value
    sig = diagm(0=>fill(diag_val, length(mu)))
    # fill in values for one off diag values
    for i = 1:end_point-1
        sig[i,i+1] = one_off_diag_val
        sig[i+1,i] = one_off_diag_val
    end
    for i = end_point+1:end_point*2-1
        sig[i,i+1] = one_off_diag_val
        sig[i+1,i] = one_off_diag_val
    end
    # fill in values for two off diag values
    for i = 1:end_point-2
        sig[i,i+2] = two_off_diag_val
        sig[i+2,i] = two_off_diag_val
    end
    for i = end_point+1:end_point*2-2
        sig[i,i+2] = two_off_diag_val
        sig[i+2,i] = two_off_diag_val
    end
    return mu, sig
    # return mu, sig
end 

## generate path variants
function generate_path_varients(path_es, path_ns, num_varients)
    path_array = Array{Tuple{Array{Float64,1},Array{Float64,1}}}(undef, num_varients)
    # produce num_varients varients
    for k = 1:num_varients
        # create empty e and n set
        new_es = path_es[1]
        new_ns = path_ns[1]
        for i = 1:length(path_es)-1
            ee_1 = path_es[i]
            ee_end = path_es[i+1]
            nn_1 = path_ns[i]
            nn_end = path_ns[i+1]
            # assign diag_val based on the segment distance
            diag_val = 100. * segment_distance((ee_end, ee_1), (nn_end, nn_1))
            # generate mu and sigma
            mu, sig = generate_mu_sig(ee_1, ee_end, nn_1, nn_end, diag_val)
            # end_point is where switches from e to n
            end_point = Int(length(mu)/2)
            # create distribution
            distribution = MvNormal(mu, sig)
            # gather sample
            new_sample = rand(distribution)
            # update e and n
            new_es = vcat(new_es, new_sample[1:end_point], ee_end)
            new_ns = vcat(new_ns, new_sample[end_point+1:end], nn_end)
        end
        path_array[k] = (new_es, new_ns)

    end
    return path_array
end

## generate a path varient
function generate_path_varient(path_es, path_ns)
    # create empty e and n set
    new_es = path_es[1]
    new_ns = path_ns[1]
    wps = [1]
    for i = 1:length(path_es)-1
        ee_1 = path_es[i]
        ee_end = path_es[i+1]
        nn_1 = path_ns[i]
        nn_end = path_ns[i+1]
        # assign diag_val based on the segment distance
        diag_val = 100. * segment_distance((ee_end, ee_1), (nn_end, nn_1))
        # generate mu and segment
        mu, sig = generate_mu_sig(ee_1, ee_end, nn_1, nn_end, diag_val)
        # end_point is where switches from e to n
        end_point = Int(length(mu)/2)
        # create distribution
        distribution = MvNormal(mu, sig)
        # gather sample
        new_sample = rand(distribution)
        # update e and n
        new_es = vcat(new_es, new_sample[1:end_point], ee_end)
        new_ns = vcat(new_ns, new_sample[end_point+1:end], nn_end)
        push!(wps, length(new_es))
    end
    return new_es, new_ns, wps
end

## transform locations to es and ns
function path_locations_to_es_ns(locations)
    loc_length = length(locations)
    es = zeros(loc_length)
    ns = zeros(loc_length)
    for i = 1:loc_length
        es[i] = locations[i][1]
        ns[i] = locations[i][2]
    end
    return es, ns
end
