# LaunchAircraftMetering

This directory contains the supplement code for the paper titled "Optimal Aircraft Metering during Space Launches" by Rachael E. Tompa and Mykel J. Kochenderfer, to appear in the 2019 Digital Avionics Systems Conference.

<!-- MarkdownTOC -->

- [Overview](#overview)
    - [Dependencies](#dependencies)
    - [Layout](#layout)
- [MDP Framework](#mdp-framework)
- [Simulation Framework](#simulation-framework)

<!-- /MarkdownTOC -->

## Overview
The code in this repository supports one of two tasks:

* generating policies (mdpRun.jl)
* evaluating those policies in simulation (simRun.jl)

### Dependencies
The following Julia packages are required for running code. 
* [DisceteValueIteration]
* [Distances](https://github.com/JuliaStats/Distances.jl)
* [Geodesy](https://github.com/JuliaGeo/Geodesy.jl)
* [GridInterpolations](https://github.com/sisl/GridInterpolations.jl)
* [HDF5](https://github.com/JuliaIO/HDF5.jl)
* [JLD2](https://github.com/JuliaIO/JLD2.jl)
* [LocalApproximationValueIteration](https://github.com/JuliaPOMDP/LocalApproximationValueIteration.jl)
* [NearestNeighbors](https://github.com/KristofferC/NearestNeighbors.jl)
* [Parameters](https://github.com/mauro3/Parameters.jl)
* [POMDPs](https://github.com/JuliaPOMDP/POMDPs.jl)
* [POMDPModelTools](https://github.com/JuliaPOMDP/POMDPModelTools.jl)
* [POMDPPolicies](https://github.com/JuliaPOMDP/POMDPPolicies.jl)
* [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl)

### Layout

```
src/

    genPathFunctions.jl
    helperFunctions.jl
    mdpRun.jl
    setupMDP.jl
    simFuntions.jl
    simRun.jl

assets/
    debrisProfile_1.txt
    waypoints.jld2
    
results/

README.md

```
