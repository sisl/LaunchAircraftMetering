## For Loading Files from JLD2
Core.eval(Main, :(using Distances))
Core.eval(Main, :(using NearestNeighbors))
Core.eval(Main, :(using Random))
Core.eval(Main, :(using LocalApproximationValueIteration))
Core.eval(Main, :(using LocalFunctionApproximation))
Core.eval(Main, :(using DiscreteValueIteration))

include("../src/helperFunctions.jl")
include("../src/setupMDP.jl")

way_points = FileIO.load("/scratch/rtompa2/way_points.jld2"); # enu from cape canaveral

# 50 generated paths
paths = [
    ["LACKS", "BERGH", "MILOE", "HOBEE", "BLUFI", "MTH", "CANOA"],
    ["ONGAL", "ITEGO", "SCAPA", "VITOB", "MAROT", "PAARR", "EXTER", "ANGIL", "TORRY", "LURKS"],
    ["CANEE", "ALUTE", "NATHY", "ATTIK", "JA"],
    ["BALTN", "NETSS", "DIW"],
    ["LIMRI", "YEPSY", "QNEPA", "CRUPE", "GAGDD", "GABRA", "LIDOL", "SELAN"],
    ["TOMAZ", "PAARR", "PEKRE", "KIXAL", "CRG", "JA"],
    ["MILOK", "LAMKN", "FOORE", "SIFEN", "BALOO", "SOORY", "GRUPI", "RKDIA"],
    ["ENAMO", "MLSAP", "RABAL", "PIREX", "YEPSY", "SELIM", "VITOL", "SCUPP"],
    ["JORGG", "MILLE", "FLUPS", "DIW"],
    ["PAEPR", "SAUCR", "UKOKA", "ORL", "CANOA"],
    ["CH", "OHLAA", "BAAGR", "ZSJ", "GOVET", "TILDI", "GUYRO", "MODUX", "ALGUS", "DAREK"],
    ["ZLS", "KRTIS", "CNNOR", "EMQUE", "ORF", "OWENZ", "VITOL", "RKDIA"],
    ["MESKA", "TOMAZ", "SLUKA", "LUCTI", "DRYED", "YEPSY", "SOORY", "DOPHN"],
    ["ARMUR", "POKAK", "TARBA", "BEMOL", "DYNAH", "BAAGR", "ILIDO", "KANUX", "PAEPR", "AZEZU"],
    ["ZTC", "OMALY", "ILIDO", "ALOBI", "ROLLE", "HERIN", "DOPHN", "BOBTU"],
    ["ECG", "ZIBUT", "ANVER", "WINGZ", "KAVAX", "GABAR", "ILURI", "ALGUS", "KOTEN"],
    ["OROSA", "MILOK", "ELOPO", "OBIKE", "FIVZE", "YEPSY", "LAZEY", "SOORY", "GRUPI", "RKDIA"],
    ["OZENA", "OTTNG", "ONGOT", "PERDO", "DASER", "BDA", "AYTTE", "KAVAX", "ILURI", "ONGAL"],
    ["DYNAH", "RAMJT", "UKOKA", "HANRI", "OTTNG", "CLB", "ILM", "ROLLE", "LACKS", "WHALE"],
    ["DINIM", "LAZEY", "PIREX", "VINSO", "WOODZ", "ASIVO", "MAROT", "OTAMO", "PELRA"],
    ["ZSJ", "CLETT", "SUMRS", "OHLAA", "OLDEY", "ECG", "PENYT", "JEBBY", "DOPHN"],
    ["PANAL", "PETEE", "ORL"],
    ["ANADA", "YIYYO", "FIVZE", "SIFEN", "NUMBR", "SQUAD", "SIE"],
    ["KIKER", "KATOK", "ALERI", "LUCTI", "RUDLI", "GALVN", "OXANA", "ROLLE", "KAYYT", "GRUPI"],
    ["EXTER", "CASPR", "CHS", "CH"],
    ["OSBOX", "HERIN", "DIW", "CHS", "ATTIK", "BLUFI", "ERRCA", "SKHOT", "HARBG", "DAKES"],
    ["COLBY", "JAQUZ", "JUELE", "MANII", "MNDZ", "MLLER", "VINSO", "DRYED", "NOSID", "AZEZU"],
    ["ETEEE", "MNOLO", "NEYDU", "FOORE", "SIFEN", "AYTTE", "DASER", "LYNUS", "OWENZ", "SCUPP"],
    ["SEBIS", "DUPOX", "SHEIL", "BALOO", "JEBBY", "DOPHN"],
    ["QNEPA", "SINGL", "KASAR", "TROUT", "OZENA", "JA"],
    ["GONIS", "VALLY", "HOBEE", "METTA", "ORF", "PENYT", "GRUPI", "DOPHN", "BOBTU"],
    ["BEXET", "SOORY", "BALTN", "SINGL", "FERNA", "THANK", "GUYRO", "ELOPO", "TOXUN", "KOTEN"],
    ["SIE", "JAINS", "BROOM", "WUNUT", "GHANN", "BENET", "LIDOL", "SELAN"],
    ["PJM", "PRCHA", "LNHOM", "ROTHM", "HOBEE", "JA"],
    ["KINGG", "LYNUS", "ORF", "DIW", "CLB", "TORRY", "VALLY", "ERRCA", "JAQUZ", "HARBG"],
    ["KNEEL", "RAMJT", "TORRY", "PANAL", "ATUGI", "TILED", "DAVES", "RKDIA"],
    ["RNTRY", "DYNAH", "DAAST", "CLETT", "KANUX", "PAEPR", "BERGH", "LINND", "SAILE", "GRUPI"],
    ["RUDLI", "KASAR", "UKOKA", "CHS", "CH"],
    ["DOPHN", "LSIER", "WILYY", "OTTNG", "BAHAA", "VKZ", "RNTRY", "ALORA", "LIDOL", "VODIN"],
    ["LINND", "PANAL", "OHLAA", "MAPYL", "BRRGO", "ROSEA", "OBNOR", "SELAN"],
    ["MILOE", "HOBEE", "ZFP", "ERRCA", "SABRE", "BETIR", "SJU", "NEYDU", "OBIKE", "KAVAX"],
    ["SCUPP", "REQU", "TASNI", "BRKZZ", "OPAUL", "YIYYO", "RAYAS", "BEXER", "OROSA", "ERIKO"],
    ["DAVES", "ENAPI", "BALTN", "SEBIS", "STAAL", "ZLS", "GHANN", "MAROT", "LENOM", "ERIKO"],
    ["NABEN", "GOVET", "BENIE", "BROOM", "HANRI", "LURKS", "CHS"],
    ["KARUM", "MACKI", "NETTA", "COUKY", "RUDLI", "KANUX", "LURKS", "CH"],
    ["SLUGO", "KINCH", "MANII", "FOLLE", "STAAL", "MAPYL", "CASPR", "METTA", "ORF", "LACKS"],
    ["TOBOR", "SELIM", "ANVER", "EMAKO", "DUNIG", "SEBIS", "CANEE", "SMTTY", "GHANN", "RNTRY"],
    ["SKHOT", "MUSSH", "BAAGR", "LOUIZ", "TROUT", "METTA", "CLB", "SAVIK", "SLATN", "JEBBY"],
    ["BERGH", "BOVIC", "DAWIN", "AMENO"],
    ["SIE", "PANAL", "SQT", "VKZ", "RODRK", "RNTRY", "LEVOR", "PELRA"]]

# times for 50 randomly generated paths
time_dicts = [
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

# solve mdp for policy
function solve_mdp(interp, mdp, i, t_anom)
    # solve
    println("gonna solve!")
    approx_solver = LocalApproximationValueIterationSolver(interp, max_iterations=1, belres=1e-6)
    approx_policy = solve(approx_solver, mdp)
    # save policy
    println("saving")
    save_policy(approx_policy, "path_"*string(i)*"_00005_action_"*string(t_anom)*"_policy")
end

# run mdp for the input paths
for path_i = 1
    println(path_i)
    path = path_locations(paths[path_i], way_points);
    println(path)
    time_dict = time_dicts[path_i]
    # cycle through the relevant times
    for key in keys(time_dict)
        println("time of anomaly is ", key)
        ts_launch = collect(time_dict[key][1]:5.:time_dict[key][2])
        ts_anom = vcat(-1, key)
        mdp = MeterMDP(path_string=paths[path_i], lambda=0.00005, path=path, ts_launch=ts_launch, ts_anom=ts_anom, min_speed = 283., max_speed = 293., speed_step = 2.5) # , p_follow_action = 0.9)
        grid = RectangleGrid(1:length(keys(mdp.ts_flight_ind)), mdp.dts, mdp.ts_launch, mdp.ts_anom)
        interp = LocalGIFunctionApproximator(grid::RectangleGrid{4});
        solve_mdp(interp, mdp, path_i, key)
    end
end
