import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import random
import math
from ordered_set import OrderedSet
# if you use conda run this in your terminal: conda install -c conda-forge ordered-set
import gurobipy as gb

# local files
from door_to_door_customer import DoorToDoorCustomer
from location import Location
from store import *
from vehicle import Vehicle
from vehicle_type import VehicleType


def get_nearest_store(stores, location):
    min_distance = float("inf")
    nearest_store = stores[0]
    for store in stores:
        distance = location.euclidean_distance(store.location)
        if distance < min_distance:
            min_distance = distance
            nearest_store = store
    return nearest_store


def run_experiment_gurobi(C, L, custom_setup=False):
    map_size = 100
    ratio_locker_customers = 0.3

    # generate data
    if custom_setup:
        C = 8
        L = 2
        stores = []
        stores.append(Store(0, Location(60, 50), capacity=float("inf"), is_warehouse=True))
        stores.append(Store(1, Location(30, 50), capacity=math.ceil(0.8 * C / L)))
        stores.append(Store(2, Location(50, 20), capacity=math.ceil(0.8 * C / L)))

        customers = []
        customers.append(LockerCustomer(0, 0, Location(10, 28), stores[1]))
        customers.append(DoorToDoorCustomer(1, 0, Location(20, 40)))
        customers.append(DoorToDoorCustomer(2, 1, Location(15, 70)))
        customers.append(DoorToDoorCustomer(3, 2, Location(30, 70)))
        customers.append(DoorToDoorCustomer(4, 3, Location(80, 60)))
        customers.append(LockerCustomer(5, 1, Location(70, 40), stores[0]))
        customers.append(DoorToDoorCustomer(6, 4, Location(90, 50)))
        customers.append(LockerCustomer(7, 2, Location(40, 15), stores[2]))
    else:
        stores = []
        stores.append(
            Store(0, Location(random.randint(0, map_size), random.randint(0, map_size)), capacity=float("inf"),
                  is_warehouse=True))
        for l in range(L):
            stores.append(Store(l + 1, Location(random.randint(0, map_size), random.randint(0, map_size)),
                                capacity=math.ceil(0.8 * C / L)))

        customers = []
        counter_locker_customer = 0
        counter_door_to_door_customer = 0
        for c in range(C):
            location = Location(random.randint(0, map_size), random.randint(0, map_size))
            if random.random() < ratio_locker_customers:
                # customer locker
                customers.append(
                    LockerCustomer(c, counter_locker_customer, location, get_nearest_store(stores, location)))
                counter_locker_customer += 1
            else:
                # door to door customer
                customers.append(DoorToDoorCustomer(c, counter_door_to_door_customer, location))
                counter_door_to_door_customer += 1

    # create sets
    C_L = list(filter(lambda customer: type(customer) == LockerCustomer, customers))
    C_D = list(filter(lambda customer: type(customer) == DoorToDoorCustomer, customers))
    lockers = list(filter(lambda store: not store.is_warehouse, stores))

    sum_W_l = 0
    for store in stores:
        if not store.is_warehouse:
            sum_W_l += store.capacity

    # define all the vehicles
    if custom_setup:
        vehicles = []
        vehicles.append(Vehicle(0, VehicleType.LOCKER_SUPPLY, stores[0], math.ceil(0.8 * sum_W_l)))
        vehicles.append(Vehicle(1, VehicleType.PF, stores[0], math.ceil(0.5 * len(C_D))))
        vehicles.append(Vehicle(2, VehicleType.LF, stores[1], math.ceil(0.6 * stores[1].capacity)))
        vehicles.append(Vehicle(3, VehicleType.LF, stores[2], math.ceil(0.6 * stores[2].capacity)))
    else:
        vehicles = []
        for store in stores:
            if store.is_warehouse:
                vehicles.append(Vehicle(0, VehicleType.LOCKER_SUPPLY, store, math.ceil(0.8 * sum_W_l)))
                vehicles.append(Vehicle(1, VehicleType.PF, store, math.ceil(0.5 * len(C_D))))
            else:
                vehicles.append(Vehicle(store.index + 1, VehicleType.LF, store, math.ceil(0.6 * store.capacity)))

    # define some np.array to plot the map
    CD_location = np.array([[0, 0]])
    CL_location = np.array([[0, 0]])
    L_location = np.array([[0, 0]])
    W_location = np.array([[0, 0]])

    for store in stores:
        if store.is_warehouse:
            W_location = np.vstack([W_location, [store.location.x, store.location.y]])
        else:
            L_location = np.vstack([L_location, [store.location.x, store.location.y]])
    for cd in C_D:
        CD_location = np.vstack([CD_location, [cd.location.x, cd.location.y]])
    for cl in C_L:
        CL_location = np.vstack([CL_location, [cl.location.x, cl.location.y]])

    CD_location = np.delete(CD_location, 0, 0)
    CL_location = np.delete(CL_location, 0, 0)
    L_location = np.delete(L_location, 0, 0)
    W_location = np.delete(W_location, 0, 0)

    # define distance matrix
    all_locations = np.array([])
    all_locations = np.append(all_locations, W_location)
    all_locations = np.vstack([all_locations, L_location])
    all_locations = np.vstack([all_locations, CD_location])
    all_locations = np.vstack([all_locations, CL_location])

    all_buildings = stores + C_D + C_L

    df = pd.DataFrame(all_locations, columns=['xcord', 'ycord'], index=all_buildings)
    dist_matrix = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
    matrix_distance_converted_in_numpy = dist_matrix.to_numpy()

    # filter data
    supply_distances_matrix = dist_matrix.filter(items=stores, axis=1)
    supply_distances_matrix = supply_distances_matrix.filter(items=stores, axis=0)

    PF_distances_matrix = dist_matrix.filter(items=[stores[0]] + C_D, axis=1)
    PF_distances_matrix = PF_distances_matrix.filter(items=[stores[0]] + C_D, axis=0)

    LF_distances_matrix = dist_matrix.filter(items=lockers + C_D, axis=1)
    LF_distances_matrix = LF_distances_matrix.filter(items=lockers + C_D, axis=0)

    # Delivery Cost
    pi = 1
    pi_l = 0.85
    pi_L = 0.75

    PF_delivery_cost = pi * PF_distances_matrix
    LF_delivery_cost = pi_l * LF_distances_matrix
    supply_cost = pi_L * supply_distances_matrix

    LF_delivery_cost_multidim = np.zeros((L, len(C_D) + 1, len(C_D) + 1))

    for i in range(L):
        l = [0]
        l[0] = lockers[i]
        ls = LF_delivery_cost.filter(items=l + C_D, axis=1)
        ls = ls.filter(items=l + C_D, axis=0)
        lt = ls.to_numpy()
        for j in range(1 + len(C_D)):
            for k in range(1 + len(C_D)):
                LF_delivery_cost_multidim[i][j][k] = lt[j][k]

    col_ind = []
    d_ak_k = []
    index_close_locker = []
    min_value = []
    Sk_def = []
    OC = []
    position_sk = []
    position_cl = []
    pck = np.array([])
    locker_where_oc_goes = []
    # compensation

    compensation_matrix = dist_matrix.filter(items=stores + C_D, axis=1)
    compensation_matrix = compensation_matrix.filter(items=stores + C_D, axis=0)

    distance_matrix_customer_locker_store = dist_matrix.filter(items=C_L, axis=1)
    distance_matrix_customer_locker_store = distance_matrix_customer_locker_store.filter(items=stores, axis=0)

    closest_store_to_CL = distance_matrix_customer_locker_store.idxmin()
    min_value_col = distance_matrix_customer_locker_store.min()

    for i in range(len(closest_store_to_CL)):
        index_close_locker.append(str(closest_store_to_CL[i]))
        min_value.append(min_value_col[i])

    for i in range(len(C_L)):
        col_ind.append("D_" + str(index_close_locker[i]) + "_" + str(C_L[i].index))

    d_ak_k = pd.DataFrame(min_value, index=col_ind)

    pre_d_c_k = dist_matrix.filter(items=C_D, axis=0)
    pre_d_c_k = pre_d_c_k.filter(items=C_L, axis=1)

    d_ak_c = pd.DataFrame()

    # insert d_c_k
    for cl in C_L:
        # Using DataFrame.insert() to add a column
        store = cl.store
        array = []
        for cd in C_D:
            distance = store.location.euclidean_distance(cd.location)
            array.append(distance)
        d_ak_c.insert(len(d_ak_c.columns), store, array, True)

    d_c_k = dist_matrix.filter(items=C_D, axis=0)
    d_c_k = d_c_k.filter(items=C_L, axis=1)

    sum_dck_dakc = d_ak_c.to_numpy() + d_c_k.to_numpy()

    def find_Sk(cd, cl):
        S_K_i = []
        if sum_dck_dakc[cl][cd] <= 1.5 * d_ak_k.to_numpy()[cl]:
            S_K_i.append('Cd' + str(cd))
        return S_K_i

    for cl in range(len(C_L)):
        for cd in range(len(C_D)):
            if sum_dck_dakc[cd][cl] <= 1.5 * d_ak_k.to_numpy()[cl]:
                OC.append('Cl' + str(cl))
                Sk_def.append('CD' + str(cd))
                position_sk.append(cd)
                position_cl.append(cl)
                locker_where_oc_goes.append((index_close_locker[cl]))

    def unique(list1):
        # initialize a null list
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        return unique_list

    OC_unique = unique(OC)

    Sk = unique(Sk_def)
    Sk = sorted(Sk, key=lambda x: int("".join([i for i in x if i.isdigit()])))

    S_k = []

    for cd in C_D:
        for s in range(len(Sk)):
            if str(cd) == Sk[s]:
                S_k.append(cd)

    index_of_cl_associated_to_closest_locker = []
    for i in range(len(OC_unique)):
        index_of_cl_associated_to_closest_locker.append(
            OC_unique[i] + '->' + locker_where_oc_goes[OC.index(OC_unique[i])])

    big_matrix = np.zeros((len(C_D), len(C_L)))
    big_matrix = pd.DataFrame(big_matrix, index=C_D, columns=C_L)

    for i in range(len(position_sk)):
        big_matrix.values[position_sk[i]][position_cl[i]] = d_ak_c.values[position_sk[i]][position_cl[i]]

    OC = []
    for i in C_L:
        is_all_null = True
        for j in range(len(C_D)):
            if big_matrix.values[j][i.locker_customer_index] != 0:
                is_all_null = False
                OC.append(i)
    OC = list(OrderedSet(OC))

    filter_cd_sk = []
    for cd in C_D:
        for s in range(len(position_sk)):
            if cd.door_to_door_customer_index == position_sk[s]:
                filter_cd_sk.append(cd)

    d_cd_oc = big_matrix.filter(items=OC, axis=1)
    d_ak_c = d_cd_oc.filter(items=S_k, axis=0)
    d_ak_c.columns = index_of_cl_associated_to_closest_locker
    d_ak_c = d_ak_c.to_numpy()

    pck = 0.5 * d_ak_c
    for c in range(len(Sk)):
        for k in range(len(OC_unique)):
            if pck[c][k] == 0:
                pck[c][k] = 100000

    lockers_wrt_their_oc_matrix = dist_matrix.filter(items=OC, axis=1)
    lockers_wrt_their_oc_matrix = lockers_wrt_their_oc_matrix.filter(items=stores, axis=0)

    lockers_wrt_their_oc = lockers_wrt_their_oc_matrix.idxmin()

    lockers_wrt_their_oc_array = []
    for i in range(len(OC_unique)):
        lockers_wrt_their_oc_array.append(lockers_wrt_their_oc[i])

    single_period_problem = gb.Model()
    single_period_problem.Params.LogToConsole = 0  # suppress the log of the model
    single_period_problem.modelSense = gb.GRB.MINIMIZE  # declare mimization

    I_PF = 1 + len(C_D)
    J_PF = 1 + len(C_D)

    I_L = 1 + L
    J_L = 1 + L

    I_LF = 1 + len(C_D)
    J_LF = 1 + len(C_D)

    # add var to the problem
    x_i_j = single_period_problem.addVars([(i, j) for i in range(I_PF) for j in range(J_PF)], vtype=gb.GRB.BINARY)
    x_i_j_L = single_period_problem.addVars([(i, j) for i in range(I_L) for j in range(J_L)], vtype=gb.GRB.BINARY)
    x_l_i_j = single_period_problem.addVars([(l, i, j) for i in range(I_LF) for j in range(J_PF) for l in range(L)],
                                            vtype=gb.GRB.BINARY)

    y_i_j = single_period_problem.addVars([(i, j) for i in range(I_PF) for j in range(J_PF)], vtype=gb.GRB.INTEGER)
    y_i_j_L = single_period_problem.addVars([(i, j) for i in range(I_L) for j in range(J_L)], vtype=gb.GRB.INTEGER)
    y_l_i_j = single_period_problem.addVars([(l, i, j,) for i in range(I_LF) for j in range(J_PF) for l in range(L)],
                                            vtype=gb.GRB.INTEGER)

    z_c = single_period_problem.addVars([c for c in range(len(C_D))], vtype=gb.GRB.BINARY)
    z_c_l = single_period_problem.addVars([(l, c) for c in range(len(C_D)) for l in range(L)], vtype=gb.GRB.BINARY)
    z_l_L = single_period_problem.addVars([l for l in range(L)], vtype=gb.GRB.BINARY)

    w_c_k = single_period_problem.addVars([(c, k) for k in range(len(OC_unique)) for c in range(len(Sk))],
                                          vtype=gb.GRB.BINARY)

    # define constraints Customers’ service

    # constraint eq. 2
    for c in range(len(C_D)):
        single_period_problem.addConstr(
            z_c[c] + gb.quicksum(w_c_k[s, k] for s in range(len(Sk)) for k in range(len(OC_unique))
                                 if C_D[c] == filter_cd_sk[s]) + gb.quicksum(z_c_l[l, c] for l in range(L)) == 1)
    # constraint eq. 3
    for k in range(len(OC_unique)):
        single_period_problem.addConstr(gb.quicksum(w_c_k[s_k, k] for s_k in range(len(Sk))) <= 1)

    # constraint eq. 4
    for l in lockers:
        single_period_problem.addConstr(
            gb.quicksum(cl.package_demand for cl in l.find_associated_CL(customers, stores))
            +
            gb.quicksum(S_k[sk].package_demand * w_c_k[sk, k] for sk in range(len(Sk)) for k in range(len(OC_unique))
                        if lockers[l.index - 1] == lockers_wrt_their_oc_array[k]) +
            gb.quicksum(C_D[cd].package_demand * z_c_l[l.index - 1, cd] for cd in range(len(C_D)))
            <= l.capacity * z_l_L[l.index - 1]
        )

    # Professional fleet constraint

    # constraint eq. 5.1 A == C
    for i in range(len(C_D)):
        single_period_problem.addConstr(
            gb.quicksum(x_i_j[i + 1, j] for j in range(len(C_D) + 1))
            == z_c[i]
        )

    # constraint eq. 5.2 B == C
    for i in range(len(C_D)):
        single_period_problem.addConstr(
            gb.quicksum(x_i_j[j, i + 1] for j in range(len(C_D) + 1))
            == z_c[i]
        )

    # constraint eq. 6
    single_period_problem.addConstr(
        gb.quicksum(x_i_j[0, j + 1] for j in range(len(C_D)))
        - gb.quicksum(x_i_j[j + 1, 0] for j in range(len(C_D)))
        == 0
    )

    # constraint eq. 7
    for i in range(len(C_D)):
        single_period_problem.addConstr(
            gb.quicksum(y_i_j[j, i + 1] for j in range(1 + len(C_D)))
            - gb.quicksum(y_i_j[i + 1, j] for j in range(1 + len(C_D)))
            == C_D[i].package_demand * z_c[i]
        )

    # constraint eq. 8
    single_period_problem.addConstr(
        gb.quicksum(y_i_j[j + 1, 0] for j in range(len(C_D)))
        - gb.quicksum(y_i_j[0, j + 1] for j in range(len(C_D)))
        == - gb.quicksum(C_D[i].package_demand * z_c[i] for i in range(len(C_D)))
    )

    # constraint eq. 9
    for i in range(1 + len(C_D)):
        for j in range(1 + len(C_D)):
            single_period_problem.addConstr(
                y_i_j[i, j]
                <= vehicles[1].capacity * x_i_j[i, j]
            )

    # constraint eq. 10
    for i in range(len(C_D)):
        single_period_problem.addConstr(
            y_i_j[i + 1, 0]
            == 0
        )

    # Supply routes constraints

    # constraint eq. 11.1 A == C
    for i in range(L):
        single_period_problem.addConstr(
            gb.quicksum(x_i_j_L[i + 1, j] for j in range(1 + L))
            == z_l_L[i]
        )

    # constraint eq. 11.2 B == C
    for i in range(L):
        single_period_problem.addConstr(
            gb.quicksum(x_i_j_L[j, i + 1] for j in range(1 + L))
            == z_l_L[i]
        )

    # constraint eq. 12
    single_period_problem.addConstr(
        gb.quicksum(x_i_j_L[0, j + 1] for j in range(L))
        - gb.quicksum(x_i_j_L[j + 1, 0] for j in range(L))
        == 0
    )

    # constraint eq. 13
    for l in lockers:
        single_period_problem.addConstr(
            gb.quicksum(y_i_j_L[j, l.index] for j in range(1 + L))
            - gb.quicksum(y_i_j_L[l.index, j] for j in range(1 + L))
            ==
            gb.quicksum(cl.package_demand for cl in l.find_associated_CL(customers, stores))
            +
            gb.quicksum(S_k[sk].package_demand * w_c_k[sk, k] for sk in range(len(Sk)) for k in range(len(OC_unique))
                        if lockers[l.index - 1] == lockers_wrt_their_oc_array[k])
            + gb.quicksum(C_D[c].package_demand * z_c_l[l.index - 1, c] for c in range(len(C_D)))
        )

    # constraint eq. 14
    single_period_problem.addConstr(
        gb.quicksum(y_i_j_L[j + 1, 0] for j in range(L))
        - gb.quicksum(y_i_j_L[0, j + 1] for j in range(L))
        == - gb.quicksum(
            gb.quicksum(cl.package_demand for cl in l.find_associated_CL(customers, stores))
            +
            gb.quicksum(S_k[sk].package_demand * w_c_k[sk, k] for sk in range(len(Sk)) for k in range(len(OC_unique))
                        if lockers[l.index - 1] == lockers_wrt_their_oc_array[k])
            + gb.quicksum(C_D[c].package_demand * z_c_l[l.index - 1, c] for c in range(len(C_D)))
            for l in lockers
        )
    )

    # constraint eq. 15
    for i in range(1 + L):
        for j in range(1 + L):
            single_period_problem.addConstr(
                y_i_j_L[i, j]
                <= vehicles[0].capacity * x_i_j_L[i, j]
            )

    # constraint eq. 16
    for i in range(L):
        single_period_problem.addConstr(
            y_i_j_L[i + 1, 0]
            == 0
        )

    # Local fleet constraints
    for l in range(L):
        # constraint eq. 17.1 A == C
        for i in range(len(C_D)):
            single_period_problem.addConstr(
                gb.quicksum(x_l_i_j[l, i + 1, j] for j in range(len(C_D) + 1))
                == z_c_l[l, i]
            )

        # constraint eq. 17.2 B == C
        for i in range(len(C_D)):
            single_period_problem.addConstr(
                gb.quicksum(x_l_i_j[l, j, i + 1] for j in range(len(C_D) + 1))
                == z_c_l[l, i]
            )

        # constraint eq. 18
        single_period_problem.addConstr(
            gb.quicksum(x_l_i_j[l, 0, j + 1] for j in range(len(C_D)))
            - gb.quicksum(x_l_i_j[l, j + 1, 0] for j in range(len(C_D)))
            == 0
        )

        # constraint eq. 19
        for i in range(len(C_D)):
            single_period_problem.addConstr(
                gb.quicksum(y_l_i_j[l, j, i + 1] for j in range(1 + len(C_D)))
                - gb.quicksum(y_l_i_j[l, i + 1, j] for j in range(1 + len(C_D)))
                == C_D[i].package_demand * z_c_l[l, i]
            )

        # constraint eq. 20
        single_period_problem.addConstr(
            gb.quicksum(y_l_i_j[l, j + 1, 0] for j in range(len(C_D)))
            - gb.quicksum(y_l_i_j[l, 0, j + 1] for j in range(len(C_D)))
            == - gb.quicksum(C_D[i].package_demand * z_c_l[l, i] for i in range(len(C_D)))
        )

        # constraint eq. 21
        for i in range(1 + len(C_D)):
            for j in range(1 + len(C_D)):
                single_period_problem.addConstr(
                    y_l_i_j[l, i, j]
                    <= vehicles[l + 2].capacity * x_l_i_j[l, i, j]
                )

        # constraint eq. 22
        for i in range(len(C_D)):
            single_period_problem.addConstr(
                y_l_i_j[l, i + 1, 0]
                == 0
            )

    single_period_problem.setObjective(
        gb.quicksum(gb.quicksum(supply_cost.values[i][j] * x_i_j_L[i, j] for j in range(1 + L)) for i in range(1 + L)) +
        gb.quicksum(gb.quicksum(pck[c][k] * w_c_k[c, k] for c in range(len(Sk))) for k in range(len(OC_unique))) +
        gb.quicksum(gb.quicksum(PF_delivery_cost.values[i][j] * x_i_j[i, j] for j in range(1 + len(C_D))) for i in
                    range(1 + len(C_D))) +
        gb.quicksum(gb.quicksum(gb.quicksum(LF_delivery_cost_multidim[l][i][j] * x_l_i_j[l, i, j]
                                            for j in range(1 + len(C_D))) for i in range(1 + len(C_D))) for l in
                    range(L))
    )

    single_period_problem.optimize()  # equivalent to solve() for xpress

    if single_period_problem.status == 2:
        return single_period_problem.status, single_period_problem.Runtime, single_period_problem.ObjVal
    else:
        print("--------Gurobi did not find a optiml solution-----------")
        return single_period_problem.status, single_period_problem.Runtime, None
