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


def solve_period(stores, vehicles, customers):
    L = len(stores) - 1

    # create sets
    C_L = list(filter(lambda customer: type(customer) == LockerCustomer, customers))
    C_D = list(filter(lambda customer: type(customer) == DoorToDoorCustomer, customers))
    lockers = list(filter(lambda store: not store.is_warehouse, stores))

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

    # calcolo la matrice delle distanze:
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
    min_value = []
    Sk = []
    Sk_with_duplicate = []
    Sk_not_ordered = []
    OC = []
    OC_with_duplcate =[]
    OC_not_ordered = []
    position_sk = []
    position_cl = []
    pck = np.array([])
    locker_where_oc_goes=[]
    d_cd_oc = []


    #creation of d_ak_k
    distance_matrix_customer_locker_store = dist_matrix.filter(items=C_L,axis=1)
    distance_matrix_customer_locker_store = distance_matrix_customer_locker_store.filter(items=stores,axis=0)

    closest_store_to_CL = distance_matrix_customer_locker_store.idxmin()
    min_value_col = distance_matrix_customer_locker_store.min()
    for i in range(len(closest_store_to_CL)):
        min_value.append(min_value_col[i])
        col_ind.append("Dist_"+str(closest_store_to_CL[i])+"_"+str(C_L[i]))

    d_ak_k = pd.DataFrame()
    d_ak_k = pd.DataFrame(min_value, index=col_ind)


    #creation of d_ak_c
    d_ak_c = pd.DataFrame()
    d_ak_c = dist_matrix.filter(items = C_D, axis=1)
    d_ak_c = d_ak_c.filter(items=closest_store_to_CL,axis=0)


    #creation of d_c_k --> per essere piu precisi questa sarebbe la d_k_c cosi che le dimensioni siano coerenti quando poi si fa la somma
    # è intuitivo che la distanza da A a B è la stessa che da B ad A
    d_c_k = pd.DataFrame()
    d_c_k = dist_matrix.filter(items=C_L,axis=0)
    d_c_k = d_c_k.filter(items=C_D,axis=1)


    # find the 2 sets OC and Sk
    sum_dck_dakc=[]
    sum_dck_dakc = d_ak_c.to_numpy()+d_c_k.to_numpy()

    for cl in range(len(C_L)):
        for cd in range(len(C_D)):
            if sum_dck_dakc[cl][cd]<=1.5*d_ak_k.to_numpy()[cl]:
                OC_with_duplcate.append(C_L[cl])
                Sk_with_duplicate.append(C_D[cd])
                position_sk.append(cd)
                position_cl.append(cl)
                locker_where_oc_goes.append((closest_store_to_CL[cl]))

    def unique(list1):
        # initialize a null list
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        return unique_list

    OC_not_ordered = unique(OC_with_duplcate)
    Sk_not_ordered = unique(Sk_with_duplicate)
    def bubble_sort(array):
        n = len(array)
        for i in range(n):
            already_sorted = True
            for j in range(n - i - 1):
                if array[j].index > array[j + 1].index:
                    array[j], array[j + 1] = array[j + 1], array[j]
                    already_sorted = False
            if already_sorted:
                break
        return array
    OC = bubble_sort(OC_not_ordered)
    Sk = bubble_sort(Sk_not_ordered)

    #LAST STEP  create p_c_k
    empty_matrix_Cd_Cl = np.zeros((len(C_D),len(C_L)))
    empty_matrix_Cd_Cl = pd.DataFrame(empty_matrix_Cd_Cl, index=C_D, columns=C_L)

    for i in range(len(position_sk)):
        empty_matrix_Cd_Cl.values[position_sk[i]][position_cl[i]]=d_ak_c.values[position_cl[i]][position_sk[i]]


    d_sk_oc = empty_matrix_Cd_Cl.filter(items = OC, axis=1)
    d_sk_oc = d_sk_oc.filter(items = Sk,axis=0)

    #piccola digressione
    lockers_wrt_their_oc_matrix = dist_matrix.filter(items=OC,axis=1)
    lockers_wrt_their_oc_matrix = lockers_wrt_their_oc_matrix.filter(items=stores,axis=0)

    lockers_wrt_their_oc = lockers_wrt_their_oc_matrix.idxmin()

    lockers_wrt_their_oc_array =[]
    for i in range(len(OC)):
        lockers_wrt_their_oc_array.append(lockers_wrt_their_oc[i])
    index_of_cl_associated_to_closest_locker =[]
    for i in range(len(OC)):
        index_of_cl_associated_to_closest_locker.append(str(OC[i])+'->'+str(lockers_wrt_their_oc_array[i]))
    #fine

    d_sk_oc.columns=index_of_cl_associated_to_closest_locker

    pck = 0.5*d_sk_oc.to_numpy()

    for c in range(len(Sk)):
        for k in range(len(OC)):
            if pck[c][k]==0:
                pck[c][k]=100000

    model = gb.Model()
    model.Params.LogToConsole = 0  # suppress the log of the model
    model.modelSense = gb.GRB.MINIMIZE  # declare mimization

    I_PF = 1 + len(C_D)
    J_PF = 1 + len(C_D)

    I_L = 1 + L
    J_L = 1 + L

    I_LF = 1 + len(C_D)
    J_LF = 1 + len(C_D)

    # add var to the problem
    x_i_j   =   model.addVars([(i,j) for i in range(I_PF) for j in range(J_PF)], vtype=gb.GRB.BINARY)
    x_i_j_L =   model.addVars([(i,j) for i in range(I_L) for j in range(J_L) ], vtype=gb.GRB.BINARY)
    x_l_i_j =   model.addVars([(l,i,j) for i in range(I_LF) for j in range(J_PF) for l in range(L)], vtype=gb.GRB.BINARY)

    y_i_j   =   model.addVars([(i,j) for i in range(I_PF) for j in range(J_PF)], vtype=gb.GRB.INTEGER)
    y_i_j_L =   model.addVars([(i,j) for i in range(I_L) for j in range(J_L) ], vtype=gb.GRB.INTEGER)
    y_l_i_j =   model.addVars([(l,i,j,) for i in range(I_LF) for j in range(J_PF) for l in range(L)], vtype=gb.GRB.INTEGER)

    z_c     =   model.addVars([c for c in range(len(C_D))],vtype=gb.GRB.BINARY)
    z_c_l   =   model.addVars([(l,c)for c in range(len(C_D)) for l in range(L)],vtype=gb.GRB.BINARY)
    z_l_L   =   model.addVars([l for l in range(L)],vtype=gb.GRB.BINARY)

    w_c_k   =   model.addVars([(c,k)for k in range(len(OC)) for c in range(len(Sk))],vtype=gb.GRB.BINARY, name="w_c_k")

    # define constraints Customers’ service

    # constraint eq. 2
    for c in range(len(C_D)):
        model.addConstr( z_c[c] + gb.quicksum(w_c_k[s,k] for s in range(len(Sk)) for k in range(len(OC))
                                                              if C_D[c] == Sk[s]) + gb.quicksum(z_c_l[l,c] for l in range(L)) == 1)
    # constraint eq. 3
    for k in range(len(OC)):
        model.addConstr( gb.quicksum(w_c_k[s_k,k] for s_k in range(len(Sk))) <= 1)

    # constraint eq. 4
    for l in lockers:
        model.addConstr(
            gb.quicksum(cl.package_demand for cl in l.find_associated_CL(customers, stores))
            +
            gb.quicksum(Sk[sk].package_demand*w_c_k[sk,k] for sk in range(len(Sk)) for k in range(len(OC))
                        if lockers[l.index-1] == lockers_wrt_their_oc_array[k] )+
            gb.quicksum(C_D[cd].package_demand*z_c_l[l.index-1,cd] for cd in range(len(C_D)))
            <= l.capacity * z_l_L[l.index-1]
        )

    # Professional fleet constraint

    # constraint eq. 5.1 A == C
    for i in range(len(C_D)):
        model.addConstr(
            gb.quicksum( x_i_j[i+1,j] for j in range(len(C_D)+1))
            == z_c[i]
        )


    # constraint eq. 5.2 B == C
    for i in range(len(C_D)):
        model.addConstr(
            gb.quicksum( x_i_j[j,i+1] for j in range(len(C_D)+1))
            == z_c[i]
        )



    # constraint eq. 6
    model.addConstr(
        gb.quicksum( x_i_j[0,j+1] for j in range(len(C_D)))
        - gb.quicksum( x_i_j[j+1,0] for j in range(len(C_D)))
        == 0
    )

    # constraint eq. 7
    for i in range(len(C_D)):
        model.addConstr(
            gb.quicksum( y_i_j[j,i+1] for j in range(1+len(C_D)))
            - gb.quicksum( y_i_j[i+1,j] for j in range(1+len(C_D)))
            == C_D[i].package_demand*z_c[i]
        )

    # constraint eq. 8
    model.addConstr(
        gb.quicksum( y_i_j[j+1,0] for j in range(len(C_D)))
        - gb.quicksum( y_i_j[0,j+1] for j in range(len(C_D)))
        == - gb.quicksum( C_D[i].package_demand*z_c[i] for i in range(len(C_D)))
    )

    # constraint eq. 9
    for i in range(1 + len(C_D)):
        for j in range(1 + len(C_D)):
            model.addConstr(
                y_i_j[i,j]
                <= vehicles[1].capacity * x_i_j[i,j]
            )

    # constraint eq. 10
    for i in range(len(C_D)):
        model.addConstr(
            y_i_j[i+1,0]
            == 0
        )

    # Supply routes constraints

    # constraint eq. 11.1 A == C
    for i in range(L):
        model.addConstr(
            gb.quicksum( x_i_j_L[i+1,j] for j in range(1 + L))
            == z_l_L[i]
        )

    # constraint eq. 11.2 B == C
    for i in range(L):
        model.addConstr(
            gb.quicksum( x_i_j_L[j,i+1] for j in range(1 + L))
            == z_l_L[i]
        )



    # constraint eq. 12
    model.addConstr(
        gb.quicksum( x_i_j_L[0,j+1] for j in range(L))
        - gb.quicksum( x_i_j_L[j+1,0] for j in range(L))
        == 0
    )

    # constraint eq. 13
    for l in lockers:
        model.addConstr(
            gb.quicksum( y_i_j_L[j,l.index] for j in range(1+L))
            - gb.quicksum( y_i_j_L[l.index,j] for j in range(1+L))
            ==
            gb.quicksum(cl.package_demand for cl in l.find_associated_CL(customers, stores))
            +
            gb.quicksum(Sk[sk].package_demand*w_c_k[sk,k] for sk in range(len(Sk)) for k in range(len(OC))
                        if lockers[l.index-1] == lockers_wrt_their_oc_array[k] )
            + gb.quicksum(C_D[c].package_demand*z_c_l[l.index-1,c] for c in range(len(C_D)))
        )

    # constraint eq. 14
    model.addConstr(
        gb.quicksum( y_i_j_L[j+1,0] for j in range(L))
        - gb.quicksum( y_i_j_L[0,j+1] for j in range(L))
        == - gb.quicksum(
            gb.quicksum(cl.package_demand for cl in l.find_associated_CL(customers, stores))
            +
            gb.quicksum(Sk[sk].package_demand*w_c_k[sk,k] for sk in range(len(Sk)) for k in range(len(OC))
                        if lockers[l.index-1] == lockers_wrt_their_oc_array[k] )
            + gb.quicksum(C_D[c].package_demand*z_c_l[l.index-1,c] for c in range(len(C_D)))
            for l in lockers
        )
    )


    # constraint eq. 15
    for i in range(1 + L):
        for j in range(1 + L):
            model.addConstr(
                y_i_j_L[i,j]
                <= vehicles[0].capacity * x_i_j_L[i,j]
            )

    # constraint eq. 16
    for i in range(L):
        model.addConstr(
            y_i_j_L[i+1,0]
            == 0
        )

    # Local fleet constraints
    for l in range(L):
        # constraint eq. 17.1 A == C
        for i in range(len(C_D)):
            model.addConstr(
                gb.quicksum( x_l_i_j[l,i+1,j] for j in range(len(C_D)+1))
                == z_c_l[l,i]
            )


        # constraint eq. 17.2 B == C
        for i in range(len(C_D)):
            model.addConstr(
                gb.quicksum( x_l_i_j[l,j,i+1] for j in range(len(C_D)+1))
                == z_c_l[l,i]
            )


        # constraint eq. 18
        model.addConstr(
            gb.quicksum( x_l_i_j[l,0,j+1] for j in range(len(C_D)))
            - gb.quicksum( x_l_i_j[l,j+1,0] for j in range(len(C_D)))
            == 0
        )

        # constraint eq. 19
        for i in range(len(C_D)):
            model.addConstr(
                gb.quicksum( y_l_i_j[l,j,i+1] for j in range(1+len(C_D)))
                - gb.quicksum( y_l_i_j[l,i+1,j] for j in range(1+len(C_D)))
                == C_D[i].package_demand*z_c_l[l,i]
            )

        # constraint eq. 20
        model.addConstr(
            gb.quicksum( y_l_i_j[l,j+1,0] for j in range(len(C_D)))
            - gb.quicksum( y_l_i_j[l,0,j+1] for j in range(len(C_D)))
            == - gb.quicksum(C_D[i].package_demand*z_c_l[l,i] for i in range(len(C_D)))
        )

        # constraint eq. 21
        for i in range(1 + len(C_D)):
            for j in range(1 + len(C_D)):
                model.addConstr(
                    y_l_i_j[l,i,j]
                    <= vehicles[l+2].capacity * x_l_i_j[l,i,j]
                )

        # constraint eq. 22
        for i in range(len(C_D)):
            model.addConstr(
                y_l_i_j[l,i+1,0]
                == 0
            )

    model.setObjective(
        gb.quicksum( gb.quicksum(supply_cost.values[i][j]*x_i_j_L[i,j]   for j in range(1+L) )  for i in range(1+L))+
        gb.quicksum( gb.quicksum(pck[c][k] *w_c_k[c,k]  for c in range(len(Sk)))for k in range(len(OC)))+
        gb.quicksum( gb.quicksum(PF_delivery_cost.values[i][j]*x_i_j[i,j] for j in range(1+len(C_D) ) )for i in range(1+len(C_D))) +
        gb.quicksum( gb.quicksum(gb.quicksum(LF_delivery_cost_multidim[l][i][j]*x_l_i_j[l,i,j]
                                             for j in range(1+len(C_D)))for i in range(1+len(C_D)))for l in range(L))
    )

    model.optimize() #equivalent to solve() for xpress

    CL_dictionary = {}
    j = 0
    for oc in OC:
        i = 0
        for sk in Sk:
            string = "w_c_k["+str(i)+","+str(j)+"]"
            if model.getVarByName(string).x == 1:
                CL_dictionary[oc] = sk
            i += 1
        j += 1


    return model.status, model.Runtime, model.ObjVal, OC, CL_dictionary
