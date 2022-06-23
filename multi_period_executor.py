import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix
import random
import math
from operator import attrgetter
from ordered_set import OrderedSet
# if you use conda run this in your terminal: conda install -c conda-forge ordered-set
import gurobipy as gb

from door_to_door_customer import DoorToDoorCustomer
from location import Location
from store import *
from vehicle import Vehicle
from vehicle_type import VehicleType
from single_period_of_multi_period_executor import solve_period
'''
input C, L, periods

output sum(runtime), status=2 if all status == 2

'''

def run_experiment(C, L, periods):
    sum_Runtime = 0
    sum_ObjVal = 0
    status_alway_2 = True

    map_size = 100
    ratio_locker_customers = 0.3
    custom_setup = False

    # define gamma and current_day
    gamma = []
    gamma.append(0.5)

    for day in range(1, periods-1):
        gamma.append((50 + (100-50)/(periods-day))/100)
    gamma.append(1)
    current_day = 0

    C_per_period = []
    for p in range(periods-1):
        C_per_period.append(C // periods)

    C_per_period.append(C-sum(C_per_period))

    def get_nearest_store(stores, location):
        min_distance = float("inf")
        nearest_store = stores[0]
        for store in stores:
            distance = location.euclidean_distance(store.location)
            if distance < min_distance:
                min_distance = distance
                nearest_store = store
        return nearest_store

    def generate_C_customers(C, stores):
        customers = []
        counter_locker_customer = 0
        counter_door_to_door_customer = 0
        for c in range(C):
            location = Location(random.randint(0, map_size), random.randint(0, map_size))
            if random.random() < ratio_locker_customers:
                # customer locker
                customers.append(LockerCustomer(c, counter_locker_customer, location, get_nearest_store(stores, location)))
                counter_locker_customer += 1
            else:
                # door to door customer
                customers.append(DoorToDoorCustomer(c, counter_door_to_door_customer, location))
                counter_door_to_door_customer += 1
        return customers

    # generates stores
    capacity_constant = 0.8 + gamma[0]
    if custom_setup:
        L = 2
        stores = []
        stores.append(Store(0, Location(60, 50), capacity=float("inf"), is_warehouse=True))
        stores.append(Store(1, Location(30, 50), capacity=math.ceil(capacity_constant * C_per_period[0] / L)))
        stores.append(Store(2, Location(50, 20), capacity=math.ceil(capacity_constant * C_per_period[0] / L)))

    else:
        stores = []
        stores.append(Store(0, Location(random.randint(0, map_size), random.randint(0, map_size)), capacity=float("inf"), is_warehouse=True))
        for l in range(L):
            stores.append(Store(l+1, Location(random.randint(0, map_size), random.randint(0, map_size)), capacity=math.ceil(capacity_constant * C_per_period[0] / L)))


    # generate all_customer for all the periods and put them in the right period to be served
    all_customers = generate_C_customers(C, stores)
    customers_per_period = []
    counter_customer = 0
    for day in range(periods):
        customers_per_period.append(all_customers[0+counter_customer:counter_customer+C_per_period[day]])
        counter_customer += C_per_period[day]

    customers = customers_per_period[current_day]
    # create sets
    C_L = list(filter(lambda customer: type(customer) == LockerCustomer, customers))
    C_D = list(filter(lambda customer: type(customer) == DoorToDoorCustomer, customers))
    lockers = list(filter(lambda store: not store.is_warehouse, stores))

    # define all the vehicles
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
                vehicles.append(Vehicle(store.index+1, VehicleType.LF, store, math.ceil(0.6 * store.capacity)))


    current_day = 0
    for day in range(periods):
        if day == 0:
            customers = customers_per_period[current_day]
        C_L = list(filter(lambda customer: type(customer) == LockerCustomer, customers))
        C_D = list(filter(lambda customer: type(customer) == DoorToDoorCustomer, customers))

        status, Runtime, ObjVal, OC, w_c_k_variables = solve_period(stores, vehicles, customers)

        sum_Runtime += Runtime
        sum_ObjVal += ObjVal

        if status != 2:
            status_alway_2 = False
            print("-----Gurobi did not find the optimal solution for a single period------")
            #raise Exception("-----Gurobi did not find the optimal solution for a single period------")

        # discover the customers that did not get the package
        customer_next_period = []
        customers_did_not_get_the_package = []
        for oc in OC:
            if random.random() > gamma[current_day]:
                customers_did_not_get_the_package.append(oc)
                if oc in w_c_k_variables:
                    customers_did_not_get_the_package.append(w_c_k_variables[oc])

        # make all the CD of the previous period prime
        for c in customers_did_not_get_the_package:
            if type(c) == DoorToDoorCustomer:
                c.set_prime(True)

        customers_did_not_get_the_package_CL = list(filter(lambda customer: type(customer) == LockerCustomer, customers_did_not_get_the_package))
        for c in customers_did_not_get_the_package_CL:
            c.set_did_not_show_up(True)

        # add the new customer of the next period
        if current_day+1 < periods:
            customer_next_period = customers_per_period[current_day+1]
            customers = customers_did_not_get_the_package + customer_next_period


        current_day += 1

    # edn of: for day in range(periods):
    return status_alway_2, sum_Runtime, sum_ObjVal
