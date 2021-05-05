##############################################
# Programmer: Adrian Rabadam & Michael Finch
# Class: CptS 322-01, Spring 2021
# Final Project
# 4/21/21
# 
# 
# Description: This file defines utility 
# functions for preparing the dataset
##############################################

import copy 
import numpy as np 
import matplotlib.pyplot as plt

def group_by(names, values):
    group_names = sorted(list(set(names)))
    group_subtables = [[] for _ in group_names]
    
    for row in values:
        index = values.index(row)
        group_by_value = names[index]

        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row.copy()) 
    
    return group_names, group_subtables

def calculate_avg_salaries(data):
    header = ["id", "avg_salary"]
    avg_salaries = []
    player_ids = []
    for instance in data: 
        player_ids.append(instance[1])
    player_names, player_info = group_by(player_ids, data)
    for name in player_names:
        inst = []
        player_salaries = []
        index = player_names.index(name)
        inst.append(name)
        for year in player_info[index]:
            player_salaries.append(year[2])
        avg = sum(player_salaries)/len(player_salaries)
        avg = round(avg, 2)
        inst.append(avg)
        avg_salaries.append(inst)
    return header, avg_salaries
def remove_players(data, header):
    year = header.index("draft_year")
    games = header.index("career_G")
    data_copy = copy.deepcopy(data)
    for instance in data: 
        if instance[games] < 164 or instance[year] < 1985:
            data_copy.remove(instance)
    return data_copy

def compute_equal_width_cutoffs(values, num_bins):
    # first compute the range of the values
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    # bin_width is likely a float
    # if your application allows for ints, use them
    # we will use floats
    # np.arange() is like the built in range() but for floats
    cutoffs = list(np.arange(min(values), max(values), bin_width)) 
    cutoffs.append(max(values))
    # optionally: might want to round
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 
    
def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1

    return freqs

def generate_histogram(data, title, xlabel, ylabel): 
    plt.figure()
    plt.hist(data, bins=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()