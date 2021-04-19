import copy 
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
    header = ["_id", "avg_salary"]
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
        if instance[year] < 1985:
            data_copy.remove(instance)
        elif instance[games] < 50:
            data_copy.remove(instance)
    return data_copy