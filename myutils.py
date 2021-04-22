import numpy as np
import math

def compute_euclidean_distance(v1, v2):
    assert len(v1) == len(v2)
    dist = 0
    #print(v1)
    for i in range(len(v1)):
        if isinstance(v1[i], str):
            if v1[i] == v2[i]:
                dist = 0
            else:
                dist = 1
        else:
            dist += (v1[i] - v2[i]) ** 2
    dist = np.sqrt(dist)
    # dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist

def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[col_index] != "NA" and row[col_index] != "":
            value = row[col_index]
            col.append(value)
    return col
def get_nbayes_xy(table, header, y_col_name):
    y_index = header.index(y_col_name)
    x =[]
    for row in table:
        temp_row = []
        for i in range(len(row)):
            # add every entry not in y_col
            if (i != y_index):
                temp_row.append(row[i])
        x.append(temp_row)
    y = get_column(table, header, y_col_name)
    return x, y

def get_frequencies(col):
    # print(col)
    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def get_table_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)
    # print(col)
    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def group_by(table, header, group_by_col_name):
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)

    group_names = sorted(list(set(col))) 
    group_subtables = [[] for _ in group_names]

    for row in table:
        group_by_value = row[col_index]
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row.copy())

    return group_names, group_subtables

def convert_to_categorical(col, cutoffs):
    categorical_col = []
    for value in col:
        for i in range(len(cutoffs)):
            if value < cutoffs[i]: 
                categorical_col.append(i) # rank
                break
            elif value == cutoffs[len(cutoffs) - 1]: # max should be in top rank
                categorical_col.append(len(cutoffs)-1) # rank
                break
    return categorical_col 

def select_attribute(instances, available_attributes, heading):
    # for now, choose attribute randomly
    # TODO: after successfully building a tree, replave w/ entropy
    if len(available_attributes) == 0:
        return available_attributes[0]
    index = entropy_index(instances, available_attributes, heading)
    return available_attributes[index]
def partition_instances(instances, split_attribute, attribute_domains, header):
    # comments reger to split_attibure "leve"
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute)

    partitions = {} # key (attribute value): value (partition)
    # task: try to finish this
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)

    return partitions

def tdidt(current_instances, available_attributes, header, att_domain):
    # basic approach (uses recursion!!):

    # select an attribute to split on
    split_attribute  = select_attribute(current_instances, available_attributes, header)
    # remove split attributes from available attribute because we cannot split on the same attribute twice in a branch
    available_attributes.remove(split_attribute)
    tree = ["Attribute", split_attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, att_domain, header)
    values  =  list(partitions.values())
    total_items = 0
    for row in values:
        total_items += len(row)
    # for each partition, repeat unless one of the following base cases occurs
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
    # group data by attribute domains (creates pairwise disjoint partitions)
    # for each partition, repeat unless one of the following occurs (base case)
    #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            #print("CASE 1")
            leaf = ["Leaf", partition[0][-1], len(partition), total_items]
            values_subtree.append(leaf)
    #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2")
            class_value = majority_voting(partition)
            leaf = ["Leaf", class_value, len(partition), total_items]
            values_subtree.append(leaf)
    #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            #print("CASE 3")
            class_value = majority_voting(current_instances)
            leaf = ["Leaf", class_value, len(partition), total_items]
            tree = leaf
            values_subtree = []
        else: # all base cases are fakse, so recurse
            subtree = tdidt(partition, available_attributes.copy(), header, att_domain)
            values_subtree.append(subtree)
        if values_subtree != []: # only append when not Case 3
            tree.append(values_subtree)
    return tree

def all_same_class(instances):
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True

def entropy_index(instances, available_attributes, header):
    entropies = []
    E_nus = []
    header.append("class")
    for i in range(len(available_attributes)):
        group_names, group_subtables = group_by(instances, header, available_attributes[i])
        for j in range(len(group_subtables)):
            class_column = get_column(group_subtables[j], header, "class")
            y_values, y_counts = get_frequencies(class_column)
            y_total = sum(y_counts)
            E = 0
            for count in y_counts:
                E-= (count/y_total) * math.log(count/y_total, 2)
                if E == 0:
                    break
            entropies.append(E)
        values, counts = get_table_frequencies(instances, header, available_attributes[i])
        total = sum(counts)
        E_nu = 0
        for k in range(len(entropies)):
            E_nu += entropies[k]*counts[k]/total
        E_nus.append(E_nu)
        entropies = []
    return E_nus.index(min(E_nus))

def majority_voting(instances):
    header = instances[0]
    values, counts = get_table_frequencies(instances, header, header[-1])
    return values[counts.index(max(counts))]

def tdidt_predict(header, tree, instance):
    # returns "True" or "False" if a leaf node is hit
    # None otherwise 
    info_type = tree[0]
    if info_type == "Attribute":
        # get the value of this attribute for the instance
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # recurse, we have a match!!
                return tdidt_predict(header, value_list[2], instance)
    else: # Leaf
        return tree[1] # label

def decision_rules_rec(attribute_names, tree, rule, classifier):
    if tree[0] == "Attribute":  # AND 'att'
        if attribute_names != None:
            temp = tree[1]
            index = int(temp[len(temp) - 1])
            att = attribute_names[index]
        else:
            att = tree[1]
        rule += "AND " + str(att) + " "
        for i in range(2, len(tree)):  # visit each subtree
            decision_rules_rec(attribute_names, tree[i], rule, classifier)
    elif tree[0] == "Value":  # == 'value'
        rule += "== " + str(tree[1]) + " "
        decision_rules_rec(attribute_names, tree[2], rule, classifier)
    elif tree[0] == "Leaf":  # THEN 'classifier' == 'classifier val' 
        rule += "THEN " + classifier + " == " + str(tree[1])
        print(rule)