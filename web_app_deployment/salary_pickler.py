import pickle 

header = ["per", "games", "ws"]
salary_tree = ['Attribute', 'per', ['Value', 1, ['Leaf', 1, 0, 135]], ['Value', 2, ['Attribute', 'games', ['Value', 1, ['Leaf', 1, 0, 93]], ['Value', 2, ['Leaf', 1, 0, 73]], ['Value', 3, ['Leaf', 1, 0, 87]], ['Value', 4, ['Leaf', 2, 0, 65]], ['Value', 5, ['Leaf', 2, 0, 68]], ['Value', 6, ['Leaf', 2, 0, 42]], ['Value', 7, ['Leaf', 4, 0, 17]], ['Value', 8, ['Leaf', 3, 0, 7]], ['Value', 9, ['Leaf', 2, 0, 4]], ['Value', 10, ['Leaf', 3, 1, 457]]]], ['Value', 3, ['Attribute', 'games', ['Value', 1, ['Leaf', 1, 0, 27]], ['Value', 2, ['Leaf', 1, 0, 21]], ['Value', 3, ['Leaf', 3, 0, 28]], ['Value', 4, ['Leaf', 3, 0, 36]], ['Value', 5, ['Leaf', 3, 0, 23]], ['Value', 6, ['Leaf', 6, 0, 36]], ['Value', 7, ['Leaf', 3, 0, 22]], ['Value', 8, ['Leaf', 5, 0, 10]], ['Value', 9, ['Leaf', 2, 0, 6]], ['Value', 10, ['Leaf', 4, 0, 4]]]], ['Value', 4, ['Leaf', 7, 0, 38]], ['Value', 5, ['Leaf', 8, 0, 13, ['Value', 10, ['Leaf', 4, 0, 2, ['Value', 5, ['Leaf', 4, 2, 2]]]]]]]

packaged_object = [header, salary_tree]
# pickle packaged_object
outfile = open("web_app_deployment/tree.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()