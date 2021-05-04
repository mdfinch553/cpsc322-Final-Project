import pickle 

header = ["per", "games", "ws"]
salary_tree = ['Attribute', 'games', ['Value', 1, ['Leaf', 1, 141, 771]], ['Value', 2, ['Attribute', 'per', ['Value', 1, ['Leaf', 1, 22, 104]], ['Value', 2, ['Leaf', 1, 60, 104]], ['Value', 3, ['Leaf', 1, 18, 104]], ['Value', 4, ['Leaf', 6, 2, 104]], ['Value', 5, ['Leaf', 4, 2, 104]]]], ['Value', 3, ['Attribute', 'per', ['Value', 1, ['Leaf', 1, 20, 136]], ['Value', 2, ['Leaf', 1, 79, 136]], ['Value', 3, ['Leaf', 5, 0, 27]], ['Value', 4, ['Leaf', 8, 0, 9]], ['Value', 5, ['Leaf', 8, 1, 136]]]], ['Value', 4, ['Leaf', 2, 0, 121]], ['Value', 5, ['Attribute', 'per', ['Value', 1, ['Leaf', 1, 5, 98]], ['Value', 2, ['Leaf', 4, 0, 63]], ['Value', 3, ['Leaf', 3, 23, 98]], ['Value', 4, ['Leaf', 3, 4, 98]], ['Value', 5, ['Leaf', 7, 3, 98]]]], ['Value', 6, ['Attribute', 'per', ['Value', 1, ['Leaf', 2, 2, 87]], ['Value', 2, ['Leaf', 2, 0, 41]], ['Value', 3, ['Leaf', 6, 0, 36]], ['Value', 4, ['Leaf', 7, 0, 6]], ['Value', 5, ['Leaf', 10, 2, 87]]]], ['Value', 7, ['Attribute', 'per', ['Value', 1, ['Leaf', 1, 1, 45]], ['Value', 2, ['Leaf', 4, 0, 17]], ['Value', 3, ['Leaf', 5, 0, 22]], ['Value', 4, ['Leaf', 10, 0, 3]], ['Value', 5, ['Leaf', 5, 0, 2]]]], ['Value', 8, ['Leaf', 4, 0, 21, ['Value', 2, ['Leaf', 4, 7, 21]], ['Value', 3, ['Leaf', 3, 0, 10]], ['Value', 4, ['Leaf', 5, 0, 3]], ['Value', 5, ['Leaf', 10, 1, 21]]]], ['Value', 9, ['Leaf', 2, 0, 11]], ['Value', 10, ['Leaf', 6, 0, 7, ['Value', 2, ['Leaf', 3, 1, 7]], ['Value', 3, ['Leaf', 6, 3, 7]], ['Value', 4, ['Leaf', 4, 1, 7]], ['Value', 5, ['Leaf', 8, 2, 7]]]]]

packaged_object = [header, salary_tree]
# pickle packaged_object
outfile = open("web_app_deployment/tree.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()