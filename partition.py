def partition(data):
    height = data[:,2]
    index = [0]
    i = 0; j = 1
    while j <len(height):
        if height[j] == height[i]:
            j += 1
        else:
            index.append(j)
            i = j
            j += 1
    index.append(len(height))
    return index
