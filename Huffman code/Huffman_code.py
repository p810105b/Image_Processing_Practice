import numpy as np
import cv2 as cv
import queue
import matplotlib.pyplot as plt
import igraph as ig

class Node:
	def __init__(self):
		self.freq  = None
		self.code  = None
		self.data  = None
		self.left  = None
		self.right = None 
	def __lt__(self, other):
		if (self.freq < other.freq):		
			return 1
		else:
			return 0
	def __gt__(self, other):
		if (self.freq > other.freq):
			return 1
		else:
			return 0
        

def bulid_Huffman_tree(frequency):
    # synchronized queue function
    syn_queue = queue.PriorityQueue()
    for data, freq in enumerate(frequency):
        if freq != 0:
            leaf = Node()
            leaf.data = data
            leaf.freq = freq
            # put the new leaf into the queue
            syn_queue.put(leaf)
            
    while syn_queue.qsize() > 1:
        # create new node
        new_node = Node()
        # get and remove 2 smallest frequency for leaves
        l = syn_queue.get()
        r = syn_queue.get()
        new_node.left  = l 		
        new_node.right = r
        # the sum of the smallest frequency as a new leaf into the queue
        new_freq = l.freq + r.freq	
        new_node.freq = new_freq
        syn_queue.put(new_node)	
        
    return syn_queue.get()


def Huffman_traversal(root_node, data_code_pair_file, node_array, Huffman_code_file, bit_numbers_file, data_value_file):	
    # 1 for right of branches of Huffman tree
    if root_node.right is not None:
        # array for nodes 
        node_array[Huffman_traversal.count] = 1
        # bis numbers
        Huffman_traversal.count += 1
        # write one value at a time
        Huffman_traversal(root_node.right, data_code_pair_file, node_array, Huffman_code_file, bit_numbers_file, data_value_file)
        Huffman_traversal.count -= 1
        
    # 0 for left of branches of Huffman tree	
    if root_node.left is not None:
        # array for nodes
        node_array[Huffman_traversal.count] = 0
        # bis numbers
        Huffman_traversal.count += 1
        # write one value at a time
        Huffman_traversal(root_node.left, data_code_pair_file, node_array, Huffman_code_file, bit_numbers_file, data_value_file)
        Huffman_traversal.count -= 1
        
    # write data and Huffman code into a file , data represents pixel intensity  
    else:	
        Huffman_code = ''.join(str(i) for i in node_array[1 : Huffman_traversal.count]) 
        data = str(root_node.data)
        
        bit_numbers = str(len(Huffman_code)) + '\n'
        data_value  = data + '\n'
        
        if root_node.data < 100:
            data_code_pair = data + '  : ' + Huffman_code + '\n' 
        else:
            data_code_pair = data + ' : '  + Huffman_code + '\n'
            
        Huffman_code = Huffman_code + '\n'
        # show the encoding result    
        #print(data_code_pair) #------------check------------#
        #print(Huffman_code)   #------------check------------#
        #print(bit_numbers)    #------------check------------#
        #print(data_value)     #------------check------------#
        # store into files
        data_code_pair_file.write(data_code_pair)
        Huffman_code_file.write(Huffman_code)
        bit_numbers_file.write(bit_numbers)
        data_value_file.write(data_value)
   
    
# Read image for gray scale
img = cv.imread('lena.bmp', 0)


#------------       function instatiation       ------------#
# the numbers of occurrence for each intensity
hist = np.bincount(img.flatten())

# the frequency of occurrence
frequency = hist / np.sum(hist)		

# build Huffman tree and initial condition setting
root_node = bulid_Huffman_tree(frequency)			
node_array = np.zeros([256], dtype = int)
Huffman_traversal.count = 1

# open files for Huffman_traversal function (writing mode)
data_code_pair_file = open('data_code_pair.txt', 'w')
Huffman_code_file   = open('Huffman_code.txt'  , 'w')
bit_numbers_file    = open('bit_numbers.txt'   , 'w')
data_value_file     = open('data_value.txt'    , 'w')

Huffman_traversal(root_node, data_code_pair_file, node_array, Huffman_code_file, bit_numbers_file, data_value_file)	

# close files
data_code_pair_file.close
Huffman_code_file.close
bit_numbers_file.close
data_value_file.close
#------------       function instatiation       ------------#


# Data processing
#------   build lists for bit_numbers and data_value  ------#
# read the data_code_pair as a list (reading mode)
Huffman_code_file = open('Huffman_code.txt' , 'r')
bit_numbers_file  = open('bit_numbers.txt'  , 'r')
data_value_file   = open('data_value.txt'   , 'r')

# bulid two lists for bit_numbers and data_value respectively
Huffman_code_list = []
bit_numbers_list  = []
data_value_list   = []

# find non-zeor intensity value of the img
num_nonzero_value = 0
for i in range(len(hist)):
    if hist[i] != 0:
        num_nonzero_value += 1
#print(num_nonzero_value)  #------------check------------#

for i in range(num_nonzero_value):
    with open("Huffman_code.txt") as f:
        Huffman_code_list.append((f.readlines()[i]))
        Huffman_code_list[i] = Huffman_code_list[i].strip()
        
    with open("bit_numbers.txt") as f:
        bit_numbers_list.append(int(f.readlines()[i]))
        
    with open("data_value.txt") as f:
        data_value_list.append(int(f.readlines()[i]))
    
#print(Huffman_code_list) #------------check------------#    
#print(bit_numbers_list)  #------------check------------#
#print(data_value_list)   #------------check------------#

# build data code pair dictionary
dict_data_code_pair = dict(zip(data_value_list, Huffman_code_list))
dict_data_code_pair = sorted(dict_data_code_pair.items(), key = lambda x : x[0])
dict_data_code_pair = dict(dict_data_code_pair)
#print(dict_data_code_pair)

Sorted_Huffman_file = open('Sorted_Huffman.txt', 'w') 
for key, value in dict_data_code_pair.items():
    key   = str(key)
    value = str(value)
    Sorted_Huffman_file.write(str(key) + ' : ' + str(value) + '\n')
    print(key + ' : ' + value) #------------result------------#
Sorted_Huffman_file.close()
#------   build lists for bit_numbers and data_value  ------#




#------------         plot histogram            ------------#
# build dictionary for data_code pair
hist = hist.tolist()
hist_dict = {}
for i in range(len(hist)):
    hist_dict[i] = hist[i]
#print(hist_dict)  #------------check------------#

# Original img diagram
x = []
y = []
for i in range(len(hist_dict)):
    x.append(i)
    y.append(hist_dict[i])
    
plt.subplot(1, 2, 1)
plt.title('Origin img')
plt.xlabel("Pixel intensity (0~255)")
plt.ylabel("Time of occurrence")
plt.bar(x, y)

# compressed img diagram
x = []
y = []
for i in range(len(data_value_list)):
    x.append(data_value_list[i])
    y.append(bit_numbers_list[i])
    
plt.subplot(1, 2, 2)
plt.title('Compressed img')
plt.xlabel("Pixel intensity(0~255)")
plt.ylabel("Bit numbers")
plt.bar(x, y)

#--------------  plot histogram  --------------#




#------------  Plot Huffman tree  -------------#
# remove the frequency = 0
for i in range(len(hist)): 
    if hist_dict.get(i) == 0:
        del hist_dict[i]
        
# Create graph
g = ig.Graph(directed = True)

# sort the hist for each frequency
hist_dict_sorted = sorted(hist_dict.items(), key = lambda x : x[1])

# transfer hist into list data form
hist_list = []
for i in range(207):
    hist_list += [list(hist_dict_sorted[i])]
    
#hist_list = [[name, freq], [name, freq], [name, freq],...], name == intensity
hist_list_edge = hist_list.copy()

##############  add vertices  ###############
# Add  vertices : (intensity numbers + combinations )
g.add_vertices(len(hist_list) + (len(hist_list) - 1))

# label intensity as name
for i in range(len(hist_list)):
    g.vs[i]["label"] = str(hist_list[i][0])
    

new_freq = 0  
# for thinking
father   = []  # vertices 207 ~ 412 generated by sum up the lowest 2 frequency
children = []  # vertices   0 ~ 206 for intensity
for i in range(205):
    # Huffman tree algorithm 
    new_freq = hist_list[0][1] + hist_list[1][1]
    
    del hist_list[0]
    del hist_list[0]
    
    hist_list.append(['plus', new_freq]) 
    hist_list.sort(key = lambda x : x[1])
    
    
    g.vs[(i + 207)]["label"] = 'f : '  + str(new_freq)
    
    father   += [[i + 207, new_freq]]  # start from vertex 207
    children += [[hist_list[0][1], hist_list[1][1]]]
# The last item
father += [[205+207, hist_list[0][1] + hist_list[1][1]]]       
# label frequency as name
g.vs[(205 + 207)]["label"] = 'f : ' + str(hist_list[0][1] + hist_list[1][1])   

##############  add edges  ###############
# [index, frequncy], index = 0 ~ 206 contains intensity
vertex_position = []  
for i in range(len(hist_list_edge)):
    vertex_position += [[i, hist_list_edge[i][1]]]
    
compare = vertex_position.copy()      # 0 ~ 206
vertex_position += father             # 0 ~ 412
compare_full = vertex_position.copy() # 0 ~ 412

# first sum frequency
g.add_edges([(207, 0),(207, 1)])

# algorithm for connect vertices
for i in range(205):
    # remove the lowest 2 frequency
    del compare[0]
    del compare[0]
    # add the sum frequency into origin intensity(0 ~ 206)
    compare += [compare_full[207 + i]]
    compare.sort(key = lambda x: x[1])
    
    g.add_edges([(208 + i, compare[0][0]),(208 + i, compare[1][0])]) 


# visual setting
visual_style = {}

img_name = "Huffman_tree.png"

# set bbox and margin
visual_style["bbox"] = (10000,6000)
visual_style["margin"] = 100

# set vertex colours
visual_style["vertex_color"] = 'white'
visual_style["edge_color"]   = 'red'

# set vertex size
visual_style["vertex_size"] = 60

# set vertex lable size
visual_style["vertex_label_size"] = 15

# don't curve the edges
visual_style["edge_curved"] = False

# set the layout
layout = g.layout_reingold_tilford()
visual_style["layout"] = layout

# plot the graph
ig.plot(g, img_name, **visual_style)

#------------  Plot Huffman tree  -------------#