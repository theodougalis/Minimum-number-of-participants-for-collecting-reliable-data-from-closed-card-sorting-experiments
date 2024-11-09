import csv
import random
import Mantel # https://jwcarr.github.io/MantelTest/
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from clusim.clustering import Clustering
import clusim.sim as sim
import matplotlib.pyplot as plt
import statistics
import math


count_of_samples_for_each_n=100 # Average of count_of_samples_for_each_n for n participants
perms_of_mantel_test=10000 # Default is 10000
plt_ylim_min=0.5
plt_xticks=np.arange(0, 71, 5)


skroutz_categories_dict = {
  "Κινητή τηλεφωνία": 1,
  "Ηλ. Υπολογιστές": 2,
  "Ηλεκτρονικά": 3,
  "Ήχος": 4,
  "Εικόνα": 5,
  "Tablets & Αξεσουάρ": 6,
  "Gadgets": 7
}
tourismos_categories_dict = {
  "Μετακινήσεις": 1,
  "Μέρη για επίσκεψη": 2,
  "Καιρός": 3,
  "Υγεία και ασφάλεια": 4,
  "Οργάνωση ταξιδιού": 5,
  "Πράγματα να κάνεις": 6,
  "Σχετικά με τη χώρα": 7,
  "Κάρτες & Wallpapers": 8
}

cards=38 # Count of cards depending on the card sorting
total_categories=8 # Count of categories depending on the card sorting
total_participants=185 # Count of all participants depending on the card sorting

# participants_range: A list with items that are selected every time from the total number of participants, in order the code to be executed
# Each of its items is a number from 1 to total_participants and is not repeated
participants_range=range(1, total_participants+1)

# participants_range_for_error_bar: A list with items that are selected every time from the total number of participants for creation of the error bars
participants_range_for_error_bar=[1, 3, 5, 8, 12, 15, 20, 30, 40, 50, 60, 70] #[1, 3, 5, 8, 12, 15, 20, 30, 40, 50, 60, 70]

column_category_label=3 # Column 4 in csv (python starts from 0)

# Column 1 of csv: participant_id
# Column 2 of csv: card_index
# Column 3 of csv: card_label
# Column 4 of csv: category_label


# The row data of csv is sorted firstly by participant_id and secondly by card_index
all_data = []
with open(r"tourismos.csv", encoding="utf-8") as csvfile: #skroutz.csv for the first card sorting, tourismos.csv for the second one
    reader = csv.reader(csvfile)
    for row in reader: # each row is a list
        all_data.append(row)

        

def dissimilarity_matrix(some_participants):
    
    global column_category_label, all_data, cards, total_participants
    
    # participants_range: A list with items that are selected every time from the total number of participants, in order the code to be executed
    # Each of its items is a number from 1 to total_participants and is not repeated (python starts counting from 0)
    # With random.sample, UNIQUE numbers are taken from the list in the first parameter of sample method
    # Examples of random.sample() can be found here: https://www.geeksforgeeks.org/python-random-sample-function
    participants_range=random.sample(range(0, total_participants), some_participants)
    
    # Initialization of the dissimilarity matrix with zeros
    # The elements of the main diagonal will be always zero
    dissimilarity_matrix = [[0 for x in range(cards)] for y in range(cards)]
    
    for participant in participants_range:
        for x in range(participant*cards, (participant+1)*cards): # Take cards/rows only form current participant
            
            for y in range(x+1,(participant+1)*cards):# Because the dissimilarity_matrix is a symmetric matrix, the itteration is started form x+1 
                
                # If the cards have been sorted in differents groups from the current user
                if all_data[x][column_category_label]!=all_data[y][column_category_label]: 
                    
                    # x-participant*cards and y-participant*cards are compared in this itteration
                    # x-participant*cards is the card_index (0, 1, ... , cards-1) of the 1st compared card from participant
                    # y-participant*cards is the card_index (0, 1, ... , cards-1) of the 2nd compared card from participant                    
                    dissimilarity_matrix[x-participant*cards][y-participant*cards]=dissimilarity_matrix[x-participant*cards][y-participant*cards]+1
                    
                    # With the following line, it is converted to a symmetric matrix
                    dissimilarity_matrix[y-participant*cards][x-participant*cards]=dissimilarity_matrix[x-participant*cards][y-participant*cards]

    return dissimilarity_matrix



def cards_x_categories_matrix(some_participants):
    
    global column_category_label, all_data, cards, total_participants, total_categories, skroutz_categories_dict, tourismos_categories_dict
    
    # participants_range: A list with items that are selected every time from the total number of participants, in order the code to be executed
    # Each of its items is a number from 1 to total_participants and is not repeated (python starts counting from 0)
    # With random.sample, UNIQUE numbers are taken from the list in the first parameter of sample method
    # Examples of random.sample() can be found here: https://www.geeksforgeeks.org/python-random-sample-function
    participants_range=random.sample(range(0, total_participants), some_participants)
    
    # Initialization of the cards x categories matrix with zeros
    cards_x_categories_matrix = [[0 for x in range(cards)] for y in range(total_categories)]
    
    #For every participant, add 1 to the matrix on the [x,y] value where x is the card id and y is the category id of the category it has been placed in
    for participant in participants_range:
        for x in range(participant*cards, (participant+1)*cards): # Take cards/rows only form current participant
            
            for y in range(total_categories): 

                if tourismos_categories_dict[all_data[x][column_category_label]]-1==y: #categories_dict changes depending on which card sorting we analyse

                    cards_x_categories_matrix[y][int(all_data[x][1])-1]+=1  

    #Divide every value with the sample size to get values between 0 and 1
    for x in range (cards):
        for y in range(total_categories):
            cards_x_categories_matrix[y][x]/=some_participants
    return cards_x_categories_matrix



def cxcmatrix2clu(cxc_matrix):

    #takes as input the cards x categories matrix and returns elm2clu dictionary
    
    global column_category_label, all_data, cards, total_participants, total_categories, skroutz_categories_dict, tourismos_categories_dict
    
    #create the elm2clu dictionary and initialize it with zeros
    #key is card_id-1 and value is category_id-1
    matrix2clu={}
    for i in range(cards):
        matrix2clu[i]=0

    #for every card, check the frequency of appearance in each category. If current category's number is higher than the number of 
    #the saved in the dict category, save this in the dict instead
    #meaning: if cxc_matrix[y][x] > cxc_matrix[matrix2clu[x]][x]  -->  matrix2clu.update({x: y})
    for x in range(cards):
        for y in range(total_categories):
            if cxc_matrix[y][x] > cxc_matrix[matrix2clu[x]][x]:
                matrix2clu.update({x: y})

    matrix2clu_dict = {key: [value] for key, value in matrix2clu.items()}
    return matrix2clu_dict



def mantel_elsim_r_average_and_errors(some_participants):
    
    global column_category_label, all_data, cards, total_participants, count_of_samples_for_each_n, perms_of_mantel_test
    dis2=dissimilarity_matrix(total_participants)
    
    m2=cards_x_categories_matrix(total_participants)
    dict2=cxcmatrix2clu(m2)
    c2=Clustering(elm2clu_dict=dict2)
        
    
    
    mantel_r_table=[]
    elsim_r_table=[]
    
    for i in range(count_of_samples_for_each_n):
        dis1=dissimilarity_matrix(some_participants)
        
        # Mantel Method
        mantel=Mantel.test(dis1, dis2, perms_of_mantel_test, method='pearson', tail='two-tail')
        mantel_r = mantel[0]
        
        mantel_r_table.append(mantel_r)

        m1=cards_x_categories_matrix(some_participants)
        dict1=cxcmatrix2clu(m1)
        c1=Clustering(elm2clu_dict=dict1)
        
        # Element-centric Similarity
        elsim_r = sim.element_sim(c1, c2, r=1.0, alpha=0.9)

        elsim_r_table.append(elsim_r)
        
        
    mantel_average = statistics.mean(mantel_r_table) # average of mantel_r
    mantel_l_error = mantel_average - min(mantel_r_table) # mantel_lower_error
    mantel_u_error = max(mantel_r_table) - mantel_average # mantel_upper_error
    mantel_sd=statistics.stdev(mantel_r_table)

    elsim_average = statistics.mean(elsim_r_table) # average of elsim_r
    elsim_l_error = elsim_average - min(elsim_r_table) # mantel_lower_error
    elsim_u_error = max(elsim_r_table) - elsim_average # mantel_upper_error
    elsim_sd=statistics.stdev(elsim_r_table)

    
    return mantel_average, mantel_l_error, mantel_u_error, mantel_sd, elsim_average, elsim_l_error, elsim_u_error, elsim_sd




def mantel_elsim_r_average_and_errors_in_participants_range(participants_range):
    
    global total_participants
    
    # Initialization with zeros
    
    mantel_r_average_of_each_n=[]
    mantel_r_lower_error_of_each_n=[]
    mantel_r_upper_error_of_each_n=[]
    mantel_r_sd_of_each_n=[]
    
    elsim_r_average_of_each_n=[]
    elsim_r_lower_error_of_each_n=[]
    elsim_r_upper_error_of_each_n=[]
    elsim_r_sd_of_each_n=[]
    
    for y in range(0,total_participants+1):
        mantel_r_average_of_each_n.append(0)
        mantel_r_lower_error_of_each_n.append(0)
        mantel_r_upper_error_of_each_n.append(0)
        mantel_r_sd_of_each_n.append(0)
        
        elsim_r_average_of_each_n.append(0)
        elsim_r_lower_error_of_each_n.append(0)
        elsim_r_upper_error_of_each_n.append(0)
        elsim_r_sd_of_each_n.append(0)
        
        
        
        
    # x is the number/id of participant (possible values: 1 , ... , total_participants)
    for x in participants_range:
        if x==0: 
            #0 ~ O(0,0) is the axis origin
            mantel_r_average_of_each_n[0]=0
            mantel_r_lower_error_of_each_n[0]=0
            mantel_r_upper_error_of_each_n[0]=0
            mantel_r_sd_of_each_n[0]=0
            
            elsim_r_average_of_each_n[0]=0
            elsim_r_lower_error_of_each_n[0]=0
            elsim_r_upper_error_of_each_n[0]=0
            elsim_r_sd_of_each_n[0]=0
        else:
            mantel_average, mantel_l_error, mantel_u_error, mantel_sd, elsim_average, elsim_l_error, elsim_u_error, elsim_sd = mantel_elsim_r_average_and_errors(x)
            
            # Put the values in an array
            mantel_r_average_of_each_n[x]=mantel_average
            mantel_r_lower_error_of_each_n[x]=mantel_l_error
            mantel_r_upper_error_of_each_n[x]=mantel_u_error
            mantel_r_sd_of_each_n[x]=mantel_sd

            elsim_r_average_of_each_n[x]=elsim_average
            elsim_r_lower_error_of_each_n[x]=elsim_l_error
            elsim_r_upper_error_of_each_n[x]=elsim_u_error
            elsim_r_sd_of_each_n[x]=elsim_sd
            
    # Return of the arrays
    return mantel_r_average_of_each_n, mantel_r_lower_error_of_each_n, mantel_r_upper_error_of_each_n, mantel_r_sd_of_each_n, elsim_r_average_of_each_n, elsim_r_lower_error_of_each_n, elsim_r_upper_error_of_each_n, elsim_r_sd_of_each_n




def save_errorbar(r_average_of_each_n, r_sd_of_each_n,  participants_range, title, xlabel, ylabel, save, clear):
    global total_participants, plt_ylim_min, plt_xticks
    
    x=[np.array(range(0,total_participants+1))[i] for i in participants_range] 
    y=[r_average_of_each_n[i] for i in participants_range]

    #Why we choose 2*(standard error) to calculate error bars
    #2*r_sd_of_each_n[i]/sqrt(count_of_samples_for_each_n): https://www.spss-tutorials.com/confidence-intervals/

    lower_error = [2*r_sd_of_each_n[i]/math.sqrt(count_of_samples_for_each_n) for i in participants_range] 
    upper_error = [2*r_sd_of_each_n[i]/math.sqrt(count_of_samples_for_each_n) for i in participants_range]
    asymmetric_error = [lower_error, upper_error]
    
    plt.errorbar(x, y, yerr=asymmetric_error, capsize=4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
   
    plt.ylim(plt_ylim_min, 1)
    plt.xticks(plt_xticks)
    
    if save==True:
        plt.savefig(title+".png",dpi=300)
    if clear==True:
        plt.show()




mantel_r_average_of_each_n, mantel_r_lower_error_of_each_n, mantel_r_upper_error_of_each_n, mantel_r_sd_of_each_n, elsim_r_average_of_each_n, elsim_r_lower_error_of_each_n, elsim_r_upper_error_of_each_n, elsim_r_sd_of_each_n = mantel_elsim_r_average_and_errors_in_participants_range(participants_range)

print("participants;", " ", "mantel lower;", " ", "mantel upper;", "mantel sd;", " ", "mantel average;", " ", "elsim lower;", " ", "elsim upper;", " ", "elsim sd;", " ", "elsim average")
for i in participants_range:
    if i==0:
        continue
    mantel_r_min = mantel_r_average_of_each_n[i] - mantel_r_lower_error_of_each_n[i]
    mantel_r_max = mantel_r_average_of_each_n[i] + mantel_r_upper_error_of_each_n[i]
    elsim_r_min = elsim_r_average_of_each_n[i] - elsim_r_lower_error_of_each_n[i]
    elsim_r_max = elsim_r_average_of_each_n[i] + elsim_r_upper_error_of_each_n[i]
    print(i,"; ", mantel_r_min, "; ", mantel_r_max, "; ", mantel_r_sd_of_each_n[i], "; ", mantel_r_average_of_each_n[i], "; ", elsim_r_min, "; ", elsim_r_max, "; ", elsim_r_sd_of_each_n[i], "; ", elsim_r_average_of_each_n[i]) 

# Copy the printed data with the results in csv file
    
    

# CASE1: Showing of both plots in one figure
save_errorbar(mantel_r_average_of_each_n, mantel_r_sd_of_each_n, participants_range_for_error_bar, "Mantel", "Sample Size", "Average correlation", False, False)
save_errorbar(elsim_r_average_of_each_n, elsim_r_sd_of_each_n, participants_range_for_error_bar, "Elsim & Mantel Test Results", "Sample Size", "Average correlation", True, True)

# CASE2: Showing plots in different figures
save_errorbar(mantel_r_average_of_each_n, mantel_r_sd_of_each_n, participants_range_for_error_bar, "Mantel Test Results", "Sample Size", "Average correlation", True, True)
save_errorbar(elsim_r_average_of_each_n, elsim_r_sd_of_each_n, participants_range_for_error_bar, "Elsim Test Results", "Sample Size", "Average correlation", True, True)


