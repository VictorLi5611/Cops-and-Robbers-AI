import matplotlib.pyplot as plt
import os
import numpy as np



#5 cops data
rawdata_for_5_cops = np.loadtxt("results/result-Tau0.005.txt")
# 8 cops data
rawdata_for_8_cops = np.loadtxt("results/result-Tau0.01.txt")
# 10 cops data
rawdata_for_10_cops = np.loadtxt("results/result-Tau0.015.txt")

rawdata_for_1024 = np.loadtxt("results/result-Tau0.02.txt")

rawdata = {5: rawdata_for_5_cops, 8: rawdata_for_8_cops, 10: rawdata_for_10_cops, 1024: rawdata_for_1024}
output = {5: [], 8: [], 10: [], 1024: []}

for key in rawdata:
    temp = []
    print(rawdata[key])
    for data in rawdata[key]:
        temp.append(data)
        # if temp is less than 100, find the average of the size of temp
        if len(temp) <= 100:
            output[key].append(sum(temp)/len(temp))
        else:
            # if temp is larger than 100, find the average of the last 100 elements
            output[key].append(sum(temp[-100:])/100)
    
    
    

# plot all data into one graph
# make size of the graph bigger
plt.figure(figsize=(10,10))
plt.figure()
plt.plot(output[5], label="Tau = 0.005", color="green")
plt.plot(output[8], label="Tau = 0.01", color="blue")
plt.plot(output[10], label="Tau = 0.015", color="red")
plt.plot(output[1024], label="Tau = 0.02", color="orange")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Average Rewards for last 100 episodes with Tau adjusted")
plt.savefig("results/rewards.png")
plt.show()







