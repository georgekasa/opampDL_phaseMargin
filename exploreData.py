import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#           p31                    p33
# inn-<p45       p46->inp          out
#  n40             n39             n28

data_dir = r'/home/gkasap/Documents/Python/opampDL/pmosOpAmp/'
# training data inputs: x and targets: y
df_path = os.path.join(data_dir, 'dataframeTrain.pkl')
df_mosfet_path = os.path.join(data_dir, 'dfMosfet.pkl')


df = pd.read_pickle(df_path)
df_mosfet = pd.read_pickle(df_mosfet_path)
df_mosfet.xs("DiffPairInp", level=1)["ids"].describe()
df_mosfet.xs("DiffPairInn", level=1)["ids"].describe()
df_mosfet.xs("ActLoadn", level=1)["ids"].describe()
df_mosfet.xs("ActLoadp", level=1)["ids"].describe()
df_mosfet.xs("MosOutput", level=1)["ids"].describe()

y = df_mosfet.xs("MosOutput", level=1)["ids"]
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2)
ax1.plot(range(0,len(y)),df_mosfet.xs("MosOutput", level=1)["ids"])
ax2.plot(range(0,len(y)),df_mosfet.xs("DiffPairInn", level=1)["ids"])
ax3.scatter(range(0,len(y)),df["phaseMargin"])


rng = np.random.default_rng(seed=11)

random.seed(9)
checkCase = []
[checkCase.append(random.randint(0,len(df))) for _ in range(0,4)]

# na checkaro kai mes 50
print(df_mosfet.loc[df_mosfet.index[checkCase][0]])
for i in checkCase:
    print(df["M_out"][i],df["Mdiff"][i], df["CapMiller"][i])
    print(df_mosfet.loc[df_mosfet.index[[i*7]][0][0]])
    print(df.iloc[i]["phaseMargin"], end =",")
    print(df.iloc[i]["ResistorMiller"])
    print("#######################")
# plt.show()
# plt.xlabel("N")
# plt.ylabel("I ")
# for key in df.keys():
#     print(df[key].describe())
#     print("################")


filteredData = df.drop(df[df.phaseMargin < 60].index) #from 1701 -> 646
#df.describe mean = 55
#wantedData mean = 68.98
filteredData.to_pickle("filterData.pkl")