import pandas as pd
from math import pi
import numpy as np
import matplotlib.pyplot as plt



# SETTINGS
path_csv = "C:\\Users\\lurde\\Downloads\\summary\\summary\\"
name_csv = "summary_Breslow.xlsx"

#graphic = "In situ"
graphic = "Breslow"
#graphic = "Multiclass"

fDN = 3
fRN = 10
fVGG = 17




def createRadar(label, data, color):
        #Attributes = ["         F1-score"," Kappa","Precision   ","     Recall          ","Specificity    ","AUC"]
    
    if graphic != "Multiclass":    
        Attributes = ["           Balanced \n          accuracy","          Accuracy","F1-score","Kappa    ","Precision          ","Recall     ","AUC","            Specificity"]
        k = 8
    else:
        Attributes = ["                  ACC-balanced","ACC","F1-score    ","Kappa   ","Precision     ","   Recall"]
        k = 6   
    data += data [:1]
        
    angles = [n / k * 2 * pi for n in range(k)]
    angles += angles [:1]
        
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1],Attributes)    
    ax.plot(angles,data,linewidth=1.5,color=color,label=label)
        #ax.fill(angles, data, 'blue', alpha=0.1)
    
    #ax.set_title(graphic, fontsize=15, pad=20)
        #plt.figtext(0.2,0.9,"Messi",color="red")
        #plt.figtext(0.2,0.8,"Ronaldo",color="teal")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    #ax.set_yticks(np.arange(0, 1.0, 0.25))
    ax.set_ylim(0, 1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.rcParams["axes.axisbelow"] = False
    ax.tick_params(axis='y', labelsize=9)
        #ax.tick_params(axis='y', pad=50, direction='in', length=50)
        #ax.set_rlabel_position(-22.5) 
        
        
csv_file = path_csv + name_csv
df = pd.read_excel(csv_file)
    
    # densenet121
row = df.iloc[fDN][1:]
data = []
for x in row:
    data.append(float(x[0:6]))
    
print(data)
createRadar("DenseNet121",data,'#71C615')
    
    # resnet50
row = df.iloc[fRN][1:]
data = []
for x in row:
     data.append(float(x[0:6]))
    
print(data)
createRadar("ResNet50",data,'#F24A19')
    
    
    # VGG16
row = df.iloc[fVGG][1:]
data = []
for x in row:
    data.append(float(x[0:6]))
    
print(data)
createRadar("VGG16",data,'#1BA8E6')
    
    #createRadar("ResNet50",[81,67,85,24,91,86],'#F24A19')
    #createRadar("VGG16",[56,98,44,79,12,39],'#23D5E1')
    
    #createRadar("fedgvf",[38,27,98,68,73,29],'#C71BE6')
plt.tight_layout()
plt.savefig(path_csv + graphic + '.png')
plt.show()


   
# else:
        
#     def createRadar(label, data, color):
#         #Attributes = ["         F1-score"," Kappa","Precision   ","     Recall          ","Specificity    ","AUC"]
#         Attributes = ["                  ACC-balanced","ACC","F1-score    ","Kappa   ","Precision     ","   Recall"]
        
#         data += data [:1]
        
#         angles = [n / 6 * 2 * pi for n in range(6)]
#         angles += angles [:1]
        
#         ax = plt.subplot(111, polar=True)
    
#         plt.xticks(angles[:-1],Attributes)    
#         ax.plot(angles,data,linewidth=2.5,color=color)#,label=label)
#         #ax.fill(angles, data, 'blue', alpha=0.1)
    
#         ax.set_title(graphic, fontsize=15, pad=20)
#         #plt.figtext(0.2,0.9,"Messi",color="red")
#         #plt.figtext(0.2,0.8,"Ronaldo",color="teal")
#         ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
#         #ax.set_ylim(0.35, 0.75)
#         ax.set_yticks(np.arange(0.15, 0.75, 0.2))
#         plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#         plt.rcParams["axes.axisbelow"] = False
        
        
        
#     csv_file = path_csv + name_csv
#     df = pd.read_excel(csv_file)
    
#     # densenet121
#     row = df.iloc[3][1:]
#     data = []
#     for x in row:
#         data.append(float(x[0:6]))
    
#     print(data)
#     createRadar("DenseNet121",data,'#71C615')
    
#     # resnet50
#     row = df.iloc[10][1:]
#     data = []
#     for x in row:
#         data.append(float(x[0:6]))
    
#     print(data)
#     createRadar("ResNet50",data,'#F24A19')
    
    
#     # VGG16
#     row = df.iloc[17][1:]
#     data = []
#     for x in row:
#         data.append(float(x[0:6]))
    
#     print(data)
#     createRadar("VGG16",data,'#23D5E1')
    
#     #createRadar("ResNet50",[81,67,85,24,91,86],'#F24A19')
#     #createRadar("VGG16",[56,98,44,79,12,39],'#23D5E1')
    
#     #createRadar("fedgvf",[38,27,98,68,73,29],'#C71BE6')
#     plt.tight_layout()
#     plt.savefig(path_csv + graphic + '.png')
#     plt.show()
    
    
