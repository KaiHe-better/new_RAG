import matplotlib.pyplot as plt
import numpy as np

save_file = "ULMS.png"
llm = "3"

num =                   [1,      2,      3,     4,        5,      6,       7,       8,       9,    10 ]

if llm == "3":
    RAG_ACC          =  [00.00,  44.15,  00.00,  51.77,   00.00,   56.09,   00.00,   58.76,   00.00, 60.25  ]
    RA_len           =  [212.15,  418.76, 623.21,821.72,  1021.91, 1224.35, 1426.58, 1626.46, 00.00, 2032.21]
    RAG_uninst      =   [00.00,  27.34,  00.00,  14.53,   00.00,   5.34,    00.00,   1.49,    00.00, 0.31   ]

    EAG_ACC          =  [62.37,  63.63,  63.39,  64.34,   63.79,   65.83,   65.91,  66.46,   00.00, 67.32]
    EAG_len          =  [132.21, 246.76, 371.37, 453.86,  602.59,  738.21,  871.29, 969.16,  00.00, 1268.97]
    EAG_uninst       =  [13.90,  13.83,  11.86 , 9.82,    7.46 ,   5.58,    4.16,   3.3,     00.00, 1.1]

if llm == "2":
    RAG_ACC          =  [34.01,  00.00,  34.64,   00.00,   34.96,   00.00,   35.43,   00.00,  00.00,  00.00]
    RA_len           =  [260.26, 513.89, 765.02,  1008.67, 1253.75, 1501.88, 1749.87, 1994.92,00.00,  00.00]
    RAG_uninst      =   [00.00,  00.00,  00.00,   00.00,   00.00,   00.00,   00.00,   00.00,  00.00,  00.00]

    EAG_ACC          =  [00.00,  41.79,  00.00,  43.21,   00.00,   43.99,   00.00,  44.01,    00.00,  00.00]
    EAG_len          =  [00.00,  308.36, 00.00, 609.45,   00.00,   914.020, 00.00,  1202.62,  00.00,  00.00]
    EAG_uninst       =  [00.00,  00.00,  00.00,  00.00,   00.00,   00.00,   00.00,  00.00,    00.00,  00.00]
   

# if save_file == "pop.png":
#     RAG_ACC_l3       =  [00.00, 47.19,  00.00,  51.26,   00.00,   50.42,   00.00,  50.49,   00.00,  50.56]
#     RA_len_l3        =  [00.00, 225.00, 00.00,  450.39,  00.00,   672.34,  00.00,  896.11,  00.00,  00.00]
    
#     EAG_ACC_l3       =  [00.00, 48.35,  00.00,  49.68,   00.00,   50.53,   00.00,  50.74, 00.00,  00.00]
#     EAG_len_l3       =  [00.00, 235.66, 00.00,  441.31,  00.00,   667.04,  00.00,  934.34, 00.00,  00.00]
   


def col_2_1(save_file):
    line1 = RA_len
    line2 = EAG_len

    bar1= RAG_ACC
    bar2= EAG_ACC


    # number of hops of EntailmentBank
    fig, ax1 = plt.subplots(1,1,figsize=(8, 3.6),dpi=300)
    plt.subplots_adjust(left=0.1, right=0.88, top=0.98, bottom=0.135)

    plt.grid(zorder=-100)

    bar_width = 0.35
    tick_label = num
    x = num

    ax1_xlabel = "The Number of Retrievals (Hyper-parameters)"

    ax1_ylabel = "Accuracy"


    ax1.set_ylim(40, 80)


    ax1.bar([i-bar_width/2 for i in x], bar1, bar_width, align="center", color='lightsteelblue', edgecolor='black', zorder=100, hatch='')
    ax1.bar([i+bar_width/2 for i in x], bar2, bar_width, align="center", color='coral', edgecolor='black', zorder=100, hatch='//')

    ax1.set_ylabel(ax1_ylabel,size=18)
    ax1.set_xlabel(ax1_xlabel,size=18)
    plt.yticks(size=16)
    plt.legend([' T-RAG','  EAG'], fontsize=10, loc=(0.01, 0.84))


    # ====================================================================================================================
    ax2 = ax1.twinx()
    ax2_ylabel = "The Length of Inputs"



    ax2.plot(x, line1, color="green",marker="o",linestyle='-.')
    ax2.plot(x, line2, color="red",marker="s",linestyle='--')
    ax2.set_ylabel(ax2_ylabel,size=16)
    

    plt.xticks(x,tick_label,size=16)
    plt.yticks(size=16)
    
    plt.legend([' T-RAG','  EAG'], fontsize=10,  loc=(0.82, 0.84))
    plt.savefig(save_file)



col_2_1(save_file)
