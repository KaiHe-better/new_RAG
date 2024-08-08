
import matplotlib.pyplot as plt


def read_data_from_file(filename):
    data_dict = {}
    with open(filename, 'r') as file:
        data = file.readlines()
    
        data_dict[data[0][:-1]] =  eval(data[1][:-1])
        data_dict[data[2][:-1]] =  eval(data[3][:-1])
        data_dict[data[4][:-1]] =  eval(data[5][:-1])
       
    return data_dict


# filename = "/hpc/home/kai_he/workshop/My_project/new_RAG/results/output/ID_0_USMLE_1_gpu_4_RA_method_No_RA_dataset_USMLE_n_docs_10/MI_0.txt"
# filename = "/hpc/home/kai_he/workshop/My_project/new_RAG/results/output/ID_0_USMLE_2_gpu_4_RA_method_Only_RA_dataset_USMLE_n_docs_10/MI_0.txt"
# filename = "/hpc/home/kai_he/workshop/My_project/new_RAG/results/output/ID_0_USMLE_3_gpu_6_RA_method_Gate_RA_dataset_USMLE_n_docs_10_loss_list_kl_soft+k/MI_66.85.txt"
# filename =  "/hpc/home/kai_he/workshop/My_project/new_RAG/results/output/ID_0_USMLE_4_gpu_5_RA_method_Gate_MI_RA_dataset_USMLE_quantile_num_0.99_if_hiera/MI_67.16.txt"

# filename = "/hpc/home/kai_he/workshop/My_project/new_RAG/results/output/ID_0_MedMCQA_1_gpu_7_RA_method_No_RA_dataset_MedMCQA/MI_0.txt"
# filename = "/hpc/home/kai_he/workshop/My_project/new_RAG/results/output/ID_0_MedMCQA_2_gpu_7_RA_method_Only_RA_dataset_MedMCQA/MI_0.txt"


filename = "/hpc/home/kai_he/workshop/My_project/new_RAG/results/arxiv/output_arxiv/ID_0_HEADQA_1_gpu_4_RA_method_No_RA_dataset_HEADQA/MI_0.txt"
# filename = "/hpc/home/kai_he/workshop/My_project/new_RAG/results/arxiv/output_arxiv/ID_0_HEADQA_2_gpu_7_RA_method_Only_RA_dataset_HEADQA/MI_0.txt"
# filename = "/hpc/home/kai_he/workshop/My_project/new_RAG/results/arxiv/output_arxiv/ID_0_HEADQA_4_gpu_4_RA_method_Gate_MI_RA_dataset_HEADQA_quantile_num_0.80/MI_65.03.txt"

data = read_data_from_file(filename)


# 假设 data 是您的数据集
y = data['r3_pior_list']
x = data['orginal_pior_list']

plt.subplots_adjust(left=0.13, right=0.97, top=0.98, bottom=0.135)

colors = ['coral' if value == 1 else 'lightsteelblue' for value in data['right_wrong']]
plt.scatter(x, y, color=colors)


# plt.title('Scatter Plot of Data')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# plt.savefig("U_no_RA.png" )
# plt.savefig("U_RA.png" )
# plt.savefig("U_gate_RA.png" )
# plt.savefig("U_gate_MI_RA.png" )

# plt.savefig("M_no_RA.png" )
# plt.savefig("M_RA.png" )
# plt.savefig("M_gate_RA.png" )
# plt.savefig("M_gate_MI_RA.png" )

plt.ylabel('Probility from T-RAG', fontsize=18)
# plt.ylabel('Probility from EAG', fontsize=18)
plt.xlabel('Probility from Orginal LLM', fontsize=18)

plt.savefig("H_no_RA1.png" )
# plt.savefig("H_RA1.png" )
# plt.savefig("H_gate_MI_RA1.png" )