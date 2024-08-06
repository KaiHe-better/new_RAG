
import matplotlib.pyplot as plt


def read_data_from_file(filename):
    data_dict = {}
    with open(filename, 'r') as file:
        data = file.readlines()
    
        data_dict[data[0][:-1]] =  eval(data[1][:-1])
        data_dict[data[2][:-1]] =  eval(data[3][:-1])
        data_dict[data[4][:-1]] =  eval(data[5][:-1])
       
    return data_dict


filename = "/hpc/home/kai_he/workshop/My_project/new_RAG/results/output/ID_0_USMLE_4_gpu_6_RA_method_Gate_MI_RA_dataset_USMLE_quantile_num_0.99_if_hiera/MI_41.67.txt"
data = read_data_from_file(filename)


x = data['r3_pior_list']
y = data['orginal_pior_list']
colors = ['green' if value == 0 else 'red' for value in data['right_wrong']]

plt.scatter(x, y, color=colors)
# plt.title('Scatter Plot of Data')
plt.xlabel('r3_pior_list')
plt.ylabel('orginal_pior_list')
plt.savefig("a.png" )