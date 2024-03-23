import pandas as pd
import numpy as np
import random

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',1000)

def make_pdata(rand_pre,multiple=50,ratio=0.05):
    """利用数据增强获得更多阳性数据"""
    mirna_list,gene_seq_list,change_num_list=[],[],[]
    acgt_dic={0:'A',1:'C',2:'G',3:'T'}
    for i in range(rand_pre.shape[0]):
        start_num,end_num,gene_seq,mirna_seq=rand_pre.iloc[i]['target_start'],rand_pre.iloc[i]['target_end'],\
                                             rand_pre.iloc[i]['gene_seq'],rand_pre.iloc[i]['miRNA_seq']
        gene_seq_list.append(gene_seq)
        change_num_list.append([])
        mirna_list.append(mirna_seq)
        # 单个序列改多少次，
        changed=0
        while changed<multiple-1:
            # 每次改的碱基数量为基因序列长度的5%
            change_num = []
            while len(change_num)<int(len(gene_seq)*ratio):
                rand_num=random.randint(1,len(gene_seq))
                # 改变的序列位置位于匹配区域50个序列外
                if rand_num not in range(start_num-50,end_num+50) and rand_num not in change_num:
                    change_num.append(rand_num)
            gene_seq2=list(gene_seq)
            # 改变原序列的碱基为随机的其他碱基
            for j in change_num:
                tem1=random.randint(0,3)
                gene_seq2[j-1]=acgt_dic[tem1]
            gene_seq_list.append(''.join(gene_seq2))
            change_num_list.append(change_num)
            mirna_list.append(mirna_seq)
            changed+=1
    more_data=pd.DataFrame()
    more_data['miRNA_seq']=mirna_list
    more_data['gene_seq']=gene_seq_list
    more_data['change_num']=change_num_list
    more_data['result']=1

    return more_data


def make_ndata(rand_pre,multiple=1,ratio=0.1):
    """利用数据增强获得更多阴性数据"""
    mirna_list,gene_seq_list,change_num_list=[],[],[]
    acgt_dic={0:'A',1:'C',2:'G',3:'T'}
    for i in range(rand_pre.shape[0]):
        gene_seq,mirna_seq=rand_pre.iloc[i]['gene_seq'],rand_pre.iloc[i]['miRNA_seq']
        # 单个序列改多少次，
        changed=0
        while changed<multiple:
            # 每次改的碱基数量为基因序列长度的5%
            change_num = random.sample(range(len(gene_seq)),int(len(gene_seq)*ratio))
            gene_seq2=list(gene_seq)
            # 改变原序列的碱基为随机的其他碱基
            for j in change_num:
                tem1=random.randint(0,3)
                gene_seq2[j]=acgt_dic[tem1]
            gene_seq_list.append(''.join(gene_seq2))
            change_num_list.append(change_num)
            mirna_list.append(mirna_seq)
            changed+=1
    more_data=pd.DataFrame()
    more_data['miRNA_seq']=mirna_list
    more_data['gene_seq']=gene_seq_list
    more_data['change_num']=change_num_list
    more_data['result']=0


    return more_data

'''
# 去重
data=pd.read_csv('all_filtered.csv')
data_unique2=data.drop_duplicates(subset=['miRNA_seq','gene_id'],inplace=False,keep='first')
data_unique2.to_csv('root_pdata2.csv',index=False)
'''

all_data=pd.read_csv('root_pdata2.csv')

# 提取所有的miRNA和gene，以符合psRNATarget网站的要求，获取匹配的区域
# all_miRNA_data=all_data.iloc[:,[0,1]].drop_duplicates()
# with open('all_mirna.txt','w') as f:
#     for i in range(all_miRNA_data.shape[0]):
#         f.write(f'>{all_miRNA_data.iloc[i,0]}\n')
#         f.write(f'{all_miRNA_data.iloc[i,1]}\n')
# all_gene_data=all_data.iloc[:,[3,4]].drop_duplicates()
# with open('all_gene.txt','w') as f:
#     for i in range(all_gene_data.shape[0]):
#         f.write(f'>{all_gene_data.iloc[i,0]}\n')
#         f.write(f'{all_gene_data.iloc[i,1]}\n')

# 根据结果获取匹配区域
# match_data=pd.read_csv('psRNATargetJob-1665475526130594.txt',sep='\t')
# all_data['target_start']=[0 for i in range(all_data.shape[0])]
# all_data['target_end']=[0 for i in range(all_data.shape[0])]
# for i in range(all_data.shape[0]):
#     # 存在多个匹配区域，取最可能的区域
#     tem=match_data[(match_data['miRNA_Acc.']==all_data['miRNA'][i]) &
#                                            (match_data['Target_Acc.']==all_data['gene_id'][i])]
#     if tem.shape[0]!=1:
#         all_data['target_start'][i]=tem.sort_values(by='Expectation',ascending=True)['Target_start'].iloc[0]
#         all_data['target_end'][i]=tem.sort_values(by='Expectation',ascending=True)['Target_end'].iloc[0]
#     else:
#         all_data['target_start'][i]=tem['Target_start']
#         all_data['target_end'][i] = tem['Target_end']


# 创造更多的阳性数据
all_process_data=pd.read_csv('all_data.csv')
random_test_l=random.sample(range(all_process_data.shape[0]),int(all_process_data.shape[0]*0.2))
random_train_l=[i for i in range(all_process_data.shape[0]) if i not in random_test_l]
all_process_train_data=all_process_data.iloc[random_train_l,:].reset_index()
all_process_test_data=all_process_data.iloc[random_test_l,:].reset_index()

# 进行数据增强  *（49+1）  比例0.05
# 训练303 测试75
more_process_train_data=make_pdata(all_process_train_data,multiple=50,ratio=0.2)
more_process_test_data=make_pdata(all_process_test_data,multiple=50,ratio=0.2)
more_process_train_data.to_csv('more_process_train_data_0.2_new.csv',index=False)
more_process_test_data.to_csv('more_process_test_data_0.2_new.csv',index=False)

# 阳性样本创建完毕 共378*20=7560   378*50=18900



# 创建阴性样本
match_data=pd.read_csv('psRNATargetJob-1665475526130594.txt',sep='\t')
data_mirna=all_data['miRNA'].drop_duplicates(inplace=False)
data_gene=all_data['gene_id'].drop_duplicates(inplace=False)
data_mirna_f,data_gene_f=[],[]
for i in data_mirna:
    for j in data_gene:
        if sum((match_data['miRNA_Acc.']==i) & (match_data['Target_Acc.']==j))==0:
            data_mirna_f.append(i)
            data_gene_f.append(j)
more_data_f=pd.DataFrame()
more_data_f['miRNA']=data_mirna_f
more_data_f['gene']=data_gene_f
data_mirna_f_seq,data_gene_f_seq=[],[]
for i in data_mirna_f:
    data_mirna_f_seq.append(all_data[all_data['miRNA']==i]['miRNA_seq'].iloc[0])
for i in data_gene_f:
    data_gene_f_seq.append(all_data[all_data['gene_id']==i]['gene_seq'].iloc[0])
more_data_f['miRNA_seq']=data_mirna_f_seq
more_data_f['gene_seq']=data_gene_f_seq
more_data_f['result']=[0 for i in range(more_data_f.shape[0])]
# 阴性样本创建完毕，共20335


# 以0.05的比例修改阴性数据，防止模型只识别哪些gene是被修改的
all_n_data=pd.read_csv('more_data_n.csv')
all_process_n_data=make_ndata(all_n_data,ratio=0.05)
all_process_n_data.to_csv('more_process_n_data_0.05.csv',index=False)
# data=pd.read_csv('more_process_n_data_0.05.csv')
# l_train=random.sample(range(data.shape[0]),12200)
# l_test=[]
#
# while len(l_test)<147:
#     tem=random.randint(0,data.shape[0]-1)
#     if tem not in l_train:
#         l_test.append(tem)
# data.iloc[l_train].to_csv('contrast_n_train_0.05_data.csv',index=False)
# data.iloc[l_test].to_csv('contrast_n_test_0.05_data.csv',index=False)