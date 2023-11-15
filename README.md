# Intro

UPMEM_DLRM is a end-to-end implementation of DLRM in UPMEM system.
The Framework of this repo is based upon PIM-Embedding-Lookup from: (https://github.com/UBC-ECE-Sasha/PIM-Embedding-Lookup)

# Pre-requirment
To set up the environment, please follow comands below. For reference, version of different packages are provided. 
'''
Python -m virtualenv fold_name
source ~/fold_name/bin/activate
pip install numpy
pip install torch
pip install tqdm
pip install scikit-learn
pip install tensorboard
pip install six
pip install psutil
'''

![image](https://github.com/Wu0103/UPMEM-DLRM/assets/94586355/01351910-5fb4-40e1-97b5-e7c1936b3a61)


**You need to change the file path in **"/PIM-Embedding-Lookup/upmem/run.sh"**" into your specific settings---please refer to the figure below.**(Default path is “/home/kaist_icn/wuxiangyu/upload/dlrm/PIM-Embedding-Lookup/upmem/PIM-common/common/include/common.h”)

![image](https://github.com/Wu0103/UPMEM-DLRM/assets/94586355/8badf847-01bd-4daa-b051-252971df53e5)

# Usage

To use it, please config parameters in file **"/PIM-Embedding-Lookup/upmem/run.sh"**.**"run.sh"** is a file of shell script, bascially all you need to modify is in this file.

By changing the parameters in **random_env() and random_run()**, you should be able to configure the DLRM model. For example, You can modify the number of embedding tables by changing the value of **"NR_TABLES"**. It is important to note that the value of **"NR_TABLES"*** should match the number of embedding tables you input. Table's dimension is changeable as well as input's batch size and pooling factor.

![image](https://github.com/Wu0103/UPMEM-DLRM/assets/94586355/2d38e5ed-bb2b-41ef-be68-99623df8a3f3)

By configure **"NR_DPUS"**, you can decide how many DPUs are used for one table; **There are 3 types of available partition scheme, column-wise, row-wise and column-row hybrid partition** ; To use different partition scheme, you should configure **"COL_DPU"**, which indicate how many DPUs you are using to do column-wise partition. For example, for a 200*10 elements table (200 is the length of the table and 10 is the width/dimension of the table), if you configure **"COL_DPU"** as 10, it's column-wise partition; If you configure **"COL_DPU"** as 5 or 2, it's hybrid partition; If you configure **"COL_DPU"** as 1, it's row partition.

The relation between **"NR_DPUS"** and **"COL_DPU"** is, firstly **"COL_DPU"** should not bigger than **"NR_DPUS"**; For column wise partition, **"NR_DPUS"** and **"COL_DPU"** need to be the same value; For row wise partition, **"NR_DPUS"** is the number of DPUs you want to use and  **"COL_DPU"** should be 1; For hybrid partition, the results of **"NR_DPUS"/"COL_DPU"** should be an integer number;


**You have the options to configure the number of DPUs and tasklets to be used. It is important to ensure that enough DPUs are allocated to save all embedding tables, otherwise, errors may occur.**

After configing all the parameters,use **./run.sh -br random** in the terminal to try with synthetic input; For now, only synthetic is supported.

To see the performance of running DLRM in CPU, you can simply change **python "${dlrm}/dlrm_dpu_pytorch.py"** in file **"run.sh"** to **python "${dlrm}/dlrm_s_pytorch.py"**.

# Results

DLRM latency will be automatically print to terminal, just like the figure below shows: CPU-DPU indicates the latency of transfer input data from CPU to DPUs. kernel indicates DPU program running latency and DPU-CPU indicates the latency of get results back to CPU. For detailed latency breakdown please uncomment some part of code in dlrm_dpu_pytorch.py. It will save the results to your local file.

![image](https://github.com/Wu0103/UPMEM-DLRM/assets/94586355/522fea37-ca9a-4e53-b811-49a2d9fbab3a)


# For your customization

If you want to implement something new, my modification point can be good starting point. Below are the functions I modified.
  
  /PIM-Embedding-Lookup/UPMEM/run.sh: 
  
      changed random_env() to make system configureable as showed in README
      
  /PIM-Embedding-Lookup/UPMEM/scr/emb_host.c:
  
      changed populate_mram() function to preload table data to dpu_set_t
      changed lookup() function to first transfer index data and offset data to DPUs, and then invoke kernel program to do embedding lookup in DPUs; After that will get results back to CPU and convert the results into float point data

  /PIM-Embedding-Lookup/UPMEM/src/dpu/emb_dpu_lookup.c:
  
      changed main() funcion to load data from MRAM into WRAM
      changed lookup() function to do embedding lookup


# Contact

Please email me at wuxiangyu@kaist.ac.kr if you have any problem using it.
