# Intro

UPMEM_DLRM is a end-to-end implementation of DLRM in UPMEM system.
The Framework of this repo is based upon PIM-Embedding-Lookup from: (https://github.com/UBC-ECE-Sasha/PIM-Embedding-Lookup)

# Usage

To use it, please config parameters in file **"run.sh"**.**"run.sh"** is a file of shell script, bascially all you need to modify is in this file.

By changing the parameters in **random_env() and random_run()**, you should be able to configure the DLRM model. For example, You can modify the number of embedding tables by changing the value of **"NR_TABLES"**. It is important to note that the value of **"NR_TABLES"*** should match the number of embedding tables you input. Table's dimension is changeable as well as input's batch size and pooling factor.
![image](https://github.com/Wu0103/UPMEM-DLRM/assets/94586355/2d38e5ed-bb2b-41ef-be68-99623df8a3f3)
By configure **"NR_DPUS"**, you can decide how many DPUs are used for one table; To use different partition scheme, you should configure **"COL_DPU"**, which indicate how many DPUs you are using to do column-wise partition. For example, for a 200*10 elements table, if you configure **"COL_DPU"** as 10, it's column-wise partition; If you configure **"COL_DPU"** as 5 or 2, it's hybrid partition; If you configure **"COL_DPU"** as 1, it's row partition.

**One thing to mention is you need to change the file path in "run.sh" into your specific settings.**

**You have the options to configure the number of DPUs and tasklets to be used. It is important to ensure that enough DPUs are allocated to save all embedding tables, otherwise, errors may occur.**

After configing all the parameters,use **./run.sh -br random** in the terminal to try with syntheic input.

To see the performance of running DLRM in CPU, you can simply change **python "${dlrm}/dlrm_dpu_pytorch.py"** in file **"run.sh"** to **python "${dlrm}/dlrm_s_pytorch.py"**.

# Results

DLRM latency will be automatically print to terminal, for detailed latency breakdown please uncomment some part of code in dlrm_dpu_pytorch.py. It will save the results to your local file.
