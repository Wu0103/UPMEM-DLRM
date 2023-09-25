# UPMEM-DLRM

UPMEM_DLRM is a end-to-end implementation of DLRM in UPMEM system.
The Framework of this repo is based upon PIM-Embedding-Lookup from:https://github.com/UBC-ECE-Sasha/PIM-Embedding-Lookup

To use it, please config parameters in run.sh. By changing the parameters in random_env() and random_run(), you should be able to configure the DLRM model.
You can modify the number of embedding tables by changing the value of "NR_TABLES". It is important to note that the value of "NR_TABLES" should match the number of embedding tables you input. Table's dimension is changeable as well as input's batch size and pooling factor.
You have the option to configure the number of DPUs and tasklets to be used. It is important to ensure that enough DPUs are allocated to save all embedding tables, otherwise, errors may occur.
After configing all the parameters,use ./run.sh -br random to try with syntheic input.

DLRM latency will be automatically print to terminal, for detailed latency breakdown please uncomment some part of code in dlrm_dpu_pytorch.py. It will save the results to your local file.
