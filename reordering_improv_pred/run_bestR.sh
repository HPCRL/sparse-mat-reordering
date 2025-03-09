#!/bin/bash

source /media/datassd/sina/spmv-omid/myenv/bin/activate
cd /media/datassd/sina/spmv-omid/proj/reordering_improv_pred

python exp_mnsplit_bestR2.py --th 1.25 > logs/log_rep_mnsplit_bestR_th125.txt

python exp_mnsplit_bestR2.py --th 1.10 > logs/log_rep_mnsplit_bestR_th110.txt

python exp_mnsplit_bestR2.py --th 1.15 > logs/log_rep_mnsplit_bestR_th115.txt

python exp_mnsplit_bestR2.py --th 1.20 > logs/log_rep_mnsplit_bestR_th120.txt

python exp_mnsplit_bestR2.py --th 1.50 > logs/log_rep_mnsplit_bestR_th150.txt


