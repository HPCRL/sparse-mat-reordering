#!/bin/bash

source /media/datassd/sina/spmv-omid/myenv/bin/activate
cd /media/datassd/sina/spmv-omid/proj/reordering_improv_pred

# python exp_mnsplit_bestR.py --th 1.05 > logs/log_rep_mnsplit_bestR_th100.txt

# python exp_mnsplit_bestR_easy.py --th 1.05 > logs/easy/log_rep_mnsplit_bestR_easy_th105.txt

# python exp_mnsplit_bestR_easy.py --th 1.00 > logs/easy/log_rep_mnsplit_bestR_easy_th100.txt

# python exp_mnsplit_bestR_easy.py --th 1.10 > logs/easy/log_rep_mnsplit_bestR_easy_th110.txt

# python exp_mnsplit_bestR_easy.py --th 1.25 > logs/easy/log_rep_mnsplit_bestR_easy_th125.txt

# python exp_mnsplit_bestR_easy.py --th 1.15 > logs/easy/log_rep_mnsplit_bestR_easy_th115.txt

# python exp_mnsplit_bestR_easy.py --th 1.20 > logs/easy/log_rep_mnsplit_bestR_easy_th120.txt

# python exp_mnsplit_bestR_easy.py --th 1.50 > logs/easy/log_rep_mnsplit_bestR_easy_th150.txt

# python exp_mnsplit_bestR_easy.py --th 2.00 > logs/easy/log_rep_mnsplit_bestR_easy_th200.txt




python exp_mnsplit_bestR_easy_1d2d.py --th 1.05 > logs/easy_1d2d/log_rep_mnsplit_bestR_easy_th105.txt

python exp_mnsplit_bestR_easy_1d2d.py --th 1.00 > logs/easy_1d2d/log_rep_mnsplit_bestR_easy_th100.txt

python exp_mnsplit_bestR_easy_1d2d.py --th 1.10 > logs/easy_1d2d/log_rep_mnsplit_bestR_easy_th110.txt

python exp_mnsplit_bestR_easy_1d2d.py --th 1.25 > logs/easy_1d2d/log_rep_mnsplit_bestR_easy_th125.txt

python exp_mnsplit_bestR_easy_1d2d.py --th 1.15 > logs/easy_1d2d/log_rep_mnsplit_bestR_easy_th115.txt

python exp_mnsplit_bestR_easy_1d2d.py --th 1.20 > logs/easy_1d2d/log_rep_mnsplit_bestR_easy_th120.txt

python exp_mnsplit_bestR_easy_1d2d.py --th 1.50 > logs/easy_1d2d/log_rep_mnsplit_bestR_easy_th150.txt

python exp_mnsplit_bestR_easy_1d2d.py --th 2.00 > logs/easy_1d2d/log_rep_mnsplit_bestR_easy_th200.txt