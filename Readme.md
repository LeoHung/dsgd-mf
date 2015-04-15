# DSGD-MF on pyspark

## Author 
San-Chuan Hung (sanchuah at andrew.cmu.edu)

## Introduction
This is the implementation of Distributed Stochastic Gradient Descent Matrix Factorization (DSGD-MF)(Gemulla et al. 2011)[1] on pyspark platform. 

## Usage

You can run DSGD-MF in local mode or cluster mode. To make it simple, you can configure parameters in Makefile to run with "make run" or "make run-cluster"

Or, you can run the code in following command:

### Local Mode

	spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

### Cluster Mode

	Master=$(master) spark-submit --master $(master) --executor-memory $(memory) --driver-memory $(memory) $(num_workers) --supervise dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

## Reference

[1] Gemulla, R., Nijkamp, E., Haas, P. J., & Sismanis, Y. (2011). Large-scale matrix factorization with distributed stochastic gradient descent (pp. 69–77). New York, New York, USA: ACM. doi:10.1145/2020408.2020426