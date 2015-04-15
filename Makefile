num_factors=10
num_workers=2
num_iterations=30
beta_value=0.6
lambda_value=0.1
# output_w_filepath=data/autolab_w.txt
# output_h_filepath=data/autolab_h.txt
# input_v_filepath=data/autolab_train.csv

#input_v_filepath=data/test_v.txt
#output_w_filepath=data/test_out_w.txt
#output_h_filepath=data/test_out_h.txt

#input_v_filepath=s3://sanchuah-bigml-hw7/netflix.50000.txt
#input_v_filepath=data/netflix.50000.txt
#output_w_filepath=data/netflix_50000_w.txt
#output_h_filepath=data/netflix_50000_h.txt

input_v_filepath=data/nf_subsample.csv
output_w_filepath=data/nf_subsample_w.txt
output_h_filepath=data/nf_subsample_h.txt

master=spark://ip-172-31-34-87.ec2.internal:7077
memory=6g

run:
	spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

run-exp3:
	for this_num_factors in 10 20 30 40 50 60 70 80 90 100 ; do\
	    spark-submit dsgd_mf.py $$this_num_factors $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath) 1> result_subsample_F_$$this_num_factors 2> error ; done;

run-exp4:
	for this_beta in 0.5 0.6 0.7 0.8 0.9 ; do\
	    spark-submit dsgd_mf.py 20 10 30 $$this_beta $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath) 1> result_subsample_B_$$this_beta 2> error ; done;



run-ec2:
	Master=$(master) spark-submit --master $(master) --executor-memory $(memory) --driver-memory $(memory) --total-executor-cores $(num_workers) --supervise dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

run-mf:
	python mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath).mf $(output_h_filepath).mf

eval:
	python eval.pyc /tmp/eval_acc.log spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)
	python eval_acc.py /tmp/eval_acc.log spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

test:
	spark-submit test-partitions.py

lnzsl:
	python LNZSL.py $(input_v_filepath)
