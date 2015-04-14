num_factors=100
num_workers=100
num_iterations=100
beta_value=0.9
lambda_value=0.1
# input_v_filepath=data/test_v.txt
# output_w_filepath=data/autolab_w.txt
# output_h_filepath=data/autolab_h.txt
# input_v_filepath=data/autolab_train.csv

#output_w_filepath=data/test_out_w.txt
#output_h_filepath=data/test_out_h.txt

input_v_filepath=s3://sanchuah-bigml-hw7/netflix.50000.txt
output_w_filepath=data/netflix_50000_w.txt
output_h_filepath=data/netflix_50000_h.txt

master=spark://ip-172-31-34-87.ec2.internal:7077 
memory=6g

run:
	spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

run-ec2:
	Master=$(master) spark-submit --master $(master) --executor-memory $(memory) --driver-memory $(memory) --total-executor-cores $(num_workers) --supervise dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)
	

eval:
	python eval.pyc /tmp/eval_acc.log spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)
	python eval_acc.py /tmp/eval_acc.log spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

test:
	spark-submit test-partitions.py

lnzsl:
	python LNZSL.py $(input_v_filepath)
