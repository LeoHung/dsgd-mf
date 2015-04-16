num_factors=10
num_workers=2
num_iterations=30
beta_value=0.6
lambda_value=0.1

input_v_filepath=data/nf_subsample.csv
output_w_filepath=data/nf_subsample_w.txt
output_h_filepath=data/nf_subsample_h.txt

master=spark://[your spark master ip with port]
memory=6g

run:
	spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

run-cluster:
	Master=$(master) spark-submit --master $(master) --executor-memory $(memory) --driver-memory $(memory) --supervise dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

run-mf:
	python mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath).mf $(output_h_filepath).mf

eval:
	python eval.pyc /tmp/eval_acc.log spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)
	python eval_acc.py /tmp/eval_acc.log spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

output_beta=0.5
output_iteration=15
output_lambda=0.0

run-output:
	python eval.pyc eval_acc.log spark-submit dsgd_mf.py 20 3 $(output_iteration) $(output_beta) $(output_lambda) data/autolab_train.csv w.csv h.csv > spark_dsgd.log 

lnzsl:
	python LNZSL.py data/autolab_train.csv w.csv h.csv

tar:
	tar -cvf hw7.tar *.py *.csv *.log sanchuah-report.pdf
