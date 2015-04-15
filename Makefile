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

