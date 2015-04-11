num_factors=5
num_workers=1
num_iterations=10
beta_value=0.8
lambda_value=1.0
input_v_filepath=data/test_v.txt
output_w_filepath=data/test_out_w.txt
output_h_filepath=data/test_out_h.txt

default: run

run:
	spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) \
    $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

eval:
	python eval.pyc /tmp/eval_acc.log spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)
	python eval_acc.py /tmp/eval_acc.log spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

test:
	spark-submit test-partitions.py