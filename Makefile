num_factors=1
num_workers=2
num_iterations=1
beta_value=0.8
lambda_value=1.0
input_v_filepath=data/test_v.txt
output_w_filepath=data/test_out_w.txt
output_h_filepath=data/test_out_h.txt

default: run

run:
	spark-submit dsgd_mf.py $(num_factors) $(num_workers) $(num_iterations) $(beta_value) \
    $(lambda_value) $(input_v_filepath) $(output_w_filepath) $(output_h_filepath)

test:
	spark-submit test-partitions.py
