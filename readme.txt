## Requirements

Python 3.10+ with `numpy`, `scipy`, `pandas` and `matplotlib`.

## Running

python run_all.py --stage verification
python run_all.py --stage calibration
python run_all.py --stage baseline
python run_all.py --stage policy_sweep
python run_all.py --stage factor --factor q_2

Available factors: lambda_1, lambda_2, q_1, q_2, theta_applicants, theta_1, theta_2, theta_e, composition, composition_equal_q.

Results are saved in results/.