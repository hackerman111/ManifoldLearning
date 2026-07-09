• Да, по Tests.md команды лучше делать так: для эксперимента 5 отдельно full ADP и отдельно ablation, потому что в файле заданы разные размеры.

Общие параметры из Tests.md:
K_max=8, m_max=20, gamma_h=0.8, lambda_rel=1e-2, для серьезного финального подтверждения J/n=1.0, то есть --center-fraction 1.0.

Для сетки с d=50,100,200 беру консервативный вариант из блока для d=200: n_phi=128, n_min=512.

mkdir -p logs outputs

Эксперимент 4: rho_k/aniso
Из Tests.md: 162 сценария × 100 seed = 16200 full ADP запусков.

MPLCONFIGDIR=/tmp/matplotlib nohup python experiments/adp_experiment_4_rho.py \
 --out outputs/adp_experiment_4_rho_serious_tests_md \
 --d 50,100,200 \
 --n-over-d 10,20 \
 --corr 0.0,0.3,0.7 \
 --snr 20,10,5 \
 --links linear,tanh,sin \
 --q 0.3 \
 --seeds 100 \
 --n-directions 128 \
 --min-neighbors 512 \
 --center-fraction 1.0 \
 --lambda-rel 1e-2 \
 --outer-steps 8 \
 --inner-steps 20 \
 --gamma-h 0.8 \
 --jobs 16 \
 --bootstrap-reps 1000 \
 --backend numpy \

> logs/adp_experiment_4_rho_serious_tests_md.log 2>&1 &

Эксперимент 5a: core full ADP
Из Tests.md: 72 сценария × 200 seed = 14400 full ADP запусков.

MPLCONFIGDIR=/tmp/matplotlib nohup python experiments/adp_experiment_5_cos_growth.py \
 --out outputs/adp_experiment_5_cos_growth_core_tests_md \
 --d 50,100,200 \
 --n-over-d 10,20 \
 --corr 0.0,0.3,0.7 \
 --snr 20,10,5 \
 --links linear,tanh,sin \
 --q 0.3,1.0 \
 --max-scenarios 72 \
 --seeds 200 \
 --methods full_adp \
 --n-directions 128 \
 --min-neighbors 512 \
 --center-fraction 1.0 \
 --lambda-rel 1e-2 \
 --outer-steps 8 \
 --inner-steps 20 \
 --gamma-h 0.8 \
 --jobs 16 \
 --bootstrap-reps 2000 \
 --backend numpy \

> logs/adp_experiment_5_cos_growth_core_tests_md.log 2>&1 &

Эксперимент 5b: ablations
Из Tests.md: 36 сценариев × 200 seed × 6 методов = 43200 запусков.

MPLCONFIGDIR=/tmp/matplotlib nohup python experiments/adp_experiment_5_cos_growth.py \
 --out outputs/adp_experiment_5_cos_growth_ablation_tests_md \
 --d 50,100,200 \
 --n-over-d 10,20 \
 --corr 0.0,0.3,0.7 \
 --snr 20,10,5 \
 --links linear,tanh,sin \
 --q 0.3,1.0 \
 --max-scenarios 36 \
 --seeds 200 \
 --methods full_adp,step0_only,no_anisotropy,fixed_h,no_regularization,random_beta_init \
 --n-directions 128 \
 --min-neighbors 512 \
 --center-fraction 1.0 \
 --lambda-rel 1e-2 \
 --outer-steps 8 \
 --inner-steps 20 \
 --gamma-h 0.8 \
 --jobs 16 \
 --bootstrap-reps 2000 \
 --backend numpy \

> logs/adp_experiment_5_cos_growth_ablation_tests_md.log 2>&1 &

Эксперимент 6: финальный success protocol
В Tests.md для него нет отдельной таблицы запусков. Поэтому беру размер из confirmatory/cos-growth блока: 72 сценария × 200 seed, без ablations.

MPLCONFIGDIR=/tmp/matplotlib nohup python experiments/adp_experiment_6_final_success.py \
 --out outputs/adp_experiment_6_final_success_tests_md \
 --d 50,100,200 \
 --n-over-d 10,20 \
 --corr 0.0,0.3,0.7 \
 --snr 20,10,5 \
 --links linear,tanh,sin \
 --q 0.3,1.0 \
 --max-scenarios 72 \
 --seeds 200 \
 --n-directions 128 \
 --min-neighbors 512 \
 --center-fraction 1.0 \
 --lambda-rel 1e-2 \
 --outer-steps 8 \
 --inner-steps 20 \
 --gamma-h 0.8 \
 --jobs 16 \
 --bootstrap-reps 2000 \
 --backend numpy \

> logs/adp_experiment_6_final_success_tests_md.log 2>&1 &

Мониторинг:

tail -f logs/adp_experiment_4_rho_serious_tests_md.log
tail -f logs/adp_experiment_5_cos_growth_core_tests_md.log
tail -f logs/adp_experiment_5_cos_growth_ablation_tests_md.log
tail -f logs/adp_experiment_6_final_success_tests_md.log

python experiments/adp_experiment_4_rho.py \
 --out outputs/adp_experiment_4_rho_serious \
 --d 100 \
 --n-over-d 10,20 \
 --corr 0.0,0.3,0.7 \
 --snr 20,10,5 \
 --links linear,tanh,sin \
 --q 0.3 \
 --seeds 100 \
 --n-directions 64 \
 --min-neighbors 256 \
 --center-fraction 0.5 \
 --lambda-rel 1e-2 \
 --outer-steps 8 \
 --inner-steps 20 \
 --gamma-h 0.8 \
 --jobs "$(( $(nproc) - 1 ))" \
 --bootstrap-reps 1000 \
 --backend numpy \
