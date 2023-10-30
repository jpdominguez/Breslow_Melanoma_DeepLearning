python Breslow/full_supervision/test/Student_Inference_patches.py -n 0 -b 32
python Breslow/full_supervision/test/Student_Inference_patches.py -n 1 -b 32
python Breslow/full_supervision/test/Student_Inference_patches.py -n 2 -b 32
python Breslow/full_supervision/test/Student_Inference_patches.py -n 3 -b 32
python Breslow/full_supervision/test/Student_Inference_patches.py -n 4 -b 32

python Breslow/full_supervision/test/get_mean_metrics.py -n 0
python Breslow/full_supervision/test/get_mean_metrics.py -n 1
python Breslow/full_supervision/test/get_mean_metrics.py -n 2
python Breslow/full_supervision/test/get_mean_metrics.py -n 3
python Breslow/full_supervision/test/get_mean_metrics.py -n 4



python Breslow/semi_supervision/test/Student_Inference_patches.py -n 0 -b 32
python Breslow/semi_supervision/test/Student_Inference_patches.py -n 1 -b 32
python Breslow/semi_supervision/test/Student_Inference_patches.py -n 2 -b 32
python Breslow/semi_supervision/test/Student_Inference_patches.py -n 3 -b 32
python Breslow/semi_supervision/test/Student_Inference_patches.py -n 4 -b 32

python Breslow/semi_supervision/test/get_mean_metrics.py -n 0
python Breslow/semi_supervision/test/get_mean_metrics.py -n 1
python Breslow/semi_supervision/test/get_mean_metrics.py -n 2
python Breslow/semi_supervision/test/get_mean_metrics.py -n 3
python Breslow/semi_supervision/test/get_mean_metrics.py -n 4

python Breslow/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 0
python Breslow/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 1
python Breslow/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 2
python Breslow/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 3
python Breslow/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 4




python InSitu_Invassive/full_supervision/test/Student_Inference_patches.py -n 10 -b 32
python InSitu_Invassive/full_supervision/test/Student_Inference_patches.py -n 11 -b 32
python InSitu_Invassive/full_supervision/test/Student_Inference_patches.py -n 12 -b 32
python InSitu_Invassive/full_supervision/test/Student_Inference_patches.py -n 13 -b 32
python InSitu_Invassive/full_supervision/test/Student_Inference_patches.py -n 14 -b 32

python InSitu_Invassive/full_supervision/test/get_mean_metrics.py -n 10
python InSitu_Invassive/full_supervision/test/get_mean_metrics.py -n 11
python InSitu_Invassive/full_supervision/test/get_mean_metrics.py -n 12
python InSitu_Invassive/full_supervision/test/get_mean_metrics.py -n 13
python InSitu_Invassive/full_supervision/test/get_mean_metrics.py -n 14



python InSitu_Invassive/semi_supervision/test/Student_Inference_patches.py -n 10 -b 32
python InSitu_Invassive/semi_supervision/test/Student_Inference_patches.py -n 11 -b 32
python InSitu_Invassive/semi_supervision/test/Student_Inference_patches.py -n 12 -b 32
python InSitu_Invassive/semi_supervision/test/Student_Inference_patches.py -n 13 -b 32
python InSitu_Invassive/semi_supervision/test/Student_Inference_patches.py -n 14 -b 32

python InSitu_Invassive/semi_supervision/test/get_mean_metrics.py -n 10
python InSitu_Invassive/semi_supervision/test/get_mean_metrics.py -n 11
python InSitu_Invassive/semi_supervision/test/get_mean_metrics.py -n 12
python InSitu_Invassive/semi_supervision/test/get_mean_metrics.py -n 13
python InSitu_Invassive/semi_supervision/test/get_mean_metrics.py -n 14

python InSitu_Invassive/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 10
python InSitu_Invassive/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 11
python InSitu_Invassive/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 12
python InSitu_Invassive/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 13
python InSitu_Invassive/semi_supervision/test/get_mean_metrics_with_best_from_teachers.py -n 14