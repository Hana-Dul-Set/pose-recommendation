
$model_path = "vendors/Automated-objects-removal-inpainter/checkpoints/places2"
$input_path = "datas/intermediate_datas/silhouette_inputs_medoids200_1013"
$output_path = "datas/silhouette_masks/medoids200_1013"

python vendors/Automated-objects-removal-inpainter/test.py --path $model_path --input $input_path  --output $output_path --remove 15