
$model_path = "vendors/Automated-objects-removal-inpainter/checkpoints/places2"
$input_path = "datas/intermediate_datas/silhouette_inputs"
$output_path = "datas/silhouette_masks"

python vendors/Automated-objects-removal-inpainter/test.py --path $model_path --input $input_path  --output $output_path --remove 15