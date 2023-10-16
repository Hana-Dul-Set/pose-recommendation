param (
    [string]$cluster_result_path
)


#$cluster_result_path = "../../datas/cluster_results/kmedoids_pam_200_test_1010.json"
$dir_identifier = Get-Item $cluster_result_path | Select-Object -ExpandProperty BaseName

$raw_images_dir = "datas/intermediate_datas/" + $dir_identifier + "/raw_images"
$temp_dir = "datas/intermediate_datas/temp"
$mask_output_dir = "datas/intermediate_datas/" + $dir_identifier + "/masks"
$pidinet_output_dir = "datas/intermediate_datas/" + $dir_identifier + "/pidinet_outputs"

$silhouette_dir = "datas/sihlouettes/" + $dir_identifier

#select images
if (-not (Test-Path -Path $raw_images_dir -PathType Container))  { 
    New-Item -Path $raw_images_dir -ItemType Directory
}
python src/asset_converter/copy_center_images.py $cluster_result_path $raw_images_dir


#run mask
$mask_model_path = "vendors/Automated-objects-removal-inpainter/checkpoints/places2"

if (-not (Test-Path -Path $mask_output_dir -PathType Container))  { 
    New-Item -Path $mask_output_dir -ItemType Directory
}
python vendors/Automated-objects-removal-inpainter/test.py --path $mask_model_path --input $raw_images_dir  --output $mask_output_dir --remove 15

#run pidinet
$pidinet_model_path= "vendors/pidinet/trained_models/table5_pidinet.pth"

if (-not (Test-Path -Path $pidinet_output_dir -PathType Container))  { 
    New-Item -Path $pidinet_output_dir -ItemType Directory
}

if (-not (Test-Path -Path $temp_dir -PathType Container))  { 
    New-Item -Path $temp_dir -ItemType Directory
}
python vendors/pidinet/main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir $temp_dir --datadir $raw_images_dir --dataset Custom --evaluate $pidinet_model_path --evaluate-converted

Get-ChildItem -Path $temp_dir"\eval_results\imgs_epoch_019" -Filter *.png | Move-Item -Destination $pidinet_output_dir
Remove-Item -Path $temp_dir -Recurse -Force

#gen images

if (-not (Test-Path -Path $silhouette_dir -PathType Container))  { 
    New-Item -Path $silhouette_dir -ItemType Directory
}
python src/asset_converter/silhouette.py $pidinet_output_dir $mask_output_dir $silhouette_dir