$input_dir = "datas/intermediate_datas/silhouette_inputs_medoids200_1013"
$temp_dir = "datas/intermediate_datas/silhouettes"
$output_dir = "datas\silhouettes\medoids200_1013"
$model = "vendors/pidinet/trained_models/table5_pidinet.pth"

if (-not (Test-Path -Path $temp_dir -PathType Container))  { 
    New-Item -Path $temp_dir -ItemType Directory
}


python vendors/pidinet/main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir $temp_dir --datadir $input_dir --dataset Custom --evaluate $model --evaluate-converted

Get-ChildItem -Path $temp_dir"\eval_results\imgs_epoch_019" -Filter *.png | Move-Item -Destination $output_dir
Remove-Item -Path $temp_dir -Recurse -Force