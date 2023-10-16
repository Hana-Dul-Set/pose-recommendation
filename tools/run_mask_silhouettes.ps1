param (
    [string]$cluster_result_path
)


#$cluster_result_path = "../../datas/cluster_results/kmedoids_pam_200_test_1010.json"
$dir_identifier = Get-Item $cluster_result_path | Select-Object -ExpandProperty BaseName

$raw_images_dir = "datas/intermediate_datas/" + $dir_identifier + "/raw_images"
$temp_dir = "datas/intermediate_datas/temp"
$mask_output_dir = "datas/intermediate_datas/" + $dir_identifier + "/masks"
$pidinet_output_dir = "datas/intermediate_datas/" + $dir_identifier + "/pidinet_outputs"
$silhouette_dir = "datas/silhouettes/" + $dir_identifier


if (-not (Test-Path -Path $silhouette_dir -PathType Container))  { 
    New-Item -Path $silhouette_dir -ItemType Directory
}
python src/asset_converter/silhouette.py $pidinet_output_dir $mask_output_dir $silhouette_dir