import os
import subprocess
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

base_data_dir = "D:/monai-project/asa/data"

input_volumes_dir = os.path.join(base_data_dir, "test_volume")
ground_truth_seg_dir = os.path.join(base_data_dir, "test_label")

totalseg_temp_output_dir = "D:/monai-project/asa/totalseg_temp_output"
final_liver_masks_dir = "D:/monai-project/asa/final_liver_masks"
output_slice_images_dir = "D:/monai-project/asa/output_slice_images"

os.makedirs(totalseg_temp_output_dir, exist_ok=True)
os.makedirs(final_liver_masks_dir, exist_ok=True)
os.makedirs(output_slice_images_dir, exist_ok=True)


def dice_score(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0
    return 2.0 * intersection / total

def jaccard_score(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0
    return intersection / union


print("Starting Liver Segmentation and Evaluation")

all_dice_scores = []
all_jaccard_scores = []
processed_case_names = []

volume_files = sorted([f for f in os.listdir(input_volumes_dir)
                       if f.endswith(".nii") or f.endswith(".nii.gz")])

if not volume_files:
    print(f"No NIfTI files found in {input_volumes_dir}. Please check the path and file extensions.")
else:
    for vol_file in volume_files:
        base_name = vol_file.replace(".nii.gz", "").replace(".nii", "")
        print(f"\nProcessing case: {base_name}")

        input_path = os.path.join(input_volumes_dir, vol_file)
        totalseg_case_output_dir = os.path.join(totalseg_temp_output_dir, base_name)
        
        print(f"Running TotalSegmentator on {vol_file}...")
        try:
            os.makedirs(totalseg_case_output_dir, exist_ok=True)
            
            subprocess.run([
                "TotalSegmentator",
                "-i", input_path,
                "-o", totalseg_case_output_dir,
                "--fast",
                "--roi_subset", "liver"
            ], check=True)

            total_seg_liver_mask_path = os.path.join(totalseg_case_output_dir, "liver.nii.gz")

            if os.path.exists(total_seg_liver_mask_path):
                final_mask_filename = f"{base_name}_liver_seg.nii.gz"
                destination_final_mask_path = os.path.join(final_liver_masks_dir, final_mask_filename)
                
                shutil.copyfile(total_seg_liver_mask_path, destination_final_mask_path)
                print(f"Saved TotalSegmentator liver mask to: {destination_final_mask_path}")

                ct_img = nib.load(input_path)
                ct_data = ct_img.get_fdata()

                gt_filename = vol_file
                gt_path = os.path.join(ground_truth_seg_dir, gt_filename)


                if not (os.path.exists(gt_path) and os.path.exists(total_seg_liver_mask_path)):
                    print(f"Skipping {base_name}: Missing ground truth ({gt_path}) or TotalSegmentator output ({total_seg_liver_mask_path}).")
                    continue

                gt_img = nib.load(gt_path)
                total_seg_img = nib.load(total_seg_liver_mask_path)

                gt_data = gt_img.get_fdata()
                total_seg_data = total_seg_img.get_fdata()

                gt_binary = (gt_data > 0).astype(np.uint8)
                total_seg_binary = (total_seg_data > 0).astype(np.uint8)

                if gt_binary.shape != total_seg_binary.shape:
                    print(f"Shape mismatch for {base_name}: Ground Truth shape {gt_binary.shape} vs TotalSegmentator shape {total_seg_binary.shape}. Skipping metric calculation for this case.")
                    continue

                dice = dice_score(total_seg_binary, gt_binary)
                jaccard = jaccard_score(total_seg_binary, gt_binary)

                all_dice_scores.append(dice)
                all_jaccard_scores.append(jaccard)
                processed_case_names.append(base_name)

                print(f"  Dice Score: {dice:.4f}")
                print(f"  Jaccard Score: {jaccard:.4f}")

                slices_with_liver = np.where(np.any(gt_binary > 0, axis=(0, 1)))[0]
                if len(slices_with_liver) > 0:
                    slice_idx = slices_with_liver[len(slices_with_liver)//2]
                else:
                    slice_idx = ct_data.shape[2] // 2

                plt.figure(figsize=(15, 5))
                plt.suptitle(f"{base_name} | Slice {slice_idx} | Dice: {dice:.4f} | Jaccard: {jaccard:.4f}", fontsize=14)

                plt.subplot(1, 3, 1)
                plt.imshow(ct_data[:, :, slice_idx], cmap="gray")
                plt.title("Input CT")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(gt_binary[:, :, slice_idx], cmap="Greens", alpha=0.7)
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(total_seg_binary[:, :, slice_idx], cmap="Reds", alpha=0.7)
                plt.title("TotalSegmentator Output")
                plt.axis("off")

                plt.tight_layout()
                slice_output_path = os.path.join(output_slice_images_dir, f"{base_name}_slice_{slice_idx}.png")
                plt.savefig(slice_output_path)
                plt.close()
                print(f"Saved slice visualization to: {slice_output_path}")

            else:
                print(f"Liver mask ('liver.nii.gz') not found for {vol_file} in {totalseg_case_output_dir}")

        except subprocess.CalledProcessError as e:
            print(f"Error running TotalSegmentator for {vol_file}: {e}")
        except FileNotFoundError:
            print("TotalSegmentator command not found. Ensure it's installed and added to your system's PATH.")
            break
        except Exception as e:
            print(f"An unexpected error occurred while processing {vol_file}: {e}")


print("\n Summary of Liver Segmentation Results")

if processed_case_names:
    for name, d, j in zip(processed_case_names, all_dice_scores, all_jaccard_scores):
        print(f"  {name}: Dice={d:.4f}, Jaccard={j:.4f}")

    mean_dice = np.mean(all_dice_scores)
    mean_jaccard = np.mean(all_jaccard_scores)

    print(f"\nOverall Mean Dice Score: {mean_dice:.4f}")
    print(f"Overall Mean Jaccard Score: {mean_jaccard:.4f}")

    plt.figure(figsize=(8, 5))
    metrics = ['Mean Dice', 'Mean Jaccard']
    values = [mean_dice, mean_jaccard]
    plt.bar(metrics, values, color=['skyblue', 'lightcoral'])
    plt.ylim(0, 1)
    plt.title("Average Liver Segmentation Scores (TotalSegmentator)")
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    average_scores_plot_path = os.path.join(output_slice_images_dir, "average_scores_bar_chart.png")
    plt.savefig(average_scores_plot_path)
    plt.close()
    print(f"Saved average scores bar chart to: {average_scores_plot_path}")

else:
    print("No volumes were successfully processed for evaluation.")

print("\n Processing Complete ")
