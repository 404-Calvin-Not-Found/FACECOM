import os
import sys

# Add root directory to import config properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import PathConfig  # <- This must succeed

TASK_B_VAL = PathConfig.TASK_B_VAL  # <- Explicitly define the variable

def validate_task_b_dataset():
    """Validate the structure and content of Task B dataset"""
    print("Validating Task B dataset structure...")

    required_files = {
        'root': ['.jpg', 'distortion/'],
        'distortion': ['.jpg']
    }

    issues_found = 0

    for person_id in sorted(os.listdir(TASK_B_VAL)):
        person_path = os.path.join(TASK_B_VAL, person_id)
        if not os.path.isdir(person_path):
            print(f"[!] {person_id}: Not a directory")
            issues_found += 1
            continue

        # Check main image
        main_img = os.path.join(person_path, f"{person_id}.jpg")
        if not os.path.exists(main_img):
            print(f"[!] {person_id}: Missing main image {person_id}.jpg")
            issues_found += 1

        # Check distortion folder
        distortion_dir = os.path.join(person_path, "distortion")
        if not os.path.isdir(distortion_dir):
            print(f"[!] {person_id}: Missing distortion directory")
            issues_found += 1
            continue

        # Check distortion images
        distortion_imgs = [f for f in os.listdir(distortion_dir)
                           if f.lower().endswith('.jpg') and f.startswith(person_id)]

        #if len(distortion_imgs) != 6:
         #   print(f"[!] {person_id}: Has {len(distortion_imgs)}/6 distortion images")
          #  issues_found += 1

        # Check for misassigned images
        for img in os.listdir(distortion_dir):
            if img.lower().endswith('.jpg') and not img.startswith(person_id):
                print(f"[!] {person_id}: Contains misassigned image {img}")
                issues_found += 1

    if issues_found == 0:
        print("✅ Validation passed - no issues found!")
    else:
        print(f"\n[!] Found {issues_found} issue(s) in the dataset structure.")

if __name__ == "__main__":
    validate_task_b_dataset()
