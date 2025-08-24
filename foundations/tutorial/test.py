import os

root = r"D:\PlantVillage"

class_counts = {}
total_count = 0

for class_name in os.listdir(root):
    class_path = os.path.join(root, class_name)
    if os.path.isdir(class_path):  
        count = 0
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
        class_counts[class_name] = count
        total_count += count

for cls, cnt in class_counts.items():
    print(f"{cls}: {cnt} images")

print(f"\nTotal images in PlantVillage: {total_count}")
