import os
from PIL import Image
from torchvision import transforms

from homework5.SaltPepperNoise import SaltPepperNoise

if __name__ == "__main__":
    transform_methods = {
        transforms.Resize((224, 224)): "Resize",
        transforms.RandomHorizontalFlip(p=1): "RandomHorizontalFlip",
        transforms.RandomVerticalFlip(p=1): "RandomVerticalFlip",
        transforms.RandomRotation((0, 90)): "RandomRotation",
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.8, hue=0): "ColorJitter",
        transforms.RandomGrayscale(p=1): "RandomGrayscale",
        transforms.Grayscale(): "Grayscale",
        transforms.FiveCrop((224, 224)): "FiveCrop",
        transforms.GaussianBlur(3, sigma=(1, 2)): "GaussianBlur",
        transforms.RandomErasing(): "RandomErasing",
        SaltPepperNoise(0.7, 1): "SaltPepperNoise",
    }

    cwd = os.getcwd()
    src_dir = os.path.join(cwd, "transform_img_RMB", "src")
    tar_dir = os.path.join(cwd, "transform_img_RMB", "tar")
    img_name = "myRMB100.jpg"

    # 创建目标文件夹（如果不存在）
    os.makedirs(tar_dir, exist_ok=True)

    # 打开源图像
    img_path = os.path.join(src_dir, img_name)
    img = Image.open(img_path)

    for transform_method, transform_name in transform_methods.items():
        try:
            print("Trying to apply " + transform_name)
            # 对于需要张量的操作，将图像转换为张量
            if isinstance(transform_method, transforms.RandomErasing):
                img_tensor = transforms.ToTensor()(img)
                transformed_img_tensor = transform_method(img_tensor)
                transformed_img = transforms.ToPILImage()(transformed_img_tensor)
            elif isinstance(transform_method, transforms.FiveCrop):
                crops = transform_method(img)  # 返回5个裁剪后的图像
                for i, crop in enumerate(crops):
                    crop.save(os.path.join(tar_dir, f"{img_name.split('.')[0]}_{transform_name}_part{i}.jpg"))
                continue
            else:
                transformed_img = transform_method(img)

            # 保存转换后的图像
            transformed_img.save(os.path.join(tar_dir, f"{img_name.split('.')[0]}_{transform_name}.jpg"))

        except Exception as e:
            print(f"Unable to apply transform {transform_name}: {str(e)}")
