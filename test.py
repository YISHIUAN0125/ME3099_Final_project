import torch
from transform import transform_test
from torchvision.utils import save_image
from PIL import Image
import os
from model import Generator

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints/G_AB_epoch150.pth"  
# input_dir = "dataset/testA" #資料集
output_dir = "output/testB"
os.makedirs(output_dir, exist_ok=True)

# === Load Model ===
G_AB = Generator(3, 3, 6).to(device)
G_AB.load_state_dict(torch.load(checkpoint_path, map_location=device))
G_AB.eval()


# === test all images ===
# for img_name in os.listdir(input_dir):
#     if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
#         continue
#     img_path = r"cycleGAN_dataset/processed_unpair_images/0-130-_jpg.jpg"#os.path.join(input_dir, img_name)
#     img = Image.open(img_path).convert("RGB")
#     input_tensor = transform_test(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         fake_img = G_AB(input_tensor)

#     # 從 [-1, 1] 轉回 [0, 1]
#     fake_img = (fake_img + 1) / 2.0
#     save_image(fake_img, os.path.join(output_dir, img_name))

# print("✅ Finished")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img_path = input("請輸入圖片路徑: ")
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform_test(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_img = G_AB(input_tensor)

    # 從 [-1, 1] 轉回 [0, 1]
    fake_img = (fake_img + 1) / 2.0
    save_image(fake_img, os.path.join(output_dir, "test.jpg"))

    print("✅ Finished")
    
    # 顯示圖片
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(fake_img.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title("Generated Image")
    plt.axis("off")
    plt.show()
