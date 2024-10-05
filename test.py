import depth_pro
import time
import torch
import torchvision.transforms.functional as TF


def show(depth):
    depth = 1 / depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    TF.to_pil_image(depth).show()


def test(img_size):
    N = 10
    model, transform = depth_pro.create_model_and_transforms(
        device=torch.device("cuda"),
        precision=torch.float16,
        img_size=img_size
    )
    model.eval()
    print("resolution", model.encoder.img_size)

    image, _, f_px = depth_pro.load_rgb("data/example.jpg")
    image = transform(image)

    prediction = model.infer(image)
    show(prediction["depth"])

    t = time.time()
    for _ in range(N):
        prediction = model.infer(image)
    print(round(1.0 / ((time.time() - t) / N), 3), "FPS")


if __name__ == "__main__":
    # on RTX3070ti
    # 1.911 FPS
    test(img_size=384)
    # 4.901 FPS
    test(img_size=256)
    # 18.01 FPS
    test(img_size=128)
