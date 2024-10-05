import depth_pro
import time
import torch
import torchvision.transforms.functional as TF


def show(depth):
    depth = 1 / depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    TF.to_pil_image(depth).show()


def test(img_size, N):
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

    torch.cuda.synchronize()
    t = time.time()
    for _ in range(N):
        prediction = model.infer(image)
    torch.cuda.synchronize()
    print(round(1.0 / ((time.time() - t) / N), 3), "FPS")


if __name__ == "__main__":
    N = 10
    # on RTX3070ti
    # 1.857 FPS
    test(img_size=384, N=N)
    # 4.291 FPS
    test(img_size=256, N=N)
    # 15.63 FPS
    test(img_size=128, N=N)
