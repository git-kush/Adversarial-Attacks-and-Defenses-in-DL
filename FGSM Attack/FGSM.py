import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import torch.optim as optim

## data preprocessing and model loading

zebra_img = Image.open("zebra.jpeg")
preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])   #224 x 224, default size for imageNet images and scaling RGB values [0,1]
zebra_tensor = preprocess(zebra_img)[None, :, :, :]      # bs x C X H x W
plt.imshow(zebra_tensor[0].numpy().transpose(1, 2, 0))
#plt.show()
norm = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

out = model(norm(zebra_tensor))
#print(out.max(dim=1)[1].item())

with open("imagenet_class_index.json") as f:
    imagenet_classes = {int(i): x[1] for i, x in json.load(f).items()}
    #print(imagenet_classes)

max_class = out.max(dim=1)[1].item()
print("initial prediction :", imagenet_classes[max_class])
print("prediction probability :", torch.nn.Softmax(dim=1)(out)[0, max_class].item())


## preparing adv. example

epsilon= 2./255
delta = torch.zeros_like(zebra_tensor, requires_grad=True)
opt = optim.SGD([delta], lr= 1e-1)

for t in range(30):
    p = zebra_tensor + delta
    out = model(norm(p))
    loss = -torch.nn.CrossEntropyLoss()(out, torch.LongTensor([340]))

    # if t%5 == 0:
    #     print(t, loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()
    delta.data.clamp_(-epsilon,epsilon)
    p.data.clamp_(0,1)

max_class = out.max(dim=1)[1].item()
print("new prediction :", imagenet_classes[max_class])
print("Prediction probability:", torch.nn.Softmax(dim=1)(out)[0, max_class].item())

plt.imshow((p)[0].detach().numpy().transpose(1,2,0))
plt.show()

out_file_path = "adv_example.jpg"
adv_image = (p)[0].detach().numpy().transpose(1,2,0)
plt.imsave(out_file_path, adv_image)

