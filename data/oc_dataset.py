import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import torch
import torchvision.transforms as transforms

def load_as_float(path, sequence_length):
    #rgb
    stack = np.array(Image.open(path)).astype(np.float32)
    h, w, _ = stack.shape
    w_img = int(w/(sequence_length))
    imgs = [stack[:, i*w_img:(i+1)*w_img] for i in range(sequence_length)]
    #tgt_index = sequence_length//2
    return imgs


class OC_Dataset(data.Dataset):
    """
    adjacent frames in the same picture
    """
    def __init__(self, root, width=512, height=512, depth_transform=None):
        super(OC_Dataset, self).__init__()
        self.root = root
        self.width = width
        self.height = height
        self.dataset = []

        imgs = os.listdir(root)
        #imgs.sort()
        #2900-3050
        """
        for i in range(2900, 3050):
            self.dataset.append(os.path.join(self.root, str(i).zfill(8) + '.png'))
        """
        for im in imgs:
            self.dataset.append(os.path.join(self.root, im))
        print("oc flow dataset size:" + str(len(self.dataset)))

        self.depth_transform = depth_transform


    def __getitem__(self, index):
        sample_path = self.dataset[index]
        imgs = load_as_float(sample_path, 2)

        #flow image is BGR?
        flow_source_img = torch.FloatTensor(np.ascontiguousarray(imgs[0][:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
        flow_target_img = torch.FloatTensor(np.ascontiguousarray(imgs[1][:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))


        if self.depth_transform is not None:
            depth_source_img = self.depth_transform(imgs[0])
            depth_target_img = self.depth_transform(imgs[1])
        return depth_source_img, depth_target_img, flow_source_img, flow_target_img

    def __len__(self):
        return len(self.dataset)


"""
depth_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

oc_dataset = OC_Dataset(root='/data12T/kcheng/oc_dataset', depth_transform=depth_transform)
dataloader = torch.utils.data.DataLoader(dataset=oc_dataset, batch_size=1, shuffle=True, num_workers=1)
dataiter = iter(dataloader)

num = 0
while True:
  try:
    source_for_depth, target_for_depth, source_for_op, target_for_op = dataiter.next()
  except StopIteration:
    print("yes")
    dataiter = iter(dataloader)
"""
