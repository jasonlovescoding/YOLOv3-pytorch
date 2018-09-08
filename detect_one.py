import torch
import cv2
from darknet import darknet53
from utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser('YOLOv3')

    parser.add_argument('--cuda', action="store_true", default=False)
    parser.add_argument('--modelpath', type=str, default='model/darknet53.pth.tar')
    parser.add_argument('--classfile', type=str, default='model/coco.names')
    parser.add_argument('--image', type=str, default='images/dog-cycle-car.png')
    parser.add_argument('--resolution', type=int, default=416)

    return parser.parse_args()

def main():
    args = parse_args()

    model = darknet53(args.modelpath).eval()

    img_ori = cv2.imread(args.image)
    inp_dim = args.resolution, args.resolution
    img = img_prepare(img_ori, inp_dim)

    if args.cuda:
        model = model.cuda()
        img = img.cuda()
    with torch.no_grad():
        out0, out1, out2 = model(img)
    output = {'out0': out0.cpu().numpy(),
              'out1': out1.cpu().numpy(),
              'out2': out2.cpu().numpy()}
    rects = rects_prepare(output)
    mapping = get_classname_mapping(args.classfile)

    scaling_factor = min(1, args.resolution / img_ori.shape[1])
    for pt1, pt2, cls, prob in rects:
        pt1[0] -= (args.resolution - scaling_factor*img_ori.shape[1])/2
        pt2[0] -= (args.resolution - scaling_factor*img_ori.shape[1])/2
        pt1[1] -= (args.resolution - scaling_factor*img_ori.shape[0])/2
        pt2[1] -= (args.resolution - scaling_factor*img_ori.shape[0])/2

        pt1[0] = np.clip(int(pt1[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
        pt2[0] = np.clip(int(pt2[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
        pt1[1] = np.clip(int(pt1[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
        pt2[1] = np.clip(int(pt2[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])

        label = "{}:{:.2f}".format(mapping[cls], prob)
        color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))

        cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
        cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, -1)
        cv2.putText(img_ori, label, (pt1[0], t_size[1] + 4 + pt1[1]), cv2.FONT_HERSHEY_PLAIN,
                    cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)
    cv2.imshow(args.image, img_ori)
    cv2.waitKey()


if __name__ == '__main__':
    main()
