import argparse
import torch
import numpy as np

import sys

sys.path.append('../SCDM')

from torch.utils.data import DataLoader
from ddpm import script_utils
from datasets.hyper import HyperDataset
from osgeo import gdal


def save_img(result_file,img,im_width,im_height,im_bands,geoTranfsorm):
    driver = gdal.GetDriverByName("GTIFF")
    dataset = driver.Create(result_file, im_width, im_height, im_bands, gdal.GDT_UInt16)
    if geoTranfsorm:
        dataset.SetGeoTransform(geoTranfsorm)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(img[i])

def main():
    args = create_argparser().parse_args()
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load(args.model_path))

        test_dataset = HyperDataset(
           root_path=args.test_dir,
           data_file=args.test_txt,
           input_channel=args.input_channels,
           output_channel=args.output_channels,
           cropsize=args.input_size,
           gt_exist=True
        )

        batch_size = args.batch_size
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=0)

        batch_cnt = 0
        for x, rgb, all in test_loader:
            x_list = []
            all_list = []
            batch_cnt = batch_cnt + 1
            print(batch_cnt)
            diffusion.eval()
            x = x.to(device)
            all = all.to(device)
            x_list.append(x)
            all_list.append(all)
            rgb = rgb.to(device)
            samples = diffusion.sample(batch_size, device, rgb)

            for image_id in range(len(samples)):

                img = (samples[image_id] * 7410 + 3693).data.cpu()
                img = np.asarray(img)

                x_img = (x_list[0] * 7410 + 3693).data.cpu()
                x_img = np.asarray(x_img)
                x_img = x_img[image_id]

                save_img(f"{args.save_dir}/{image_id+(batch_cnt-1)*batch_size}.tif", img, args.input_size, args.input_size, args.output_channels, None)
                save_img(f"{args.save_dir}/x_{image_id+(batch_cnt-1)*batch_size}.tif", x_img, args.input_size, args.input_size, args.output_channels, None)

    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=1, batch_size=1, device=device, schedule_low=0, schedule_high=1)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./", type=str)
    parser.add_argument("--save_dir", default='./', type=str)
    parser.add_argument("--test_dir", default='./', type=str)
    parser.add_argument("--test_txt", default='./', type=str)

    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()