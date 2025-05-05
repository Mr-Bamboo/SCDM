import argparse
import datetime
import torch

import sys

sys.path.append('../SCDM')

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ddpm import script_utils
from datasets.hyper import HyperDataset


def main():
    args = create_argparser().parse_args()
    device = args.device
    log_txt = open(args.log_txt, 'w')

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        batch_size = args.batch_size

        ## you need to fill the root_path and data_file
        train_dataset = HyperDataset(
           root_path='./',
           data_file='./',
           input_channel=args.input_channels,
           output_channel=args.output_channels,
           cropsize=args.input_size
        )

        test_dataset = HyperDataset(
           root_path='./',
           data_file='./',
           input_channel=args.input_channels,
           output_channel=args.output_channels,
           cropsize=args.input_size
        )

        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        ))
        test_loader = DataLoader(test_dataset, batch_size=4, drop_last=True, num_workers=0)

        acc_train_loss = 0
        train_loss_arr = []
        cnt = 0
        writer = SummaryWriter()

        for iteration in range(1, args.iterations + 1):
            diffusion.train()

            x, rgb, _ = next(train_loader)

            x = x.to(device)
            rgb = rgb.to(device)

            if args.use_labels:
                loss = diffusion(x)
            else:
                loss, sam_loss = diffusion(x, rgb)
                total_loss = loss + 1.0 * sam_loss

            acc_train_loss += total_loss.item()
            writer.add_scalar('l1', loss.item(), iteration)
            writer.add_scalar('sam', sam_loss.item(), iteration)
            writer.add_scalar('total', total_loss.item(), iteration)
            print("iter" + str(iteration) + " total_loss:" + str(total_loss.item()) + " l1_loss:" + str(loss.item()) + " sam_loss:" + str(sam_loss.item()) )
            train_loss_arr.append(float(total_loss.item()))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            diffusion.update_ema()

            if iteration % args.log_rate == 0:
                test_loss = 0
                print("start valid")
                with torch.no_grad():
                    diffusion.eval()
                    for x, rgb, _ in test_loader:
                        x = x.to(device)
                        rgb = rgb.to(device)

                        if args.use_labels:
                            loss = diffusion(x)
                        else:
                            loss, sam_loss = diffusion(x, rgb)
                            total_loss = loss + 1.0*sam_loss

                        test_loss += total_loss.item()

                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                writer.add_scalar('train', acc_train_loss, iteration)
                writer.add_scalar('val', test_loss, iteration)
                print("test_loss:" + str(test_loss) + "," + "train_loss:" + str(acc_train_loss))
                log_txt.write("test_loss:" + str(test_loss) + "," + "train_loss:" + str(acc_train_loss) + '\n')
                acc_train_loss = 0

            save_log = args.log_dir
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{save_log}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{save_log}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
        log_txt.close()

    except KeyboardInterrupt:
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=1e-4,
        batch_size=4,
        iterations=100000,

        log_to_wandb=False,
        log_rate=100,
        checkpoint_rate=5000,
        log_dir="./",
        log_txt='./',
        project_name="scdm",
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()