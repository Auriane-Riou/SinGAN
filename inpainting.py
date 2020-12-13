from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
import torch
import cv2


def main():
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Inpainting')
    parser.add_argument('--inpainting_start_scale', help='inpainting injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='inpainting')
    parser.add_argument('--radius', help='radius harmonization', type=int, default=10)
    parser.add_argument('--ref_name', help='training image name', type=str, default="")
    parser.add_argument('--initialization', help='initialization technique', type=str, default="mean")
    parser.add_argument('--x1_mask', type=float, help='lower x bound for occlusion in inpainting', default=0.25)
    parser.add_argument('--x2_mask', type=float, help='upper x bound for occlusion in inpainting', default=0.5)
    parser.add_argument('--y1_mask', type=float, help='lower y bound for occlusion in inpainting', default=0.3)
    parser.add_argument('--y2_mask', type=float, help='upper y bound for occlusion in inpainting', default=0.5)

    opt = parser.parse_args(raw_args)
    opt = functions.post_config(opt)
    if opt.ref_name == "":
        opt.ref_name = opt.input_name

    # ajout vérifier si initialisation dans celles possibles
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if dir2save is None:
        print('task does not exist')
    # elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        real = functions.read_image(opt)
        real_occluded = real.clone()

        real = functions.adjust_scales2image(real, opt)
        real_np = functions.convert_image_np(real)

        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)

        if (opt.inpainting_start_scale < 1) | (opt.inpainting_start_scale > (len(Gs) - 1)):

            print("injection scale should be between 1 and %d" % (len(Gs) - 1))
        else:

            mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir, opt.input_name[:-4], opt.input_name[-4:]),
                                            opt)

            # saving original image in output folder for comparison
            plt.imsave('%s/start_scale=%d_original.png' % (dir2save, opt.inpainting_start_scale),
                       functions.convert_image_np(real), vmin=0, vmax=1)

            # saving occluded image in output folder for comparison
            plt.imsave('%s/start_scale=%d_occluded.png' % (dir2save, opt.inpainting_start_scale),
                       functions.convert_image_np((1 - functions.denorm(mask)) * real), vmin=0, vmax=1)

            # random initialization
            if opt.initialization == "random":
                unique_pixel_values = torch.unique(real)
                real_occluded[mask == 1] = torch.tensor(
                    [random.choice(unique_pixel_values) for _ in range(len(real[mask == 1]))])

            # mean initialization
            elif opt.initialization == "mean":
                # mean taken for each channel
                for j in range(3):
                    real_occluded[0, j, :, :][mask[0, j, :, :] == 1] = real_occluded[0, j, :, :][
                        mask[0, j, :, :] == 1].mean()

            real_occluded_np = functions.convert_image_np(real_occluded)
            plt.imsave(
                '%s/%s_averaged_init=%s%s' % (opt.input_dir, opt.ref_name[:-4], opt.initialization, opt.ref_name[-4:]),
                real_occluded_np, vmin=0, vmax=1)
            ref = functions.read_image_dir(
                '%s/%s_averaged_init=%s%s' % (opt.input_dir, opt.ref_name[:-4], opt.initialization, opt.ref_name[-4:]),
                opt)

            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.inpainting_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]

            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            # saving generator output
            plt.imsave(
                '%s/start_scale=%d_init=%s_generated.png' % (dir2save, opt.inpainting_start_scale, opt.initialization),
                functions.convert_image_np(out.detach()), vmin=0, vmax=1)

            # saving final image reconstitution (replacing occluded areas by the generated image)
            out = (1 - mask) * real + mask * out
            plt.imsave(
                '%s/start_scale=%d_init=%s_inpainted.png' % (dir2save, opt.inpainting_start_scale, opt.initialization),
                functions.convert_image_np(out.detach()), vmin=0, vmax=1)


if __name__ == '__main__':
    main()

