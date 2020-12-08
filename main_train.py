from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--inpainting', action='store_true', help='store true to train for inpainting')
    parser.add_argument('--x1_mask', type=float, help='lower x bound for occlusion in inpainting', default=0.25)
    parser.add_argument('--x2_mask', type=float, help='upper x bound for occlusion in inpainting', default=0.5)
    parser.add_argument('--y1_mask', type=float, help='lower y bound for occlusion in inpainting', default=0.3)
    parser.add_argument('--y2_mask', type=float, help='upper y bound for occlusion in inpainting', default=0.5)
    parser.add_argument('--ref_dir', help='inpainting reference dir', default='Input/Inpainting')

    opt = parser.parse_args()
    print(opt)
    opt = functions.post_config(opt)
    print(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')

    elif opt.x1_mask > opt.x2_mask or opt.y1_mask > opt.y2_mask:
        print("Wrong dimensions for occlusion mask for inpainting")

    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
