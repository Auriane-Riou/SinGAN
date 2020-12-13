from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


def main(raw_args=None):

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--inpainting', action='store_true', help='store true to train for inpainting')
    parser.add_argument('--ref_dir', help='inpainting reference dir', default='Input/Inpainting')

    opt = parser.parse_args(raw_args)
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

    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)



if __name__ == '__main__':
    main()
