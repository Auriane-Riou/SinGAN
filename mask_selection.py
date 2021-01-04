import SinGAN.functions as functions
from config import get_arguments
from SinGAN.functions import computes_mask_inpainting
import cv2


def main(raw_args=None):

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--ref_dir', help='inpainting reference dir', default='Input/Inpainting/Masks')

    opt = parser.parse_args(raw_args)
    print(opt)
    opt = functions.post_config(opt)
    print(opt)



    source_img = cv2.imread('%s/%s' % (opt.input_dir, opt.input_name))

    # FOR LOCAL EXECUTION (only for images):
    # lets user select areas to occlude
    functions.get_occluded_area(source_img, opt)

    # saves mask image corresponding to those occluded areas
    computes_mask_inpainting(opt)

if __name__ == '__main__':
    main()
