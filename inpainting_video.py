from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import cv2
import torchvision as tv
import main_train
import inpainting

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input video dir', default='Input/Videos')
    parser.add_argument('--input_name', help='training video name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Inpainting/Videos')
    parser.add_argument('--inpainting_start_scale', help='inpainting injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='inpainting_video')
    parser.add_argument('--radius', help='radius harmonization', type=int, default=10)
    parser.add_argument('--ref_name', help='training video name', type=str, default="")
    parser.add_argument('--initialization', help='initialization technique', type=str, default="mean")

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    if opt.ref_name == "":
        opt.ref_name = opt.input_name

    opt.input_dir += "/%s" % (opt.input_name[:-4])

    if not os.path.exists('%s/%s/' % (opt.ref_dir, opt.input_name[:-4])):
        os.makedirs('%s/%s/' % (opt.ref_dir, opt.input_name[:-4]))

    COCO_CLASS_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # pretrained model for mask RCNN
    model_mask = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model_mask.eval()


    def automated_person_detection(source_video):
        """
        Saves for each frame of the video a binary mask
        The mask corresponds to the occlusion of detected person (using a pretrained mask RCNN model)
        Args:
            source_video:

        Returns:

        """
        cap = cv2.VideoCapture(source_video)

        # check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")

        frame_count = 0

        # read until video is completed

        while cap.isOpened():  # and frame_count < 10:
            # capture frame-by-frame
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # we check if mask for this frame has already been computed
                if not os.path.exists('%s/%s/%s_%d_mask%s' % (opt.ref_dir, opt.input_name[:-4], opt.input_name[:-4],
                                                              frame_count, ".png")):

                    occlusions = []
                    transform = transforms.Compose([transforms.ToTensor()])

                    img = transform(frame)
                    pred = model_mask([img])
                    pred_score = list(pred[0]['scores'].detach().numpy())

                    pred_t = [pred_score.index(x) for x in pred_score if x > 0.7][-1]
                    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()

                    pred_class = [COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].numpy())]
                    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]

                    masks = masks[:pred_t + 1]
                    pred_boxes = pred_boxes[:pred_t + 1]
                    pred_class = pred_class[:pred_t + 1]

                    # we check if there are persons amongst detected objects
                    for i in range(len(pred_boxes)):

                        if pred_class[i] == "person":
                            x1, y1, x2, y2 = pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i][1][0], \
                                             pred_boxes[i][1][1]

                            occlusions.append([x1, y1, x2, y2])

                    mask = np.zeros((frame.shape[0], frame.shape[1], 3))

                    # we fill white squares corresponding to the different occluded areas
                    for occlusion in occlusions:
                        x1, x2 = occlusion[1], occlusion[3]
                        y1, y2 = occlusion[0], occlusion[2]

                        for i in range(int(x1), int(x2)):
                            for j in range(int(y1), int(y2)):
                                mask[i, j] = [255, 255, 255]

                    mask = mask.astype('uint8')

                    plt.imsave('%s/%s/%s_%d_mask%s' % (opt.ref_dir, opt.input_name[:-4], opt.input_name[:-4],
                                                       frame_count, ".png"), mask, vmin=0, vmax=1)
                    plt.imsave('%s/%s_%d%s' % (opt.input_dir, opt.input_name[:-4], frame_count, ".png"), frame, vmin=0,
                               vmax=1)

                    frame_count += 1

                    # press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

                # if mask for current frame has already been computed
                else:
                    frame_count += 1

            else:
                break

        # when everything done, release the video capture object
        cap.release()

        # closes all the frames
        cv2.destroyAllWindows()


    print("Start of person mask computing for video")
    automated_person_detection("%s/%s" % (opt.input_dir, opt.input_name))
    print("End of person mask computing for video")

    # number of masks corresponds to number of frames
    # (more generally this would be number of frames in initial video)
    nb_masked_frames = len(os.listdir('%s/%s/' % (opt.ref_dir, opt.input_name[:-4]))) // 2

    for i in range(nb_masked_frames):

        # training is performed on the first frame only
        if i == 0:
            print("Start of training")

            main_train_args = ["--input_dir", opt.input_dir,
                               "--input_name", '%s_%d%s' % (opt.input_name[:-4], 0, ".png"),
                               "--inpainting",
                               "--ref_dir", "Input/Inpainting/Videos/%s" % (opt.input_name[:-4])
                               ]
            if opt.not_cuda:
                main_train_args.append("--not_cuda")

            main_train.main(main_train_args)

            print("End of training")

        print("Inpainting of image %d" % (i))
        inpainting_args = ["--input_dir", opt.input_dir,
                           "--input_name", '%s_%d%s' % (opt.input_name[:-4], i, ".png"),
                           "--ref_dir","Input/Inpainting/Videos/%s" % (opt.input_name[:-4]),
                           "--inpainting_start_scale", str(opt.inpainting_start_scale),
                           "--initialization", opt.initialization,
                           "--mode", opt.mode
                           ]

        if opt.not_cuda:
            inpainting_args.append("--not_cuda")

        inpainting.main_inpainting(inpainting_args)

    img_array = []
    for i in range(nb_masked_frames):
        dir2save = '%s/Inpainting/%s/%s_%d' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4], i)
        path_to_image = '%s/start_scale=%d_init=%s_inpainted.png' % (dir2save, opt.inpainting_start_scale, opt.initialization)

        img = cv2.imread(path_to_image)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('%s/Inpainting/%s_inpainted_%s_initialization.avi' % (opt.out, opt.input_name[:-4], opt.initialization), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
