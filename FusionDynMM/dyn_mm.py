from argparse import Namespace
import warnings
import torch
from torch import nn
from src.models.model_skip_mod import SkipESANet
from src.models.model_skip_mod_globalgate import SkipGateESANet
from src.models.resnet import ResNet
from src.classif_head import ClassifierHead

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(args, n_classes):
    if (
        not args.pretrained_on_imagenet
        or args.last_ckpt
        or args.pretrained_scenenet != ""
    ):
        pretrained_on_imagenet = False
    else:
        pretrained_on_imagenet = True

    # set the number of channels in the encoder and for the
    # fused encoder features
    if "decreasing" in args.decoder_channels_mode:
        if args.decoder_channels_mode == "decreasing":
            channels_decoder = [512, 256, 128]

        warnings.warn(
            "Argument --channels_decoder is ignored when "
            "--decoder_chanels_mode decreasing is set."
        )
    else:
        channels_decoder = [args.channels_decoder] * 3

    if isinstance(args.nr_decoder_blocks, int):
        nr_decoder_blocks = [args.nr_decoder_blocks] * 3
    elif len(args.nr_decoder_blocks) == 1:
        nr_decoder_blocks = args.nr_decoder_blocks * 3
    else:
        nr_decoder_blocks = args.nr_decoder_blocks
        assert len(nr_decoder_blocks) == 3

    block_rule = []
    for s in args.block_rule:
        block_rule.append(int(s))
    assert len(block_rule) == 4
    if args.encoder_depth in [None, "None"]:
        args.encoder_depth = args.encoder

    if args.global_gate:
        model = SkipGateESANet(
            height=args.height,
            width=args.width,
            num_classes=n_classes,
            pretrained_on_imagenet=pretrained_on_imagenet,
            pretrained_dir=args.pretrained_dir,
            encoder_rgb=args.encoder,
            encoder_depth=args.encoder_depth,
            encoder_block=args.encoder_block,
            activation=args.activation,
            encoder_decoder_fusion=args.encoder_decoder_fusion,
            context_module=args.context_module,
            nr_decoder_blocks=nr_decoder_blocks,
            channels_decoder=channels_decoder,
            fuse_depth_in_rgb_encoder=args.fuse_depth_in_rgb_encoder,
            upsampling=args.upsampling,
            temp=args.temp,
            block_rule=block_rule,
            nb_channel_nir=args.nb_channel_nir,
            nb_channel_rgb=args.nb_channel_rgb,
        )

    else:
        model = SkipESANet(
            height=args.height,
            width=args.width,
            num_classes=n_classes,
            pretrained_on_imagenet=pretrained_on_imagenet,
            pretrained_dir=args.pretrained_dir,
            encoder_rgb=args.encoder,
            encoder_depth=args.encoder_depth,
            encoder_block=args.encoder_block,
            activation=args.activation,
            encoder_decoder_fusion=args.encoder_decoder_fusion,
            context_module=args.context_module,
            nr_decoder_blocks=nr_decoder_blocks,
            channels_decoder=channels_decoder,
            fuse_depth_in_rgb_encoder=args.fuse_depth_in_rgb_encoder,
            upsampling=args.upsampling,
            temp=args.temp,
            block_rule=block_rule,
            nb_channel_nir=args.nb_channel_nir,
            nb_channel_rgb=args.nb_channel_rgb,
        )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)

    if args.he_init:
        module_list = []

        # first filter out the already pretrained encoder(s)
        for c in model.children():
            if pretrained_on_imagenet and isinstance(c, ResNet):
                continue
            for m in c.modules():
                module_list.append(m)

        # iterate over all the other modules
        # output layers, layers followed by sigmoid (in SE block) and
        # depthwise convolutions (currently only used in learned upsampling)
        # are not initialized with He method
        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if (
                    m.out_channels == n_classes
                    or isinstance(module_list[i + 1], nn.Sigmoid)
                    or m.groups == m.in_channels
                ):
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print("Applied He init.")

    if args.pretrained_scenenet != "":
        checkpoint = torch.load(args.pretrained_scenenet)

        weights_scenenet = checkpoint["state_dict"]

        # (side) outputs and learned upsampling
        keys_to_ignore = [
            k
            for k in weights_scenenet
            if "out" in k or "decoder.upsample1" in k or "decoder.upsample2" in k
        ]
        if args.context_module not in ["ppm", "appm"]:
            keys_to_ignore.extend(
                [k for k in weights_scenenet if "context_module.features" in k]
            )

        for key in keys_to_ignore:
            weights_scenenet.pop(key)

        weights_model = model.state_dict()

        # just for verification that weight loading/ updating works
        # import copy
        # weights_before = copy.deepcopy(weights_model)

        weights_model.update(weights_scenenet)
        model.load_state_dict(weights_model)

        print(f"Loaded pretrained SceneNet weights: {args.pretrained_scenenet}")

    if args.finetune is not None:
        checkpoint = torch.load(args.finetune)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Loaded weights for finetuning: {args.finetune}")

        # print('Freeze the encoder(s).')
        # for name, param in model.named_parameters():
        #     if 'encoder_rgb' in name or 'encoder_depth' in name or 'se_layer' in name:
        #         param.requires_grad = False
    return model, device


def get_dyn_mm(**kwargs):
    data_size = kwargs.get("data_size", (224, 224))
    args = Namespace(
        results_dir="./results",
        last_ckpt="",
        pretrained_dir="./trained_models/imagenet",
        pretrained_scenenet="",
        pretrained_on_imagenet=False,
        finetune=None,
        batch_size=8,
        batch_size_valid=None,
        height=data_size[0],
        width=data_size[1],
        class_weighting="median_frequency",
        c_for_logarithmic_weighting=1.02,
        he_init=False,
        valid_full_res=False,
        dynamic=True,
        global_gate=True,
        block_rule="1111",
        temp=1.0,
        end_temp=0.001,
        loss_ratio=0.0001,
        flop_budget=0.0,
        epoch_ini=0,
        epoch_hard=500,
        eval_every=1,
        save_every=100,
        baseline=False,
        freeze=False,
        soft_eval=False,
        activation="relu",
        encoder="resnet50",
        encoder_block="BasicBlock",
        nr_decoder_blocks=[3],
        encoder_depth=None,
        modality="rgbd",
        encoder_decoder_fusion=kwargs.get("encoder_decoder_fusion", "add"),
        context_module="ppm",
        channels_decoder=128,
        decoder_channels_mode="decreasing",
        fuse_depth_in_rgb_encoder=kwargs.get("fuse_depth_in_rgb_encoder", "SE-add"),
        # upsampling='learned-3x3-zeropad',
        upsampling="nearest",
        # dataset='nyuv2',
        # dataset_dir='./datasets/nyuv2',
        raw_depth=False,
        aug_scale_min=1.0,
        aug_scale_max=1.4,
        workers=32,
        debug=False,
        nb_channel_rgb=kwargs.get("nb_channel_rgb", 3),
        nb_channel_nir=kwargs.get("nb_channel_nir", 1),
    )

    return build_model(args, n_classes=kwargs.get("num_classes", 6))


class dynnModel(nn.Module):
    def __init__(self, model, num_classes=6):
        super(dynnModel, self).__init__()
        self.model = model
        self.classifier = ClassifierHead(
            embed_dim=6,
            num_classes=num_classes,
            dropout_rate=0.1,
            norm_layer=nn.BatchNorm2d,
            width=7,
        )
        self.classifier.to(device)

    def forward(self, rgb, nir):
        x, _ = self.model(rgb, nir)
        x = x[-1]
        x = self.classifier(x)
        return x


def get_dyn_mm_model(nb_channel_rgb, nb_channel_nir, num_classes, **kwargs):
    data_size = kwargs.pop("data_size", (224, 224))
    encoder_decoder_fusion = kwargs.pop("encoder_decoder_fusion", "add")
    fuse_depth_in_rgb_encoder = kwargs.pop("fuse_depth_in_rgb_encoder", "SE-add")

    model, _ = get_dyn_mm(
        num_classes=num_classes,
        data_size=data_size,
        encoder_decoder_fusion=encoder_decoder_fusion,
        nb_channel_rgb=nb_channel_rgb,
        nb_channel_nir=nb_channel_nir,
        fuse_depth_in_rgb_encoder=fuse_depth_in_rgb_encoder,
        **kwargs,
    )

    dynMm = dynnModel(model, num_classes=num_classes)
    return dynMm


if __name__ == "__main__":
    nb_channel_rgb = 3
    nb_channel_nir = 1
    fusion_method = "second_degree"
    model = get_dyn_mm_model(
        num_classes=6,
        data_size=(224, 224),
        encoder_decoder_fusion="add",
        fuse_depth_in_rgb_encoder=fusion_method,
        nb_channel_rgb=nb_channel_rgb,
        nb_channel_nir=nb_channel_nir,
    )

    dummy_input_rgb = torch.randn(2, nb_channel_rgb, 224, 224).to(device)
    dummy_input_nir = torch.randn(2, nb_channel_nir, 224, 224).to(device)

    out = model(dummy_input_rgb, dummy_input_nir)
    print(out.shape)

    # from torch.utils.tensorboard import SummaryWriter

    # writer = SummaryWriter("torchlogs/")
    # writer.add_graph(model, (dummy_input_rgb, dummy_input_nir))
    # writer.close()

    # from torchviz import make_dot
    # make_dot(out, params=dict(model.named_parameters())).render("model_default", format="png")
