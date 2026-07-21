from detr3d.models.checkpoint import translate_fcos3d_key


def test_translate_fcos3d_backbone_and_modulated_dcn_keys():
    targets = {
        "backbone.stem.0.weight",
        "backbone.stage4.0.conv2.deform_conv.weight",
        "backbone.stage4.0.conv2.conv_offset.weight",
    }

    assert translate_fcos3d_key("img_backbone.conv1.weight", targets) == "backbone.stem.0.weight"
    assert (
        translate_fcos3d_key("img_backbone.layer3.0.conv2.weight", targets)
        == "backbone.stage4.0.conv2.deform_conv.weight"
    )
    assert (
        translate_fcos3d_key("img_backbone.layer3.0.conv2.conv_offset.weight", targets)
        == "backbone.stage4.0.conv2.conv_offset.weight"
    )


def test_translate_fcos3d_fpn_keys_and_drop_extra_level():
    targets = {
        "neck.lateral_convs.0.weight",
        "neck.output_convs.2.bias",
        "neck.extra_conv.weight",
    }

    assert (
        translate_fcos3d_key("img_neck.lateral_convs.0.conv.weight", targets)
        == "neck.lateral_convs.0.weight"
    )
    assert (
        translate_fcos3d_key("img_neck.fpn_convs.2.conv.bias", targets)
        == "neck.output_convs.2.bias"
    )
    assert (
        translate_fcos3d_key("img_neck.fpn_convs.3.conv.weight", targets)
        == "neck.extra_conv.weight"
    )
    assert translate_fcos3d_key("img_neck.fpn_convs.4.conv.weight", targets) is None
