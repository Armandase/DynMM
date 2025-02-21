import torch
from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def addition(x1, x2):
    return torch.add(x1, x2)


def element_wise_multiplication(x1, x2):
    return torch.mul(x1, x2)


def concatenation(x1, x2):
    return torch.cat((x1, x2), dim=1)


# bilinear fusion
def dot_product(x1, x2):
    z = torch.matmul(x1, x2)
    return z


# bilinear fusion with a bias term
def augmented_dot_product(x1, x2):
    x1 = torch.cat(
        (x1, torch.ones(x1.shape[0], 1, x1.shape[-2], x1.shape[-1]).to(device)), dim=1
    )
    x2 = torch.cat(
        (x2, torch.ones(x2.shape[0], 1, x2.shape[-2], x2.shape[-1]).to(device)), dim=1
    )
    z = torch.matmul(x1, x2)
    return z


def polynomial_fusion(x1, x2):
    return x1 + x2 + x1 * x2


def se_add(channel):
    return SqueezeAndExciteFusionAdd(channel).to(device)


methods_list = {
    "add": addition,
    "mul": element_wise_multiplication,
    "concat": concatenation,
    "dot_product": dot_product,
    "augmented_dot_product": augmented_dot_product,
    "second_degree": polynomial_fusion,
    "SE-add": se_add,
}


def select_fusion_method(name, channel=None):
    if name not in methods_list:
        raise ValueError(
            f"Invalid fusion method name, options are: {list(methods_list.keys())}"
        )
    if name == "SE-add":
        if not channel:
            raise ValueError("Channel size is required for SE-add")
        return methods_list[name](channel)
    return methods_list[name]


if __name__ == "__main__":
    x1 = torch.randn(1, 64, 224, 224).to(device)
    x2 = torch.randn(1, 64, 224, 224).to(device)

    se_add = select_fusion_method("SE-add", 64)
    a = se_add(x1, x2)
    print(a.shape)

    add = select_fusion_method("add")
    b = add(x1, x2)
    print(b.shape)

    agm_dot = select_fusion_method("augmented_dot_product")
    c = agm_dot(x1, x2)
    print(c.shape)

    concat = select_fusion_method("concat")
    d = concat(x1, x2)
    print(d.shape)
