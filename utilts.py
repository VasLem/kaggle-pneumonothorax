from torch import nn
def pad_div32(tensor):
    """
    Pads last 2 dimensions of a tensor so they can be divisible with 32
    """
    h, w = tensor.size()[-2:]
    if h < 32:
        y_p = 32 - h
    else:
        y_p = 32 - (h % 32)
    if w < 32:
        x_p = 32 - w
    else:
        x_p = 32 - (w % 32)
    pads = (x_p // 2, x_p - x_p // 2, y_p // 2, y_p - y_p // 2)
    return nn.ZeroPad2d(pads)(tensor), pads