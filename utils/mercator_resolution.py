# m
mercator_xrange = [-20037508.3427892, 20037508.3427892]
mercator_yrange = [-20037508.3427892, 20037508.3427892]
# 中间本初子午线

earth = 20037508.3427892 * 2  # 40075016.6855784，恰好为 赤道周长


def level_pixel_num(z):
    return 256 * 2 ** z


def level_resolution(z):
    return earth / level_pixel_num(z)


def demo_levels():
    for z in range(22):
        print(f'z={z}, {level_pixel_num(z)}, {level_resolution(z)}')


demo_levels()
