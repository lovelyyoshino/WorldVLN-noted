# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import cv2


def clip_boxes_to_image(boxes, height, width):
    """
        按图像的 height 和 width 将边界框裁剪到图像范围内。

        参数：
            boxes (ndarray): 需要裁剪的边界框，维度为 `num boxes` x 4。
            height (int): 图像高度。
            width (int): 图像宽度。

        返回：
            boxes (ndarray): 裁剪后的边界框。

    """
    boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return boxes


def random_short_side_scale_jitter_list(images, min_size, max_size, boxes=None):
    """
        对给定图像列表及对应边界框执行短边随机缩放抖动。

        参数：
            images (list): 需要缩放抖动的图像列表，维度为
                说明：`height` x `width` x `channel`。
            min_size (int): 帧短边缩放后的最小尺寸。
            max_size (int): 帧短边缩放后的最大尺寸。
            boxes (list): 可选，图像对应的边界框，维度为 `num boxes` x 4。

        返回：
            (list): 缩放后的图像列表，维度为
                说明：`new height` x `new width` x `channel`。
            (ndarray or None): 缩放后的边界框，维度为 `num boxes` x 4。

    """
    size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))

    height = images[0].shape[0]
    width = images[0].shape[1]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images, boxes
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = [
                proposal * float(new_height) / height for proposal in boxes
            ]
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = [proposal * float(new_width) / width for proposal in boxes]
    return (
        [
            cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            for image in images
        ],
        boxes,
    )


def scale(size, image):
    """
        将图像短边缩放到指定 size，并保持宽高比。

        参数：
            size (int): 图像短边要缩放到的尺寸。
            image (array): 要执行短边缩放的图像，维度为
                说明：`height` x `width` x `channel`。

        返回：
            (ndarray): 缩放后的图像，维度为 `height` x `width` x `channel`。

    """
    height = image.shape[0]
    width = image.shape[1]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return image
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    img = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return img.astype(np.float32)


def scale_boxes(size, boxes, height, width):
    """
        按图像短边缩放比例同步缩放边界框。

        参数：
            size (int): 图像短边要缩放到的尺寸。
            boxes (ndarray): 需要缩放的边界框，维度为 `num boxes` x 4。
            height (int): 原图高度。
            width (int): 原图宽度。

        返回：
            boxes (ndarray): 缩放后的边界框。

    """
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return boxes

    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        boxes *= float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        boxes *= float(new_width) / width
    return boxes


def horizontal_flip_list(prob, images, order="CHW", boxes=None):
    """
        按概率水平翻转图像列表，并同步翻转可选边界框。

        参数：
            prob (float): 执行翻转的概率。
            image (list): 待处理图像列表，维度为
                说明：`height` x `width` x `channel` or `channel` x `height` x `width`.
            order (str): `height`、`channel` 和 `width` 的排列顺序。
            boxes (list): 可选，图像对应的边界框，维度为 `num boxes` x 4。

        返回：
            (ndarray): 翻转后的图像，维度为 `height` x `width` x `channel`。
            (list): 可选，图像对应的边界框，维度为 `num boxes` x 4。

    """
    _, width, _ = images[0].shape
    if np.random.uniform() < prob:
        if boxes is not None:
            boxes = [flip_boxes(proposal, width) for proposal in boxes]
        if order == "CHW":
            out_images = []
            for image in images:
                image = np.asarray(image).swapaxes(2, 0)
                image = image[::-1]
                out_images.append(image.swapaxes(0, 2))
            return out_images, boxes
        elif order == "HWC":
            return [cv2.flip(image, 1) for image in images], boxes
    return images, boxes


def spatial_shift_crop_list(size, images, spatial_shift_pos, boxes=None):
    """
        对给定图像列表执行左/中/右（或上/中/下）位置裁剪。

        参数：
            size (int): 裁剪尺寸。
            image (list): 待裁剪图像列表，维度为
                说明：`height` x `width` x `channel` or `channel` x `height` x `width`.
            spatial_shift_pos (int): 裁剪位置，0 表示左/上，1 表示中间，
                2 表示右/下。
            boxes (list): 可选，图像对应的边界框，维度为 `num boxes` x 4。

        返回：
            cropped (ndarray): 裁剪后的图像列表，维度为
                说明：`height` x `width` x `channel`。
            boxes (list): 可选，图像对应的边界框，维度为 `num boxes` x 4。

    """

    assert spatial_shift_pos in [0, 1, 2]

    height = images[0].shape[0]
    width = images[0].shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_shift_pos == 0:
            y_offset = 0
        elif spatial_shift_pos == 2:
            y_offset = height - size
    else:
        if spatial_shift_pos == 0:
            x_offset = 0
        elif spatial_shift_pos == 2:
            x_offset = width - size

    cropped = [
        image[y_offset : y_offset + size, x_offset : x_offset + size, :]
        for image in images
    ]
    assert cropped[0].shape[0] == size, "图像高度裁剪不正确"
    assert cropped[0].shape[1] == size, "图像宽度裁剪不正确"

    if boxes is not None:
        for i in range(len(boxes)):
            boxes[i][:, [0, 2]] -= x_offset
            boxes[i][:, [1, 3]] -= y_offset
    return cropped, boxes


def CHW2HWC(image):
    """
        将维度从 `channel` x `height` x `width` 转置为
            说明：`height` x `width` x `channel`.

        参数：
            image (array): 需要转置的图像。

        返回：
            (array): 转置后的图像。

    """
    return image.transpose([1, 2, 0])


def HWC2CHW(image):
    """
        将维度从 `height` x `width` x `channel` 转置为
            说明：`channel` x `height` x `width`.

        参数：
            image (array): 需要转置的图像。

        返回：
            (array): 转置后的图像。

    """
    return image.transpose([2, 0, 1])


def color_jitter_list(
    images, img_brightness=0, img_contrast=0, img_saturation=0
):
    """
        对图像列表执行颜色抖动。

        参数：
            images (list): 需要颜色抖动的图像列表。
            img_brightness (float): 亮度抖动比例。
            img_contrast (float): 对比度抖动比例。
            img_saturation (float): 饱和度抖动比例。

        返回：
            images (list): 抖动后的图像列表。

    """
    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_list(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_list(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_list(img_saturation, images)
    return images


def lighting_list(imgs, alphastd, eigval, eigvec, alpha=None):
    """
        对给定图像列表执行 AlexNet 风格的 PCA lighting 抖动。

        参数：
            images (list): 需要执行 lighting 抖动的图像列表。
            alphastd (float): PCA 抖动比例。
            eigval (list): PCA 抖动使用的特征值。
            eigvec (list[list]): PCA 抖动使用的特征向量。

        返回：
            out_images (list): 抖动后的图像列表。

    """
    if alphastd == 0:
        return imgs
    # 生成 alpha1、alpha2、alpha3。
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = []
    for img in imgs:
        for idx in range(img.shape[0]):
            img[idx] = img[idx] + rgb[2 - idx]
        out_images.append(img)
    return out_images


def color_normalization(image, mean, stddev):
    """
        使用给定 mean 和 stddev 对图像做颜色归一化。

        参数：
            image (array): 需要归一化的图像。
            mean (float): 要减去的均值。
            stddev (float): 要除以的标准差。

    """
    # 输入图像应为 CHW 格式。
    assert len(mean) == image.shape[0], "通道均值计算不正确"
    assert len(stddev) == image.shape[0], "通道标准差计算不正确"
    for idx in range(image.shape[0]):
        image[idx] = image[idx] - mean[idx]
        image[idx] = image[idx] / stddev[idx]
    return image


def pad_image(image, pad_size, order="CHW"):
    """
        按 pad_size 给定的宽度填充图像四周。

        参数：
            image (array): 需要填充的图像。
            pad_size (int): 填充宽度。
            order (str): `height`、`channel` 和 `width` 的排列顺序。

        返回：
            img (array): 填充后的图像。

    """
    if order == "CHW":
        img = np.pad(
            image,
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode=str("constant"),
        )
    elif order == "HWC":
        img = np.pad(
            image,
            ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode=str("constant"),
        )
    return img


def horizontal_flip(prob, image, order="CHW"):
    """
        按概率水平翻转单张图像。

        参数：
            prob (float): 执行翻转的概率。
            image (array): 需要翻转的图像。
            order (str): `height`、`channel` 和 `width` 的排列顺序。

        返回：
            img (array): 翻转后的图像。

    """
    assert order in ["CHW", "HWC"], "不支持 order {}".format(order)
    if np.random.uniform() < prob:
        if order == "CHW":
            image = image[:, :, ::-1]
        elif order == "HWC":
            image = image[:, ::-1, :]
        else:
            raise NotImplementedError("未知 order {}".format(order))
    return image


def flip_boxes(boxes, im_width):
    """
        按图像宽度水平翻转边界框。

        参数：
            boxes (array): 需要翻转的边界框。
            im_width (int): 图像宽度。

        返回：
            boxes_flipped (array): 翻转后的边界框。

    """

    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped


def crop_boxes(boxes, x_offset, y_offset):
    """
        根据 x/y 偏移量裁剪边界框坐标。

        参数：
            boxes (array): 需要裁剪的边界框。
            x_offset (int): x 方向偏移。
            y_offset (int): y 方向偏移。

    """
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset
    return boxes


def random_crop_list(images, size, pad_size=0, order="CHW", boxes=None):
    """
        对图像列表执行随机裁剪。

        参数：
            images (list): 需要随机裁剪的图像列表。
            size (int): 裁剪尺寸。
            pad_size (int): 填充尺寸。
            order (str): `height`、`channel` 和 `width` 的排列顺序。
            boxes (list): 可选，图像对应的边界框，维度为 `num boxes` x 4。

        返回：
            cropped (ndarray): 裁剪后的图像列表，维度为
                说明：`height` x `width` x `channel`.
            boxes (list): 可选，图像对应的边界框，维度为 `num boxes` x 4。

    """
    # 按图像维度顺序显式处理，避免意外翻转图像。
    if pad_size > 0:
        images = [
            pad_image(pad_size=pad_size, image=image, order=order)
            for image in images
        ]

    # 图像格式应为 CHW。
    if order == "CHW":
        if images[0].shape[1] == size and images[0].shape[2] == size:
            return images, boxes
        height = images[0].shape[1]
        width = images[0].shape[2]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = [
            image[:, y_offset : y_offset + size, x_offset : x_offset + size]
            for image in images
        ]
        assert cropped[0].shape[1] == size, "图像裁剪不正确"
        assert cropped[0].shape[2] == size, "图像裁剪不正确"
    elif order == "HWC":
        if images[0].shape[0] == size and images[0].shape[1] == size:
            return images, boxes
        height = images[0].shape[0]
        width = images[0].shape[1]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = [
            image[y_offset : y_offset + size, x_offset : x_offset + size, :]
            for image in images
        ]
        assert cropped[0].shape[0] == size, "图像裁剪不正确"
        assert cropped[0].shape[1] == size, "图像裁剪不正确"

    if boxes is not None:
        boxes = [crop_boxes(proposal, x_offset, y_offset) for proposal in boxes]
    return cropped, boxes


def center_crop(size, image):
    """
        对输入图像执行中心裁剪。

        参数：
            size (int): 裁剪后的高度和宽度。
            image (array): 需要中心裁剪的图像。

    """
    height = image.shape[0]
    width = image.shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))
    cropped = image[y_offset : y_offset + size, x_offset : x_offset + size, :]
    assert cropped.shape[0] == size, "图像高度裁剪不正确"
    assert cropped.shape[1] == size, "图像宽度裁剪不正确"
    return cropped


# ResNet 风格的缩放抖动：从 [1/max_size, 1/min_size] 中随机选择缩放比例。
def random_scale_jitter(image, min_size, max_size):
    """
        执行 ResNet 风格的随机缩放抖动。

        缩放比例会从 [1/max_size, 1/min_size] 中随机选择。

        参数：
            image (array): 需要随机缩放的图像。
            min_size (int): 缩放后的最小尺寸。
            max_size (int) 缩放后的最大尺寸。

        返回：
            image (array): 缩放后的图像。

    """
    img_scale = int(
        round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
    )
    image = scale(img_scale, image)
    return image


def random_scale_jitter_list(images, min_size, max_size):
    """
        对图像列表执行 ResNet 风格的随机缩放抖动。

        缩放比例会从 [1/max_size, 1/min_size] 中随机选择；所有图像共享同一个比例。

        参数：
            images (list): 需要随机缩放的图像列表。
            min_size (int): 缩放后的最小尺寸。
            max_size (int) 缩放后的最大尺寸。

        返回：
            images (list): 缩放后的图像列表。

    """
    img_scale = int(
        round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
    )
    return [scale(img_scale, image) for image in images]


def random_sized_crop(image, size, area_frac=0.08):
    """
        对给定图像执行随机面积裁剪。

        随机裁剪面积为原图的 8% - 100%，长宽比在 [3/4, 4/3] 范围内。

        参数：
            image (array): 需要裁剪的图像。
            size (int): 裁剪后缩放到的尺寸。
            area_frac (float): 最小面积比例。

        返回：
            (array): 裁剪后的图像。

    """
    for _ in range(0, 10):
        height = image.shape[0]
        width = image.shape[1]
        area = height * width
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w, h = h, w
        if h <= height and w <= width:
            if height == h:
                y_offset = 0
            else:
                y_offset = np.random.randint(0, height - h)
            if width == w:
                x_offset = 0
            else:
                x_offset = np.random.randint(0, width - w)
            y_offset = int(y_offset)
            x_offset = int(x_offset)
            cropped = image[y_offset : y_offset + h, x_offset : x_offset + w, :]
            assert (
                cropped.shape[0] == h and cropped.shape[1] == w
            ), "裁剪尺寸不正确"
            cropped = cv2.resize(
                cropped, (size, size), interpolation=cv2.INTER_LINEAR
            )
            return cropped.astype(np.float32)
    return center_crop(size, scale(size, image))


def lighting(img, alphastd, eigval, eigvec):
    """
        对给定图像执行 AlexNet 风格的 PCA lighting 抖动。

        参数：
            image (array): 需要执行 lighting 抖动的图像。
            alphastd (float): PCA 抖动比例。
            eigval (array): PCA 抖动使用的特征值。
            eigvec (list): PCA 抖动使用的特征向量。

        返回：
            img (tensor): 抖动后的图像。

    """
    if alphastd == 0:
        return img
    # 生成 alpha1、alpha2、alpha3。
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    for idx in range(img.shape[0]):
        img[idx] = img[idx] + rgb[2 - idx]
    return img


def random_sized_crop_list(images, size, crop_area_fraction=0.08):
    """
        对给定图像列表执行随机面积裁剪。

        随机裁剪面积为原图的 8% - 100%，长宽比在 [3/4, 4/3] 范围内。

        参数：
            images (list): 需要裁剪的图像列表。
            size (int): 裁剪后缩放到的尺寸。
            area_frac (float): 最小面积比例。

        返回：
            (list): 裁剪后的图像列表。

    """
    for _ in range(0, 10):
        height = images[0].shape[0]
        width = images[0].shape[1]
        area = height * width
        target_area = np.random.uniform(crop_area_fraction, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w, h = h, w
        if h <= height and w <= width:
            if height == h:
                y_offset = 0
            else:
                y_offset = np.random.randint(0, height - h)
            if width == w:
                x_offset = 0
            else:
                x_offset = np.random.randint(0, width - w)
            y_offset = int(y_offset)
            x_offset = int(x_offset)

            croppsed_images = []
            for image in images:
                cropped = image[
                    y_offset : y_offset + h, x_offset : x_offset + w, :
                ]
                assert (
                    cropped.shape[0] == h and cropped.shape[1] == w
                ), "裁剪尺寸不正确"
                cropped = cv2.resize(
                    cropped, (size, size), interpolation=cv2.INTER_LINEAR
                )
                croppsed_images.append(cropped.astype(np.float32))
            return croppsed_images

    return [center_crop(size, scale(size, image)) for image in images]


def blend(image1, image2, alpha):
    """按 alpha 权重混合两张图像，常用于颜色增强。"""
    return image1 * alpha + image2 * (1 - alpha)


def grayscale(image):
    """
        将图像转换为灰度图。

        参数：
            image (tensor): 需要转换为灰度的图像，维度为
                说明：`channel` x `height` x `width`.

        返回：
            img_gray (tensor): 灰度图像。

    """
    # 红色通道权重为 0.299，绿色为 0.587，蓝色为 0.114。
    img_gray = np.copy(image)
    gray_channel = 0.299 * image[2] + 0.587 * image[1] + 0.114 * image[0]
    img_gray[0] = gray_channel
    img_gray[1] = gray_channel
    img_gray[2] = gray_channel
    return img_gray


def saturation(var, image):
    """
        对给定图像执行饱和度增强。

        参数：
            var (float): 随机扰动范围。
            image (array): 需要调整饱和度的图像。

        返回：
            (array): 调整饱和度后的图像。

    """
    img_gray = grayscale(image)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def brightness(var, image):
    """
        对给定图像执行亮度增强。

        参数：
            var (float): 随机扰动范围。
            image (array): 需要调整亮度的图像。

        返回：
            (array): 调整亮度后的图像。

    """
    img_bright = np.zeros(image.shape).astype(image.dtype)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_bright, alpha)


def contrast(var, image):
    """
        对给定图像执行对比度增强。

        参数：
            var (float): 随机扰动范围。
            image (array): 需要调整对比度的图像。

        返回：
            (array): 调整对比度后的图像。

    """
    img_gray = grayscale(image)
    img_gray.fill(np.mean(img_gray[0]))
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def saturation_list(var, images):
    """
        对给定图像列表执行饱和度增强。

        参数：
            var (float): 随机扰动范围。
            images (list): 需要调整饱和度的图像列表。

        返回：
            (list): 调整饱和度后的图像列表。

    """
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def brightness_list(var, images):
    """
        对给定图像列表执行亮度增强。

        参数：
            var (float): 随机扰动范围。
            images (list): 需要调整亮度的图像列表。

        返回：
            (array): 调整亮度后的图像列表。

    """
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_bright = np.zeros(image.shape).astype(image.dtype)
        out_images.append(blend(image, img_bright, alpha))
    return out_images


def contrast_list(var, images):
    """
        对给定图像列表执行对比度增强。

        参数：
            var (float): 随机扰动范围。
            images (list): 需要调整对比度的图像列表。

        返回：
            (array): 调整对比度后的图像列表。

    """
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        img_gray.fill(np.mean(img_gray[0]))
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def color_jitter(image, img_brightness=0, img_contrast=0, img_saturation=0):
    """
        对给定图像执行颜色抖动。

        参数：
            image (array): 需要颜色抖动的图像。
            img_brightness (float): 亮度抖动比例。
            img_contrast (float): 对比度抖动比例。
            img_saturation (float): 饱和度抖动比例。

        返回：
            image (array): 抖动后的图像。

    """
    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                image = brightness(img_brightness, image)
            elif jitter[order[idx]] == "contrast":
                image = contrast(img_contrast, image)
            elif jitter[order[idx]] == "saturation":
                image = saturation(img_saturation, image)
    return image


def revert_scaled_boxes(size, boxes, img_height, img_width):
    """
        将缩放后的输入框还原到原始图像尺寸。

        参数：
            size (int): 裁剪图像的尺寸。
            boxes (array): shape 为 (num_boxes, 4)。
            img_height (int): 原始图像高度。
            img_width (int): 原始图像宽度。

        返回：
            reverted_boxes (array): 缩放回原始图像尺寸的边界框。

    """
    scaled_aspect = np.min([img_height, img_width])
    scale_ratio = scaled_aspect / size
    reverted_boxes = boxes * scale_ratio
    return reverted_boxes
