
def random_sample_ratio(img_scale, ratio):
    """Randomly sample an img_scale when ``ratio`` is specified.

    A ratio will be randomly sampled from the range specified by
    ``ratio``. Then it would be multiplied with ``img_scale`` to
    generate sampled scale.

    Args:
        img_scale (tuple): Images scale base to multiply with ratio.
        ratio (float): The ratio to scale
            the ``img_scale``.

    Returns:
        scale: scale is sampled ratio multiplied with ``img_scale``
    """
    scale = (int(img_scale[0] * ratio), int(img_scale[1] * ratio))
    return scale
