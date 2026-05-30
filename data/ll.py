from PIL import Image


def fix_png_icc(image_path, output_path=None):
    """修复PNG文件中的iCCP问题"""
    img = Image.open(image_path)
    # 移除ICC配置
    info = img.info.copy()
    info.pop("icc_profile", None)

    # 保存修复后的图片
    if output_path is None:
        output_path = image_path
    img.save(output_path, format='PNG', icc_profile=None, optimize=True)


# 使用示例
fix_png_icc("your_image.png", "fixed_image.png")