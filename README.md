# nanoowl_utils
nanoowl_utils


# 使用 nanoowl_utils 包的示例代码

以下示例展示了如何使用封装的 `ImageProcessor` 工具类。

```bash
pip install git+https://github.com/tebie6/nanoowl_utils.git
```

## 示例代码

```python
from nanoowl_utils.image_processor import ImageProcessor

prompt = "[a person(sitting,stand)(waving,not wave)]"
threshold = 0.15
model = "google/owlvit-base-patch32"
image_encoder_engine = "/opt/nanoowl/data/owl_image_encoder_patch32.engine"

processor = ImageProcessor(
    prompt=prompt,
    threshold=threshold,
    model=model,
    image_encoder_engine=image_encoder_engine
)

specific_image_path = "specific_image.jpg"

# 处理图片并输出结果为 JSON
try:
    output = processor.process_single_image(specific_image_path)
    print(output)
except Exception as e:
    print(f"处理图片时出错: {e}")
```