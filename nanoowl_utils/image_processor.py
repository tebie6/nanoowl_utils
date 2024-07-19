# nanoowl_utils/image_processor.py

import PIL.Image
import numpy as np
import json
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_predictor import TreePredictor, Tree, TreeOutput, TreeDetection
from datetime import datetime
import time
import os


class ImageProcessor:
    def __init__(self,
                 prompt="[a person(sitting,stand)(waving,not wave)]",
                 threshold=0.15,
                 model="google/owlvit-base-patch32",
                 image_encoder_engine="/opt/nanoowl/data/owl_image_encoder_patch32.engine"):
        try:
            # 初始化参数
            self.prompt = prompt
            self.threshold = threshold
            self.model = model
            self.image_encoder_engine = image_encoder_engine

            # 初始化预测器
            self.predictor = TreePredictor(
                owl_predictor=OwlPredictor(self.model, image_encoder_engine=self.image_encoder_engine)
            )

            # 解析提示并编码文本
            self.tree = Tree.from_prompt(self.prompt)
            self.clip_text_encodings = self.predictor.encode_clip_text(self.tree)
            self.owl_text_encodings = self.predictor.encode_owl_text(self.tree)
        except Exception as e:
            print(f"初始化 ImageProcessor 时出错: {e}")
            raise

    def process_single_image(self, image_path, output_type="json"):

        try:
            # 处理单张图片
            image = PIL.Image.open(image_path)
            output = self.predictor.predict(
                image=image,
                tree=self.tree,
                clip_text_encodings=self.clip_text_encodings,
                owl_text_encodings=self.owl_text_encodings,
                threshold=self.threshold
            )

            if output_type == "json":
                output = self._convert_output_to_json(output)

            return output
        except FileNotFoundError:
            print(f"未找到指定的图片文件: {image_path}")
        except Exception as e:
            print(f"处理图片时出错: {e}")
            raise

    def process_image(self, image, output_format="json"):

        try:
            # 预测输出
            output = self.predictor.predict(
                image=image,
                tree=self.tree,
                clip_text_encodings=self.clip_text_encodings,
                owl_text_encodings=self.owl_text_encodings,
                threshold=self.threshold
            )

            # 根据指定格式转换输出
            if output_format == "json":
                return self._convert_output_to_json(output)
            return output
        except FileNotFoundError:
            print("未找到指定的图片文件")
        except Exception as e:
            print(f"处理图片时出错: {e}")
            raise

    def _convert_output_to_json(self, output):
        try:
            if not isinstance(output, TreeOutput):
                return json.dumps([])  # 如果输出不是 TreeOutput 实例，返回空的 JSON 数组

            label_map = self.tree.get_label_map()
            result = [
                {
                    "detection_id": detection.id,
                    "parent_id": detection.parent_id,
                    "box": detection.box,
                    "labels_scores": [
                        {"label": label_map[label], "score": score}
                        for label, score in zip(detection.labels, detection.scores)
                    ]
                }
                for detection in output.detections
            ]

            return json.dumps(result, indent=4)
        except Exception as e:
            print(f"将输出转换为 JSON 时出错: {e}")
            raise
