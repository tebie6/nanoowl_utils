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
                 prompt="(seated,standing)(waving,no wave)",
                 group_key=["seated", "waving"],
                 threshold=0.15,
                 model="google/owlvit-base-patch32",
                 image_encoder_engine="/opt/nanoowl/data/owl_image_encoder_patch32.engine"):
        try:
            # 初始化参数
            self.prompt = prompt
            self.group_key = group_key
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
            elif output_format == "text":
                return self._convert_output_to_text(output)
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

    def _convert_output_to_group(self, output):
        try:
            if not isinstance(output, TreeOutput):
                return ""  # 如果输出不是 TreeOutput 实例，返回空的字符串

            label_map = self.tree.get_label_map()
            result = {}

            for detection in output.detections:
                for index, label in enumerate(detection.labels[1:], start=1):  # 跳过第一个 label 并从 1 开始索引
                    label_text = label_map.get(label, "Unknown Label")
                    key = self.group_key[index - 1]  # 调整索引以匹配 group_key
                    result["is_" + key] = (label_text == key)  # 直接比较并赋值

            return result
        except Exception as e:
            print(f"Error: {e}")
            raise

    def _convert_output_to_text(self, output):
        try:
            if not isinstance(output, TreeOutput):
                return ""  # 如果输出不是 TreeOutput 实例，返回空的字符串

            label_map = self.tree.get_label_map()
            result = []

            for detection in output.detections:
                for label in detection.labels[1:]:  # 跳过第一个 label
                    label_text = label_map.get(label, "Unknown Label")
                    result.append(label_text)

            return ", ".join(result) + "" if result else ""
        except Exception as e:
            print(f"将输出转换为 text 时出错: {e}")
            raise
