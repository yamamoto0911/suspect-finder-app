import cv2
import numpy as np
from ultralytics import YOLO
import json
import datetime
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import math

@dataclass
class DetectedObject:
    """検出された物体の情報"""
    id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]
    first_seen_frame: int
    last_seen_frame: int
    
@dataclass
class DumpingEvent:
    """不法投棄イベントの情報"""
    timestamp: str
    frame_number: int
    suspect_id: int
    suspect_class: str
    object_id: int
    object_class: str
    suspect_bbox: Tuple[int, int, int, int]
    object_bbox: Tuple[int, int, int, int]
    distance: float
    confirmed: bool = False
    confirmation_frame: Optional[int] = None
    # 投棄行為の詳細情報
    initial_distance: float = 0.0
    final_distance: float = 0.0
    object_dropped: bool = False  # 物体が落下したか
    suspect_moved_away: bool = False  # 容疑者が離れたか

@dataclass 
class ObjectRelationship:
    """人と物体の関係性を追跡"""
    suspect_id: int
    object_id: int
    start_frame: int
    initial_distance: float
    min_distance: float
    current_distance: float
    distance_history: List[float]
    object_y_history: List[float]  # 物体のY座標履歴（落下検知用）
    frames_together: int = 0
    is_carrying: bool = False

class DumpWatcher:
    """不法投棄検知システムのメインクラス"""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        # YOLOモデルの初期化
        self.model = YOLO(model_path)
        
        # 検出対象クラス
        self.person_classes = ['person']
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        # すべてのCOCOクラスを対象にして、ごみ袋の可能性がある物体をすべて検出
        self.object_classes = ['backpack', 'handbag', 'suitcase', 'bottle', 'cup', 
                              'bowl', 'chair', 'potted plant', 'tv', 'laptop', 
                              'keyboard', 'cell phone', 'book', 'vase', 'bag',
                              'sports ball', 'teddy bear', 'hair drier', 'toothbrush',
                              'scissors', 'remote', 'wine glass', 'fork', 'knife',
                              'spoon', 'banana', 'apple', 'sandwich', 'orange',
                              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                              # ごみ袋として認識される可能性のあるクラスを追加
                              'refrigerator', 'microwave', 'oven', 'toaster', 'sink',
                              'toilet', 'couch', 'bed', 'dining table', 'umbrella',
                              'tie', 'skis', 'snowboard', 'kite', 'baseball bat',
                              'baseball glove', 'skateboard', 'surfboard', 'tennis racket']
        
        # 追跡管理
        self.next_object_id = 1
        self.tracked_objects: Dict[int, DetectedObject] = {}
        self.current_frame = 0
        
        # エリア内の物体管理
        self.area_persons: Dict[int, DetectedObject] = {}
        self.area_vehicles: Dict[int, DetectedObject] = {}
        self.area_objects: Dict[int, DetectedObject] = {}
        
        # 不法投棄イベント管理
        self.dumping_events: List[DumpingEvent] = []
        self.confirmed_events: List[DumpingEvent] = []
        
        # 人と物体の関係性追跡
        self.object_relationships: Dict[str, ObjectRelationship] = {}
        
        # パラメータ（より検知しやすく調整）
        self.max_distance_for_tracking = 150  # トラッキング用の最大距離
        self.dumping_distance_threshold = 200  # 不法投棄判定用の距離閾値（広げる）
        self.carrying_distance_threshold = 120  # 物を持っているとみなす距離（広げる）
        self.drop_distance_threshold = 100  # 投棄とみなす距離の増加量（下げる）
        self.drop_y_threshold = 20  # 落下とみなすY座標の変化量（下げる）
        self.confirmation_frames = 5 * 30   # 5秒（30fps想定）に短縮
        
        # 出力設定
        self.output_video_writer = None
        self.evidence_count = 0
    
    def calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """2点間の距離を計算"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def frame_to_time(self, frame_number: int) -> str:
        """フレーム番号を時間文字列に変換 (例: 585 -> "19.5秒" または "0:19.5")"""
        if not hasattr(self, 'video_fps') or self.video_fps == 0:
            return f"フレーム{frame_number}"
        
        total_seconds = frame_number / self.video_fps
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        
        if minutes > 0:
            return f"{minutes}:{seconds:.1f}"
        else:
            return f"{seconds:.1f}秒"
    
    def get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """バウンディングボックスの中心点を計算"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def track_objects(self, detections: List[Tuple]) -> None:
        """物体のトラッキングを実行"""
        current_detections = []
        
        # 検出結果を処理
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            class_name = self.model.names[int(cls)]
            center = self.get_bbox_center((int(x1), int(y1), int(x2), int(y2)))
            
            current_detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'center': center,
                'class_name': class_name,
                'confidence': float(conf)
            })
        
        # 既存の物体との対応付け
        used_object_ids = set()
        new_objects = []
        
        for detection in current_detections:
            best_match_id = None
            best_distance = float('inf')
            
            # 既存の物体との距離を計算
            for obj_id, obj in self.tracked_objects.items():
                if obj_id in used_object_ids:
                    continue
                if obj.class_name != detection['class_name']:
                    continue
                
                distance = self.calculate_distance(obj.center, detection['center'])
                if distance < self.max_distance_for_tracking and distance < best_distance:
                    best_distance = distance
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # 既存物体を更新
                obj = self.tracked_objects[best_match_id]
                obj.bbox = detection['bbox']
                obj.center = detection['center']
                obj.confidence = detection['confidence']
                obj.last_seen_frame = self.current_frame
                used_object_ids.add(best_match_id)
            else:
                # 新しい物体として追加
                new_objects.append(detection)
        
        # 新しい物体を追加
        for detection in new_objects:
            new_obj = DetectedObject(
                id=self.next_object_id,
                class_name=detection['class_name'],
                confidence=detection['confidence'],
                bbox=detection['bbox'],
                center=detection['center'],
                first_seen_frame=self.current_frame,
                last_seen_frame=self.current_frame
            )
            self.tracked_objects[self.next_object_id] = new_obj
            self.next_object_id += 1
        
        # 古い物体を削除（30フレーム見えなかったら削除 - 条件緩和）
        to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            if self.current_frame - obj.last_seen_frame > 30:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
    
    def update_area_status(self) -> None:
        """エリア内の物体状況を更新"""
        # エリア内の人物、車両、物体を分類
        self.area_persons.clear()
        self.area_vehicles.clear()
        self.area_objects.clear()
        
        for obj_id, obj in self.tracked_objects.items():
            if obj.class_name in self.person_classes:
                self.area_persons[obj_id] = obj
            elif obj.class_name in self.vehicle_classes:
                self.area_vehicles[obj_id] = obj
            elif obj.class_name in self.object_classes:
                self.area_objects[obj_id] = obj
        
        # デバッグ情報（50フレームごとに頻繁に確認）
        if self.current_frame % 50 == 0:
            print(f"フレーム{self.current_frame}: 全物体={len(self.tracked_objects)}, 人物={len(self.area_persons)}, 物体={len(self.area_objects)}, 関係性={len(self.object_relationships)}")
            # 検出されている全ての物体を表示
            for obj_id, obj in self.tracked_objects.items():
                print(f"  検出物体: {obj.class_name} (ID:{obj_id}, フレーム:{obj.first_seen_frame}-{obj.last_seen_frame})")
            if self.area_objects:
                for obj_id, obj in self.area_objects.items():
                    print(f"  対象物体: {obj.class_name} (ID:{obj_id})")
            if self.area_persons:
                for person_id, person in self.area_persons.items():
                    print(f"  人物: {person.class_name} (ID:{person_id})")
    
    def update_object_relationships(self) -> None:
        """人と物体の関係性を更新"""
        suspects = {**self.area_persons, **self.area_vehicles}
        
        # 既存の関係性を更新
        relationships_to_remove = []
        for rel_key, relationship in self.object_relationships.items():
            suspect_id = relationship.suspect_id
            object_id = relationship.object_id
            
            # 容疑者と物体がまだ存在するかチェック
            if (suspect_id not in suspects or 
                object_id not in self.area_objects):
                relationships_to_remove.append(rel_key)
                continue
            
            suspect = suspects[suspect_id]
            obj = self.area_objects[object_id]
            current_distance = self.calculate_distance(suspect.center, obj.center)
            
            # 距離履歴を更新
            relationship.current_distance = current_distance
            relationship.distance_history.append(current_distance)
            relationship.object_y_history.append(obj.center[1])
            relationship.frames_together += 1
            
            # 最小距離を更新
            if current_distance < relationship.min_distance:
                relationship.min_distance = current_distance
            
            # 持ち運び状態の判定
            if current_distance <= self.carrying_distance_threshold:
                relationship.is_carrying = True
        
        # 不要な関係性を削除
        for rel_key in relationships_to_remove:
            del self.object_relationships[rel_key]
        
        # 新しい関係性を検出
        for suspect_id, suspect in suspects.items():
            for obj_id, obj in self.area_objects.items():
                rel_key = f"{suspect_id}_{obj_id}"
                
                if rel_key not in self.object_relationships:
                    distance = self.calculate_distance(suspect.center, obj.center)
                    
                    # 十分近い場合は関係性を開始
                    if distance <= self.dumping_distance_threshold:
                        self.object_relationships[rel_key] = ObjectRelationship(
                            suspect_id=suspect_id,
                            object_id=obj_id,
                            start_frame=self.current_frame,
                            initial_distance=distance,
                            min_distance=distance,
                            current_distance=distance,
                            distance_history=[distance],
                            object_y_history=[obj.center[1]]
                        )
                        print(f"新しい関係性開始: 人物{suspect_id} - 物体{obj_id} (距離:{distance:.1f})")

    def detect_dumping_scenario(self) -> None:
        """改良された不法投棄シナリオの検知"""
        suspects = {**self.area_persons, **self.area_vehicles}
        
        # 関係性を基にした投棄検知
        for rel_key, relationship in self.object_relationships.items():
            suspect_id = relationship.suspect_id
            object_id = relationship.object_id
            
            # 最低限のフレーム数が経過していることを確認（条件緩和）
            if relationship.frames_together < 5:
                continue
            
            # 投棄の兆候をチェック
            if self.detect_dropping_behavior(relationship):
                suspect = suspects.get(suspect_id)
                obj = self.area_objects.get(object_id)
                
                if suspect and obj:
                    # 不法投棄イベントを記録
                    event = DumpingEvent(
                        timestamp=datetime.datetime.now().isoformat(),
                        frame_number=self.current_frame,
                        suspect_id=suspect_id,
                        suspect_class=suspect.class_name,
                        object_id=object_id,
                        object_class=obj.class_name,
                        suspect_bbox=suspect.bbox,
                        object_bbox=obj.bbox,
                        distance=relationship.current_distance,
                        initial_distance=relationship.initial_distance,
                        final_distance=relationship.current_distance,
                        object_dropped=self.check_object_dropped(relationship),
                        suspect_moved_away=self.check_suspect_moved_away(relationship)
                    )
                    self.dumping_events.append(event)
                    time_str = self.frame_to_time(self.current_frame)
                    print(f"投棄行為を検知: {time_str} (フレーム{self.current_frame}), "
                          f"{suspect.class_name}が{obj.class_name}を投棄 "
                          f"(距離変化: {relationship.min_distance:.1f}→{relationship.current_distance:.1f})")
    
    def detect_dropping_behavior(self, relationship: ObjectRelationship) -> bool:
        """投棄行為の検知（より厳密な条件）"""
        # 十分な観察期間があることを確認
        if relationship.frames_together < 30:  # 1秒以上の関係性が必要
            return False
            
        # 1. 物を持っていた状態から離れた状態への変化
        was_carrying = relationship.is_carrying
        distance_increased = (relationship.current_distance - relationship.min_distance) > self.drop_distance_threshold
        
        # 2. 物体の落下（Y座標の増加）
        object_dropped = self.check_object_dropped(relationship)
        
        # 3. 距離の急激な増加かつ継続的な分離
        rapid_distance_increase = False
        if len(relationship.distance_history) >= 10:
            recent_distances = relationship.distance_history[-10:]
            distance_trend_increasing = recent_distances[-1] > recent_distances[0]  # 距離が増加傾向
            significant_increase = (max(recent_distances) - min(recent_distances)) > self.drop_distance_threshold / 2
            rapid_distance_increase = distance_trend_increasing and significant_increase
        
        # 4. 最小距離が十分近かったことを確認（実際に持ち運んでいた証拠）
        was_really_close = relationship.min_distance < self.carrying_distance_threshold / 2  # より近い距離
        
        # すべての条件を満たす場合のみ投棄と判定
        return (was_carrying and was_really_close and 
                (distance_increased or object_dropped) and rapid_distance_increase)
    
    def check_object_dropped(self, relationship: ObjectRelationship) -> bool:
        """物体が落下したかチェック"""
        if len(relationship.object_y_history) < 5:
            return False
        
        # 最近のY座標の変化をチェック（Y座標が増加 = 下方向）
        recent_y = relationship.object_y_history[-5:]
        y_change = max(recent_y) - min(recent_y)
        
        return y_change > self.drop_y_threshold
    
    def check_suspect_moved_away(self, relationship: ObjectRelationship) -> bool:
        """容疑者が離れたかチェック"""
        return relationship.current_distance > self.carrying_distance_threshold * 2
    
    def check_dumping_confirmation(self) -> None:
        """不法投棄の確定チェック"""
        for event in self.dumping_events[:]:
            if event.confirmed:
                continue
            
            # 物体がまだ存在するかチェック
            if event.object_id not in self.tracked_objects:
                # 物体が消えた場合はイベントを削除
                self.dumping_events.remove(event)
                continue
            
            # 容疑者が立ち去ったかチェック
            suspect_present = event.suspect_id in self.tracked_objects
            
            # 確定条件を緩和：時間経過または容疑者が十分離れた場合
            frames_passed = self.current_frame - event.frame_number
            suspect_far_away = False
            if suspect_present and event.object_id in self.area_objects:
                suspect = self.tracked_objects[event.suspect_id]
                obj = self.area_objects[event.object_id]
                current_distance = self.calculate_distance(suspect.center, obj.center)
                suspect_far_away = current_distance > self.carrying_distance_threshold * 3
            
            if (frames_passed >= self.confirmation_frames or 
                not suspect_present or 
                suspect_far_away) and event.object_id in self.area_objects:
                
                # 不法投棄を確定
                event.confirmed = True
                event.confirmation_frame = self.current_frame
                self.confirmed_events.append(event)
                
                confirmation_time = self.frame_to_time(self.current_frame)
                initial_time = self.frame_to_time(event.frame_number)
                print(f"不法投棄を確定: {initial_time}に発生した{event.suspect_class}による{event.object_class}の投棄 (確定時刻: {confirmation_time})")
                
                # 証拠画像を保存
                self.save_evidence_image(event)
    
    def save_evidence_image(self, event: DumpingEvent) -> None:
        """証拠画像を保存"""
        if hasattr(self, 'current_frame_image'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evidence_{timestamp}_{self.evidence_count}.jpg"
            
            # 検出した物体に枠を描画
            img_with_boxes = self.current_frame_image.copy()
            
            # 物体の枠を描画
            if event.object_id in self.tracked_objects:
                obj = self.tracked_objects[event.object_id]
                x1, y1, x2, y2 = obj.bbox
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_with_boxes, f"DUMPED: {obj.class_name}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imwrite(filename, img_with_boxes)
            self.evidence_count += 1
            print(f"証拠画像を保存: {filename}")
    
    
    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """検出結果を描画"""
        result_frame = frame.copy()
        
        # 追跡中の物体を描画
        for obj_id, obj in self.tracked_objects.items():
            x1, y1, x2, y2 = obj.bbox
            
            # クラスによって色を変える
            if obj.class_name in self.person_classes:
                color = (255, 0, 0)  # 青: 人物
                label = f"Person_{obj_id}"
            elif obj.class_name in self.vehicle_classes:
                color = (0, 255, 0)  # 緑: 車両
                label = f"Vehicle_{obj_id}"
            elif obj.class_name in self.object_classes:
                color = (0, 0, 255)  # 赤: 物体
                label = f"Object_{obj_id}"
            else:
                color = (128, 128, 128)  # グレー: その他
                label = f"Other_{obj_id}"
            
            # バウンディングボックスを描画
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_frame, f"{label}: {obj.class_name}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 確定した不法投棄イベントを強調表示
        for event in self.confirmed_events:
            if event.object_id in self.tracked_objects:
                obj = self.tracked_objects[event.object_id]
                x1, y1, x2, y2 = obj.bbox
                # 赤い太枠で強調
                cv2.rectangle(result_frame, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 5)
                cv2.putText(result_frame, "ILLEGAL DUMPING DETECTED!", 
                           (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_frame
    
    def process_video(self, input_path: str, output_path: str = "output.mp4") -> None:
        """動画ファイルを処理"""
        print(f"動画処理開始: {input_path}")
        print(f"ファイル存在確認: {os.path.exists(input_path)}")
        print(f"ファイルサイズ: {os.path.getsize(input_path) if os.path.exists(input_path) else 'N/A'} bytes")
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"エラー: 動画ファイル '{input_path}' を開けませんでした")
            print("OpenCVがサポートしているコーデック情報:")
            print(f"OpenCV version: {cv2.__version__}")
            return
        
        # 動画の情報を取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # FPSをインスタンス変数として保存（時間変換用）
        self.video_fps = fps if fps > 0 else 30  # デフォルト30fps
        
        print(f"動画情報: {width}x{height}, {fps}fps, {total_frames}フレーム")
        
        if total_frames == 0 or fps == 0:
            print("警告: 動画の基本情報が取得できませんでした。コーデックの問題の可能性があります。")
            cap.release()
            return
        
        # 出力動画の設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.output_video_writer.isOpened():
            print(f"エラー: 出力動画ファイル '{output_path}' を作成できませんでした")
            cap.release()
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.current_frame_image = frame
            
            # YOLO検出を実行（信頼度をさらに下げてより多く検出）
            results = self.model(frame, verbose=False, conf=0.1)
            
            # 検出結果を抽出
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = boxes.cls[i].cpu().numpy()
                    detections.append((x1, y1, x2, y2, conf, cls))
            
            # 物体追跡を実行
            self.track_objects(detections)
            
            # エリア状況を更新
            self.update_area_status()
            
            # 人と物体の関係性を更新
            self.update_object_relationships()
            
            # 不法投棄シナリオを検知
            self.detect_dumping_scenario()
            
            # 不法投棄の確定をチェック
            self.check_dumping_confirmation()
            
            # 結果を描画
            result_frame = self.draw_detections(frame)
            
            # 出力動画に書き込み
            self.output_video_writer.write(result_frame)
            
            # 進行状況を表示
            if self.current_frame % 100 == 0:
                progress = (self.current_frame / total_frames) * 100
                print(f"処理中... {progress:.1f}% ({self.current_frame}/{total_frames})")
            
            self.current_frame += 1
        
        # リソースを解放
        cap.release()
        self.output_video_writer.release()
        
        print(f"動画処理完了: {output_path}")
        print(f"検知した不法投棄イベント: {len(self.confirmed_events)}件")
    
    def process_video_simple(self, input_path: str, output_path: str) -> None:
        """シンプルな動画処理（ログあり）"""
        try:
            self.process_video(input_path, output_path)
            
            # 結果をJSONファイルに保存
            self.save_analysis_log()
            
        except Exception as e:
            print(f"動画処理でエラーが発生しました: {str(e)}")
            # エラーが発生した場合でもJSONログを保存
            self.save_analysis_log()
            raise
    
    def save_analysis_log(self) -> None:
        """解析結果をJSONファイルに保存"""
        analysis_data = {
            "analysis_info": {
                "total_frames": self.current_frame,
                "total_events": len(self.dumping_events),
                "confirmed_events": len(self.confirmed_events),
                "timestamp": datetime.datetime.now().isoformat()
            },
            "confirmed_dumping_events": [
                {
                    "timestamp": event.timestamp,
                    "frame_number": event.frame_number,
                    "time_in_video": self.frame_to_time(event.frame_number),
                    "suspect_class": event.suspect_class,
                    "object_class": event.object_class,
                    "distance": event.distance,
                    "confirmed": event.confirmed,
                    "confirmation_frame": event.confirmation_frame,
                    "confirmation_time": self.frame_to_time(event.confirmation_frame) if event.confirmation_frame else None
                }
                for event in self.confirmed_events
            ],
            "pending_events": [
                {
                    "timestamp": event.timestamp,
                    "frame_number": event.frame_number,
                    "time_in_video": self.frame_to_time(event.frame_number),
                    "suspect_class": event.suspect_class,
                    "object_class": event.object_class,
                    "distance": event.distance,
                    "confirmed": event.confirmed
                }
                for event in self.dumping_events if not event.confirmed
            ]
        }
        
        with open("dump_log.json", "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)