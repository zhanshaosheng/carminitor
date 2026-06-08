import math


class Tracker: #Tracker类用于跟踪视频中的目标对象
    def __init__(self):#初始化创建两个属性，一个字典，一个计数器
        # 存储目标的中心位置
        self.center_points = {}
        # ID计数
        # 每当检测到一个新的目标id时, 计数将增加1
        self.id_count = 0

    # update用于接收当前帧中检测到的目标矩形框列表，objects_rect矩形框包含了左上角坐标，宽度和高度
    def update(self, objects_rect):
        # 存储目标的方框和ID，
        objects_bbs_ids = []#用于存储每个目标的矩形框和对应的ID。

        # 获取新目标的中心点
        #遍历每个检测到的目标矩形框，计算中心点坐标
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

         # 看看这个目标是否已经被检测到过
            same_object_detected = False
            #遍历已有的目标中心点，计算当前目标中心点与已有中心点的距离。
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
        #如果距离小于35像素，则认为是同一个目标，更新该目标的中心点，并将该目标的矩形框和ID添加到objects_bbs_ids列表中。
                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True#表示当前目标已经被检测到过。
                    break

            # 检测到新目标，分配ID给新目标
            #如果 same_object_detected 仍然为 False，说明当前目标没有与已有的ID匹配，是一个新检测到的目标。
            if same_object_detected is False:
            #为新目标分配一个新的ID，并将其中心点和矩形框信息添加到objects_bbs_id列表中。
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1#计数器加1，为检测下一个新目标做好准备

        # 按中心点清理字典, 删除不再使用的ID
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:#遍历当前帧所有检测到的目标的矩形框和ID信息
            _, _, _, _, object_id = obj_bb_id#从 obj_bb_id 中提取目标的ID
            center = self.center_points[object_id]#从 self.center_points 字典中获取与 object_id 对应的中心点坐标。
            new_center_points[object_id] = center#将当前目标的ID和中心点坐标添加到 new_center_points 字典中。

        # 更新字典, 删除未使用的ID
        self.center_points = new_center_points.copy()
        return objects_bbs_ids