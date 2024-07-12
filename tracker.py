import math

class Tracker:
    def __init__(self):
        # Nesnelerin merkez pozisyonlarını depolamak için bir sözlük
        self.center_points = {}
        # ID'lerin sayısını tutmak için, her yeni nesne algılandığında sayı bir artar
        self.id_count = 0

    def update(self, objects_rect):
        # Nesne kutuları ve kimlikler
        objects_bbs_ids = []

        # Yeni nesnenin merkez noktasını al
        for rect in objects_rect:
            x, y, w, h, cl = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Bu nesnenin önceden algılanıp algılanmadığını kontrol et
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id, cl])
                    same_object_detected = True
                    break

            # Yeni nesne algılandıysa, bu nesneye bir ID atarız
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, cl])
                self.id_count += 1

        # Kullanılmayan ID'leri içeren merkez noktalarını temizle
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id,_ = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Kullanılmayan ID'leri içermeyen sözlüğü güncelle
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
