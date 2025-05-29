import pandas as pd

class Work:
    """ 定义分段类并接收excel表格中的数据 """
    attributes = ["GC", "ZD", "SG", "sand_area" ,"paint_area" ,"ZYQ", "BZ", "start_time",
                  "end_time", "pre_end_paint_time", "CX", "length", "weight", "height", "n_paint"]
    def __init__(self, data_list):
        # 将读取到的分段信息赋值给分段类的各个属性
        for name, value in zip(self.attributes, data_list):
            setattr(self, name, value)
        # 将最长边、次最长边和最短边依次定义为长、宽、高
        self.real_start_sand_time = None
        self.real_end_sand_time = None
        self.real_end_paint_time = None
        self.delay_days = 0
        self.assigned = False
        self.scheduled_day = 0
        self.size_list = []
        self.height, self.weight, self.length = self.size_change()
        # 获取分段的占地面积
        # self.s = self.get_s()
        self.paint = self.total_paint()

    def get_s(self):
        return round((self.weight + 1.0) * (self.length + 1.0) ,2)

    def size_change(self):
        self.size_list = [self.length, self.weight, self.height]
        self.size_list.sort()
        return [item for item in self.size_list]

    def total_paint(self):
        return self.paint_area * self.n_paint


def read_excel_range(file_path, sheet_name, workshop_num, max_paint_float):
    """读取指定单元格范围的内容"""
    work_list = []
    expect_paint = 0.0
    df = pd.read_excel(file_path, sheet_name)
    data = df.values.tolist()

    for lst in data:
        work_temp = Work(lst)
        expect_paint += work_temp.paint
        work_list.append(work_temp)

    return work_list, expect_paint / workshop_num + max_paint_float


def work_sort(work_list, work_data):
    """ 将分段按原加工日期分为 6 类（对应 6天）然后再将每一天的分段按涂漆时间从大到小排序 """
    new_work_list  = [[] for _ in range(len(work_data))]
    new_work_list_sort = []
    # 分为6类
    for work in work_list:
        idx = work_data.index(work.start_time.day)
        work.scheduled_day = idx
        new_work_list[idx].append(work)

    # 从大到小排序
    for item in new_work_list:
        item_sort = sorted(item, key=lambda x: -x.paint)
        new_work_list_sort.append(item_sort)

    return new_work_list_sort