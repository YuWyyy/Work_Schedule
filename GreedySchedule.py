from datetime import datetime
from Tools.ReadAndSort import work_sort, read_excel_range
from Tools.Writer import show_info, write_group_assignment
from Tools.Greedy_Algorithm import distribute_groups, painting_distribute

# === 基础参数 ===
file_path = "Data.xlsx"  # 需要读取的Excel文件名
sheet_name = 1                                      # 数据所在的工作簿序号（0，1，2）
worker_sand_efficiency = 5000                       # 单跨间一天冲砂面积
max_paint_float = 6000.0                            # 涂漆容许浮动范围
paint_efficiency = 3 * 300 * 8
start_date = datetime(2024, 8, 19)  # 起始日期
work_data = [19 ,20 ,21 ,22 ,23 ,25]                # 原方案跨间工作日
team_name = ["富纬","皓源","库众","松宇","旭丰"]       # 团队名称
wash_file_path = "冲砂方案.xlsx"                 # 冲砂新方案的Excel文件名
paint_file_path = "涂漆方案.xlsx"               # 涂漆新方案的Excel文件名
insert_cell = 'A1'                            # 新方案的起始插入位置
# ===============

if __name__ == "__main__":

    max_days = len(work_data)
    workshop_num = len(team_name)

    work_list, expect_paint = read_excel_range(file_path, sheet_name, workshop_num, max_paint_float)
    new_work_list = work_sort(work_list, work_data)
    group, total_area, pre_paint = distribute_groups(new_work_list, worker_sand_efficiency, workshop_num, max_days, expect_paint)
    show_info(group, wash_file_path, insert_cell, start_date, team_name)

    total_paint_list, total_paint_area = painting_distribute(group, workshop_num)
    write_group_assignment(total_paint_list, paint_file_path, insert_cell, team_name, paint_efficiency)
