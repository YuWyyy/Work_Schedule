import heapq

def find_first_index(workpieces, remaining_paint):
    for index, workpiece in enumerate(workpieces):
        if workpiece.paint <= remaining_paint:
            return index
    return -1


def distribute_groups(sorted_tasks, worker_efficiency, workshop_num, max_days, expect_paint):
    """分配算法"""
    # 初始化数据结构
    remaining_paint = [expect_paint] * workshop_num
    group = [[[] for _ in range(workshop_num)] for _ in range(max_days)]
    total_area = [[0.0 for _ in range(workshop_num)] for _ in range(max_days)]
    pre_painting = [0.0 for _ in range(workshop_num)]
    current_day = 0  # 当前处理的天数

    for day in range(max_days):
        stopped = [False] * workshop_num
        while not all(stopped):
            if current_day >= len(sorted_tasks):
                break  # 所有天数任务已分配完毕
            # 按车间轮流分配
            for workshop in range(workshop_num):
                if stopped[workshop]:
                    continue
                # 尝试从当前天或下一天寻找合适分段
                if not group[day][workshop]:
                    # 分配最大分段
                    if sorted_tasks[current_day]:
                        idx = find_first_index(sorted_tasks[current_day], remaining_paint[workshop])
                        task = sorted_tasks[current_day].pop(idx)
                        group[day][workshop].append(task)
                        total_area[day][workshop] += task.sand_area
                        pre_painting[workshop] += task.paint
                else:
                    # 计算剩余容量并寻找合适分段
                    remaining_sand = worker_efficiency - total_area[day][workshop]
                    remaining_paint[workshop] = expect_paint - pre_painting[workshop]

                    task_found = False
                    # 优先当前天
                    for idx, task_1 in enumerate(sorted_tasks[current_day]):
                        if task_1.sand_area <= remaining_sand and task_1.paint <= remaining_paint[workshop]:
                            group[day][workshop].append(sorted_tasks[current_day].pop(idx))
                            total_area[day][workshop] += task_1.sand_area
                            pre_painting[workshop] += task_1.paint
                            task_found = True
                            break
                    # 当前天无合适任务，尝试下一天
                    if not task_found and current_day + 1 < len(sorted_tasks):
                        for idx, task_2 in enumerate(sorted_tasks[current_day + 1]):
                            if task_2.sand_area <= remaining_sand and task_2.paint <= remaining_paint[workshop]:
                                group[day][workshop].append(sorted_tasks[current_day + 1].pop(idx))
                                total_area[day][workshop] += task_2.sand_area
                                pre_painting[workshop] += task_2.paint
                                task_found = True
                                break
                    # 标记停止
                    if not task_found:
                        stopped[workshop] = True
            # 更新当前天
            if not sorted_tasks[current_day]:
                current_day += 1
    return group, total_area, pre_painting


def distribute_elements(initial_segments, remaining_segments):
    """分配剩余分段到三个子列表，第一个子列表初始包含第19日的分段"""
    # 初始化第一个子列表并计算初始和
    list1 = initial_segments.copy()
    list1_sort = sorted(list1, key=lambda x: -x.paint)
    sum1 = sum(seg.paint for seg in list1_sort[0::2])
    sum2 = sum(seg.paint for seg in list1_sort[1::2])

    sums = [sum1, sum2]
    lists = [list1_sort[0::2], list1_sort[1::2]]
    heap = [(sum1, 0), (sum2, 1)]
    heapq.heapify(heap)

    # 按面积降序排序剩余分段
    sorted_remaining = sorted(remaining_segments, key=lambda x: -x.paint)

    for seg in sorted_remaining:
        min_sum, min_idx = heapq.heappop(heap)
        lists[min_idx].append(seg)
        new_sum = min_sum + seg.paint
        heapq.heappush(heap, (new_sum, min_idx))
        sums[min_idx] = new_sum

    return lists, sums


def painting_distribute(group, workshop_num):
    """处理三维列表，确保第19日分段在第一个子列表"""
    total_paint_list = []
    total_paint_area = []

    for workshop in range(workshop_num):
        # 提取第19日分段（必须添加到第一个子列表）
        day19_segments = group[0][workshop].copy()  # group[day][workshop]是列表

        # 提取剩余5日分段（20~24日）
        remaining = [
            seg
            for day in group[1:6]  # 取索引1~5（对应20~24日）
            for seg in day[workshop]
        ]

        # 分配剩余分段到三个子列表（第一个子列表已包含第19日分段）
        lists, sums = distribute_elements(day19_segments, remaining)
        total_paint_list.append(lists)
        total_paint_area.append(sums)

    return total_paint_list, total_paint_area