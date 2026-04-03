import os
import sys
from typing import Optional

import numpy as np

# 命令行执行该py时，传入的参数
def parse_args(argv: Optional[list] = None):
    """
    解析命令行参数。终端执行「python3 resmaple.py」时，未传入的选项使用本函数内
    add_argument(..., default=...) 的默认值（如 --grid-step、--max-gap 等）。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="使用 Open3D 对点云进行『补地面点』上采样（默认处理 demo2_outdoor.pcd）"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="outdoor_afterCutForce2.pcd",
        help="输入 PCD/PLY 文件路径（默认: outdoor_afterCut.pcd）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outdoor_afterCutForce2_upsampled.pcd",
        help="输出点云文件路径（默认: outdoor_afterCut_upsampled.pcd）",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.05,
        help="地面虚拟格点间距 (m)，越小地面越密集（默认: 0.02）",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=10.0,
        help="判断格点是否在可信地面附近的最大距离 (m)（默认: 1.2）",
    )
    parser.add_argument(
        "--ground-z-offset",
        type=float,
        default=-0.3,
        help="地面平面整体 Z 向偏移 (m)，负值向下（默认: -0.3，即下移 30cm）",
    )

    return parser.parse_args(argv)


def upsample_ground_with_open3d(
    input_pcd_path: str,
    output_pcd_path: str,
    grid_step: float = 0.02,  # 其他 py 调用时，使用该默认值
    max_gap: float = 1.2,  # 其他 py 调用时，使用该默认值
    ground_z_offset: float = -0.9,  # 其他 py 调用时，使用该默认值；地面平面 Z 向偏移 (m)，负值向下
) -> None:
    """
    重点「补地面点」的上采样方案（基于 Open3D），更贴合路径规划需求。

    默认参数说明：
      - grid_step、max_gap 的默认值（上方的函数签名）：仅在「其他 Python 文件
        直接调用本函数且不传 grid_step/max_gap」时生效。
      - 终端通过「python resmaple.py」调用时，实际使用的是 parse_args() 里
        的默认值（与上述保持一致则行为一致）。

    思路：
      1. 用 RANSAC 从全局点云里分割出主地面平面（假设近似水平）；
      2. 在地面点的 XY 包围盒内，按 grid_step 生成规则网格；
      3. 若某网格中心距离最近地面点 < max_gap，则认为该处是可信地面，
         用平面方程计算其 Z，高密度补点；
      4. 将这些「虚拟地面点」与原始点云（包含非地面 + 原始地面）合并输出。
    """
    try:
        import open3d as o3d
    except ImportError as e:
        raise SystemExit(
            "导入 open3d 失败，请先安装：\n"
            "  pip install open3d\n"
            "或使用 conda:\n"
            "  conda install -c conda-forge open3d\n"
            f"原始错误: {e}"
        )

    if not os.path.exists(input_pcd_path):
        raise SystemExit(f"输入点云不存在: {input_pcd_path}")

    # 读取点云
    pcd = o3d.io.read_point_cloud(input_pcd_path)
    if len(pcd.points) == 0:
        raise SystemExit(f"加载点云失败或点数为 0: {input_pcd_path}")

    print(f"读取点云: {input_pcd_path}, 点数: {len(pcd.points)}")

    # 先做一次体素下采样，便于稳健拟合地面
    print("进行体素下采样用于地面拟合 ...")
    pcd_ds = pcd.voxel_down_sample(voxel_size=grid_step * 1.5)

    print("RANSAC 拟合地面平面 ...")
    plane_model, inliers = pcd_ds.segment_plane(
        distance_threshold=0.1,
        ransac_n=3,
        num_iterations=1000,
    )
    a, b, c, d = plane_model
    print(f"拟合到平面: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # 要求法向量基本「竖直」，否则认为不是地面
    normal = np.array([a, b, c], dtype=float)
    normal /= np.linalg.norm(normal) + 1e-8
    if abs(normal[2]) < 0.7:
        print("警告: 拟合到的平面法向量不够接近垂直方向，"
              "可能不是地面，请检查点云或参数。")

    ground_ds = pcd_ds.select_by_index(inliers)
    non_ground = pcd  # 输出时仍保留全部原始点

    if len(ground_ds.points) == 0:
        raise SystemExit("地面内点数量为 0，无法进行地面补点。")

    ground_pts = np.asarray(ground_ds.points)
    min_x, min_y = ground_pts[:, 0].min(), ground_pts[:, 1].min()
    max_x, max_y = ground_pts[:, 0].max(), ground_pts[:, 1].max()
    print(f"地面 XY 范围: x ∈ [{min_x:.2f}, {max_x:.2f}], "
          f"y ∈ [{min_y:.2f}, {max_y:.2f}]")

    # 在 XY 平面上生成规则网格
    xs = np.arange(min_x, max_x + grid_step, grid_step)
    ys = np.arange(min_y, max_y + grid_step, grid_step)
    xx, yy = np.meshgrid(xs, ys)
    grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # 使用 KDTree 判定每个格点是否在「可可信地面」附近
    print("构建地面 KDTree ...")
    kdtree = o3d.geometry.KDTreeFlann(ground_ds)

    dense_points = []
    print("生成虚拟地面格点并过滤过远区域 ...")
    for x, y in grid_xy:
        # 查询最近地面点
        [_, _, dists] = kdtree.search_knn_vector_3d([x, y, 0.0], 1)
        if len(dists) == 0:
            continue
        dist = np.sqrt(dists[0])
        if dist > max_gap:
            # 与最近地面点相距太远，认为该格点超出可信区域
            continue

        # 用平面方程求该处的 z: a x + b y + c z + d = 0；再叠加 ground_z_offset
        if abs(c) < 1e-6:
            continue
        z = -(a * x + b * y + d) / c + ground_z_offset
        # 在理论位置附近加随机扰动：xy 在 ±grid_step 内，z 在 ±10cm（总高 20cm）矩形框内
        x_rand = x + np.random.uniform(-grid_step, grid_step)
        y_rand = y + np.random.uniform(-grid_step, grid_step)
        z_rand = z + np.random.uniform(-0.1, 0.1)
        dense_points.append([x_rand, y_rand, z_rand])

    dense_points = np.asarray(dense_points, dtype=np.float64)
    print(f"生成虚拟地面点数量: {len(dense_points)}")

    dense_ground = o3d.geometry.PointCloud()
    dense_ground.points = o3d.utility.Vector3dVector(dense_points)

    # 合并原始点云 + 虚拟地面点
    print("合并原始点云与补点地面 ...")
    merged = non_ground + dense_ground

    if not o3d.io.write_point_cloud(output_pcd_path, merged, write_ascii=False):
        raise SystemExit(f"保存点云失败: {output_pcd_path}")

    print(f"上采样完成，原始点数: {len(pcd.points)}, "
          f"补点后总点数: {len(merged.points)}")
    print(f"已保存补地面点云到: {output_pcd_path}")


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)

    input_pcd = args.input
    output_pcd = args.output

    # 如果给的是相对路径，以当前脚本所在目录为基准
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(input_pcd):
        input_pcd = os.path.join(script_dir, input_pcd)
    if not os.path.isabs(output_pcd):
        output_pcd = os.path.join(script_dir, output_pcd)

    print(f"输入点云: {input_pcd}")
    print(f"输出点云: {output_pcd}")

    upsample_ground_with_open3d(
        input_pcd_path=input_pcd,
        output_pcd_path=output_pcd,
        grid_step=args.grid_step,
        max_gap=args.max_gap,
        ground_z_offset=args.ground_z_offset,
    )


if __name__ == "__main__":
    main(sys.argv[1:])


