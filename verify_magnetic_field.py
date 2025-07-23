#!/usr/bin/env python3
"""
磁场计算验证脚本
比较理论公式计算和magpylib计算结果
"""
import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt

def theoretical_dipole_field(r, m):
    """
    计算磁偶极子场（理论公式）
    
    Args:
        r: 位置向量 [x, y, z] (m)
        m: 磁矩向量 [mx, my, mz] (A·m²)
    
    Returns:
        B: 磁感应强度向量 [Bx, By, Bz] (T)
    """
    r = np.array(r)
    m = np.array(m)
    
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-10:  # 避免除零
        return np.zeros(3)
        
    r_hat = r / r_norm
    
    # μ0/4π = 1e-7
    mu0_4pi = 1e-7
    
    # B = (μ0/4π) * [3(m·r̂)r̂ - m]/r³
    m_dot_r = np.dot(m, r_hat)
    B = mu0_4pi * (3 * m_dot_r * r_hat - m) / (r_norm**3)
    
    return B

def magpylib_cylinder_field(pos, mag_pos, magnetization, radius, height):
    """
    使用magpylib计算圆柱形磁铁的磁场
    
    Args:
        pos: 观测点位置 [x, y, z] (m)
        mag_pos: 磁铁位置 [x, y, z] (m)
        magnetization: 磁化强度向量 [Mx, My, Mz] (A/m)
        radius: 磁铁半径 (m)
        height: 磁铁高度 (m)
    
    Returns:
        B: 磁感应强度向量 [Bx, By, Bz] (T)
    """
    mag = magpy.magnet.Cylinder(
        magnetization=magnetization,
        dimension=(radius, height)
    ).move(mag_pos)
    
    return mag.getB(pos)

def main():
    # 测试点设置
    test_points = [
        [0, 0, 0.15],      # 正上方
        [0.05, -0.05, 0.15],  # 斜上方
        [0.05, 0, 0],      # 水平面
    ]
    
    # 磁铁参数（从mag_field_simulation.py）
    magnet_config = {
        'position': [-0.08, 0.2, -0.3],
        'moment_direction': [0, 0, 1],
        'radius': 0.01,  # 10mm
        'height': 0.005,  # 5mm
        'magnetization': 1e6  # A/m
    }
    
    # 计算磁矩
    volume = np.pi * magnet_config['radius']**2 * magnet_config['height']
    m = np.array(magnet_config['moment_direction']) * magnet_config['magnetization'] * volume
    
    print(f"磁铁体积: {volume*1e9:.2f} mm³")
    print(f"磁矩: {np.linalg.norm(m):.2e} A·m²")
    print(f"磁矩向量: [{m[0]:.2e}, {m[1]:.2e}, {m[2]:.2e}] A·m²")
    print("\n" + "="*50 + "\n")
    
    for point in test_points:
        print(f"测试点: {point}")
        
        # 计算相对位置向量
        r = np.array(point) - np.array(magnet_config['position'])
        print(f"相对位置向量: {r}")
        print(f"距离: {np.linalg.norm(r):.3f} m")
        
        # 理论公式计算
        B_theory = theoretical_dipole_field(r, m)
        
        # magpylib计算
        B_magpy = magpylib_cylinder_field(
            point,
            magnet_config['position'],
            np.array(magnet_config['moment_direction']) * magnet_config['magnetization'],
            magnet_config['radius'],
            magnet_config['height']
        )
        
        # 转换为μT并打印结果
        B_theory_uT = B_theory * 1e6
        B_magpy_uT = B_magpy * 1e6
        
        print("\n理论公式计算结果:")
        print(f"Bx = {B_theory_uT[0]:.3f} μT")
        print(f"By = {B_theory_uT[1]:.3f} μT")
        print(f"Bz = {B_theory_uT[2]:.3f} μT")
        print(f"|B| = {np.linalg.norm(B_theory_uT):.3f} μT")
        
        print("\nmagpylib计算结果:")
        print(f"Bx = {B_magpy_uT[0]:.3f} μT")
        print(f"By = {B_magpy_uT[1]:.3f} μT")
        print(f"Bz = {B_magpy_uT[2]:.3f} μT")
        print(f"|B| = {np.linalg.norm(B_magpy_uT):.3f} μT")
        
        # 计算相对误差
        rel_error = np.linalg.norm(B_theory - B_magpy) / np.linalg.norm(B_magpy)
        print(f"\n相对误差: {rel_error*100:.2f}%")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 