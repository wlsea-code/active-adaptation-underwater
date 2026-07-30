# BlueROV 显式控制器：数学与物理原理

本文档对应以下实现：

- `active_adaptation/control/trajectory.py`
- `active_adaptation/control/bluerov_controller.py`
- `active_adaptation/control/thruster.py`

目标是说明代码实际实现的控制链，而不是给出一套与代码无关的通用水下机器人理论。完整数据流为：

```text
目标位姿
  -> 最小加加速度参考轨迹
  -> 位姿 PD（目标加速度）
  -> 刚体/附加质量模型（目标 wrench）
  -> 有界加权推力分配（各推进器目标推力）
  -> 推力/RPM/油门逆模型（各推进器油门）
```

油门之后的推进器一阶滞后、水动力、浮力以及向 Isaac 写力由
`active_adaptation/envs/robots/underwater.py` 实现，不属于本文三个控制文件，但会在最后说明接口边界。

## 1. 符号、坐标系与约定

代码中的后缀约定：

- `_w`：世界坐标系（world frame）。
- `_b`：BlueROV 机体坐标系（body frame）。
- `_wb`：代码中的机体姿态四元数，四元数顺序为 `(w, x, y, z)`。
- `quat_rotate_inverse(q_wb, x_w)`：把世界系向量旋转到机体系，可记为
  $R_{bw}x_w$。

主要状态和参考量：

| 符号 | 代码字段 | 含义 |
|---|---|---|
| $p_w$ | `position_w` | 当前世界系位置 |
| $q_{wb}$ | `orientation_wb` | 当前姿态四元数 |
| $v_w$ | `linear_velocity_w` | 当前世界系线速度 |
| $\omega_b$ | `angular_velocity_b` | 当前机体系角速度 |
| $p_{d,w}$ | `target_position_w` | 目标世界系位置 |
| $q_{d,wb}$ | `target_orientation_wb` | 目标姿态 |
| $v_{d,w}$ | `target_linear_velocity_w` | 目标世界系线速度 |
| $\omega_{d,b}$ | `target_angular_velocity_b` | 目标机体系角速度 |

六维加速度和 wrench 的排列固定为：

$$
\dot\nu_b =
\begin{bmatrix}
a_b \\
\alpha_b
\end{bmatrix},
\qquad
\tau_b =
\begin{bmatrix}
F_x & F_y & F_z & M_x & M_y & M_z
\end{bmatrix}^{T}.
$$

其中 $a_b$ 是机体系线加速度，$\alpha_b$ 是机体系角加速度。

## 2. 最小加加速度参考轨迹

对应实现：

- `trajectory.PoseReference`
- `trajectory.MinimumJerkPoseTrajectory`

### 2.1 平移五次多项式

每个位置轴使用五次多项式：

$$
p(t) = a_0 + a_1t + a_2t^2 + a_3t^3 + a_4t^4 + a_5t^5.
$$

其速度和加速度为：

$$
v(t) = a_1 + 2a_2t + 3a_3t^2 + 4a_4t^3 + 5a_5t^4,
$$

$$
a(t) = 2a_2 + 6a_3t + 12a_4t^2 + 20a_5t^3.
$$

代码用 `_coefficients[..., 6, 3]` 同时保存 XYZ 三个轴的六组系数，`sample()` 使用 `torch.einsum` 批量计算位置、速度和加速度。

### 2.2 重新规划时的边界条件

调用 `retarget()` 时，轨迹从当前参考状态继续规划，而不是从机器人实测状态突然重启。起点条件是：

$$
a_0 = p(0),\qquad
a_1 = v(0),\qquad
a_2 = \frac{1}{2}a(0).
$$

终点 $t=T$ 的约束是：

$$
p(T)=p_g,\qquad v(T)=0,\qquad a(T)=0.
$$

因此剩余的 $a_3,a_4,a_5$ 通过下面的线性系统求解：

$$
\begin{bmatrix}
T^3 & T^4 & T^5 \\
3T^2 & 4T^3 & 5T^4 \\
6T & 12T^2 & 20T^3
\end{bmatrix}
\begin{bmatrix}
a_3 \\ a_4 \\ a_5
\end{bmatrix}
=
\begin{bmatrix}
p_g-(a_0+a_1T+a_2T^2) \\
-(a_1+2a_2T) \\
-2a_2
\end{bmatrix}.
$$

这保证平移参考在重新规划点保持位置、速度和加速度连续，即达到 $C^2$ 连续。五次多项式也对应经典 minimum-jerk 边界形式。

### 2.3 姿态插值

姿态没有使用多项式直接插值四元数，而是使用球面线性插值（SLERP）：

$$
q_d(t)=\operatorname{SLERP}(q_0,q_g,h(u)),
\qquad u=\frac{t}{T}.
$$

插值参数使用五次平滑函数：

$$
h(u)=10u^3-15u^4+6u^5.
$$

它满足：

$$
h(0)=0,\quad h(1)=1,\quad
h'(0)=h'(1)=0,\quad
h''(0)=h''(1)=0.
$$

因此姿态插值比例在起点和终点平滑起停。

实现限制：`PoseReference` 当前只保存平移速度和加速度，没有从 SLERP 计算目标角速度与角加速度。当前 `PoseReferenceCommand` 将角速度参考和角加速度前馈设为零。

## 3. 位姿 PD：误差到目标加速度

对应实现：

- `bluerov_controller.PoseControllerCfg`
- `bluerov_controller.PoseAccelerationController`

### 3.1 平移控制

世界系位置和速度误差定义为：

$$
e_{p,w}=p_{d,w}-p_w,
\qquad
e_{v,w}=v_{d,w}-v_w.
$$

世界系目标线加速度为：

$$
a_{cmd,w}
=K_p^p\odot e_{p,w}
+K_d^p\odot e_{v,w}
+a_{ff,w}.
$$

$\odot$ 表示逐分量乘法。代码中的增益不是完整矩阵，而是 XYZ 三个独立增益。

随后进行逐分量限幅：

$$
a_{cmd,w,i}
\leftarrow
\operatorname{clip}
\left(a_{cmd,w,i},-a_{max,i},a_{max,i}\right).
$$

最后把目标线加速度转换到机体系：

$$
a_{cmd,b}=R_{bw}(q_{wb})a_{cmd,w}.
$$

### 3.2 姿态控制

姿态误差四元数为：

$$
q_{e,b}=q_{wb}^{-1}\otimes q_{d,wb}.
$$

$\otimes$ 表示四元数乘法。代码通过 `axis_angle_from_quat()` 把误差四元数转换成机体系轴角误差向量：

$$
e_{R,b}=\operatorname{Log}_{SO(3)}(q_{e,b}).
$$

角速度误差为：

$$
e_{\omega,b}=\omega_{d,b}-\omega_b.
$$

目标角加速度为：

$$
\alpha_{cmd,b}
=K_p^R\odot e_{R,b}
+K_d^R\odot e_{\omega,b}
+\alpha_{ff,b}.
$$

同样进行逐分量限幅：

$$
\alpha_{cmd,b,i}
\leftarrow
\operatorname{clip}
\left(\alpha_{cmd,b,i},-\alpha_{max,i},\alpha_{max,i}\right).
$$

### 3.3 该控制器的性质

这是带前馈和加速度限幅的 PD 控制器，不包含：

- 积分项；
- 在线参数辨识；
- 抗积分饱和（因为本身没有积分器）；
- 严格的非线性水下机器人全动力学反馈线性化。

稳态外扰如果没有在 `external_wrench_b` 中补偿，可能产生稳态误差。

## 4. 目标加速度到目标 wrench

对应实现：

- `bluerov_controller.RigidBodyWrenchController`

定义目标六维加速度：

$$
\dot\nu_{cmd,b}=
\begin{bmatrix}
a_{cmd,b} \\
\alpha_{cmd,b}
\end{bmatrix}.
$$

刚体质量矩阵为：

$$
M_{RB}=
\begin{bmatrix}
mI_3 & 0 \\
0 & J_b
\end{bmatrix},
$$

其中 $m$ 是总质量，$J_b$ 是机体系转动惯量。若启用附加质量，则再加入 $M_A\in\mathbb{R}^{6\times6}$：

$$
M=M_{RB}+M_A.
$$

代码首先计算惯性 wrench：

$$
\tau_{inertia,b}=M\dot\nu_{cmd,b}.
$$

随后只对角运动加入欧拉刚体陀螺项：

$$
\tau_{gyro,b}=
\begin{bmatrix}
0_3 \\
\omega_b\times(J_b\omega_b)
\end{bmatrix}.
$$

如果调用方给出已知的非推进器外部 wrench $\tau_{ext,b}$，控制器使用约定：

$$
M\dot\nu_b=\tau_{thruster,b}+\tau_{ext,b},
$$

所以目标推进器 wrench 为：

$$
\boxed{
\tau_{d,b}
=M\dot\nu_{cmd,b}
+\tau_{gyro,b}
-\tau_{ext,b}
}.
$$

重要实现约束：若 $M_A$ 已加入质量矩阵，就不能再把同一个附加质量惯性 wrench 放进 `external_wrench_b`，否则会重复补偿。代码注释明确要求调用方避免该问题。

该文件没有完整实现经典水下机器人模型

$$
M\dot\nu+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau,
$$

而是只在控制器内实现质量/附加质量惯性项、刚体陀螺项和调用方显式传入的外部 wrench。水动力阻尼、浮力等是否补偿由 Action 层组装 `external_wrench_b` 决定。

## 5. Wrench 通道掩码

对应实现：

- `bluerov_controller.BlueROVExplicitController.compute()`
- 配置项 `wrench_command_mask`

在进入推力分配前，代码可对六维目标 wrench 逐项屏蔽：

$$
\tau_{masked,b}=m_w\odot\tau_{d,b}.
$$

例如当前 BlueROV 配置使用：

```yaml
wrench_command_mask: [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
```

对应：

$$
[F_x,F_y,F_z,M_x,M_y,M_z]
\longrightarrow
[F_x,F_y,F_z,M_x,0,M_z].
$$

即不命令 pitch 力矩 $M_y$。掩码发生在推力分配之前，所以被屏蔽的通道不参与目标求解。

## 6. 推进器几何与分配矩阵

对应实现：

- `thruster.ThrusterAllocator`

设 BlueROV 有 $N$ 个推进器。第 $i$ 个推进器在机体系中的位置为 $r_i\in\mathbb{R}^3$，单位推力方向为 $d_i\in\mathbb{R}^3$，标量推力为 $f_i$。

它产生的机体系力为：

$$
F_i=d_if_i.
$$

由安装位置产生的力矩为：

$$
M_{arm,i}=r_i\times d_i f_i.
$$

代码还允许配置单位推力对应的反作用力矩系数 $k_i$：

$$
M_{reaction,i}=k_id_if_i.
$$

因此第 $i$ 个推进器对应的六维分配列向量为：

$$
b_i=
\begin{bmatrix}
d_i \\
r_i\times d_i+k_id_i
\end{bmatrix}.
$$

将所有列拼接得到分配矩阵：

$$
B=\begin{bmatrix}b_1&b_2&\cdots&b_N\end{bmatrix}
\in\mathbb{R}^{6\times N}.
$$

推进器推力向量 $f=[f_1,\ldots,f_N]^T$ 产生的 wrench 为：

$$
\tau_{achieved,b}=Bf.
$$

`allocator.rank` 返回 $\operatorname{rank}(B)$；`singular_values` 返回 $B$ 的奇异值。它们可以用来判断推进器布局能独立控制多少个 wrench 方向，以及布局是否接近奇异。

## 7. 有界加权阻尼最小二乘分配

对应实现：

- `ThrusterAllocator.allocate()`
- `ThrusterAllocator._allocate_one()`

理想目标是找到 $f$ 使 $Bf\approx\tau_d$，同时满足每个推进器的推力边界：

$$
f_{min,i}\le f_i\le f_{max,i}.
$$

代码采用加权阻尼最小二乘思想：

$$
\min_f
\left\|W(Bf-\tau_d)\right\|_2^2
+\lambda\left\|f\right\|_2^2,
$$

其中：

- $W=\operatorname{diag}(w_1,\ldots,w_6)$ 来自 `wrench_weights`；
- $\lambda$ 来自 `damping`；
- 较大的 $w_j$ 表示更重视第 $j$ 个 wrench 分量；
- 阻尼项缓解矩阵病态，并抑制过大的推力解。

### 7.1 无边界时的解

令：

$$
B_w=WB,\qquad \tau_w=W\tau_d,
$$

则阻尼正规方程解为：

$$
f=
\left(B_w^TB_w+\lambda I\right)^{-1}
B_w^T\tau_w.
$$

代码没有显式求逆，而是使用 `torch.linalg.solve()` 求解线性系统，数值上更合适。

### 7.2 Active-set 饱和处理

推进器有边界，代码使用简化 active-set 过程：

1. 初始时所有推进器都是自由变量。
2. 在自由推进器子矩阵上求阻尼最小二乘解。
3. 将越界推力截断到上下限，并把这些推进器标记为固定。
4. 从目标 wrench 中减去固定推进器已经贡献的 wrench。
5. 用剩余自由推进器重新分配剩余 wrench。
6. 直到无越界、无自由推进器，或达到最多 $N+1$ 次迭代。

若固定集合为 $S$、自由集合为 $F$，剩余目标为：

$$
\tau_{res}=\tau_d-B_Sf_S.
$$

自由变量求解：

$$
f_F=
\left((WB_F)^T(WB_F)+\lambda I\right)^{-1}
(WB_F)^TW\tau_{res}.
$$

最终诊断量为：

$$
\tau_{achieved}=Bf,
\qquad
\tau_{residual}=\tau_d-\tau_{achieved}.
$$

残差大说明目标 wrench 超出推进器布局或推力极限的实现能力。

当前实现逐环境调用 `_allocate_one()`，因此并行环境数量较大时会成为性能瓶颈。

## 8. 油门、RPM 与推力模型

对应实现：

- `thruster.ThrusterModelCfg`
- `thruster.BlueROVThrusterModel`

该模型是双向可用的标定模型：

```text
正向（仿真执行）：throttle -> RPM -> thrust
反向（显式控制）：thrust -> RPM -> throttle
```

### 8.1 油门到 RPM

设归一化油门为 $u\in[-1,1]$，死区为 $\delta$。代码先执行：

$$
u\leftarrow\operatorname{clip}(u,-1,1).
$$

RPM 的分段仿射模型为：

$$
n(u)=
\begin{cases}
s_+u+b_+, & u>\delta,\\
0, & |u|\le\delta,\\
s_-u+b_-, & u<-\delta.
\end{cases}
$$

最后执行：

$$
n\leftarrow\operatorname{clip}(n,n_{min},n_{max}).
$$

对应配置字段：

- `throttle_deadband`：$\delta$；
- `positive_rpm_slope`、`positive_rpm_intercept`：$s_+,b_+$；
- `negative_rpm_slope`、`negative_rpm_intercept`：$s_-,b_-$；
- `min_rpm`、`max_rpm`：RPM 上下限。

### 8.2 RPM 到推力

正反转分别使用二次标定曲线。若系数记为 $(a_+,b_+,c_+)$ 和 $(a_-,b_-,c_-)$，推力为：

$$
T(n)=
\begin{cases}
s_T(a_+n^2+b_+n+c_+), & n>0,\\
0, & n=0,\\
s_T(a_-n^2+b_-n+c_-), & n<0.
\end{cases}
$$

$s_T$ 对应 `thrust_scale`。`ThrusterModelCfg.__post_init__()` 会检查：

- RPM 范围跨过零；
- 死区属于 $[0,1)$；
- 两个分支在有效 RPM 区间单调递增；
- 正转推力保持为正，反转推力保持为负；
- 标定系数有限且迭代次数为正。

### 8.3 推力到 RPM：二分反演

二次曲线虽然可解析求根，但代码选择在单调分支上做二分反演：

1. 将目标推力截断到可实现的 $[T_{min},T_{max}]$。
2. 根据推力正负选择正转或反转 RPM 区间。
3. 迭代 `inversion_iterations` 次，每次取中点。
4. 用 `rpm_to_thrust(mid)` 判断应该保留区间的哪一半。
5. 比较活动分支解与零 RPM 的误差；死区附近若零更接近目标，则返回零。

二分法的区间误差大约随迭代次数按 $2^{-K}$ 缩小，其中 $K$ 是 `inversion_iterations`。

### 8.4 RPM 到油门

在非零分支上反解仿射映射：

$$
u(n)=
\begin{cases}
\dfrac{n-b_+}{s_+}, & n>0,\\
0, & n=0,\\
\dfrac{n-b_-}{s_-}, & n<0.
\end{cases}
$$

由于 $|u|\le\delta$ 会被正向模型解释为零 RPM，代码使用 `torch.nextafter()` 把非零油门推到死区边界之外，保证反演得到的非零 RPM 再经过正向模型时不会重新落回死区。最后仍将油门截断到 $[-1,1]$。

## 9. 总控制器组合

对应实现：

- `bluerov_controller.BlueROVExplicitController`
- `bluerov_controller.ExplicitControlOutput`

`compute()` 的严格执行顺序为：

1. `PoseAccelerationController.compute()`：

   $$
   (p,q,v,\omega,p_d,q_d,v_d,\omega_d)
   \longrightarrow
   (a_{cmd,b},\alpha_{cmd,b}).
   $$

2. `RigidBodyWrenchController.compute()`：

   $$
   (a_{cmd,b},\alpha_{cmd,b},\omega_b,\tau_{ext,b})
   \longrightarrow
   \tau_{d,b}.
   $$

3. 应用 `wrench_command_mask`：

   $$
   \tau_{masked,b}=m_w\odot\tau_{d,b}.
   $$

4. `ThrusterAllocator.allocate()`：

   $$
   \tau_{masked,b}\longrightarrow f.
   $$

5. 推进器逆模型：

   $$
   f\longrightarrow n\longrightarrow u.
   $$

6. 再经过正向模型计算量化/死区之后真正可实现的推力：

   $$
   f_{realized}=T(n).
   $$

7. 计算可实现 wrench 和执行器残差：

   $$
   \tau_{realized,b}=Bf_{realized},
   $$

   $$
   \tau_{actuation\_residual,b}
   =\tau_{masked,b}-\tau_{realized,b}.
   $$

`ExplicitControlOutput` 保留上述中间量，便于画图、记录日志和定位控制误差来自哪一层。

## 10. 与实际仿真执行层的边界

控制文件计算出的 `throttle` 是目标油门。之后 `UnderwaterThrottle` 把它写入 `UnderwaterRobotData.throttle_cmd`，`UnderwaterRobot.write_data_to_sim()` 才执行实际执行器和流体仿真。

### 10.1 推进器一阶滞后

Robot 层使用离散一阶模型：

$$
\alpha_i=\exp\left(-\frac{\Delta t}{\tau_i}\right),
$$

$$
u_{i,k}=\alpha_i u_{i,k-1}+(1-\alpha_i)u_{cmd,i,k}.
$$

因此 `ExplicitControlOutput.realized_wrench_b` 是根据静态推进器模型计算的即时可实现值，不包含 Robot 层的一阶响应滞后。

### 10.2 单推进器力常数缩放

Robot 层还会按每个推进器的 `force_constants` 相对 `nominal_force_constant` 缩放推力：

$$
T_{sim,i}
=\frac{k_i}{k_{nom}}T(n_i).
$$

因此若各推进器 $k_i$ 不相等，控制器内的 `realized_wrench_b` 与 Isaac 最终施加的 wrench 之间仍可能存在差异。

### 10.3 水下机器人动力学

Robot 层另行计算阻尼、附加质量、科氏项和浮力。控制器是否补偿这些项由 Action 层传入的 `external_wrench_b` 决定。调试时应区分：

- 控制器期望的 `desired_wrench_b`；
- 推力分配后的 `allocation.achieved_wrench_b`；
- 推进器静态模型后的 `realized_wrench_b`；
- Robot 层考虑滞后、个体缩放和水动力后实际写入 Isaac 的 wrench。

## 11. 配置字段到公式的映射

当前配置位于 `cfg/task/UW/BlueROVExplicit.yaml`；共享的物理与观测配置位于
`cfg/task/UW/BlueROVBase.yaml`。

| 配置字段 | 数学含义 |
|---|---|
| `pose_controller.position_kp` | $K_p^p$ |
| `pose_controller.position_kd` | $K_d^p$ |
| `pose_controller.attitude_kp` | $K_p^R$ |
| `pose_controller.attitude_kd` | $K_d^R$ |
| `pose_controller.max_linear_acceleration` | $a_{max}$ |
| `pose_controller.max_angular_acceleration` | $\alpha_{max}$ |
| `wrench_weights` | $W$ 的六个对角元素 |
| `wrench_command_mask` | $m_w$ |
| `reaction_torque_per_thrust` | 各推进器 $k_i$ |
| `allocation_damping` | 阻尼最小二乘中的 $\lambda$ |
| `use_added_mass` | 是否将 $M_A$ 加入控制器质量矩阵 |
| `compensate_hydrodynamics` | Action 是否把已估计水动力加入外部 wrench 补偿 |
| `thruster_model.throttle_deadband` | 油门死区 $\delta$ |
| `thruster_model.*rpm_slope/intercept` | 油门到 RPM 的分段仿射参数 |
| `thruster_model.*thrust_coefficients` | RPM 到推力的正反二次曲线参数 |
| `thruster_model.inversion_iterations` | 推力逆模型二分次数 $K$ |

## 12. 调试建议

按照控制链逐层排查，不要只观察最终轨迹：

1. **轨迹层**：检查 `PoseReference.position_w`、`linear_velocity_w`、`linear_acceleration_w` 是否连续，是否在 $T$ 秒到达目标。
2. **位姿层**：检查位置误差、姿态轴角误差，以及加速度是否长期撞到限幅。
3. **Wrench 层**：检查 `desired_wrench_b`，确认外部 wrench 的符号和坐标系。
4. **可控性层**：检查 `allocator.rank` 和奇异值；rank 不足意味着某些 wrench 方向无法独立实现。
5. **饱和层**：检查推进器推力是否频繁达到上下限，以及 `allocation.residual_wrench_b`。
6. **标定层**：检查推力、RPM、油门是否处于死区或饱和区。
7. **执行层**：比较目标油门与 Robot 层实际油门，判断一阶滞后是否主导误差。

项目现有诊断量包括：

- `controller/allocator_rank`
- `controller/residual_wrench`
- `controller/max_abs_throttle`

## 13. 当前实现的主要限制

- 位姿控制器是 PD，没有积分项，未补偿的恒定扰动可能导致稳态误差。
- 姿态轨迹当前不提供目标角速度和角加速度前馈。
- 控制器内部不是完整的六自由度水下机器人动力学模型；部分流体项位于 Robot 层。
- 推力分配器逐环境求解，尚未完全向量化。
- 推力边界 active-set 是工程化迭代算法，不是通用二次规划求解器。
- 控制器内的静态 `realized_wrench_b` 不包含推进器时间常数和每个推进器的个体力常数缩放。
- 坐标系和外部 wrench 符号必须与 `UnderwaterRobot` 保持一致，否则补偿项会放大而不是抵消扰动。

## 14. 代码导航

| 控制阶段 | 类/方法 |
|---|---|
| 平滑参考 | `MinimumJerkPoseTrajectory.sample/retarget/advance` |
| 位姿误差到加速度 | `PoseAccelerationController.compute` |
| 加速度到 wrench | `RigidBodyWrenchController.compute` |
| 推力分配 | `ThrusterAllocator.allocate/_allocate_one` |
| 油门到 RPM | `BlueROVThrusterModel.throttle_to_rpm` |
| RPM 到推力 | `BlueROVThrusterModel.rpm_to_thrust` |
| 推力反演 | `BlueROVThrusterModel.thrust_to_rpm` |
| RPM 反演为油门 | `BlueROVThrusterModel.rpm_to_throttle` |
| 总流程 | `BlueROVExplicitController.compute` |
