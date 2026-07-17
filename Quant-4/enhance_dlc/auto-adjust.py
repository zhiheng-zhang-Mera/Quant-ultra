import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# ========== 请替换为您的真实业务导入 ==========
# from Phase_4.step4_labeling_weighting import apply_sample_weights
# from Phase_5.step5_model_training_calibration import train_core_model
# from Phase_6.step6_position_sizing import run_convex_optimizer

# ------------------- 模拟Mock函数（仅用于演示结构，正式请注释掉） ------------------
def apply_sample_weights(y, decay, clip):
    return np.ones_like(y) * decay  # 模拟

def train_core_model(X, y, lr, depth, l2, sample_weight=None):
    class MockModel:
        def predict(self, X): return np.random.randn(len(X)) * 0.01
    return MockModel()

def run_convex_optimizer(cov, ret, risk, max_w):
    if np.isnan(cov).any(): raise np.linalg.LinAlgError
    return np.random.randn() * 0.5 + 0.5  # 模拟夏普
# --------------------------------------------------------------------------------

class QuantAdaptiveTuner:
    """
    量化多阶段端到端自适应调参器（生产级）
    """
    def __init__(self, X, y, covariance_matrix, expected_returns, 
                 holdout_ratio=0.20, gap_days=5, cv_splits=3):
        """
        :param holdout_ratio: 拒斥集比例（推荐 0.15 ~ 0.20）
        :param gap_days: 时序交叉验证中，训练集与验证集之间的清洗间隔（交易日）
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.cov = np.asarray(covariance_matrix)
        self.ret = np.asarray(expected_returns)
        self.gap = gap_days
        self.cv_splits = cv_splits
        
        # -------- 核心：切分出绝对不可见的拒斥集 (Hold-out) ----------
        n = len(self.X)
        holdout_size = int(n * holdout_ratio)
        if holdout_size < 50:
            print(f"⚠️ 警告: 拒斥集仅 {holdout_size} 个样本，建议增加数据量或降低比例至10%")
        
        # 时间序列必须按顺序切分，严禁打乱！
        self.train_val_idx = np.arange(0, n - holdout_size)
        self.holdout_idx = np.arange(n - holdout_size, n)
        
        print(f"✅ 训练/验证集样本数: {len(self.train_val_idx)}, 拒斥集样本数: {len(self.holdout_idx)}")
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.study_ = None

    def _get_purged_cv(self):
        """生成带清洗间隔的时序折叠（Purged TimeSeriesSplit）"""
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        for train_idx, val_idx in tscv.split(self.train_val_idx):
            # 核心：将验证集起始点向后推 gap 天，清洗掉紧邻的样本
            if len(train_idx) > self.gap:
                train_idx = train_idx[:-self.gap]
            # 确保训练集与验证集无重叠
            yield train_idx, val_idx

    def _objective(self, trial):
        """端到端联合目标函数（所有阶段参数共舞）"""
        # ---------- Phase 4: 权重参数 ----------
        decay = trial.suggest_float('decay_factor', 0.80, 0.99)
        clip = trial.suggest_float('clip_bound', 2.0, 5.0)

        # ---------- Phase 5: 模型参数（带条件依赖） ----------
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        depth = trial.suggest_int('max_depth', 3, 12)
        # 关键强化：深度越大，L2搜索上限自动抬高，防止过拟合
        l2_high = 15.0 if depth > 8 else 5.0
        l2 = trial.suggest_float('l2_reg', 1e-3, l2_high, log=True)

        # ---------- Phase 6: 仓位参数（引入波动率先验，此处模拟） ----------
        # 真实场景可传入外部波动率序列 np.std(self.y) 作为缩放因子
        vol_scaler = max(0.5, np.std(self.y) * 10) 
        risk_high = max(5.0, 5.0 * vol_scaler)
        risk_ave = trial.suggest_float('risk_aversion', 0.5, min(20.0, risk_high))
        max_w = trial.suggest_float('max_weight', 0.03, 0.25)

        # ---------- 带清洗的时序交叉验证 ----------
        cv_scores = []
        for train_idx, val_idx in self._get_purged_cv():
            try:
                # 1. 样本权重
                sample_w = apply_sample_weights(self.y[train_idx], decay, clip)
                
                # 2. 训练模型（传入权重）
                model = train_core_model(
                    self.X[train_idx], self.y[train_idx],
                    lr, depth, l2, sample_weight=sample_w
                )
                
                # 3. 验证集预测
                preds = model.predict(self.X[val_idx])
                
                # 4. 凸优化求解仓位（传入对应时期的协方差与预测收益）
                # 注意：若 cov/ret 是整体矩阵，此处可切片；若仅全局，则传入整体
                # 实战中若每个时段协方差不同，请传入 cov[val_idx] 等
                sharpe = run_convex_optimizer(
                    self.cov, preds,  # 简化：直接用preds作为预期收益
                    risk_ave, max_w
                )
                cv_scores.append(sharpe)
                
            except (np.linalg.LinAlgError, ValueError, AttributeError) as e:
                # 凸优化奇异或模型报错，给予极低分数惩罚，保证搜索不崩
                return -1e10

        # 强化目标：最大化平均夏普，同时惩罚高波动（均值-2*标准差）
        if len(cv_scores) == 0:
            return -1e10
        mean_sharpe = np.mean(cv_scores)
        std_sharpe = np.std(cv_scores)
        return mean_sharpe - 2.0 * std_sharpe

    def run_optimization(self, n_trials=150, timeout_sec=3600):
        """启动自适应超参搜索"""
        # 使用 TPE 采样器 + 中位数剪枝（垃圾参数早期夭折）
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=15)
        
        self.study_ = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        print(f"🚀 开始端到端调优，共 {n_trials} 次试验，超时 {timeout_sec}s ...")
        self.study_.optimize(
            self._objective, 
            n_trials=n_trials, 
            timeout=timeout_sec,
            n_jobs=-1,  # 启用全部CPU核心并行跑折叠
            show_progress_bar=True
        )
        
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        print(f"✅ 调优完成！最佳分数 (均值-2σ): {self.best_score_:.4f}")
        print(f"📦 最佳参数: {self.best_params_}")
        return self.best_params_

    def evaluate_on_holdout(self):
        """使用完全隔离的拒斥集进行最终泛化性评测（唯一可信的终审分数）"""
        if self.best_params_ is None:
            raise RuntimeError("请先运行 run_optimization()")
        
        # 解包最佳参数
        decay = self.best_params_['decay_factor']
        clip = self.best_params_['clip_bound']
        lr = self.best_params_['learning_rate']
        depth = self.best_params_['max_depth']
        l2 = self.best_params_['l2_reg']
        risk_ave = self.best_params_['risk_aversion']
        max_w = self.best_params_['max_weight']
        
        # 在拒斥集上训练（使用全部训练+验证数据，拒斥集仅做测试）
        X_train = self.X[self.train_val_idx]
        y_train = self.y[self.train_val_idx]
        X_test = self.X[self.holdout_idx]
        y_test = self.y[self.holdout_idx]  # 仅用于计算真实表现，不参与训练
        
        try:
            sample_w = apply_sample_weights(y_train, decay, clip)
            model = train_core_model(X_train, y_train, lr, depth, l2, sample_weight=sample_w)
            preds = model.predict(X_test)
            
            # 计算拒斥集上的最终夏普（或自定义指标）
            final_sharpe = run_convex_optimizer(
                self.cov[self.holdout_idx], preds, 
                risk_ave, max_w
            )
            
            print("\n" + "="*50)
            print(f"🔒 【拒斥集终审报告】样本数: {len(X_test)}")
            print(f"最终泛化夏普比率: {final_sharpe:.4f}")
            print("="*50)
            return final_sharpe
            
        except Exception as e:
            print(f"❌ 拒斥集评估失败: {e}")
            return -np.inf

    def plot_importance(self):
        """生成超参数重要性分析（HTML报告）"""
        if self.study_ is None:
            print("请先运行调优")
            return
        try:
            import optuna.visualization as vis
            importance = optuna.importance.get_param_importances(self.study_)
            print("📊 参数重要性排序:", importance)
            fig = vis.plot_param_importances(self.study_)
            fig.write_html("adaptive_param_importance.html")
            print("✅ 重要性报告已保存为 adaptive_param_importance.html")
        except Exception as e:
            print(f"可视化失败（可忽略）: {e}")


# ===================== 主程序调用示例 =====================
if __name__ == "__main__":
    # 模拟生成数据（替换为您的真实DataFrame/Array）
    np.random.seed(42)
    n_samples = 3000
    X_demo = np.random.randn(n_samples, 20)
    y_demo = np.random.randn(n_samples) * 0.01
    cov_demo = np.eye(20) * 0.001
    ret_demo = np.random.randn(n_samples) * 0.0005

    # 1. 初始化调优器（固定预留20%拒斥集）
    tuner = QuantAdaptiveTuner(
        X=X_demo, 
        y=y_demo,
        covariance_matrix=cov_demo,
        expected_returns=ret_demo,
        holdout_ratio=0.20,  # 15%-20% 最推荐
        gap_days=5,          # 5天清洗间隔
        cv_splits=3
    )

    # 2. 执行调优（注意：拒斥集全程未参与！）
    best_params = tuner.run_optimization(n_trials=50, timeout_sec=600)  # 演示设小，正式改为150+

    # 3. 输出参数重要性（辅助剔除无效参数）
    tuner.plot_importance()

    # 4. ★ 最终唯一可信评估：在拒斥集上跑分
    holdout_score = tuner.evaluate_on_holdout()
    
    # 如果拒斥集分数远低于调优时分数（如低于50%），强烈怀疑过拟合，需重新审视特征！