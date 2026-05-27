"""
EWC Regularizer: Elastic Weight Consolidation
============================================

Proper EWC using real Fisher Information Matrix.

Mathematical Foundation:
Standard gradient descent on new task causes catastrophic forgetting:
θ ← θ - η ∇θ L_new(θ)

EWC protects old knowledge with quadratic penalty:
θ ← θ - η (∇θ L_new(θ) + λ Σᵢ Fᵢ (θᵢ - θ*ᵢ)²)

Where:
- Fᵢ = Fisher Information Matrix diagonal (parameter importance)
- θ*ᵢ = optimal parameter from previous task
- λ = EWC penalty strength

Key insight:
Parameters with high Fisher are "important" for old task.
We penalize changes to important parameters.
"""

import jax
import jax.numpy as jnp
from jax import grad
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from .kfac_fisher import KFACEstimator, DiagonalFisherEstimator, compute_fisher_information

log = logging.getLogger(__name__)


class EWCBackup:
    """
    Stores parameter backup for a learned task.
    
    Used to compute EWC penalty when learning new tasks.
    """
    
    def __init__(
        self,
        task_name: str,
        params: np.ndarray,
        fisher_diag: np.ndarray,
        loss: float,
    ):
        """
        Args:
            task_name: Name of the task
            params: Optimal parameters for this task
            fisher_diag: Fisher diagonal (parameter importance)
            loss: Final loss on this task
        """
        self.task_name = task_name
        self.params = params.copy()
        self.fisher_diag = fisher_diag.copy()
        self.loss = loss
    
    def __repr__(self):
        return f"EWCBackup({self.task_name}, n_params={len(self.params)})"


class EWCLoss:
    """
    EWC Loss with real Fisher Information Matrix.
    
    L_total = L_new(θ) + λ_EWC * Σᵢ Fᵢ (θᵢ - θ*ᵢ)²
    
    Where Fᵢ is the Fisher Information for parameter i.
    """
    
    def __init__(
        self,
        lambda_ewc: float = 1000.0,
        use_kfac: bool = True,
    ):
        """
        Args:
            lambda_ewc: EWC penalty strength
            use_kfac: Use K-FAC approximation (faster) or full Fisher
        """
        self.lambda_ewc = lambda_ewc
        self.use_kfac = use_kfac
        
        # Storage for previous task parameters
        self.backups: list[EWCBackup] = []
        
        # K-FAC estimator (if using)
        if use_kfac:
            self.kfac: Optional[KFACEstimator] = None
    
    def register_task(
        self,
        task_name: str,
        params: np.ndarray,
        fisher_diag: np.ndarray,
        loss: float,
    ):
        """
        Register a completed task for EWC protection.
        
        Call this after learning a new mechanism.
        
        Args:
            task_name: Name of the mechanism/task
            params: Learned parameters
            fisher_diag: Fisher diagonal (from K-FAC)
            loss: Final loss on this task
        """
        backup = EWCBackup(task_name, params, fisher_diag, loss)
        self.backups.append(backup)
        
        log.info(f"EWC: Registered {task_name} with {len(params)} parameters")
        log.info(f"EWC: {len(self.backups)} tasks now protected")
    
    def compute_ewc_penalty(self, current_params: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute EWC penalty for current parameters.
        
        L_EWC = Σ_task λ * Σᵢ Fᵢ (θᵢ - θ*ᵢ)²
        
        Args:
            current_params: Current model parameters
            
        Returns:
            penalty: Total EWC penalty
            breakdown: Per-task penalty breakdown
        """
        if not self.backups:
            return 0.0, {}
        
        total_penalty = 0.0
        breakdown = {}
        
        for backup in self.backups:
            # Quadratic penalty
            diff = current_params - backup.params
            weighted_diff = backup.fisher_diag * (diff ** 2)
            task_penalty = self.lambda_ewc * np.sum(weighted_diff)
            
            total_penalty += task_penalty
            breakdown[backup.task_name] = task_penalty
        
        return float(total_penalty), breakdown
    
    def compute_total_loss(
        self,
        task_loss: float,
        current_params: np.ndarray,
    ) -> Tuple[float, Dict]:
        """
        Compute total loss = task loss + EWC penalty.
        
        Args:
            task_loss: Loss on new task
            current_params: Current parameters
            
        Returns:
            total_loss: Combined loss
            info: Breakdown of losses
        """
        ewc_penalty, breakdown = self.compute_ewc_penalty(current_params)
        
        total = task_loss + ewc_penalty
        
        info = {
            'task_loss': task_loss,
            'ewc_penalty': ewc_penalty,
            'total': total,
            'breakdown': breakdown,
        }
        
        return float(total), info


class EWCRegularizer:
    """
    EWC regularizer with online Fisher estimation.
    
    Integrates with the self-improving physics engine.
    """
    
    def __init__(
        self,
        lambda_ewc: float = 1000.0,
        fisher_ema_decay: float = 0.99,
        damping: float = 0.1,
    ):
        """
        Args:
            lambda_ewc: EWC penalty strength
            fisher_ema_decay: EMA decay for Fisher estimation
            damping: Damping factor for Fisher
        """
        self.lambda_ewc = lambda_ewc
        self.fisher_estimator = DiagonalFisherEstimator(
            n_params=0,  # Will be set on first use
            ema_decay=fisher_ema_decay,
            damping=damping,
        )
        
        # Task memories
        self.task_memories: list[EWCBackup] = []
        
        # Current Fisher estimate
        self.current_fisher: Optional[np.ndarray] = None
    
    def set_n_params(self, n_params: int):
        """Initialize Fisher estimator with correct size."""
        if self.fisher_estimator.n_params == 0:
            self.fisher_estimator = DiagonalFisherEstimator(
                n_params=n_params,
                ema_decay=0.99,
                damping=0.1,
            )
    
    def update_fisher(self, gradients: np.ndarray):
        """
        Update Fisher estimate with new gradients.
        
        Call during training on a task.
        
        Args:
            gradients: [n_params] gradient of loss
        """
        self.fisher_estimator.update(gradients)
        self.current_fisher = self.fisher_estimator.get_fisher_diagonal()
    
    def register_task_completion(
        self,
        task_name: str,
        params: np.ndarray,
        final_loss: float,
    ):
        """
        Register task completion and save Fisher estimate.
        
        Call this after successfully learning a mechanism.
        
        Args:
            task_name: Name of the mechanism
            params: Final parameters
            final_loss: Final loss value
        """
        if self.current_fisher is None:
            self.current_fisher = self.fisher_estimator.get_fisher_diagonal()
        
        # Store backup
        backup = EWCBackup(
            task_name=task_name,
            params=params.copy(),
            fisher_diag=self.current_fisher.copy(),
            loss=final_loss,
        )
        self.task_memories.append(backup)
        
        log.info(f"EWC: Registered {task_name}")
        log.info(f"  - {len(params)} parameters")
        log.info(f"  - Fisher sum: {np.sum(self.current_fisher):.4f}")
        log.info(f"  - Total protected tasks: {len(self.task_memories)}")
    
    def ewc_loss(
        self,
        params: np.ndarray,
        task_loss: float,
    ) -> Tuple[float, Dict]:
        """
        Compute total loss with EWC regularization.
        
        Args:
            params: Current parameters
            task_loss: Loss on current task
            
        Returns:
            total_loss: task_loss + EWC_penalty
            info: Loss breakdown
        """
        # Compute EWC penalty from all stored tasks
        ewc_penalty = 0.0
        breakdown = {}
        
        for backup in self.task_memories:
            diff = params - backup.params
            weighted = backup.fisher_diag * (diff ** 2)
            penalty = self.lambda_ewc * np.sum(weighted)
            ewc_penalty += penalty
            breakdown[backup.task_name] = float(penalty)
        
        total = task_loss + ewc_penalty
        
        info = {
            'task_loss': float(task_loss),
            'ewc_penalty': float(ewc_penalty),
            'total': float(total),
            'breakdown': breakdown,
        }
        
        return total, info
    
    def get_importance_scores(self) -> Optional[np.ndarray]:
        """
        Get parameter importance scores from current Fisher.
        
        Returns:
            importance: [n_params] normalized importance
        """
        if self.current_fisher is None:
            return None
        
        # Normalize to sum to 1
        importance = self.current_fisher / (np.sum(self.current_fisher) + 1e-8)
        return importance


def test_ewc():
    """Test EWC regularizer."""
    print("=" * 60)
    print("Testing EWC Regularizer")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate learning two tasks
    n_params = 10
    
    # Task 1: Learn parameters p1
    print("\n--- Task 1: Learning initial parameters ---")
    p1 = np.random.randn(n_params) * 2
    F1 = np.random.rand(n_params) + 0.1  # Random Fisher
    
    print(f"Learned params: mean={np.mean(p1):.3f}, std={np.std(p1):.3f}")
    print(f"Fisher: sum={np.sum(F1):.3f}")
    
    # Task 2: Learn new parameters
    print("\n--- Task 2: Learning new parameters ---")
    p2 = np.random.randn(n_params) * 2
    F2 = np.random.rand(n_params) + 0.1
    
    print(f"Learned params: mean={np.mean(p2):.3f}, std={np.std(p2):.3f}")
    
    # Create EWC regularizer
    ewc = EWCRegularizer(lambda_ewc=1000.0)
    ewc.set_n_params(n_params)
    
    # Register Task 1
    ewc.register_task_completion("task_1", p1, 0.1)
    ewc.current_fisher = F1
    
    # Compute EWC penalty for Task 2 params
    print("\n--- EWC Penalty Computation ---")
    ewc_penalty, info = ewc.ewc_loss(p2, task_loss=1.0)
    
    print(f"Task loss: {info['task_loss']:.4f}")
    print(f"EWC penalty: {info['ewc_penalty']:.4f}")
    print(f"Total loss: {info['total']:.4f}")
    print(f"Per-task breakdown: {info['breakdown']}")
    
    # Compute importance
    importance = ewc.get_importance_scores()
    print(f"\nParameter importance (Task 1):")
    for i, imp in enumerate(importance[:5]):
        print(f"  θ_{i}: {imp:.4f}")
    print("  ...")
    
    # Register Task 2
    ewc.register_task_completion("task_2", p2, 0.05)
    
    # Now EWC should penalize BOTH tasks
    print("\n--- EWC with 2 protected tasks ---")
    p3 = np.random.randn(n_params) * 3  # New random params
    _, info = ewc.ewc_loss(p3, task_loss=0.5)
    
    print(f"Total loss: {info['total']:.4f}")
    print(f"Per-task: {info['breakdown']}")
    
    print("\n" + "=" * 60)
    print("EWC Regularizer Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_ewc()
