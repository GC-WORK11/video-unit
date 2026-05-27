"""VideoUnit Self-Healing Pipeline.

This package provides self-healing capabilities for the VideoUnit AI video
testing framework. It analyzes evaluation failures, diagnoses root causes,
and generates fix proposals to improve test contracts.

Example usage:

    from videounit.healer import SelfHealingPipeline, PipelineConfig
    from videounit.healer.diagnosis import create_default_diagnosis_engine
    from videounit.healer.fixing import create_default_fix_generators

    config = PipelineConfig(max_iterations=3)
    pipeline = SelfHealingPipeline(config=config)

    report = await pipeline.heal(evaluation_result, contract, context)

    print(f"Diagnoses: {len(report.diagnoses)}")
    print(f"Fix proposals: {len(report.fix_proposals)}")
    print(f"Converged: {report.converged}")
"""

from videounit.healer._base import (
    DiagnosisEngine,
    DiagnosisResult,
    FailureAnalyzer,
    FixGenerator,
    FixProposal,
    FixProposalGenerator,
    RepairStrategy,
    SelfHealingReport,
)
from videounit.healer.pipeline import (
    PipelineConfig,
    SelfHealingPipeline,
    create_pipeline,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DiagnosisEngine",
    "DiagnosisResult",
    "FailureAnalyzer",
    "FixGenerator",
    "FixProposal",
    "FixProposalGenerator",
    "PipelineConfig",
    "RepairStrategy",
    "SelfHealingPipeline",
    "SelfHealingReport",
    "create_pipeline",
]
