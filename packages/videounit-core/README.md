# VideoUnit Core SDK

AI video testing framework core SDK for VideoUnit.

## Installation

```bash
pip install videounit-core
```

## Quick Start

```python
from videounit import VideoUnitClient, VideoContract

# Connect to backend
client = VideoUnitClient("http://localhost:8000", api_key=None)

# Load contract from YAML
contract = VideoContract.from_yaml("tests/fixtures/sample_contract.yaml")

# Evaluate video
result = await client.evaluate("path/to/video.mp4", contract)

print(f"Overall score: {result.overall}")
for failure in result.failures:
    print(f"  [{failure.severity}] {failure.message}")
```

## Features

- Pydantic-based data models for video contracts and evaluation results
- Async HTTP client for VideoUnit backend communication
- Contract loading from YAML
- Perception pipeline integration (SAM2, CoTracker3, MiDaS, YOLO)
- ChromaDB knowledge base integration
- LLM client for contract generation
