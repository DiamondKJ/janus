# JANUS-CORE: A Cognitive Architecture for Dual-Process Reasoning

JANUS-CORE is a neuromorphic cognitive architecture that combines analytical and associative processing with metacognitive oversight. The system uses a dual-hemisphere approach inspired by human cognition, with a resonance actuator that detects cognitive dissonance and triggers higher-order reasoning when needed.

## 🧠 Architecture Overview

JANUS-CORE implements a three-tier cognitive system:

1. **Dual-Process Engines**: Analytical and associative language models that process information in parallel
2. **Resonance Actuator**: A neural network that detects cognitive dissonance between the two processing streams
3. **Meta-Cognitive Module**: A specialized model that synthesizes conflicting thoughts when dissonance is detected

### Cognitive Flow

```
Input Prompt → Limbic Scan → Dual Processing → Dissonance Detection → Response Generation
     ↓              ↓              ↓                ↓                    ↓
Emotional      System         Analytical &    Resonance Actuator    Final Response
Valence      Configuration   Associative     (0-1 metacognitive    (Analytical/
Analysis                     Thoughts        gain score)           Associative/
                                                                    Meta-cognitive)
```

## 📁 Project Structure

```
JANUS-CORE/
├── assets/                          # Pre-trained models
│   ├── engines/
│   │   ├── analytical_engine_v1.0/  # Analytical processing model
│   │   └── associative_engine_v1.0/ # Associative processing model
│   └── modules/
│       ├── meta_module_v1/          # Meta-cognitive synthesis model
│       └── resonance_actuator_v1.pt # Trained dissonance detector
├── datasets/                        # Training data (generated)
├── janus/                          # Core Python package
│   ├── __init__.py
│   ├── architecture.py             # Model definitions
│   ├── harness.py                  # Main cognitive system
│   ├── training/                   # Training utilities
│   │   ├── __init__.py
│   │   └── train_actuator.py       # Resonance actuator trainer
│   └── data_creation/              # Data generation
│       ├── __init__.py
│       ├── generate_curriculum.py  # Prompt curriculum generator
│       └── generate_training_data.py # Dissonance dataset generator
├── reports/                        # Test reports and outputs
├── requirements.txt                # Python dependencies
├── run_janus.py                   # Main entry point
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Google API Key** for Gemini (limbic scanning)
3. **GPU** (optional, for faster processing)

### Installation

1. **Clone and setup**:
```bash
cd JANUS-CORE
pip install -r requirements.txt
```

2. **Set environment variable**:
```bash
export GOOGLE_API_KEY='your-google-api-key-here'
```

3. **Run in test mode**:
```bash
python run_janus.py --test
```

### Usage Examples

#### Interactive Mode
```bash
python run_janus.py --interactive
```

#### Training Pipeline
```bash
# Generate training curriculum
python run_janus.py --generate-curriculum

# Generate dissonance dataset
python run_janus.py --generate-dataset

# Train resonance actuator
python run_janus.py --train-actuator

# Or run the full pipeline
python run_janus.py --full-pipeline
```

#### Custom Assets Path
```bash
python run_janus.py --test --assets-path /path/to/your/assets
```

## 🧪 Training Your Own Models

### 1. Generate Training Data

The system uses a curriculum of prompts to generate dissonance vectors:

```python
from janus.data_creation import generate_prompt_curriculum

# Generate 1000 diverse prompts
curriculum = generate_prompt_curriculum(
    output_path="datasets/prompt_curriculum.jsonl",
    num_prompts=1000
)
```

### 2. Create Dissonance Dataset

Run prompts through both engines to create interference vectors:

```python
from janus.data_creation import generate_dissonance_dataset

# Generate dissonance vectors from curriculum
samples = generate_dissonance_dataset(
    curriculum_path="datasets/prompt_curriculum.jsonl",
    output_path="datasets/dissonance_vectors.pt"
)
```

### 3. Train Resonance Actuator

Train the dissonance detection model:

```python
from janus.training import train_resonance_actuator

# Train the model
model = train_resonance_actuator(
    dataset_path="datasets/dissonance_vectors.pt",
    output_path="assets/modules/resonance_actuator_v1.pt"
)
```

## 🔧 Configuration

### Cognitive Parameters

The system can be configured through the `CognitiveConfig` class:

```python
from janus.architecture import CognitiveConfig

config = CognitiveConfig(
    temperature=0.7,           # Generation temperature
    conflict_threshold=0.5,    # Dissonance threshold for meta-cognition
    force_strategy=None,       # Force specific strategy ('analytical', 'associative')
    max_tokens=128            # Maximum response length
)
```

### Model Paths

Update model paths in the harness:

```python
from janus.harness import JanusHarness

harness = JanusHarness(assets_path="path/to/your/assets")
```

## 📊 Understanding the Output

### Test Reports

Test runs generate detailed reports in `reports/`:

```
JANUS-CORE TEST REPORT - SESSION: 2024-01-15 14:30:25
================================================================================

--- Test ---
PROMPT: I need the fastest route to the hospital, now!
VALENCE: {'analytical_focus': 0.9, 'creative_focus': 0.1, 'urgency': 0.95, 'curiosity': 0.2}
STRATEGY: Forced Analytical (High Urgency)
RESPONSE: [Generated response...]
```

### Cognitive Strategies

The system can employ three strategies:

1. **Forced Analytical**: High urgency prompts trigger direct analytical processing
2. **Corpus Callosum Synthesis**: Low dissonance prompts use associative processing
3. **Meta-Cognitive Override**: High dissonance triggers meta-cognitive synthesis

## 🧠 Technical Details

### Resonance Actuator Architecture

```python
ResonanceActuator(
    input_dim=384,      # Sentence transformer embedding dimension
    hidden_dim_1=128,   # First hidden layer
    hidden_dim_2=64,    # Second hidden layer
    output_dim=1        # Metacognitive gain score (0-1)
)
```

### Dissonance Detection

The system computes interference vectors between analytical and associative responses:

```python
interference_vector = |analytical_embedding - associative_embedding|
metacognitive_gain = actuator_model(interference_vector)
```

### Limbic Scanning

Emotional valence analysis using Google's Gemini model:

- **analytical_focus**: Demand for logic and facts
- **creative_focus**: Demand for imagination and metaphor
- **urgency**: Need for fast, actionable answers
- **curiosity**: Invitation for open-ended exploration

## 🔬 Research Applications

JANUS-CORE is designed for research in:

- **Cognitive Architecture**: Dual-process reasoning systems
- **Meta-Cognition**: Higher-order thinking and self-reflection
- **Neural-Symbolic Integration**: Combining neural networks with symbolic reasoning
- **Adversarial Robustness**: Handling conflicting information sources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by dual-process theories of human cognition
- Built on the shoulders of the transformer architecture
- Leverages Google's Gemini for emotional intelligence

## 📞 Support

For questions, issues, or contributions, please open an issue on the repository.

---

**JANUS-CORE**: Where analytical precision meets associative creativity, guided by metacognitive wisdom. 