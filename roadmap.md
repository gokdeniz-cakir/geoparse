# geoparse - Project Roadmap

## The Vision

Build an end-to-end geometry problem solver that takes an image of a YKS (Turkish university entrance exam) geometry problem and outputs the answer.

**Image → Structured Representation → Answer**

---

## Background

### Why YKS Geometry?

- **Abundant data**: Thousands of problems in YKS prep textbooks
- **Constrained domain**: Standardized exam with predictable problem types
- **Clear evaluation**: Right or wrong answer
- **Underexplored**: Everyone works on IMO/AIME, YKS is a gap
- **Domain expertise**: We know Turkish, we know the exam

### Two Eras of Problems

- **Pre-2018**: Pure geometry. "Here's a triangle, find the angle."
- **Post-2018**: World problems. Parse the story, extract the geometry, then solve.

This gives us a natural ablation: test pure geometric reasoning vs language + reasoning.

---

## Architecture

### Stage 1: Diagram Parser (Vision)
**Input**: Image of a geometry diagram  
**Output**: Structured JSON representation, a trivial example:

```json
{
  "vertices": {"A": [0, 0], "B": [4, 0], "C": [2, 3]},
  "segments": [{"from": "A", "to": "B", "length": 5}],
  "angles": [{"vertex": "A", "measure": 60}],
  "circles": [],
  "goal": "find angle C"
}
```

### Stage 2: Solver (Reasoning)
**Input**: Structured JSON  
**Output**: Numerical answer

Options:
- Neural network trained on (structure, answer) pairs
- Symbolic constraint solver
- Evolution-based search (continuous fitness = distance from answer)
- Hybrid approaches

### Stage 3: Integration
Chain Stage 1 → Stage 2. Just plumbing once both pieces work.

---

## Data Strategy

### Real Data
- Source: YKS prep textbooks (PDFs)
- Pipeline: PDF → extract individual problems → (image, text, answer) tuples
- Manual annotation: Create structured representations for a subset

### Synthetic Data
Generate unlimited (image, structure) pairs programmatically:

```python
def generate_training_pair():
    # Generate random geometry
    structure = random_triangle() | random_circle() | ...
    
    # Render as image
    image = render_diagram(structure)
    
    return (image, structure)
```

**Why synthetic works here:**
- Geometry diagrams have constrained visual vocabulary (lines, circles, labels, angle markers)
- Unlike natural images, variation is limited
- Can generate unlimited training data with perfect ground truth

**The transfer question:**
Will synthetic → real generalize? This is a research question worth answering.

---

## Problem Types (YKS Coverage)

Common patterns to support:

1. **Triangles**
   - Right triangles (Pythagorean theorem)
   - Similar triangles
   - Isosceles/equilateral
   - Area calculations

2. **Circles**
   - Inscribed angles
   - Tangent lines
   - Arc lengths
   - Inscribed/circumscribed shapes

3. **Trigonometry**
   - Sin/cos/tan applications
   - Law of sines/cosines
   - Very common (3+ per exam)

4. **Coordinate Geometry**
   - Distance formula
   - Midpoint
   - Line equations

5. **Similarity**
   - Used as a component in many problems
   - Core feature that appears across types

**Note**: YKS problems often combine multiple concepts. A single problem might require similarity + Pythagorean theorem + trig.

---

## Implementation Plan

### Phase 1: Foundation 

**1.1 PDF Extractor**
- Script to pull individual problems from testbook PDFs
- Output: (image, text, answer) for each problem
- This is engineering, not ML

**1.2 Synthetic Diagram Generator**
- Start simple: triangles with labels and side lengths
- Render as PNG, output structure as JSON
- Gradually add: circles, angles, more complex shapes

**1.3 Dataset Structure**
```
data/
├── synthetic/
│   ├── images/
│   └── structures/
├── real/
│   ├── raw/          # extracted from PDFs
│   └── annotated/    # manually labeled subset
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

### Phase 2: Diagram Parser 

**2.1 Model Architecture**
- Vision encoder (pretrained, e.g., ResNet, ViT)
- Output head for structured prediction
- Options: direct JSON prediction, or detect primitives then assemble

**2.2 Training**
- Train on synthetic data
- Validate on held-out synthetic
- Test on real YKS diagrams

**2.3 Iteration**
- Analyze failures on real diagrams
- Fine-tune on manually annotated real data if needed
- Iterate on synthetic generation to close the gap

### Phase 3: Solver 

**3.1 Baseline**
- Simple neural network: structure → answer
- Train on synthetic problems with known solutions

**3.2 Symbolic Approach**
- Constraint satisfaction
- Geometric theorem database
- Step-by-step reasoning

**3.3 Evolution Approach (Optional)**
- Evolve construction sequences
- Fitness = distance from correct answer
- Natural continuous reward signal

**3.4 Comparison**
- Which approach works best on which problem types?
- Where do they fail?

### Phase 4: Generalization (Scale)
**4.1 Advanced Domain Randomization (Sim2Real)**
- "Sketchy" rendering (e.g. `plt.xkcd()`)
- Background noise augmentation (paper textures, scan artifacts)
- Font randomization (handwritten styles)
- Variable line widths/styles

**4.2 Robust Architecture**
- Switch from Global Regression (ResNet -> Coords) to Local Detection (Keypoint R-CNN / Heatmaps)
- Why? Regression overfits to the "mean" style. Detection finds local features ("corners") regardless of style.
- Integration of OCR for reading values (e.g. "30°")

**4.3 Real Data Loop**
- Curate "Golden Set" of real textbook problems
- Use "Pseudo-Labeling": Train on synthetic -> Predict on real -> Filter by Solver Consistency -> Retrain

### Phase 5: Integration & Evaluation 

**5.1 End-to-End Pipeline**
- Image → Parser → Solver → Answer
- Error analysis: where do failures come from?

**5.2 Benchmark Creation**
- Curated test set of real YKS problems
- Categorized by type and difficulty
- Public release: YKS-Geometry benchmark

**5.3 Baselines**
- GPT-4o / Claude on raw images
- GPT-4o / Claude with structured input
- Compare to our pipeline

---

## Research Questions

1. **Synthetic → Real Transfer**: How well does a parser trained on synthetic diagrams generalize to real textbook images?

2. **Bottleneck Analysis**: Where do end-to-end failures come from - vision or reasoning?

3. **Compositional Problems**: Can the solver handle problems requiring multiple concepts?

4. **Pre vs Post 2018**: Do models perform differently on pure geometry vs word problems?

5. **Neural vs Symbolic vs Evolution**: Which solver approach works best, and on what problem types?

---

## Paper Angle

**Option A: Benchmark Paper**
"YKS-Geometry: A Benchmark for Visual Geometric Reasoning"
- Introduce the dataset
- Baseline results from various approaches
- Analysis of what makes problems hard

**Option B: Systems Paper**
"End-to-End Geometry Problem Solving via Synthetic Pre-training"
- Full pipeline description
- Synthetic → real transfer analysis
- Comparison to LLM baselines

**Option C: Analysis Paper**
"Where Do Vision-Language Models Fail on Geometry?"
- Deep dive into failure modes
- Vision vs reasoning decomposition
- Insights for future work

---

## Repo Structure

```
geoparse/
├── README.md
├── generator/           # Synthetic diagram generation
│   ├── primitives.py    # Points, lines, circles, etc.
│   ├── renderer.py      # Structure → image
│   └── problems.py      # Problem templates
├── extractor/           # PDF → problems
│   └── pdf_extract.py
├── parser/              # Diagram → structure (vision model)
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── solver/              # Structure → answer
│   ├── neural.py
│   ├── symbolic.py
│   └── evolution.py     # Optional: evolve solutions
├── pipeline/            # End-to-end
│   └── solve.py
├── data/
│   ├── synthetic/
│   └── real/
├── experiments/
│   └── results/
└── scripts/
    └── benchmark.py
```

---

## Success Criteria

**Minimum Viable Result:**
- Synthetic diagram generator working
- Parser achieves >80% accuracy on synthetic test set
- Some evidence of transfer to real diagrams

**Good Result:**
- Parser works on real YKS diagrams (>60% accuracy)
- Solver works on structured problems (>70% accuracy)
- End-to-end pipeline beats random guessing

**Great Result:**
- End-to-end accuracy competitive with GPT-4o on same problems
- Clear analysis of what works and what doesn't
- Publishable benchmark + results

---

## Open Questions

- How to represent geometry problems? What's the right schema?
- How many manual annotations needed for fine-tuning?
- What vision architecture works best for diagrams?
- How to handle problem text (Turkish language)?
- Should we start with just diagrams, or include text from the start?

---

## Next Steps

1. Create the repo
2. Build synthetic triangle generator (proof of concept)
3. Generate 1000 (image, structure) pairs
4. Train a small model, see if it learns anything
5. Iterate from there

---

*Document created after a productive brainstorming session. Started with evolutionary theorem proving in Lean, learned about fitness shaping and search landscapes, then pivoted to this more tractable problem with clearer path to results.*