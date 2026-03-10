# OxideXLA Architecture

This document describes the internal architecture of OxideXLA using diagrams
and prose. Each section covers one stage of the compiler pipeline.

---

## High-Level Pipeline

The compiler follows a standard frontend-middleend-backend structure.

```mermaid
flowchart LR
    A["ONNX File (.onnx)"] --> B["Parser"]
    B --> C["Graph IR (petgraph DAG)"]
    C --> D["Shape Inference"]
    D --> E["Op Mapper"]
    E --> F["Code Generator"]
    F --> G["JAX Python (.py)"]
```

---

## Detailed Data Flow

This diagram shows every data transformation from input to output,
including the intermediate representations.

```mermaid
flowchart TD
    subgraph Input
        ONNX["ONNX Protobuf File"]
    end

    subgraph Parser ["parser module"]
        PB["Decode Protobuf"]
        EXTRACT["Extract Nodes, Edges, Initializers"]
        PB --> EXTRACT
    end

    subgraph Graph ["graph module"]
        DAG["Build petgraph DAG"]
        TOPO["Topological Sort"]
        SHAPE["Shape Inference Pass"]
        DAG --> TOPO --> SHAPE
    end

    subgraph OpMapper ["ops module"]
        LOOKUP["Match ONNX OpType to JaxOp"]
        VALIDATE["Validate Input Shapes and Dtypes"]
        LOOKUP --> VALIDATE
    end

    subgraph CodeGen ["codegen module"]
        PARAMS["Generate params Dictionary Structure"]
        BODY["Generate Forward Function Body"]
        MODULE["Assemble Complete Python Module"]
        PARAMS --> BODY --> MODULE
    end

    subgraph Output
        PY["output.py"]
    end

    ONNX --> PB
    EXTRACT --> DAG
    SHAPE --> LOOKUP
    VALIDATE --> PARAMS
    MODULE --> PY
```

---

## IR Node Structure

Each node in the computation graph carries these fields:

```mermaid
classDiagram
    class OnnxNode {
        +String id
        +String op_type
        +Vec~String~ inputs
        +Vec~String~ outputs
        +HashMap~String, Attribute~ attributes
    }

    class IrNode {
        +NodeIndex id
        +JaxOp op
        +String name
        +Vec~usize~ output_shape
        +DType dtype
        +Option~String~ param_key
    }

    class JaxOp {
        <<enumeration>>
        MatMul
        Add
        Mul
        Relu
        Softmax
        Reshape
        Transpose
        Conv
    }

    OnnxNode --> IrNode : "op mapper converts"
    IrNode --> JaxOp : "contains"
```

---

## Topological Sort and Execution Order

The graph must be walked in dependency order. A node can only execute
after all its inputs have been computed.

```mermaid
flowchart TD
    INPUT["input (x)"] --> MATMUL["MatMul"]
    W["params.linear0.weight"] --> MATMUL
    MATMUL --> ADD["Add"]
    B["params.linear0.bias"] --> ADD
    ADD --> RELU["Relu"]
    RELU --> OUTPUT["output"]
```

The topological sort produces the execution order:
`[input, weight, bias, matmul, add, relu, output]`

---

## CLI Command Flow

```mermaid
flowchart LR
    subgraph CLI ["oxide_xla CLI"]
        PARSE_ARGS["Parse Arguments (clap)"]
        DISPATCH["Dispatch to Subcommand"]
    end

    subgraph Inspect
        LOAD_I["Load ONNX"]
        PRINT["Print Graph (ASCII or JSON)"]
        LOAD_I --> PRINT
    end

    subgraph Compile
        LOAD_C["Load ONNX"]
        BUILD["Build IR Graph"]
        INFER["Run Shape Inference"]
        MAP["Map Ops"]
        GEN["Generate JAX Code"]
        WRITE["Write .py File"]
        LOAD_C --> BUILD --> INFER --> MAP --> GEN --> WRITE
    end

    PARSE_ARGS --> DISPATCH
    DISPATCH -->|"inspect"| LOAD_I
    DISPATCH -->|"compile"| LOAD_C
```

---

## Stage 3: PyTorch FX Bridge (Future)

```mermaid
sequenceDiagram
    participant User
    participant FXBridge as Python FX Bridge
    participant TorchFX as torch.fx
    participant TorchONNX as torch.onnx
    participant OxideXLA as OxideXLA (Rust)

    User->>FXBridge: provide nn.Module
    FXBridge->>TorchFX: symbolic_trace(model)
    TorchFX-->>FXBridge: FX Graph
    FXBridge->>TorchONNX: export to ONNX
    TorchONNX-->>FXBridge: .onnx bytes
    FXBridge->>OxideXLA: pipe via stdin
    OxideXLA-->>FXBridge: JAX Python code
    FXBridge-->>User: output.py
```
