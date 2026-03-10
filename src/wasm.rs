use wasm_bindgen::prelude::*;
use crate::parser::load_onnx_from_bytes;
use crate::graph::IrGraph;
use crate::codegen::generate_jax_module;

#[wasm_bindgen]
pub struct CompilationResult {
    code: String,
    json: String,
}

#[wasm_bindgen]
impl CompilationResult {
    #[wasm_bindgen(getter)]
    pub fn code(&self) -> String {
        self.code.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn json(&self) -> String {
        self.json.clone()
    }
}

#[wasm_bindgen]
pub fn transpile_onnx(data: &[u8]) -> Result<CompilationResult, JsValue> {
    let onnx_model = load_onnx_from_bytes(data)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse ONNX: {:?}", e)))?;
        
    let ir_graph = IrGraph::from_onnx(&onnx_model)
        .map_err(|e| JsValue::from_str(&format!("Failed to build IR: {:?}", e)))?;
        
    let jax_code = generate_jax_module(&ir_graph)
        .map_err(|e| JsValue::from_str(&format!("Failed to generate JAX: {:?}", e)))?;
        
    let json_graph = ir_graph.to_json()
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize graph: {:?}", e)))?;

    Ok(CompilationResult {
        code: jax_code,
        json: json_graph,
    })
}
