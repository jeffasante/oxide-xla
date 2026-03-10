use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use oxide_xla::parser::load_onnx_model;
use oxide_xla::graph::IrGraph;
use oxide_xla::codegen::generate_jax_module;

/// OxideXLA -- A high-speed Rust compiler for transforming ONNX graphs
/// into pure, stateless JAX functions.
#[derive(Parser)]
#[command(name = "oxide_xla")]
#[command(version = "0.1.0")]
#[command(about = "Transpile ONNX models into runnable JAX Python code.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse and display the structure of an ONNX model.
    Inspect {
        /// Path to the .onnx file.
        model: PathBuf,

        /// Output format: "ascii" (default) or "json".
        #[arg(long, default_value = "ascii")]
        format: String,
    },

    /// Transpile an ONNX model into a JAX Python file.
    Compile {
        /// Path to the .onnx file.
        model: PathBuf,

        /// Path for the generated .py output file.
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { model, format } => {
            run_inspect(&model, &format)
        }
        Commands::Compile { model, output } => {
            run_compile(&model, &output)
        }
    }
}

/// Load an ONNX model and print its graph structure.
fn run_inspect(model_path: &PathBuf, format: &str) -> Result<()> {
    let onnx_model = load_onnx_model(model_path)?;
    let ir_graph = IrGraph::from_onnx(&onnx_model)?;

    match format {
        "json" => {
            let json = ir_graph.to_json()?;
            println!("{}", json);
        }
        "ascii" | _ => {
            println!("{}", ir_graph.to_ascii());
        }
    }

    Ok(())
}

/// Load an ONNX model, build the IR, and generate JAX Python code.
fn run_compile(model_path: &PathBuf, output_path: &PathBuf) -> Result<()> {
    log::info!("Loading ONNX model from {:?}", model_path);
    let onnx_model = load_onnx_model(model_path)?;

    log::info!("Building IR graph");
    let ir_graph = IrGraph::from_onnx(&onnx_model)?;

    log::info!("Generating JAX code");
    let jax_code = generate_jax_module(&ir_graph)?;

    log::info!("Writing output to {:?}", output_path);
    std::fs::write(output_path, &jax_code)?;

    println!(
        "Compiled {} nodes into {}",
        ir_graph.node_count(),
        output_path.display()
    );

    Ok(())
}
