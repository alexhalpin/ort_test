use ort::inputs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ort::init().commit()?;
    let model = ort::Session::builder()?
        .commit_from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("rf_iris.onnx"))?; // replace with your own model

    let model_input: Vec<f32> = vec![4.4, 3.2, 1.3, 0.2, 6.7, 2.5, 5.8, 1.8];
    let outputs = model.run(inputs!["X" => ([2usize, 4usize], model_input) ]?)?;

    let output = outputs["output_label"].try_extract_tensor::<i64>()?;

    print!("output: {}", &output);
    Ok(())
}
