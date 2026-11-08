//! Минимальный тест DirectML: грузим dll, создаём DML-сессию, гоняем инференс.

fn main() {
    let ort_dir = std::path::Path::new("ort_runtime");
    let dylib = ort_dir.join("onnxruntime.dll");
    let dml = ort_dir.join("DirectML.dll");

    unsafe {
        let lib = libloading::Library::new(&dml).expect("preload DirectML.dll");
        std::mem::forget(lib);
    }
    std::env::set_var("ORT_DYLIB_PATH", dylib.to_string_lossy().to_string());

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.onnx".to_string());

    let mut session = ort::session::Session::builder()
        .unwrap()
        .with_execution_providers([ort::ep::DirectML::default()
            .with_device_id(0)
            .build()])
        .unwrap()
        .commit_from_file(&path)
        .expect("commit_from_file");

    let planes = 256usize;
    let data = vec![0.5f32; 16 * planes * 196];
    let value = ort::value::Tensor::from_array((
        vec![16i64, planes as i64, 14, 14],
        data,
    ))
    .unwrap();
    let outputs = session
        .run(ort::inputs!["input" => value])
        .expect("run");
    let (shape, pol) = outputs["policy"].try_extract_tensor::<f32>().unwrap();
    let (vshape, val) = outputs["value"].try_extract_tensor::<f32>().unwrap();
    println!(
        "DML inference OK: policy {:?}, value {:?}",
        shape, vshape
    );
    let _ = (pol, val);
}
